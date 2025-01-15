from collections import defaultdict
import functools
import inspect
import logging
from typing import Dict, List, Tuple, overload
import torch
import torch.nn as nn
from awq.models.base import BaseAWQForCausalLM
from tqdm import tqdm, trange

from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from awq.modules.fused.norm import FasterTransformerRMSNorm
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    get_op_by_name, 
    set_op_by_name
)
from awq.modules.act import ScaledActivation
from awq.quantize.scale import scale_fc_fc, scale_fc_fcs, scale_gelu_fc, scale_ln_fcs
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer as OldQwen2DecoderLayer,
    Qwen2RMSNorm,
)
from llava.model.builder import load_pretrained_model

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

allowed_norms = [nn.LayerNorm, Qwen2RMSNorm,]
allowed_act_fns = [
    nn.GELU
]



class llava_onevision(BaseAWQForCausalLM):
    layer_type = "QwenDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @classmethod
    def from_pretrained(cls, model_path, model_type, **kwargs):
        print(f"Loading LLAVA-OneVision model from {model_path}")

        llava_model_args = {
            "multimodal": True,
            "attn_implementation": kwargs.get("attn_implementation", "eager"),
            "device_map": kwargs.get("device_map", "auto"),
            "torch_dtype": kwargs.get("torch_dtype", "bfloat16"),
        }

        customized_config = kwargs.get("customized_config", None)
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config

        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        token_strategy = kwargs.get("token_strategy", "single")
        max_frames_num = kwargs.get("max_frames_num", 32)
        mm_spatial_pool_stride = kwargs.get("mm_spatial_pool_stride", 2)
        mm_spatial_pool_mode = kwargs.get("mm_spatial_pool_mode", "bilinear")
        video_decode_backend = kwargs.get("video_decode_backend", "decord")
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")


        overwrite_config = {
            "mm_spatial_pool_stride": mm_spatial_pool_stride,
            "mm_spatial_pool_mode": mm_spatial_pool_mode,
        }


        from transformers import AutoConfig
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        llava_model_args["overwrite_config"] = overwrite_config
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path, 
            None,  
            kwargs.get("model_name", 'qwen'), 
            **llava_model_args
        )

        model.eval()
        model.to(device)
        model.config.use_cache = kwargs.get("use_cache", False)

        cls.pretrained = model_path
        cls.token_strategy = token_strategy
        cls.max_frames_num = max_frames_num
        cls.mm_spatial_pool_stride = mm_spatial_pool_stride
        cls.mm_spatial_pool_mode = mm_spatial_pool_mode
        cls.video_decode_backend = video_decode_backend
        cls.device = device
        cls.tokenizer = tokenizer
        cls.model = model
        cls.image_processor = image_processor
        cls.max_length = max_length

        return cls

    @classmethod
    def fuse_layers(model: Qwen2ForCausalLM):
        pass
    
    @staticmethod
    def get_model_layers(model: LlavaQwenForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: OldQwen2DecoderLayer):
        return dict(is_scalable=False)
    
    @staticmethod
    def move_embed(model: LlavaQwenForCausalLM, device: str):
        model.model.embed_tokens = model.get_input_embeddings().to(
            device
        )
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        
    @staticmethod
    def get_layers_for_scaling(module: OldQwen2DecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:  #512 3584 & 3584 3584 ==>GQA skip o_proj
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers
 

class LlavaOVFuser:
    def __init__(self, model: LlavaQwenForCausalLM):
        self.model = model.model

        self.qwen2_blocks: List[Tuple[str, OldQwen2DecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Qwen2DecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldQwen2DecoderLayer
        for module in tqdm(self.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon,
            )
            if hasattr(self.model.config, "max_seq_len"):
                max_seq_len = self.model.config.max_seq_len
            else:
                max_seq_len = self.model.config.max_position_embeddings
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)



class LLavaOVAwqQuantizer(AwqQuantizer):
    def __init__(self, 
                 parent,                 
                 model, 
                 tokenizer, 
                 w_bit, 
                 q_group_size, 
                 zero_point, 
                 version, 
                 calib_data, 
                 split, 
                 text_column, 
                 duo_scaling, 
                 modules_to_not_convert=None, 
                 export_compatible=False, 
                 apply_clip=False, 
                 n_parallel_calib_samples=1, 
                 max_calib_samples=None, 
                 max_calib_seq_len=None, 
                 max_chunk_memory=None, 
                 **kwargs):

        awq_model = llava_onevision
        super().__init__(awq_model, 
                         model, 
                         tokenizer, 
                         calib_data=calib_data, 
                         w_bit=w_bit, 
                         group_size=q_group_size, 
                         zero_point = zero_point, 
                         version=version, 
                         split=split, 
                         text_column=text_column,
                         duo_scaling=duo_scaling,
                         **kwargs)

    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)  # 28 x qwen2decoderlayer
        samples = self.calib_data

        inps = []
        layer_kwargs =[]

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)
                ##### pop past_key_values!!!
                kwargs.pop("past_key_value", None)
                inps.append(hidden_states)
                layer_kwargs.append(kwargs)
                raise ValueError  

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        modules[0] = Catcher(modules[0])
        for sample in samples:
            for k, v in sample.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    sample[k] = move_to_device(v, best_device)
            try:
                self.model(**sample)
            except ValueError:  
                pass
            finally:
                for k, v in sample.items():
                    if isinstance(v, (torch.Tensor, nn.Module)):
                        sample[k] = move_to_device(v, "cpu")
        
              # restore

            del sample
        modules[0] = modules[0].module
        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps
    
    def quantize(self):
        blocks = len(self.modules)
        for i in trange(blocks, desc="AWQ"):
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            
            self.awq_model.move_embed(self.model, common_device)


            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)   # output after module forward
            clear_memory()
            ### key:8('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'mlp.o_proj')
            ### value per key: n_samples(list[torch.Tensor])
            

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()
                  
    def _get_input_feat(self, layer, named_linears):
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        if  "mixtral" in self.awq_model.layer_type:
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if "deepseek_v2" in self.awq_model.layer_type:
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
            
        device = next(layer.parameters()).device
        self.inps = [inp.to(device) for inp in self.inps]# in case multi-gpu
       

        self.inps = self._module_forward(self.inps, layer, self.module_kwargs)
        for h in handles:
            h.remove()

        return input_feat
    
    @torch.no_grad()
    def _module_forward(
        self, x, module: torch.nn.Module, module_kwargs: Dict
    ): # module: qkvo mlp
        
        module_output = []
        partitioned_inputs = x  
        for x_partial, kwargs in zip(partitioned_inputs, module_kwargs):
            partial_output = module(x_partial, **kwargs)

            if isinstance(partial_output, tuple):
                partial_output = partial_output[0]

            module_output.append(partial_output.cpu())

        return module_output
    
    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: List[torch.Tensor],
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")
        device = next(module2inspect.parameters()).device
        inp = [_.to(device) for _ in inp]

    
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        weight = weight.view(-1, self.group_size)
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        w_scale = w_scale.view(org_shape)
        w_mean = w_scale.mean(0)
        clear_memory(weight)
        
        hidden_dim = inp[0].shape[-1]
        total_num_elements = sum(t.shape[1] for t in inp)
        num_channels = hidden_dim
        
        element_size_bytes = inp[0].element_size() * 2   # multiplied by 2 for fp32, original is bf16
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, max(1, total_num_elements))

        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp[0].device)
        
        processed_elements = 0 
        for input in inp:
            inp_flat = input.cpu().abs().view(-1, hidden_dim)
            local_num_elements = inp_flat.size(0)
            for start_idx in range(0, local_num_elements, chunk_size):
                end_idx = min(start_idx + chunk_size, local_num_elements)
                chunk_sum = inp_flat[start_idx:end_idx].to(torch.float32).sum(dim=0)
                x_sum += chunk_sum.to(device)
                clear_memory(x_sum)
                
            clear_memory(inp_flat)
    
            processed_elements += local_num_elements

        x_mean = (x_sum / float(processed_elements)).to(dtype=inp[0].dtype)
        clear_memory(x_sum)

        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            bf16_output = self._module_forward(inp, module2inspect, module_kwargs)

        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, bf16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )
        
    def _sanitize_kwargs(self, inputs_kwargs, module):

        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
    
        new_list = []
        for single_dict in inputs_kwargs:
            filtered_dict = {}
            for k, v in single_dict.items():
                if k in module_signature:
                    filtered_dict[k] = v
            new_list.append(filtered_dict)
        return new_list

    def _compute_best_scale(
        self,
        x: List[torch.Tensor],
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: List[torch.Tensor],
        kwargs: List[Dict],
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}  # for recovery

        device = x[0].device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            ratio = ratio / n_grid

            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            int_w_output = self._module_forward(x, module2inspect, kwargs)
            
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()
    
    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output_list: list[torch.Tensor],
        int_w_output_list: list[torch.Tensor],
        device: torch.device,
    ):
        total_loss = 0.0
        total_num_elements = 0
        for fp16_output, int_w_output in zip(fp16_output_list, int_w_output_list):
            fp16_output_flat = fp16_output.view(-1)
            int_w_output_flat = int_w_output.view(-1)
            num_elements = fp16_output_flat.size(0)
            element_size_bytes = fp16_output.element_size()

        
            chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
            chunk_size = min(chunk_size, num_elements)

            fp16_chunks = torch.split(fp16_output_flat, chunk_size)
            int_w_chunks = torch.split(int_w_output_flat, chunk_size)

            local_sum_sq = 0.0
            for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
                diff = fp16_chunk.to(device).float() - int_w_chunk.to(device).float()
                local_sum_sq += diff.pow(2).sum().item()
            total_loss += local_sum_sq
            total_num_elements += num_elements

        final_loss = total_loss / total_num_elements if total_num_elements > 0 else 0.0
        return final_loss
    
    
    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: List[torch.Tensor],
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        ci = org_w_shape[1]
        group_size = self.group_size if self.group_size > 0 else ci
        
        processed_feats = []
        for feat in input_feat:
            seq_i = feat.shape[1]
            feat_2d = feat.view(seq_i, ci)
            step_size = max(1, seq_i // n_sample_token)
            feat_4d = feat_2d.reshape(1, seq_i, -1, group_size)
            feat_4d = feat_4d[:, ::step_size]
            processed_feats.append(feat_4d)
            
            
        merged_input_feat = torch.cat(processed_feats, dim=1)
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            merged_input_feat = merged_input_feat.to(w.device)
            org_out = (merged_input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (merged_input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)
    
def apply_scale(module, scales_list, input_feat_dict=None):
        for prev_op_name, layer_names, scales in scales_list:
            prev_op = get_op_by_name(module, prev_op_name)
            layers = [get_op_by_name(module, name) for name in layer_names]

            best_device = get_best_device()
            prev_op.to(best_device)
            for layer in layers:
                layer.to(best_device)
            scales.to(best_device)

            if (
                isinstance(prev_op, nn.Linear)
                and type(layers) == list
                and isinstance(layers[0], nn.Linear)
            ):
                scale_fc_fcs(prev_op, layers, scales)

            elif isinstance(prev_op, nn.Linear):
                assert len(layers) == 1
                scale_fc_fc(prev_op, layers[0], scales)

            elif (
                any(isinstance(prev_op, t) for t in allowed_norms)
                or "rmsnorm" in str(prev_op.__class__).lower()
            ):
                scale_ln_fcs(prev_op, layers, scales)

            elif any(isinstance(prev_op, t) for t in allowed_act_fns):
                new_module = ScaledActivation(prev_op, scales)
                set_op_by_name(module, prev_op_name, new_module)
                scale_gelu_fc(prev_op, layers[0], scales)

            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

            if input_feat_dict is not None:
                for layer_name in layer_names:
                    if layer_name in input_feat_dict:
                        inp = input_feat_dict[layer_name]
                        inp = [inputs.div_(scales.view(1, -1)).to(inputs.device) for inputs in inp]

            prev_op.cpu()
            for layer in layers:
                layer.cpu()
            scales.cpu()
            
@torch.no_grad()
def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
    for name, max_val in clip_list:
        layer: nn.Linear = get_op_by_name(module, name)
        layer.to(get_best_device())
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()
        
        