import argparse
import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import os
import sys
import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

from pruner.base_pruner import LayerSparsity, BasePruner, LayerWiseBasePruner
from pruner.utils import loss_language, get_module_recursive, find_layers, WrappedGPT, prepare_sample
from pruner.build_dataset import LocalLlavaDataset, llava_collate_fn
from pruner.processing_llava import LlavaProcessor
torch.backends.cuda.matmul.allow_tf32 = True

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
from llava.model.builder import load_pretrained_model


class LLavaLLMPruner(LayerWiseBasePruner):
    

    def __init__(self,
                 sparsity_ratio=0.2,
                 max_sparsity_per_layer=0.5,
                 num_samples=4,
                 num_noise=1,
                 noise_eps=1e-3,
                 pretrained: str = "/share/home/mhma/models/llava-onevision-qwen2-7b-ov",
                 truncation: Optional[bool] = True,
                 device: Optional[str] = "cuda:0",
                 batch_size: Optional[Union[int, str]] = 1,
                 model_name: Optional[str] = 'llava_qwen',
                 attn_implementation: Optional[str] = 'sdpa',
                 device_map: Optional[str] = "auto",
                 conv_template: Optional[str] = "qwen_1_5",
                 use_cache: Optional[bool] = True,
                 truncate_context: Optional[bool] = False,
                 customized_config: Optional[str] = None,
                 max_frames_num: Optional[int] = 32,
                 mm_spatial_pool_stride: Optional[int] = 2,
                 mm_spatial_pool_mode: Optional[str] = "bilinear",
                 token_strategy: Optional[str] = "single",
                 video_decode_backend: str = "decord",
                 method: Optional[str] = None,
                 json_path: Optional[str] = None,
                 images_dir: Optional[str] = None,
                 prune_spec=None,
                 importance_scores_cache=None,
                 keep_indices_or_masks_cache=None,
                 is_strct_pruning=False,
                 is_global=False,
                 model_prefix="model",
                 sparsity_ratio_granularity='block',
                 score_method="MEZO-GradOnly_avg",
                 num_data_first_stage=4,
                 sparsity_dict=None,
                 prune_per_model=False,
                 prune_n=0,
                 prune_m=1,
                 pruner_name = "llava_ecoflap_wanda_pruner",
                 **kwargs,
                 ):
        super().__init__(
            model=None,  # Initialize with None, will be loaded later
            data_loader=None,  # Initialize with None, will be created later
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=None,
            pruner_name=pruner_name,
        )

        llava_model_args = {
            "multimodal": True,
            "attn_implementation": attn_implementation,
            "device_map": device_map,
            "torch_dtype": kwargs.get("torch_dtype", "bfloat16") 
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend
        self.device = device

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args)

        self._model.eval()
        self._model.to(device)
        self._model.config.use_cache = False # Disable cache for calibration

        processor = LlavaProcessor.from_pretrained(pretrained)
        dataset = LocalLlavaDataset(
            json_path=json_path,
            images_dir=images_dir,
            processor=processor,
            max_length=2048
        )

        self.data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: llava_collate_fn(x, self._image_processor, self._model, self._tokenizer, device=device),
        )

        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.loss_func = loss_language
        self.pruner_name = pruner_name

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    def check_sparsity(self, model, module_to_process="model.layers"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

        layers = get_module_recursive(model, module_to_process)
        count = 0
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W == 0).sum().item()
                total_params += W.numel()

                sub_count += (W == 0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        model.config.use_cache = use_cache
        return float(count)/total_params

    def forward_to_cache(self, model, batch):
        return model(**batch)

    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="model.layers"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

        layers = get_module_recursive(model, module_to_process)

        inps = []
        caches = []
        keys_to_cache = ["attention_mask","input_ids","position_ids","past_key_values"]

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):

                inps.append(inp)
                inps[-1].requires_grad = False

                cache = {}
                for k in keys_to_cache:
                    cache[k] = kwargs.get(k)
                caches.append(cache)  ####attn mask is not equal on last two dims   @mhma check this 0110

                raise ValueError
        
        layers[0] = Catcher(layers[0])
        total_samples = 0
        for i, batch in enumerate(dataloader):
            if total_samples >= n_samples:
                break
            total_samples += batch["input_ids"].shape[0]
            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                pass
        
        if any(torch.equal(t1, t2) for i, t1 in enumerate(inps) for t2 in inps[i+1:]):  # similarity is very high @mhma
            raise ValueError("Duplicate inputs found in calibration data")
        
        
        layers[0] = layers[0].module
        outs = [None] * len(inps)

        model.config.use_cache = use_cache

        return inps, outs, caches

    def _prune(self, model, dataloader, device, model_prefix="model", module_to_process="model.layers", n_samples=64, sparsity_ratio=0.5):
        use_cache = model.config.use_cache
        model.config.use_cache = False

        print("loading calibration data")
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))

        layers = get_module_recursive(model, module_to_process)
        # ModuleList(
        # (0-27): 28 x Qwen2DecoderLayer(
        #     (self_attn): Qwen2Attention(
        #     (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
        #     (k_proj): Linear(in_features=3584, out_features=512, bias=True)
        #     (v_proj): Linear(in_features=3584, out_features=512, bias=True)
        #     (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
        #     (rotary_emb): Qwen2RotaryEmbedding()
        #     )
        #     (mlp): Qwen2MLP(
        #     (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
        #     (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
        #     (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
        #     (act_fn): SiLU()
        #     )
        #     (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        #     (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        # )
        # )

        
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)  # qkvo gate up down    7 subset per layer

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], 
                                        attention_mask=caches[j]['attention_mask'], 
                                        position_ids = caches[j]['position_ids'])[0]
            for h in handles:  # remove hooks @mhma    check this done.
                h.remove()

            for name in subset:
                assert wrapped_layers[name].nsamples == len(inps)
                print(f"pruning layer {i} name {name}")
                
                
                # wanda metric
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                #### can also support structured pruning just change one line: W_metric = W_metric.sum(dim=0) @mhma

                W_mask = (torch.zeros_like(W_metric) == 1)
                if self.prune_n != 0: 
                    # structured n:m sparsity pruning
                    if self.pruner_name == "n:m structured pruning":
                        for ii in range(W_metric.shape[1]):
                            if ii % self.prune_m == 0:
                                tmp = W_metric[:,ii:(ii+self.prune_m)].float()
                                W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                else:  # unstructured pruning
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                    
                    indices = sort_res[1][:,:int(W_metric.shape[1] * sparsity_ratio.get(sparsity_key, self.max_sparsity_per_layer))]
                    W_mask.scatter_(1, indices, True)
                    
                    

                subset[name].weight.data[W_mask] = 0

            for j in range(n_samples):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], 
                                        attention_mask=caches[j]['attention_mask'],
                                        position_ids = caches[j]['position_ids'])[0]
            inps, outs = outs, inps

        model.config.use_cache = use_cache
        torch.cuda.empty_cache()

        return model

    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.safe_load(f)

        if sparsity_ratio_granularity is None:
            layer_to_group_mapping = {}

        else:
            def check(name, v):
                return len(v.shape) == 2 and ".layers" in name and "relative_attention_bias.weight" not in name and name.startswith(self.model_prefix+'.layers')

            parameters_to_prune = [k for k, v in self._model.named_parameters() if check(k, v)]

            if sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {k: k for k in parameters_to_prune}
            elif sparsity_ratio_granularity == "block":
                layer_to_group_mapping = {k: ".".join(k.split(".")[:3]) for k in parameters_to_prune}
            else:
                raise NotImplementedError

        sparsity_module = LayerSparsity(
            self._model,
            self.data_loader,
            self.pruner_name,
            loss_language,
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            prune_per_model=self.prune_per_model,
        )

        return sparsity_module.return_sparsity()

    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self._model)

        if self.prune_spec is None:
            return self._model, None

        # _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)

        # sparsity_ratio = 1 - keep_ratio[0]
        
        sparsity_ratio = self.sparsity_ratio

        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity   # maybe have bugs, fix later @mhma    check this done.
        )

        self._model = self._prune(
            self._model, self.data_loader, device,
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.layers",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )

        self.model_reset(self._model, dtype_record, requires_grad_record, device)

        return self._model, sparsity_dict

    def model_setup_and_record_attributes(self, model):
        dtype_record = {}
        requires_grad_record = {}
        device = next(model.parameters()).device
        for n, p in model.named_parameters():
            dtype_record[n] = p.dtype
            requires_grad_record[n] = p.requires_grad
        return dtype_record, requires_grad_record, device

    def convert_spec_to_list(self, prune_spec):
        if isinstance(prune_spec, str):
            num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = prune_spec.split("-")
            return int(num_layers), float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)
        elif isinstance(prune_spec, list):
            module_names = [item[0] for item in prune_spec]
            keep_ratios = [item[1] for item in prune_spec]
            prune_nums = [item[2] for item in prune_spec] if len(prune_spec[0]) > 2 else [0] * len(prune_spec)
            prune_ms = [item[3] for item in prune_spec] if len(prune_spec[0]) > 3 else [1] * len(prune_spec)
            return module_names, keep_ratios, prune_nums, prune_ms
        else:
            raise ValueError(f"Unsupported prune_spec format: {type(prune_spec)}")

    def model_reset(self, model, dtype_record, requires_grad_record, device):
        for n, p in model.named_parameters():
            if n in dtype_record:
                p.data = p.data.to(dtype_record[n]).to(device)
            if n in requires_grad_record:
                p.requires_grad = requires_grad_record[n]




import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prune LLaVA-based model.")

    parser.add_argument(
        "--pretrained_model_path", 
        type=str, 
        default="/share/home/mhma/models/llava-onevision-qwen2-7b-ov",
        help="Path to the pretrained model."
    )
    parser.add_argument(
        "--json_path", 
        type=str, 
        default="/share/home/mhma/MLLM-Efficiency/datasets/coco/caption.json",
        help="Path to the dataset JSON."
    )
    parser.add_argument(
        "--images_dir", 
        type=str, 
        default="/share/home/mhma/MLLM-Efficiency/datasets/coco/images",
        help="Path to the images directory."
    )
    parser.add_argument(
        "--result_folder", 
        type=str, 
        default="/share/home/mhma/MLLM-Efficiency/models/llava-onevision-qwen2-7b",
        help="Folder to save pruned model."
    )

    parser.add_argument(
        "--sparsity_ratio_granularity",
        type=str,
        default=None,
        help="Pruning method granularity."
    )
    
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.5,
        help="Global ratio of weights to be pruned."
    )
    parser.add_argument(
        "--max_sparsity_per_layer",
        type=float,
        default=0.9,
        help="Per-layer maximum pruning ratio."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of calibration samples to use for pruning."
    )
    parser.add_argument(
        "--num_data_first_stage",
        type=int,
        default=1,
        help="Number of samples used in the first stage of sparsity computation."
    )
    parser.add_argument(
        "--pruner_name",
        default="llava_wanda_pruner",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    
    args = parse_args()

    pretrained_model_path = args.pretrained_model_path
    json_path = args.json_path
    images_dir = args.images_dir
    result_floader = args.result_folder
    sparsity_ratio_granularity = args.sparsity_ratio_granularity
    pruner_name = args.pruner_name
    llava_pruner = LLavaLLMPruner(
        pretrained=pretrained_model_path,
        json_path=json_path,
        images_dir=images_dir,
        sparsity_ratio=args.sparsity_ratio,
        max_sparsity_per_layer=args.max_sparsity_per_layer,
        num_samples=args.num_samples,
        num_data_first_stage=args.num_data_first_stage,
        # sparsity_ratio_granularity = args.sparsity_ratio_granularity,
        prune_spec=[("model.layers", 0.5)],  # not used @mhma 0112
        device="cuda:0",
        attn_implementation="flash_attention_2",
        pruner_name=pruner_name,
    )
    dir_path = f"{result_floader}_{llava_pruner.pruner_name}_{llava_pruner.sparsity_ratio}"
    pruned_model, sparsity_dict = llava_pruner.prune()

    pruned_model.save_pretrained(dir_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    tokenizer.save_pretrained(dir_path)
    
    print("Finished pruning.")
    sparsity = llava_pruner.check_sparsity(pruned_model)
    print(f"Global sparsity: {sparsity:.4f}")
    
    
    