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
from pruner.utils import (
    loss_language, 
    get_module_recursive, 
    find_layers, 
    prepare_sample   
)
from pruner.build_dataset import LocalLlavaDataset, llava_collate_fn
from pruner.processing_llava import LlavaProcessor

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


class SparseGPT:
    
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device

        W = layer.weight.data.clone().float()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:  
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t()

        self.H *= (self.nsamples / (self.nsamples + tmp))
        self.nsamples += tmp
        scale = math.sqrt(2.0 / self.nsamples)
        inp = inp.float() * scale
        self.H += inp @ inp.t()  
 
    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        
        W = self.layer.weight.data.clone().float()    # 512 3584

        H = self.H    # 3584  3584
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)   # 512
        
        
        import time
        torch.cuda.synchronize()
        start = time.time()

        Hinv = torch.linalg.pinv(H)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"Time to compute Hinv: {end - start:.6f}s")
        # if (torch.isinf(H) * (H > 0)).float().sum() > 0:
        #     pos = torch.isinf(H) * (H > 0)
        #     H[pos] = torch.quantile(H, 0.999)
            
        # if (torch.isinf(H) * (H < 0)).float().sum() > 0:
        #     pos = torch.isinf(H) * (H < 0)
        #     H[pos] = torch.quantile(H, 0.001)
            
        # damp = percdamp * torch.mean(torch.diag(H))
        # diag = torch.arange(self.columns, device=self.dev)
        
        # while True:
        #     try:
        #         decompose_H = torch.linalg.cholesky(H)
                
        #         if not torch.isnan(decompose_H).any():
        #             H = decompose_H
        #             break
                
        #         if torch.isinf(damp).any():
        #             import pdb; pdb.set_trace()      
        #         H[diag, diag] += damp
        #     except:
        #         H[diag, diag] += damp

        # H = torch.cholesky_inverse(H)
        
        # if (torch.isinf(H) * (H > 0)).float().sum() > 0:
        #     pos = torch.isinf(H) * (H > 0)
        #     H[pos] = torch.quantile(H, 0.999)
            
        # if (torch.isinf(H) * (H < 0)).float().sum() > 0:
        #     pos = torch.isinf(H) * (H < 0)
        #     H[pos] = torch.quantile(H, 0.001)
            
        # damp = percdamp * torch.mean(torch.diag(H).abs())
        # diag = torch.arange(self.columns, device=self.dev)
        
        # while True:
        #     try:
        #         decompose_H = torch.linalg.cholesky(H, upper=True)
                
        #         if not torch.isnan(decompose_H).any():
        #             H = decompose_H
        #             break
        #         H[diag, diag] += damp
        #     except:
        #         H[diag, diag] += damp

        # Hinv = H
        
        
        s = W ** 2 / (torch.diag(Hinv).reshape((1, -1))) ** 2
        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
    def free(self):
        self.H = None
        torch.cuda.empty_cache()

class LLavaLLMPrunerSparseGPT(LayerWiseBasePruner):

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
                 pruner_name="llava_sparsegpt_pruner",
                 **kwargs,
                 ):
        super().__init__(
            model=None,  
            data_loader=None,
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
            prune_per_model=prune_per_model,
        )

        # load LLaVA model
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

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained, None, model_name, **llava_model_args
        )
        self._model.eval()
        self._model.to(device)
        self._model.config.use_cache = False  # disable cache for calibration

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
            collate_fn=lambda x: llava_collate_fn(
                x, 
                self._image_processor, 
                self._model, 
                self._tokenizer, 
                device=device
            ),
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
                caches.append(cache)
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
        
        layers[0] = layers[0].module
        outs = [None] * len(inps)
        model.config.use_cache = use_cache
        return inps, outs, caches

    def _prune(self, model, dataloader, device, model_prefix="model", module_to_process="model.layers", n_samples=64, sparsity_ratio=0.5):
        use_cache = model.config.use_cache
        model.config.use_cache = False

        print("loading calibration data")
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(
                model, dataloader, device, model_prefix, n_samples, module_to_process
            )

        n_samples = min(n_samples, len(inps))
        layers = get_module_recursive(model, module_to_process)

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = SparseGPT(subset[name])

            def add_batch_func(name):
                def tmp_hook(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp_hook

            hooks = []
            for name in wrapped_layers:
                h = subset[name].register_forward_hook(add_batch_func(name))
                hooks.append(h)


            for j in range(n_samples):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outs[j] = layer(
                            inps[j], 
                            attention_mask=caches[j]['attention_mask'],
                            position_ids=caches[j]['position_ids']
                        )[0]

            for h in hooks:
                h.remove()


            for name in subset:
                print(f"pruning layer {i} name {name}")
                s_ratio = sparsity_ratio
                if isinstance(sparsity_ratio, dict):
                    key = f"{module_to_process}.{i}.{name}.weight"
                    s_ratio = sparsity_ratio.get(key, self.max_sparsity_per_layer)

                wrapped_layers[name].fasterprune(
                    s_ratio, 
                    prune_n=self.prune_n,
                    prune_m=self.prune_m,
                    blocksize=128,
                    percdamp=0.01
                )
                wrapped_layers[name].free()

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
                return (len(v.shape) == 2 
                        and ".layers" in name 
                        and "relative_attention_bias.weight" not in name 
                        and name.startswith(self.model_prefix+'.layers'))

            parameters_to_prune = [
                k for k, v in self._model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {k: k for k in parameters_to_prune}
            elif sparsity_ratio_granularity == "block":
                layer_to_group_mapping = {
                    k: ".".join(k.split(".")[:3]) for k in parameters_to_prune
                }
            else:
                raise NotImplementedError

        sp = LayerSparsity(
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
        return sp.return_sparsity()

    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self._model)

        if self.prune_spec is None:
            return self._model, None


        global_sratio = self.sparsity_ratio
        sdict = self.get_sparsity(global_sratio, self.sparsity_ratio_granularity)

        self._model = self._prune(
            self._model,
            self.data_loader,
            device,
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.layers",
            n_samples=self.num_samples,
            sparsity_ratio=sdict if isinstance(sdict, dict) else global_sratio,
        )

        self.model_reset(self._model, dtype_record, requires_grad_record, device)
        return self._model, sdict

    def model_setup_and_record_attributes(self, model):
        dtype_record = {}
        requires_grad_record = {}
        dev = next(model.parameters()).device
        for n, p in model.named_parameters():
            dtype_record[n] = p.dtype
            requires_grad_record[n] = p.requires_grad
        return dtype_record, requires_grad_record, dev

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


def parse_args():
    parser = argparse.ArgumentParser(description="Prune LLaVA-based model with SparseGPT.")
    parser.add_argument("--pretrained_model_path", type=str,
                        default="/share/home/mhma/models/llava-onevision-qwen2-7b-ov",
                        help="Path to the pretrained model.")
    parser.add_argument("--json_path", type=str,
                        default="/share/home/mhma/MLLM-Efficiency/datasets/coco/caption.json",
                        help="Path to the dataset JSON.")
    parser.add_argument("--images_dir", type=str,
                        default="/share/home/mhma/MLLM-Efficiency/datasets/coco/images",
                        help="Path to the images directory.")
    parser.add_argument("--result_folder", type=str,
                        default="/share/home/mhma/MLLM-Efficiency/models/llava-onevision-qwen2-7b",
                        help="Folder to save pruned model.")
    parser.add_argument("--sparsity_ratio", type=float,
                        default=0.5, help="Global ratio of weights to be pruned.")
    parser.add_argument("--max_sparsity_per_layer", type=float,
                        default=0.9, help="Per-layer maximum pruning ratio.")
    parser.add_argument("--num_samples", type=int,
                        default=1, help="Number of calibration samples.")
    parser.add_argument("--num_data_first_stage", type=int,
                        default=1, help="Number of data used in the first stage.")
    parser.add_argument("--pruner_name", type=str,
                        default="llava_sparsegpt_pruner",
                        help="Name of the pruner.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    llava_pruner = LLavaLLMPrunerSparseGPT(
        pretrained=args.pretrained_model_path,
        json_path=args.json_path,
        images_dir=args.images_dir,
        sparsity_ratio=args.sparsity_ratio,
        max_sparsity_per_layer=args.max_sparsity_per_layer,
        num_samples=args.num_samples,
        num_data_first_stage=args.num_data_first_stage,
        prune_spec=[("model.layers", 0.5)],  # just a placeholder
        device="cuda:0",
        attn_implementation="flash_attention_2",
        pruner_name=args.pruner_name,
    )

    dir_path = f"{args.result_folder}_{llava_pruner.pruner_name}_{llava_pruner.sparsity_ratio}"
    pruned_model, sparsity_dict = llava_pruner.prune()

    pruned_model.save_pretrained(dir_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.save_pretrained(dir_path)

    print("Finished SparseGPT-based pruning.")
    s = llava_pruner.check_sparsity(pruned_model)
    print(f"Global sparsity: {s:.4f}")
