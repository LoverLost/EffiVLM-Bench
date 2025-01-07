import transformers
from .qwen_model import (
    qwen_attention_forward_streamingLLM,
    qwen_attention_forward_H2O,
    qwen_model_forward_vlcache,
    qwen_attention_forward_vlcache,
    qwen_attention_forward_LOOK_M,
    qwen_attention_forward_snapkv,
    qwen_model_forward_fastv,
    qwen_attention_forward_fastv
)

from .kv_cache_utils import VlCacheKVCluster


def replace_qwen(args, method):

    if method == "streamingllm":
        print('using streamingllm')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_streamingLLM
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
    elif method == "h2o":
        print('using h2o')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_H2O
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_alpha_sparsity = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_different_window_per_layer = args.vlcache_different_window_per_layer
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_head_adaptive = args.vlcache_head_adaptive

    elif method =='look-m':
        print('using look-m')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(
            args, 'hh_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(
            args, 'recent_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budget = getattr(
            args, 'budgets', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.merge = getattr(
            args, 'merge', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_LOOK_M

    elif method == 'snapkv':
        print('using snapkv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_snapkv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.snapkv_head_adaptive = args.snapkv_head_adaptive
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.snapkv_head_adaptive = args.pooling

    elif method == 'fastv':
        print('using fastv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.target_layer_idx = getattr(args, 'target_layer_idx', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.target_layer_idx = getattr(args, 'target_layer_idx', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.budgets = getattr(args, 'budgets', None)   # visual part
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.origin = getattr(args, 'origin', None)

def replace_mistral(method):
    pass


def replace_llama(method):
    pass

