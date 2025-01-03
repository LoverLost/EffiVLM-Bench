import transformers
from .qwen_model import qwen_attention_forward_streamingLLM, qwen_model_forward_vlcache,qwen_attention_forward_vlcache,qwen_attention_forward_LOOK_M
from .kv_cache_utils import VlCacheKVCluster


def replace_qwen(args, method):
    
    if method == "streamingllm":
        print('using streamingllm')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_streamingLLM
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        # transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_decode_forward

    if method == "vl-cache":
        print('using vlcache')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_alpha_sparsity = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_different_window_per_layer = args.vlcache_different_window_per_layer
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_head_adaptive = args.vlcache_head_adaptive

        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_post_vision_size = 0
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_attn_weights_importance_tuple = ()
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_sparsity_layer_tuple = ()
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_first_after_prefill = True 
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_buget_layers = None
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_sorted_attn_kv_indices = None
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.kv_cluster = VlCacheKVCluster()

        # TODO: 需要添加一个变量来记录当前的 question_id
        # 清空 vlcache_attn_weights_importance_tuple vlcache_sparsity_layer_tuple vlcache_first_after_prefill
        # 读入 

    if method =='look-m':
        print('using look-m')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(args, 'hh_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(args, 'recent_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budget = getattr(args, 'budget', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_LOOK_M
    


def replace_mistral(method):
    pass

def replace_llama(method):
    pass    