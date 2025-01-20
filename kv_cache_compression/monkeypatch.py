import transformers
from .qwen_model import (
    qwen_attention_forward_streamingLLM,
    qwen_attention_forward_H2O,
    qwen_model_forward_vlcache,
    qwen_attention_forward_vlcache,
    qwen_attention_forward_LOOK_M,
    qwen_attention_forward_snapkv,
    qwen_model_forward_fastv,
    qwen_attention_forward_fastv,
    qwen_attn_forward_PyramidKV,
    qwen_attention_forward_CSP,
    qwen_attention_forward_random
)

from .qwen_model import (
    qwen_flash_attention_forward_streamingLLM,
    qwen_flash_attention_forward_H2O,
    qwen_flash_attention_forward_PyramidKV,
    qwen_flash_attention_forward_random,
    qwen_flash_attention_forward_snapkv,
    qwen_flash_attention_forward_look_m,
)

from .kv_cache_utils import VlCacheKVCluster
from .siglip_model import *
import qwen2vl
from .qwen2vl_model import (
    qwen_vl_model_forward_fastv,
    qwen_vl_flash_attention_forward_fastv,
    qwen2vl_vision_flash_attention2_forward_visionzip,
    qwen2vl_vision_tower_forward_visionzip,
    qwen2vl_vision_block_forward_visionzip,
    qwen2vl_generation_forward_visionzip
)

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
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_budget_layer_adaptive = getattr(args, 'vlcache_budget_layer_adaptive', True)

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
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pooling = args.pooling

    elif method == 'fastv':
        print('using fastv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.target_layer_idx = getattr(args, 'target_layer_idx', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.target_layer_idx = getattr(args, 'target_layer_idx', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.budgets = getattr(args, 'budgets', None)   # visual part
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.origin = getattr(args, 'origin', None)
    
    
    elif method == "csp":
        print('using csp')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_CSP
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(args, 'hh_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(args, 'recent_ratio', None)
        # The budget ratio allocated to cross-attention. The default value is 0.1
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.cross_ratio = getattr(args, 'cross_ratio', 0.1)
        # The kv_recent_bias setting is how many times you want the recent window to be larger than the original window. The default value is 1, which means no additional increase in the window size.
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.kv_recent_bias = getattr(args, 'kv_recent_bias', 1)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = getattr(args, 'budgets', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.csp_head_adaptive = args.csp_head_adaptive


    elif method == 'pyramidkv':
        print('using pyramidkv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attn_forward_PyramidKV
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pooling = args.pooling

    elif method == 'visionzip':
        print('using visionzip')
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipEncoderLayer, SigLipAttention
        from llava.model.llava_arch import LlavaMetaForCausalLM
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        # SigLipVisionTower.dominant = getattr(args, 'dominant_num')
        # SigLipVisionTower.contextual = getattr(args, 'contextual_num')
        SigLipVisionTower.budgets = getattr(args, 'budgets', None)
        SigLipVisionTower.forward = siglip_vision_tower_forward
        SigLipEncoderLayer.forward = siglip_EncoderLayer_forward
        SigLipAttention.forward = siglip_attention_forward
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
        LlavaMetaForCausalLM.encode_images_visionzip_simple = encode_images_visionzip_simple
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip

    elif method == 'random':
        print('using random')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = getattr(args, 'budgets', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_random
    
    elif method == 'prumerge+':
        print('using prumerge+')
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.llava_arch import LlavaMetaForCausalLM
        SigLipVisionTower.budgets = getattr(args, 'budgets', None)
        SigLipVisionTower.forward = siglip_vision_tower_forward_prumerge_plus
        LlavaMetaForCausalLM.encode_images_prumerge_plus = encode_images_prumerge_plus
        LlavaMetaForCausalLM.encode_images_prumerge_plus_simple = encode_images_prumerge_plus_simple
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_prumerge_plus


    elif method == 'sparsevlm':
        print('using sparsevlm')
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.language_model.sparse_llava_qwen import LlavaQwenSparseForCausalLM
        from llava.model.language_model.sparse_modeling_qwen import Qwen2SparseModel
        LlavaQwenSparseForCausalLM.bias = 0
        LlavaQwenSparseForCausalLM.scale = 13.5
        Qwen2SparseModel.ratio = getattr(args, 'r', None)

        
        
        
def replace_qwen2vl(args, method):
     
    if method == 'streamingllm':
        print('using streamingllm')
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen_attention_forward_streamingLLM
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_streamingLLM
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = args.budgets
    
    elif method == "h2o":
        print('using h2o')
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen_attention_forward_H2O
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.h2o_head_adaptive = args.h2o_head_adaptive
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_H2O
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.h2o_head_adaptive = args.h2o_head_adaptive
        
    elif method == 'snapkv':
        print('using snapkv')
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen_attention_forward_snapkv
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.snapkv_head_adaptive = args.snapkv_head_adaptive
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.pooling = args.pooling
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_snapkv
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.snapkv_head_adaptive = args.snapkv_head_adaptive
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.pooling = args.pooling
        
    elif method == 'pyramidkv':
        print('using pyramidkv')
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen_attn_forward_PyramidKV
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.pooling = args.pooling
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_PyramidKV
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = args.budgets
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.pooling = args.pooling
    
    elif method == 'random':
        print('using random')
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.budgets = getattr(args, 'budgets', None)
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen_attention_forward_random
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = getattr(args, 'budgets', None)
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_random

    elif method == 'look-m':
        print('using look-m')
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budget = getattr(args, 'budgets', None)
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.merge = getattr(args, 'merge', False)
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.hh_ratio = getattr(args, 'hh_ratio', None)
        qwen2vl.modeling_qwen2_vl.Qwen2VLAttention.recent_ratio = getattr(args, 'recent_ratio', None)
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_look_m

    elif method == 'fastv':
        print('using fastv')
        qwen2vl.modeling_qwen2_vl.Qwen2VLModel.forward = qwen_vl_model_forward_fastv
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_vl_flash_attention_forward_fastv
        qwen2vl.modeling_qwen2_vl.Qwen2VLModel.target_layer_idx = getattr(args, 'target_layer_idx', 2)
        qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.target_layer_idx = getattr(args, 'target_layer_idx', 2)
        qwen2vl.modeling_qwen2_vl.Qwen2VLModel.budgets = getattr(args, 'budgets', None)
        qwen2vl.modeling_qwen2_vl.Qwen2VLModel.origin = getattr(args, 'origin', False)

    elif method == 'visionzip':
        print('using visionzip')
        qwen2vl.modeling_qwen2_vl.VisionFlashAttention2.forward = qwen2vl_vision_flash_attention2_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = qwen2vl_vision_tower_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VLVisionBlock.forward = qwen2vl_vision_block_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 0.01)
        qwen2vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2vl_generation_forward_visionzip


    

         
     
def replace_mistral(method):
    pass


def replace_llama(method):
    pass

