import transformers
from .cache_utils import streamingLLMCache
from .qwen_model import qwen_attention_forward_streamingLLM


def replace_qwen(method):
    if method == "streamingllm":
        print('using streamingllm')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_streamingLLM
        transformers.cache_utils.DynamicCache = streamingLLMCache
def replace_mistral(method):
    pass

def replace_llama(method):
    pass    