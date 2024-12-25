import transformers
from .qwen_model import qwen_attention_forward_streamingLLM, qwen_decode_forward


def replace_qwen(method):
    if method == "streamingllm":
        print('using streamingllm')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_streamingLLM
        # transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_decode_forward
def replace_mistral(method):
    pass

def replace_llama(method):
    pass    