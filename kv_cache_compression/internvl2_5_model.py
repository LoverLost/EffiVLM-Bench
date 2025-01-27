import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def internvlchat_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    image_flags = image_flags.squeeze(-1)
    input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

    vit_embeds = self.extract_feature(pixel_values)
    vit_embeds = vit_embeds[image_flags == 1]
    vit_batch_size = pixel_values.shape[0]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(
            f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    try:
        input_embeds[selected] = input_embeds[selected] * \
            0.0 + vit_embeds.reshape(-1, C)
    except Exception as e:
        vit_embeds = vit_embeds.reshape(-1, C)
        print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
              f'vit_embeds.shape={vit_embeds.shape}')
        n_token = selected.sum()
        input_embeds[selected] = input_embeds[selected] * \
            0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    # TODO
    # self.language_model fuzhi

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1,
                                         self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
