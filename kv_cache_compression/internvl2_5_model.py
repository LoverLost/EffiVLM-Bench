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



@torch.no_grad()
def internvl_generate_with_mask(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


@torch.no_grad()
def internvl_generate_38B(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


def internvl_naive_attn_4B_visionzip(self, x):
        B, N, C = x.shape   # batchsize， seqlen， hidden_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        origin_dtype = q.dtype
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        if self.layer_idx == 22:
            attn = ((q.float() * self.scale) @ k.transpose(-2, -1).float())
            self.attn_weights = attn[:,:,1:,1:]   # 去掉了cls
            self.metric = k.mean(dim=1)[:,1:,:]
            attn = attn.to(origin_dtype)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)             
        attn = self.attn_drop(attn).to(origin_dtype)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def internvl_attention_forward_4B_visionzip(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states)   # NOTE 此处强制使用naive attn
        return x

def internvl_extract_feature_4B_visionzip(self, pixel_values):   # 编码视觉部分
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)   # [num_patches, 256, 2048]

        attn_weights = self.vision_model.encoder.layers[-2].attn.attn_weights
        metric = self.vision_model.encoder.layers[-2].attn.metric
        metric = metric.view(metric.shape[0], 32, 32, -1)
        metric = metric.view(metric.shape[0], 16, 2, 16, 2, -1).mean(dim=(2, 4))
        metric = metric.flatten(1,2)
        hidden_states = vit_embeds

        per_patch_token_num = metric.shape[1]
        dominant_num =  max(1, int(self.budgets * per_patch_token_num * 5.4 / 6.4))
        contextual_num = max(1, int(self.budgets * per_patch_token_num * 1 / 6.4))

        attention_sum = attn_weights.mean(dim=1).mean(dim=1)
        attention_sum = attention_sum.view(attention_sum.shape[0], 32, 32)
        attention_sum = attention_sum.view(attention_sum.shape[0], 16, 2, 16, 2).mean(dim=(2, 4))
        attention_sum = attention_sum.flatten(1)

        all_indices = attention_sum.topk(dominant_num, dim=1).indices
        mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)  # [3, 729]  哪些需要被删掉
        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
        
        ### Filter
        metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, metric.shape[2])  # [3, 675, 72]
        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, hidden_states.shape[2])  # [3, 675, 1152]
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

        ## Contextual Visual Tokens
        step = max(1, metric_normalized.shape[1] // contextual_num)
        target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]  # 均匀采样
        target_tokens = metric_normalized[:, target_indices, :]  # [3, 10, 72]

        tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]   # [3, 665, 72]
        similarity = torch.bmm(tokens_to_merge.float(), target_tokens.transpose(1, 2).float())   # [3, 665, 10]    FIXME  change here
        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)       # 计算每个聚类中心分配到的token的数量
        hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
        aggregated_hidden = (torch.bmm(assign_one_hot.transpose(1, 2).float(), hidden_to_merge.float()) / counts).to(torch.bfloat16) # [3, 10, 1152]   FIXME  change here
        target_hidden = hidden_states_filtered[:, target_indices, :]  # [3, 10, 1152]
        
        contextual_tokens = target_hidden + aggregated_hidden

        # Merge with target hidden states and concatenate
        hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(pixel_values.dtype)  # [3， 64， 1152]

        return hidden_states_save

def internvl_generate_4B_visionzip(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,   # 151665 151667 151666
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        assert input_ids.shape[0] == 1, '当前只支持一个batch'
        num_visual_tokens = vit_embeds.shape[1] * vit_embeds.shape[0]
         # 将 input_ids 转为一维张量，方便后续操作
        input_ids_flat = input_ids[0]  # shape: [seq_len]
        # 找出视觉 token（即占位符）所在位置，假设 self.img_context_token_id 对应的数字为151667
        visual_positions = (input_ids_flat == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        # 如果视觉 token 占位符数量超过实际视觉 token 数量，则只保留前 num_visual_tokens 个
        if visual_positions.numel() > num_visual_tokens:
            # 构造掩码，初始全部为 True
            mask = torch.ones_like(input_ids_flat, dtype=torch.bool)
            # 将多余的视觉 token 占位符对应位置置为 False
            mask[visual_positions[num_visual_tokens:]] = False
            # 根据掩码过滤 input_ids，并还原 shape 为 [1, new_seq_len]
            input_ids = input_ids_flat[mask].unsqueeze(0)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs