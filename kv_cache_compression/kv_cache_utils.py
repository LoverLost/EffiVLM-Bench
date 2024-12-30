import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

from typing import List


from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent
     
class StreamingLLMKVCluster():
    def __init__(self, query_len, budgets, window_size_budgets=0.1, merge=None):
        self.query_len = query_len
        self.budgets = budgets # 保留比
        self.max_capacity_prompt = int(query_len * budgets)
        self.window_size_budgets = window_size_budgets # 窗口大小比
        self.window_size = int(self.max_capacity_prompt * window_size_budgets)
        self.merge = merge

    def reset(self, budgets, window_size_budgets=0.1, merge=None):
        self.query_len = None
        self.budgets = budgets # 保留比
        self.window_size_budgets = window_size_budgets # 窗口大小比
        self.window_size = None
        self.max_capacity_prompt = None
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
 
 
 
class H2OKVCluster():
    def __init__(self, query_len, budgets, window_size_budgets=0.1, head_adaptive=True, merge=None):
        self.query_len = query_len
        self.budgets = budgets # 保留比
        self.max_capacity_prompt = int(query_len * budgets)
        self.window_size_budgets = window_size_budgets # 窗口大小比
        self.window_size = int(self.max_capacity_prompt * window_size_budgets)
        self.head_adaptive = head_adaptive
        self.merge = merge

    def reset(self, budgets, window_size_budgets=0.1, head_adaptive=True, merge=None):
        self.query_len = None
        self.budgets = budgets # 保留比
        self.window_size_budgets = window_size_budgets # 窗口大小比
        self.window_size = None
        self.max_capacity_prompt = None
        self.merge = merge
        self.head_adaptive = head_adaptive
        
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
        dtype_min = torch.finfo(attn_weights.dtype).min
        mask = torch.triu(
            torch.full((self.window_size, self.window_size), dtype_min, device=attn_weights.device),
            diagonal=1,
        )
        attn_weights[:, :, -self.window_size:, -self.window_size:] += mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.head_adaptive:
            attn_weights_sum = attn_weights[:, :, :, :-self.window_size].sum(dim=-2)
            print(1111)
        else:

            attn_weights_sum = attn_weights[:, :, :, :-self.window_size].sum(dim=[1, 2]).unsqueeze(1).expand(-1, num_heads, -1)

        indices = attn_weights_sum.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states
    
   
   

class VlCacheKVCluster():

    def __init__(self,vlcache_alpha_sparsity,vlcache_different_window_per_layer,vlcache_head_adaptive):

        self.vlcache_alpha_sparsity = vlcache_alpha_sparsity
        self.vlcache_different_window_per_layer = vlcache_different_window_per_layer
        self.vlcache_head_adaptive = vlcache_head_adaptive
        self.sparsity_layer_tuple = ()
        self.attn_weights_importance_tuple = ()
        self.isprefill = True

    def allocate_budget_and_update_kv(
            self,
            buget_layers:torch.Tensor,
            sorted_attn_kv_indices:torch.Tensor,
            layer_idx:int,
            past_key_value:Cache
    ):
        """
        Called in the forward function of Qwen2Attention, after the prefill phase ends,
        Crop the kv cache based on the reallocated budget and token importance.
        """
        kv_cache_image_num = buget_layers[layer_idx] * past_key_value[layer_idx][0].shape[-2]
        if self.vlcache_different_window_per_layer:
            # different window size for each layer
            kv_cache_window_num = buget_layers[layer_idx] * past_key_value[layer_idx][0].shape[-2] * 0.1
        else:
            # same window size for each layer , 10% budget for each layer
            kv_cache_window_num =  past_key_value[layer_idx][0].shape[-2] * self.vlcache_alpha_sparsity * 0.1
        kv_cache_image_num_int = int(torch.ceil(kv_cache_image_num).item())
        kv_cache_window_num_int = math.ceil(kv_cache_window_num)
        # choose window
        sorted_attn_kv_select_window = torch.arange(
            past_key_value[layer_idx][0].shape[-2] - kv_cache_window_num_int, 
            past_key_value[layer_idx][0].shape[-2],
            device = sorted_attn_kv_indices.device
        )

        # choose other tokens
        if self.vlcache_head_adaptive:
            assert len(sorted_attn_kv_indices.shape) == 4
            # [layer, batch_size, head_num , kv_len]
            sorted_attn_kv_indices_layer = sorted_attn_kv_indices[layer_idx,0,:,:]
            sorted_attn_kv_select_window = sorted_attn_kv_select_window.repeat(sorted_attn_kv_indices.shape[2], 1) 
            mask = ~ torch.isin(sorted_attn_kv_indices_layer,sorted_attn_kv_select_window)
            sorted_attn_kv_select_image = torch.stack([
                    row[m] for row, m in zip(sorted_attn_kv_indices_layer, mask)
                ])
            sorted_attn_kv_select_image = sorted_attn_kv_select_image[:,:kv_cache_image_num_int]
        else:
            assert len(sorted_attn_kv_indices.shape) == 2
            # [layer,kv_len]
            sorted_attn_kv_indices_layer = sorted_attn_kv_indices[layer_idx]
            mask = ~ torch.isin(sorted_attn_kv_indices_layer,sorted_attn_kv_select_window)
            sorted_attn_kv_select_image = sorted_attn_kv_indices_layer[mask]
            sorted_attn_kv_select_image = sorted_attn_kv_select_image[:kv_cache_image_num_int]

        sorted_attn_kv_select = torch.cat([sorted_attn_kv_select_image, sorted_attn_kv_select_window], dim=-1)
        sorted_attn_kv_select = torch.sort(sorted_attn_kv_select, dim=-1)[0]

        if self.vlcache_head_adaptive:
            self.prefill_update_head(past_key_value, layer_idx, sorted_attn_kv_select)
        else:
            past_key_value.key_cache[layer_idx] = past_key_value.key_cache[layer_idx][:, :, sorted_attn_kv_select,:]
            past_key_value.value_cache[layer_idx] = past_key_value.value_cache[layer_idx][:, :, sorted_attn_kv_select,:]


    def get_budget_layer(
        self,
        attn_weights_postvison:torch.Tensor, 
        q_len:int,
        post_vision_size:int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Called in the forward function of Qwen2Attention, Calculate the sparsity of this layer based on attn_weights_postvision
        '''
        # get max scores from one row
        max_scores,_ = torch.max(attn_weights_postvison, axis=-1, keepdims=True)
        # filter the attn_weights_postvison
        filtered_attn_weights = (attn_weights_postvison >= 0.01 * max_scores).int()
        # get nonzero entries
        nonzero_entries = (filtered_attn_weights > 0).sum(dim=(-2, -1)).to(torch.float32)
        matrix_tril = torch.tril(torch.ones((q_len, q_len), device=attn_weights_postvison.device))[-post_vision_size:, :]
        num_elements_denominator = matrix_tril.count_nonzero().to(torch.float32)
        # get sparsity of each layer
        sparsity = (num_elements_denominator - nonzero_entries) / num_elements_denominator
        sparsity_layer = sparsity.mean()
        sparsity_layer_tuple = (sparsity_layer,)
        return sparsity_layer_tuple
    
    def get_budget_layer_and_sorted_attn_kv_indices(
        self,
        vlcache_sparsity_layer_tuple:Tuple[torch.Tensor],
        vlcache_attn_weights_importance_tuple:Tuple[torch.Tensor],
        layers:int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Called in the forward function of Qwen2Model, after the prefill phase ends, the sparsity of each layer is aggregated 
        to reallocate according to the budget, and the token importance is calculated
        '''
        sparsity_tensor = torch.stack([layer_sparsity[0] for layer_sparsity in vlcache_sparsity_layer_tuple])
        non_sparsity_sum = (1 - sparsity_tensor).sum()
        buget_layers = torch.zeros(layers)
        # Support different window sizes for each layer
        for l in range(layers):
            if self.vlcache_different_window_per_layer:
                # different window size for each layer
                buget_layers[l] = torch.clamp((1.0 - sparsity_tensor[l]) / non_sparsity_sum * self.vlcache_alpha_sparsity * layers, min=0.01, max=1.0)
            else:
                # same window size for each layer , 10% budget for each layer
                buget_layers[l] = torch.clamp((1.0 - sparsity_tensor[l]) / non_sparsity_sum * self.vlcache_alpha_sparsity * 0.9 * layers, min=0.01, max=1.0)
        
        # [ layers , batch_size , head_num , question_len]
        stacked_attn_weights_importance = torch.stack([attn_weights_importance[0] for attn_weights_importance in vlcache_attn_weights_importance_tuple])
        
        if not self.vlcache_head_adaptive:
            # if not head adaptive , sum over head and batch. 
            attn_weights_importance_sum_layer = stacked_attn_weights_importance.sum(dim=[1, 2])
            sorted_attn_kv, sorted_indices = torch.sort(attn_weights_importance_sum_layer, dim=-1, descending=True)
        else:
            sorted_attn_kv, sorted_indices = torch.sort(stacked_attn_weights_importance, dim=-1, descending=True)
        return buget_layers, sorted_indices
    
    def get_token_importance(
        self,
        attn_weights_postvison:torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Called in the forward function of Qwen2Attention , Calculate the token importance based on the attn_weights_postvisa of the current header
        '''
        attn_weights_importance = attn_weights_postvison.sum(dim=-2)
        attn_weights_importance_tuple = (attn_weights_importance,)

        return attn_weights_importance_tuple
    
    def prefill_update_head(
        self,
        past_key_value:Cache, 
        layer_idx: int,
        sorted_attn_kv_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Create new tensors to store the reordered states
        past_key_states = past_key_value.key_cache[layer_idx]
        past_value_states = past_key_value.value_cache[layer_idx]
        batch_size = past_key_states.shape[0]
        num_heads = past_key_states.shape[1] 
        head_dim = past_key_states.shape[-1]
        seq_len = sorted_attn_kv_indices.shape[-1]
        
        # Initialize new tensors with the right shape
        new_key_states = torch.zeros((batch_size, num_heads, seq_len, head_dim), 
                                dtype=past_key_states.dtype,
                                device=past_key_states.device)
        new_value_states = torch.zeros((batch_size, num_heads, seq_len, head_dim),
                                    dtype=past_value_states.dtype,
                                    device=past_value_states.device)
        
        # Reorder each head separately
        for head_idx in range(num_heads):
            new_key_states[:, head_idx] = past_key_states[:, head_idx, sorted_attn_kv_indices[head_idx]]
            new_value_states[:, head_idx] = past_value_states[:, head_idx, sorted_attn_kv_indices[head_idx]]
        
        # Update the cache
        past_key_value.key_cache[layer_idx] = new_key_states
        past_key_value.value_cache[layer_idx] = new_value_states



def init_StreamingLLM(self,
                      query_len, 
                      window_size_budgets = 0.1, 
                      budgets = 0.3, 
                      merge = None, 
                      ):

    self.kv_cluster = StreamingLLMKVCluster(
        query_len=query_len,
        budgets = budgets,
        window_size_budgets = window_size_budgets,
        merge = merge,
        )

def init_H2O(self,
            query_len, 
            head_adaptive,
            window_size_budgets = 0.1, 
            budgets = 0.3, 
            merge = None, 
            ):

    self.kv_cluster = H2OKVCluster(
        query_len=query_len,
        budgets = budgets,
        window_size_budgets = window_size_budgets,
        head_adaptive=head_adaptive,
        merge = merge,
        )   
        )

def init_Vlcache(self):

    self.kv_cluster = VlCacheKVCluster(
        vlcache_alpha_sparsity = self.vlcache_alpha_sparsity,
        vlcache_different_window_per_layer = self.vlcache_different_window_per_layer,
        vlcache_head_adaptive = self.vlcache_head_adaptive,
        )