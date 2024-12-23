import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
def compute_cosine_similarity(k_hh_pruned, k_hh_recent):
    
    return (k_hh_pruned/torch.norm(k_hh_pruned, dim=-1, keepdim=True)) @ (k_hh_recent/ torch.norm(k_hh_recent, dim=-1, keepdim=True)).T


def compute_cosine_similarity_for_layers(key_cache):

    num_layers, num_heads, _, _ = key_cache.shape 
    
    cosine_similarities = []
    
    for layer in range(num_layers):
        layer_similarities = []
        
        for head in range(num_heads):
            k_hh_pruned = key_cache[layer, head]  
            k_hh_recent = key_cache[layer, head]  
            cos_sim = compute_cosine_similarity(k_hh_pruned, k_hh_recent)  # (seq_len, seq_len)
            layer_similarities.append(cos_sim)
        
        cosine_similarities.append(torch.stack(layer_similarities))
        
    return torch.stack(cosine_similarities)

def calculate_grid_size(num_heads):

    rows = int(math.ceil(math.sqrt(num_heads))) 
    cols = int(math.ceil(num_heads / rows))      
    return rows, cols

def plot_attention_heatmaps(key_cache, base_dir):
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    num_layers, num_heads, _, _ = key_cache.shape
    cosine_similarities = compute_cosine_similarity_for_layers(key_cache)
    for layer in range(num_layers):
        rows, cols = calculate_grid_size(num_heads)  
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        axes = axes.flatten()
        
        for head in range(num_heads):
            cos_sim = cosine_similarities[layer, head].cpu().numpy()  
            
            sns.heatmap(cos_sim, ax=axes[head], cmap='viridis', square=True)
            axes[head].set_title(f'Head {head+1}')
            axes[head].set_xlabel('Token Index')
            axes[head].set_ylabel('Token Index')
            axes[head].set_xticks([])  
            axes[head].set_yticks([]) 
            
            print(f'layer_{layer+1}_head_{head+1} has done.')
            
        plt.tight_layout()
        plt.savefig(base_dir + f'layer_{layer+1}_heatmap.png')
        print(f'layer_{layer+1}_heatmap.png saved')


def visualize_key_similarity(past_key_values_from_mha, past_key_values_from_gqa, base_dir):
  
  
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  
    
    past_key_values_from_mha = torch.load(past_key_values_from_mha)
    key_cache_from_mha = past_key_values_from_mha.key_cache
    key_cache_from_mha = torch.cat(key_cache_from_mha, dim=0)   # 32 32 2988 128
    if key_cache_from_mha.dtype == torch.bfloat16:
        key_cache_from_mha = key_cache_from_mha.to(torch.float32)

        
    plot_attention_heatmaps(key_cache_from_mha, base_dir+'mha/')
    
    
    del past_key_values_from_mha
    torch.cuda.empty_cache()
    
    past_key_values_from_gqa = torch.load(past_key_values_from_gqa)
    key_cache_from_gqa = past_key_values_from_gqa.key_cache
    key_cache_from_gqa = torch.cat(key_cache_from_gqa, dim=0)   # 32 8 2988 128
    
    if key_cache_from_gqa.dtype == torch.bfloat16:
        key_cache_from_gqa = key_cache_from_gqa.to(torch.float32)
    
    plot_attention_heatmaps(key_cache_from_gqa, base_dir+'gqa/')
    

    
    
if __name__ == "__main__":
    past_key_values_from_mha = "/share/home/mhma/0000078_llama/past_key_values.pth"
    past_key_values_from_gqa = "/share/home/mhma/0000078_mistral/past_key_values.pth"
    visualize_key_similarity(past_key_values_from_mha, 
                             past_key_values_from_gqa, 
                             '/share/home/mhma/MLLM-Efficiency/visualization/k_similarity/')
