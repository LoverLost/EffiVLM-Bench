# MLLM-Efficiency


## Installation
1. Create a new conda environment and install the basic dependencies
    ```bash
    conda create -n mllm-efficiency python=3.10
    conda activate mllm-efficiency
    pip install -r requirements.txt
    pip install ninja
    conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install nvidia/label/cuda-12.1.1::cuda-nvcc
    ```

2. Change the env path 
    ```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
    ```
    Create a new file in the activate.d directory and add the following content:
    ```bash
    #!/bin/bash
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    ``` 
    Create a new file in the deactivate.d directory and add the following content:
    ```bash
    #!/bin/bash
    unset CUDA_HOME
    ```

3. Install the flash-attn
    ```bash
    conda activate mllm-efficiency
    echo $CUDA_HOME
    which nvcc
    pip install flash-attn --no-build-isolation
    ```

### Implement details for xxxkv methods
> The KV cache cluster (managed by the `xxxKVCluster` class) is responsible for selecting and compressing the KV states (keys and values), while the actual storage and dynamic updates of the compressed KV states are **still** handled by the `past_key_values` (aka. `Dynamiccache` class in `transformers` libraries), which remains the central mechanism for autoregressive decoding in Transformers. This separation allows for flexible and efficient cache management, critical for scaling LLMs in constrained environments.

> [!TIP]
> Step-by-Step Code construction(take `exampleKVCache` as example.)
> 1. Initialization of the KV Cache Compression System.
>
>    Code: `init_exampleMLLM`
>    ```python
>    def init_exampleMLLM(self):
>        # some configs setting
>
>        self.kv_cluster = exampleMLLMKVCluster(
>            # some configs setting
>        )
>    ```
> 2. Forward Pass in Attention.
>
>    Code: `qwen_attn_forward_exampleMLLM`
>    ```python
>    def llama_attn_forward_exampleMLLM(...):
>        bsz, q_len, _ = hidden_states.size()
>        init_exampleMLLM(self)
>        ...
>        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
>                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
>            )
>        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
>    ```
>    **if the method need allocate budgets from all layers, initialize in decoder layer class or do some change(like self.budgets in init_exampleMLLM). Take dynamickv from llm as example to handle this problem.**
> 3. Managing and Compressing KV Cache
>
>    Code: `exampleMLLMKVCluster` and `update_kv` Method in this class

> [!IMPORTANT]
> Above steps is the basic implementation from kv-cache-factory. BUT the disadvantage is that we can not evict cache during decoding since only current qkv are in 
