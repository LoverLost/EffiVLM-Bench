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