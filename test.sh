#!/bin/bash

#SBATCH -J ov                               # 作业名为 test
#SBATCH -o test.out                           # stdout 重定向到 test.out
#SBATCH -e llavaov-streamingllm.err                           # stderr 重定向到 test.err
#SBATCH -p gpu02                              # 作业提交的分区为 gpu02
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 1:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH -w gpu06                              # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:1                         # 申请 4 卡 A100，如果只申请CPU可以删除本行

source ~/.bashrc

# 设置运行环境
conda activate test1225

cd /share/home/mhma/MLLM-Efficiency

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.1 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/