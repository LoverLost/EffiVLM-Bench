#!/bin/bash
#SBATCH --job-name="100_ov_chartqa"
#SBATCH -o 100_ov_chartqa.out  
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 50:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --cpus-per-task=12                   # cpu数量，如果排不上队可以检查一下CPU数量够不够，命令： sinfo -N -o "%5N  %5t  %13C  %8O  %8e  %7m  %G"
                                              # 其中对应i的数量是空余数量
#SBATCH --mem=80G  
#SBATCH --gres=gpu:a100-sxm4-80gb:1

source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate lmms-efficiency
pip list

cd /home/zxwang/clash
./clash -d . & 
export http_proxy=http://127.0.0.1:8899
export HTTP_PROXY=http://127.0.0.1:8899
export https_proxy=http://127.0.0.1:8899
export HTTPS_PROXY=http://127.0.0.1:8899
export all_proxy=socks5h://127.0.0.1:8900
export ALL_PROXY=socks5h://127.0.0.1:8900

# 设置环境变量
ROOT_DIR="/home/zxwang/module/MLLM-Efficiency/result/0107"
NUM_PROCESSES=1
export CUDA_VISIBLE_DEVICES="0"

# 保留率
budgets_ratio_list=(1.0 0.4 0.2 0.1 0.05 0.01) 
# 日志后缀
log_suffix_name_list=(100 40 20 10 5 1)
task="chartqa"


export CONDA_DEFAULT_ENV="lmms-efficiency"
export PATH="/home/zxwang/anaconda3/envs/mllm-efficiency/bin:$PATH"
export PYTHONPATH="/home/zxwang/module/MLLM-Efficiency:/home/zxwang/module/MLLM-Efficiency/lmms-eval"

# 循环列表
for i in "${!budgets_ratio_list[@]}"; do
    echo "----------------------------------------"
    echo "Index: $i, 保留率: ${budgets_ratio_list[$i]}, log 后缀: ${log_suffix_name_list[$i]}"

    BUDGETS_RATIO=${budgets_ratio_list[$i]}
    LOG_SUFFIX_NAME=${log_suffix_name_list[$i]}
        
    echo "BUDGETS_RATIO: $BUDGETS_RATIO, LOG_SUFFIX_NAME: $LOG_SUFFIX_NAME"
    echo "----------------------------------------"
    date
    
    accelerate launch --num_processes=$NUM_PROCESSES \
    --mixed_precision bf16 \
    -m lmms_eval \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix logs_onevision_csp_${task}_${LOG_SUFFIX_NAME} \
    --output_path $ROOT_DIR/logs_onevision_csp_${task}_${LOG_SUFFIX_NAME} \
    --model llava_onevision_with_kvcache \
    --model_args "pretrained=/home/zxwang/huggingface/llava-onevision-qwen2-7b-ov,method=csp,cross_ratio=0.1,kv_recent_bias=1,csp_head_adaptive=False,budgets=${BUDGETS_RATIO}" \
    --tasks $task
    date
done





