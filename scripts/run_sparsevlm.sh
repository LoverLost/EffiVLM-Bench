#!/bin/bash
#SBATCH --job-name="train"
#SBATCH -o all.out
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 100:00:00                            
#SBATCH --cpus-per-task=16                                                               
#SBATCH -w gpu18
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

cd
cd clash
./clash -d . &
export http_proxy="http://127.0.0.1:12397"
export HTTP_PROXY="http://127.0.0.1:12397"
export https_proxy="http://127.0.0.1:12397"
export HTTPS_PROXY="http://127.0.0.1:12397"
export all_proxy="socks5h://127.0.0.1:12382"
export ALL_PROXY="socks5h://127.0.0.1:12382"
cd
source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate mllm-efficiency
cd /home/rcmu/read_papers/MLLM-Efficiency-main
pwd

# 设置环境变量
ROOT_DIR="/home/rcmu/rebaserebaserebase"
NUM_PROCESSES=1
export CUDA_VISIBLE_DEVICES="0"

# 保留率
budgets_ratio_list=(0.01 0.1 0.4) 
# 日志后缀
log_suffix_name_list=(1 10 40)

# 遍历的 task 列表
task_list=("textvqa_val" "infovqa_test" "ocrbench" "chartqa" "mathvista_testmini")
merge_list=(false true)

export CONDA_DEFAULT_ENV="mllm-efficiency"
export PATH="/home/rcmu/anaconda3/envs/mllm-efficiency/bin:$PATH"
export PYTHONPATH="/home/rcmu/read_papers/MLLM-Efficiency-main:/home/rcmu/read_papers/MLLM-Efficiency-main/lmms-eval"
export OPENAI_API_URL="https://api.openai-proxy.org/v1/chat/completions"
export OPENAI_API_KEY="sk-jGzkmsfwW3Cn798VcBmMMJQ9L2R6q2m7SMbl5AZRKJmWsrJF"


# 循环遍历 budgets_ratio_list 和 task_list
for merge in "${merge_list[@]}"; do
    for task in "${task_list[@]}"; do
        for i in "${!budgets_ratio_list[@]}"; do
            echo "----------------------------------------"
            echo "Task: $task"
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
            --log_samples_suffix logs_onevision_sparsevlm_${LOG_SUFFIX_NAME} \
            --output_path $ROOT_DIR/logs_onevision_sparsevlm_${task}_${LOG_SUFFIX_NAME} \
            --model qwen2_vl_with_kvcache \
            --model_args "pretrained=/home/rcmu/models/Qwen2-VL-7B-Instruct,method=look-m,merge=false,use_flash_attention_2=true,budgets=${BUDGETS_RATIO}" \
            --tasks $task
            date
        done
    done
done