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

# 存储结果和日志的目录
ROOT_DIR="/data/scir/rcmu/results"
NUM_PROCESSES=8

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CONDA_DEFAULT_ENV="mllm-efficiency"
export PATH="/data/scir/rcmu/miniconda3/envs/mllm-efficiency/bin:$PATH"
export PYTHONPATH="/data/scir/rcmu/MLLM-Efficiency:/data/scir/rcmu/MLLM-Efficiency/lmms-eval"
export OPENAI_API_URL="https://api.openai-proxy.org/v1/chat/completions"
export OPENAI_API_KEY="sk-jGzkmsfwW3Cn798VcBmMMJQ9L2R6q2m7SMbl5AZRKJmWsrJF"


BASE_COMMAND="python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache \
    --batch_size 1 \
    --log_samples"

METHODS=(
    "h2o head h2o_head_adaptive=True,use_flash_attention_2=true"
    "h2o non_head h2o_head_adaptive=False,use_flash_attention_2=true"
    "snapkv head snapkv_head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    "snapkv non_head snapkv_head_adaptive=False,pooling=avgpool,use_flash_attention_2=true"
    "pyramidkv head pyramidkv_head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    "pyramidkv non_head pyramidkv_head_adaptive=False,pooling=avgpool,use_flash_attention_2=true"
    "look-m merge merge=True,use_flash_attention_2=true"
    "look-m non_merge merge=False,use_flash_attention_2=true"
    "vl-cache head_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,vlcache_budget_layer_adaptive=True"
    "vl-cache non_head_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=False,vlcache_budget_layer_adaptive=True"
    "vl-cache head_non_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,vlcache_budget_layer_adaptive=False"
    "vl-cache non_head_non_layer vlcache_different_window_per_layer=False,vlcache_head_adaptive=False,vlcache_budget_layer_adaptive=False"
    "random random use_flash_attention_2=true"
    "streamingllm streamingllm use_flash_attention_2=true"
    "fastv fastv target_layer_idx=2,origin=false,use_flash_attention_2=true"
    "visonzip visionzip use_flash_attention_2=true"
)

# 数据集配置
TASKS=("docvqa_test" "chartqa" "textvqa_val" "ai2d_no_mask" "infovqa_test" "ocrbench" "mmmu_val" "gqa" "mme" "realworldqa" "mmstar" "mathvista_testmini" "mathverse_testmini_vision")
# TASKS=("mathvista_testmini" "mathverse_testmini_vision")
BUDGETS=(0.01 0.05 0.1 0.2 0.4)

# 模型路径
MODEL_PATH="/data/scir/mhma/models/Qwen2-VL-7B-Instruct"



# 打印日志
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

# 日志函数
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "${message}" | tee -a "${MAIN_LOG_FILE}"
}


log_message "开始执行脚本"
log_message "环境信息："
log_message "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
log_message "NUM_PROCESSES: ${NUM_PROCESSES}"
log_message "MODEL_PATH: ${MODEL_PATH}"

for TASK in "${TASKS[@]}"; do
    for METHOD_CONFIG in "${METHODS[@]}"; do
        METHOD=$(echo "$METHOD_CONFIG" | awk '{print $1}')
        FILENAME=$(echo "$METHOD_CONFIG" | awk '{print $2}')
        ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1="";$2=""; print $0}' | xargs)

        for BUDGET in "${BUDGETS[@]}"; do
            # 为每个任务创建单独的日志文件
            TASK_LOG_FILE="${LOG_DIR}/${TASK}_${METHOD}_${BUDGET}_${FILENAME}_${TIMESTAMP}.log"
            OUTPUT_PATH="$ROOT_DIR/qwen2vl_${METHOD}_${TASK}_${BUDGET}_${FILENAME}"

            # 检查文件夹是否存在
            if [ -d "$OUTPUT_PATH" ]; then
                echo "文件夹存在，跳出循环。"
                continue
            fi

            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},${ADDITIONAL_ARGS}"

            COMMAND="${BASE_COMMAND} --tasks ${TASK} --output_path ${OUTPUT_PATH} --log_samples_suffix ${TASK} --model_args \"${MODEL_ARGS}\""

            # 记录开始执行特定任务
            log_message "开始执行任务: ${TASK} - ${METHOD} - ${BUDGET}"
            log_message "输出路径: ${OUTPUT_PATH}"
            log_message "完整命令: ${COMMAND}"

            # 执行命令并记录输出
            {
                if eval "${COMMAND}" > >(tee -a "${TASK_LOG_FILE}") 2> >(tee -a "${TASK_LOG_FILE}" >&2); then
                    log_message "任务成功完成: ${TASK} - ${METHOD} - ${BUDGET}"
                else
                    log_message "任务执行失败: ${TASK} - ${METHOD} - ${BUDGET}"
                    log_message "错误详情请查看: ${TASK_LOG_FILE}"
                fi
            } || {
                log_message "任务执行出现严重错误: ${TASK} - ${METHOD} - ${BUDGET}"
                exit 1
            }
        done
    done
done

# 记录脚本结束
log_message "脚本执行完成"