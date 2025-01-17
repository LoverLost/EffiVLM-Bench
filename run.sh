#!/bin/bash

export all_proxy=http://ln01:33269

BASE_COMMAND="python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=4 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache \
    --batch_size 1 \
    --log_samples"

METHODS=(
    "h2o h2o_head_adaptive=True,use_flash_attention_2=true"
    "snapkv snapkv_head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    "random use_flash_attention_2=true"
    "pyramidkv pyramidkv_head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
)

# 数据集配置
TASKS=("chartqa" "docvqa_test" "testvqa_val")

# 模型路径
MODEL_PATH="/share/home/mhma/models/Qwen2-VL-7B-Instruct"

# 循环生成并运行命令
for TASK in "${TASKS[@]}"; do
    for METHOD_CONFIG in "${METHODS[@]}"; do
        METHOD=$(echo "$METHOD_CONFIG" | awk '{print $1}')
        ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1=""; print $0}' | xargs)
        BUDGETS=(0.05 0.2 0.3)

        for BUDGET in "${BUDGETS[@]}"; do
            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},${ADDITIONAL_ARGS}"
            OUTPUT_PATH="./logs/qwen2vl/${TASK}"
            COMMAND="${BASE_COMMAND} --tasks ${TASK} --output_path ${OUTPUT_PATH} --log_samples_suffix ${TASK} --model_args ${MODEL_ARGS}"
            echo "Executing: ${COMMAND}"
            eval ${COMMAND}
        done
    done
done
