# export HF_HOME="/share/home/mhma/.cache/huggingface" 
# export HF_TOKEN="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"
# export HF_HUB_ENABLE_HF_TRANSFER="1"



TASK=$1
# CKPT_PATH=$2
# CONV_TEMPLATE=$3
# MODEL_NAME=$4
METHDO=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX


accelerate launch --num_processes 1 --main_process_port 23415 -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=$CKPT_PATH,method=$METHOD \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path /share/home/mhma/MLLM-Efficiency/logs/ \
    # --device_map 'auto' \