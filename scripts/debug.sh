accelerate launch --num_processes 1 --main_process_port 23415 -m debugpy --wait-for-client --listen 5678 lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm \
    --tasks textvqa \
    --batch_size 1 \
    --log_samples \
    --output_path /share/home/mhma/MLLM-Efficiency/logs/
