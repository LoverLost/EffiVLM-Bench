python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.01 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.05 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.1 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.2 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/
# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.01 \
#     --tasks chartqa  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix chartqa \
#     --output_path ./logs/

# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm \
#     --tasks chartqa  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix chartqa \
#     --output_path ./logs/

# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.2 \
#     --tasks textvqa_val  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix textvqa_val \
#     --output_path ./logs/

# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.1 \
#     --tasks textvqa_val  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix textvqa_val \
#     --output_path ./logs/

# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.05 \
#     --tasks textvqa_val  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix textvqa_val \
#     --output_path ./logs/


# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.01 \
#     --tasks textvqa_val  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix textvqa_val \
#     --output_path ./logs/


# python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm\
#     --tasks textvqa_val  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix textvqa_val \
#     --output_path ./logs/