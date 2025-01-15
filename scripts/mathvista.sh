source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate mllm-efficiency

NUM_PROCESSES=1
CUDA_VISIBLE_DEVICES="0"
export OPENAI_API_URL="https://api.openai-proxy.org/v1/chat/completions"
export OPENAI_API_KEY="sk-jGzkmsfwW3Cn798VcBmMMJQ9L2R6q2m7SMbl5AZRKJmWsrJF"
export CONDA_DEFAULT_ENV="mllm-efficiency"
export PATH="/home/zxwang/anaconda3/envs/mllm-efficiency/bin:$PATH"
export PYTHONPATH="/home/zxwang/module/MLLM-Efficiency:/home/zxwang/module/MLLM-Efficiency/lmms-eval"

# 直接推理不评测的命令
accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --tasks mathvista_testmini \
    --batch_size 1 \
    --model_args pretrained=/home/zxwang/huggingface/llava-onevision-qwen2-7b-ov,method=vl-cache,budgets=0.05,vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,vlcache_budget_layer_adaptive=True \
    --log_samples \
    --limit 20 \
    --log_samples_suffix llava_onevision_with_kvcache \
    --output_path /home/zxwang/module/MLLM-Efficiency/result/0115/mathvista_testmini/ \
    --predict_only

# 只评测 不推理的脚本
accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model from_log \
    --model_args logs=/home/zxwang/module/MLLM-Efficiency/result/0115/mathvista_testmini/,model_name=llava_onevision_with_kvcache,task=mathvista_testmini \
    --tasks mathvista_testmini \
    --batch_size 1 \
    --limit 10 \
    --log_samples \
    --log_samples_suffix llava_onevision_with_kvcache \
    --output_path /home/zxwang/module/MLLM-Efficiency/result/0115/mathvista_testmini_result4/

# 推理加评测的脚本
# accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
#     --model llava_onevision_with_kvcache \
#     --tasks mathvista_testmini \
#     --batch_size 1 \
#     --model_args pretrained=/home/zxwang/huggingface/llava-onevision-qwen2-7b-ov,method=vl-cache,budgets=0.05,vlcache_different_window_per_layer=False,vlcache_head_adaptive=True,vlcache_budget_layer_adaptive=True \
#     --log_samples \
#     --limit 20 \
#     --log_samples_suffix llava_onevision_with_kvcache \
#     --output_path /home/zxwang/module/MLLM-Efficiency/result/0115/mathvista_testmini_all/ 