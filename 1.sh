python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=h2o,budgets=0.2,h2o_head_adaptive=false \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/h2o/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=h2o,budgets=0.01 \
    --tasks docvqa_test  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/docvqa_test/h2o/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=h2o,budgets=0.05 \
    --tasks docvqa_test  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/docvqa_test/h2o/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=h2o,budgets=0.01,h2o_head_adaptive=false \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/docvqa_test/h2o/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=pyramidkv,budgets=0.05,pyramidkv_head_adaptive=true,pooling=avgpool \
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/textvqa_val/pyramidkv/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=streamingllm,budgets=0.4 \
    --tasks mathvista  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mathvista \
    --output_path ./logs/mathvista/



python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache  \
    --model_args pretrained=/share/home/mhma/MLLM-Efficiency/models/origin/llava-onevision-qwen2-7b_llava_sparsegpt_pruner_0.5\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/sparsegpt/



    