## example for llava-onevision
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
    --num_processes=2 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=snapkv,budgets=0.01,snapkv_head_adaptive=false,pooling=avgpool \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chatqa/snapkv/


## example for sparsegpt-llava-onevision
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


## example for qwen2vl



python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-2B-Instruct,method=h2o,budgets=0.01,h2o_head_adaptive=True\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/qwen2vl/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model llava_onevision_with_kvcache \
    --model_args pretrained=/share/home/mhma/models/llava-onevision-qwen2-7b-ov,method=random,budgets=0.01 \
    --tasks mathvista  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/debug/



## example for sparsegpt-llava-onevision
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


## example for qwen2vl
python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-2B-Instruct,method=h2o,budgets=0.01,h2o_head_adaptive=True\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/chartqa/qwen2vl/

<<<<<<< HEAD
    
=======
>>>>>>> 3b02ec3 (add qwen2vl and adapted text-based methods, such as h2o, streamingLLM, SnapKV, and PyramidKV, to Qwen2VL, while verifying the data type of attention (torch.fp32) and the correctness of position_id)
