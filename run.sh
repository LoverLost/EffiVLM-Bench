# python3 -m accelerate.commands.launch \
#     --main_process_port=28175 \
#     --mixed_precision=bf16 \
#     --num_processes=2 \
#     -m lmms_eval \
#     --model qwen2_vl_with_kvcache  \
#     --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,use_flash_attention_2=true\
#     --tasks chartqa  \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix chartqa \
#     --output_path ./logs/qwen2vl/chatqa/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=random,budgets=0.01 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chatqa/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=streamingllm,budgets=0.01,use_flash_attention_2=true \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chatqa/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=h2o,budgets=0.01,h2o_head_adaptive=True,use_flash_attention_2=true\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chartqa

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-2B-Instruct,method=snapkv,budgets=0.01,snapkv_head_adaptive=True,pooling=avgpool\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chartqa/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=random,budgets=0.1 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chatqa/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=streamingllm,budgets=0.1 \
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chatqa/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=h2o,budgets=0.1,h2o_head_adaptive=True\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chartqa/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-2B-Instruct,method=snapkv,budgets=0.1,snapkv_head_adaptive=True,pooling=avgpool\
    --tasks chartqa  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chartqa \
    --output_path ./logs/qwen2vl/chartqa/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct\
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/




python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=random,budgets=0.01 \
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=streamingllm,budgets=0.01 \
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=h2o,budgets=0.01,h2o_head_adaptive=True\
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-2B-Instruct,method=snapkv,budgets=0.01,snapkv_head_adaptive=True,pooling=avgpool\
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=random,budgets=0.1 \
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/


python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=streamingllm,budgets=0.1 \
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-7B-Instruct,method=h2o,budgets=0.1,h2o_head_adaptive=True\
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/

python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=2 \
    -m lmms_eval \
    --model qwen2_vl_with_kvcache  \
    --model_args pretrained=/share/home/mhma/models/Qwen2-VL-2B-Instruct,method=snapkv,budgets=0.1,snapkv_head_adaptive=True,pooling=avgpool\
    --tasks textvqa_val  \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix textvqa_val \
    --output_path ./logs/qwen2vl/textvqa_val/






