CUDA_VISIBLE_DEVICES=0 python -m pruner.ecoflap_wanda \
    --sparsity_ratio 0.1 \
    --max_sparsity_per_layer 0.9 \
    --num_samples 128 \
    --num_data_first_stage 128

CUDA_VISIBLE_DEVICES=1 python -m pruner.ecoflap_wanda \
    --sparsity_ratio 0.3 \
    --max_sparsity_per_layer 0.9 \
    --num_samples 128 \
    --num_data_first_stage 128


CUDA_VISIBLE_DEVICES=1 python -m pruner.ecoflap_wanda \
    --pruner_name llava_wanda_pruner \
    --sparsity_ratio 0.8 \
    --max_sparsity_per_layer 0.9 \
    --num_samples 128 \
    --num_data_first_stage 128


CUDA_VISIBLE_DEVICES=0 python -m pruner.sparsegpt \
    --pruner_name llava_sparsegpt_pruner \
    --sparsity_ratio 0.5 \
    --max_sparsity_per_layer 0.9 \
    --num_samples 128 \
    --num_data_first_stage 128