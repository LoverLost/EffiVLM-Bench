import argparse


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
from quant.llava_onevison_for_awq import llava_onevision

from pruner.build_dataset import LocalLlavaDataset, llava_collate_fn
from torch.utils.data import DataLoader

from quant.llava_onevison_for_awq import LLavaOVAwqQuantizer
# replace registered map
AWQ_CAUSAL_LM_MODEL_MAP["llava"] = llava_onevision


def parse_args():
    parser = argparse.ArgumentParser(
        description='Quantize a causal language model')

    parser.add_argument(
        '--model_path',
        type=str,
        default='/share/home/mhma/models/llava-onevision-qwen2-7b-ov',
        help='Path to the model to quantize')

    parser.add_argument('--quant_path',
                        type=str,
                        default='/share/home/mhma/MLLM-Efficiency/models/quant/llava-onevision-qwen2-7b-ov-awq',
                        help='Path to save the quantized model')
    parser.add_argument('--json_path',
                        type=str,
                        default='/share/home/mhma/MLLM-Efficiency/datasets/coco/caption.json',
                        help='Path to calibration data file')
    parser.add_argument('--images_dir',
                        type=str,
                        default='/share/home/mhma/MLLM-Efficiency/datasets/coco/images',
                        help='Path to images directory')
    parser.add_argument('--device',
                    type=str,
                    default='cuda:0',
                    help='Device to run calibration on')
    # for quantization

    
    
    parser.add_argument('--calibrated',
                        type = int,
                        default = 128,
                        help='Number of calibration data')
    parser.add_argument('--zero_point',
                        type=bool,
                        default=True
                        )
    parser.add_argument('--q_group_size',
                        type=int,
                        default=128,
                        help='Group size for quantization')
    parser.add_argument('--w_bit',
                        type=int,
                        default=4,
                        help='Weight bitwidth')
    parser.add_argument('--version',
                        type=str,
                        default='GEMM',
                        help='Version of quantization')

    # for llava
    parser.add_argument("--attn_implementation", type=str,
                        default="eager",
                        help="Attention implementation to use")
    parser.add_argument('--device_map', type=str,
                        default="auto", help='Device map configuration.')
    parser.add_argument('--conv_template', type=str,
                        default="qwen_1_5", help='Convolution template.')
    parser.add_argument('--use_cache', type=bool,
                        default=True, help='Whether to use cache.')
    parser.add_argument('--truncate_context', type=bool,
                        default=False, help='Whether to truncate context.')
    parser.add_argument('--customized_config', type=str,
                        default=None, help='Path to customized config.')
    parser.add_argument('--max_frames_num', type=int,
                        default=32, help='Maximum number of frames.')
    parser.add_argument('--mm_spatial_pool_stride', type=int,
                        default=2, help='Spatial pool stride for multimodal.')
    parser.add_argument('--mm_spatial_pool_mode', type=str,
                        default="bilinear", help='Spatial pool mode for multimodal.')
    parser.add_argument('--token_strategy', type=str,
                        default="single", help='Token strategy.')
    parser.add_argument('--video_decode_backend', type=str,
                        default="decord", help='Backend for video decoding.')
    return parser.parse_args()



if __name__ == '__main__':
    
    args = parse_args()
    model_path = args.model_path
    quant_path = args.quant_path
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,    
        "version": args.version,
    }
    llava_config = {
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "conv_template": args.conv_template,
        "use_cache": args.use_cache,
        "truncate_context": args.truncate_context,
        "customized_config": args.customized_config,
        "max_frames_num": args.max_frames_num,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "token_strategy": args.token_strategy,
        "video_decode_backend": args.video_decode_backend,
    }
    calibrated_size = args.calibrated
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **llava_config
    )
    image_processor = model.image_processor
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = LocalLlavaDataset(
            json_path=args.json_path,
            images_dir=args.images_dir,
            processor=image_processor,
            max_length=2048
        )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: llava_collate_fn(x, image_processor, model.model, tokenizer, device='cpu'),
    )
    

    print('Preparing calibration data...')
    from itertools import islice
    
    
    
    calibrated_data = list(islice(data_loader, calibrated_size))

    # for quicklly debug and will removed in the future
    
    # import pickle
    # with open('/share/home/mhma/MLLM-Efficiency/datasets/llava_data_for_debug/calibrated_data.pkl', 'rb') as f:
    #     calibrated_data = pickle.load(f)

    # with open('/share/home/mhma/MLLM-Efficiency/datasets/llava_data_for_debug/calibrated_data.pkl', 'wb') as f:
    #     pickle.dump(calibrated_data, f)
        
    # inputs_data = {}
    
    
    model.quantize(model, calib_data = calibrated_data, quant_config=quant_config, quantizer_cls=LLavaOVAwqQuantizer)

    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')
