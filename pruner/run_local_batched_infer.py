import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
from llava.model.builder import load_pretrained_model
from build_dataset import LocalLlavaDataset, llava_collate_fn
from processing_llava import LlavaProcessor
def main():
    pretrained = "/share/home/mhma/models/llava-onevision-qwen2-7b-ov"
    llava_model_args = {
        "attn_implementation": "eager", 
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "multimodal": True
    }
    
    model_name = get_model_name_from_path(pretrained)
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args)
    model.config.tokenizer_padding_side = 'left' 
    
    
    model.eval()
    processor = LlavaProcessor.from_pretrained(pretrained)

    json_path = "/share/home/mhma/MLLM-Efficiency/datasets/coco/caption.json"
    images_dir = "/share/home/mhma/MLLM-Efficiency/datasets/coco/images"
    dataset = LocalLlavaDataset(
        json_path=json_path,
        images_dir=images_dir,
        processor=processor,
        max_length=2048
    )

    loader = DataLoader(
        dataset,
        batch_size=1,     
        shuffle=False,
        collate_fn=lambda x: llava_collate_fn(x, image_processor,model,tokenizer,device='cuda'),
    )

    results = []
    for input_ids, image_tensor, image_sizes in loader:
        # for k,v in inputs.items():
        #     inputs[k] = v.to(model.device, torch.bfloat16)

        with torch.no_grad():
            generate_ids = model.generate(input_ids, image_tensor, image_sizes, max_new_tokens=40)
        output_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)


if __name__ == "__main__":
    main()
