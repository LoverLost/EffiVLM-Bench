import json
import os
import json
import torch
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
import transformers

def build_sharegpt4v_coco(data_path, image_folder, json_path):
    data = load_dataset(data_path, data_dir="sharegpt4v(coco)", split="train")

    image_folder = image_folder
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        
    converted_data = []

    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            json_data["image"] = f"{da['id']}.jpg"
            da["image"].save(os.path.join(image_folder, json_data["image"]))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)


    with open(json_path, "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)



def load_image(image_path):
    return Image.open(image_path)


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") :
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    
    
    target_ids = [[id for id in ids if id >= 0] for ids in targets]
    tokenizer.batch_decode(target_ids)
    # input_ids = [[id for id in ids if id >= 0] for ids in input_ids]
    # tokenizer.batch_decode(input_ids)
    # check the input_ids and targets is right or not
    
    
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )





class LocalLlavaDataset(Dataset):
    """
      {
        "id": "000000000072",
        "image": "000000000072.jpg",
        "conversations": [...],
        ...
      }
    """

    def __init__(self, 
                 json_path,     
                 images_dir,    
                 processor,     
                 max_length=2048):

        self.json_path = json_path
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length
        self.conv_template = "qwen_1_5"
        

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data_list = json.load(f) 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]
        img_id = record["id"]
        img_filename = record["image"] 
        conversations = record["conversations"]  

        
        # question = conversations[0]['value']
        # conv = copy.deepcopy(conv_templates[self.conv_template])
        # conv.append_message(conv.roles[0], question)
        # conv.append_message(conv.roles[1], None)
        # prompt_question = conv.get_prompt()
        
        
        image_path = os.path.join(self.images_dir, img_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = load_image(image_path)
        
        return {
            "id": img_id,
            "image": img,
            "conversations": conversations,
        }

def llava_collate_fn(samples, processor, model, tokenizer, device):
    
    batch_ids = [_ for _ in range(len(samples))]
    batch_images = [s["image"] for s in samples]
    # batch_prompts = [s["prompt"] for s in samples]

    # inputs = processor(
    #     images=batch_images,
    #     text=batch_prompts,
    #     padding=True,
    #     return_tensors="pt"
    # )
    batch_ids = [_ for _ in range(len(samples))]
    batch_images = [s["image"] for s in samples]
    batch_conversations = [s["conversations"] for s in samples]  

    processed_data = [preprocess_qwen([convs], tokenizer, has_image=True) for convs in batch_conversations]
    
    batch_input_ids = [data["input_ids"].to(device=device) for data in processed_data][0]
    batch_labels = [data["labels"].to(device=device) for data in processed_data][0]

    image_tensor = process_images(batch_images, processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    # attntion_mask = torch.ones_like(batch_input_ids)
    
    
    # image_tensor = process_images(batch_images, processor, model.config)
    # image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    # input_ids = tokenizer_image_token(batch_prompts[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # 
    image_sizes = [batch_images.size for batch_images in batch_images]
    
    return {'input_ids':batch_input_ids, 
            'images': image_tensor, 
            'image_sizes': image_sizes,
            'labels': batch_labels,
            }


if __name__ == "__main__":
    
    build_sharegpt4v_coco("/share/home/mhma/datasets/LLaVA-OneVision-Data", "/share/home/mhma/MLLM-Efficiency/datasets/coco/images", "/share/home/mhma/MLLM-Efficiency/datasets/caption.json")