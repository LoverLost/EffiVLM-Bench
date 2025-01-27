from qwen2vl import Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
import datasets
from PIL import Image

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/share/home/mhma/models/Qwen2-VL-7B-Instruct",
    torch_dtype="bfloat16",
    # use_flash_attention_2=True,
    attn_implementation={"vision_config": "flash_attention_2", "": "eager"}, # the {"": "eager"} part doesn't work and always sets text in eager mode
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("/share/home/mhma/models/Qwen2-VL-7B-Instruct")


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image","image": "/share/home/mhma/MLLM-Efficiency/datasets/chartqa/images/image.png"},
            {"type": "text", "text": "What's the value of the lowest bar?"},
        ],
    }
]

images = Image.open("/share/home/mhma/MLLM-Efficiency/datasets/chartqa/images/image.png")
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(images=images, text=prompt, return_tensors="pt", padding=True).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)