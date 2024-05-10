import argparse
import base64
import logging
from threading import Thread
from typing import Optional

import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import os
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from vary.model import *
from vary.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from vary.model.plug.transforms import train_transform, test_transform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def load_image(image_file: str, image_type: str = 'path'):
    if image_type == 'base64':
        img_data = base64.b64decode(image_file)
        img_io = BytesIO(img_data)
        image = Image.open(img_io).convert('RGB')
    elif image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    return image


from singleton_decorator import singleton


@singleton
class Model:
    def __init__(self):
        # Model
        disable_torch_init()
        model_path = "/data/firebux/workspace/Vary-toy"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = varyQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='cuda',
                                                    trust_remote_code=True)
        model.to(device='cuda', dtype=torch.bfloat16)

        self.model = model
        self.tokenizer = tokenizer


def get_model() -> Model:
    return Model()


def eval_model(args):
    model_obj = get_model()
    tokenizer = model_obj.tokenizer
    model = model_obj.model

    # TODO download clip-vit in huggingface
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    use_im_start_end = True
    image_token_len = 256

    qs = args.query  # 'Provide the ocr results of this image.'
    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image = load_image(args.image_file, args.image_type)
    image_1 = image.copy()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor_1 = image_processor_high(image_1)

    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=True,
            num_beams=1,
            temperature=0.1,
            streamer=streamer,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
            top_k=5,
        )
    # generation_kwargs = dict(
    #     inputs=input_ids,
    #     images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
    #     do_sample=True,
    #     num_beams=1,
    #     streamer=streamer,
    #     max_new_tokens=2048,
    #     stopping_criteria=[stopping_criteria]
    # )
    # thread = Thread(target=model.generate, kwargs=generation_kwargs)
    # thread.start()

    full_content = ""
    for text in streamer:
        full_content += text
    print(full_content)

    return full_content


class ChatRequest(BaseModel):
    image_type: Optional[str] = 'path'  # url、path、base64
    image_file: str
    query: str


app = FastAPI()


@app.post("/v1/chat")
async def create_chat_completion(request: ChatRequest):
    logging.info("image_type:{}", request.image_type)
    outputs = eval_model(request)
    return Response(content=outputs)


if __name__ == "__main__":
    uvicorn.run("eval_varytoy:app", host="0.0.0.0", port=7891, log_level="info")

#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
#     parser.add_argument("--image-file", type=str, required=True)
#     parser.add_argument("--conv-mode", type=str, default=None)
#     parser.add_argument("--query", type=str, default='Written all the texts.')
#     args = parser.parse_args()
#
#     eval_model(args)
