import argparse
from threading import Thread

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
from transformers import TextStreamer
from vary.model.plug.transforms import train_transform, test_transform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
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
        model_dir = '/data/firebux/vary-llava80k/'
        model_name = os.path.expanduser(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = varyConfig.from_pretrained(model_name, trust_remote_code=True)
        with init_empty_weights():
            model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.float16)
        no_split_modules = model._no_split_modules
        print(f"no_split_modules: {no_split_modules}", flush=True)
        map_list = {0: "20GB", 1: "20GB"}
        device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)

        # 加速
        model = load_checkpoint_and_dispatch(model, checkpoint=model_dir, device_map=device_map)
        model.to(device='cuda')

        self.model = model
        self.tokenizer = tokenizer


def get_model() -> Model:
    return Model()

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = varyQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda',
                                                trust_remote_code=True)

    model.to(device='cuda')

    # TODO download clip-vit in huggingface
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = test_transform
    use_im_start_end = True
    image_token_len = 256

    qs = args.query #'Provide the ocr results of this image.'
    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image = load_image(args.image_file)
    image_1 = image.copy()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor_1 = image_processor_high(image_1)

    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        inputs=input_ids,
        images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
        do_sample=True,
        num_beams=1,
        streamer=streamer,
        max_new_tokens=2048,
        stopping_criteria=[stopping_criteria]
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    full_content = ""
    for text in streamer:
        full_content += text
    print(full_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--query", type=str, default='Written all the texts.')
    args = parser.parse_args()

    eval_model(args)
