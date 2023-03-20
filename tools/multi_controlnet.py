import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='model',
)
parser.add_argument(
    '--vae',
    type=str,
    help='vae'
)
parser.add_argument(
    '--controlnet1',
    type=str,
    required=True,
    help='controlnet1'
)
parser.add_argument(
    '--controlnet2',
    type=str,
    required=True,
    help='controlnet2'
)
parser.add_argument(
    '--controlnet1_image',
    type=str,
    required=True,
    help='image for controlnet1'
)
parser.add_argument(
    '--controlnet2_image',
    type=str,
    required=True,
    help='image for controlnet2'
)
parser.add_argument(
    '--seed',
    type=int,
    default=19,
    help='the seed (for reproducible sampling)',
)
args = parser.parse_args()

model_id = args.model
vae_folder =args.vae

controlnet1 = args.controlnet1
controlnet2 = args.controlnet2

image1 = args.controlnet1_image
image2 = args.controlnet2_image

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder, torch_dtype=torch.float16).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float16).to('cuda')

controlnet_processor1 = ControlNetModel.from_pretrained(controlnet1, torch_dtype=torch.float16).to('cuda')
controlnet_processor2 = ControlNetModel.from_pretrained(controlnet2, torch_dtype=torch.float16).to('cuda')

control_image1 = load_image(image1)  # load_image always return RGB format image
control_image2 = load_image(image2)  # refer to diffusers/src/diffusers/utils/testing_utils.py

prompt = "a beautiful girl wearing high neck sweater, best quality, extremely detailed, cowboy shot"
negative_prompt = "cowboy, monochrome, lowres, bad anatomy, worst quality, low quality"

seed = args.seed

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    controlnet=[controlnet_processor1, controlnet_processor2],
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image = [control_image1, control_image2],
    generator = torch.manual_seed(seed),
    num_inference_steps=30,
).images[0]
image.save(f"./controlnet_both_result_{seed}.png")#