import os
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
    '--controlnet',
    nargs='*',
    type=str,
    required=True,
    help='list of controlnets'
)
parser.add_argument(
    '--image',
    nargs='*',
    type=str,
    required=True,
    help='list of images'
)
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--prompt',
    type=str,
    help='prompt'
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
)
args = parser.parse_args()

model_id = args.model
vae_folder =args.vae

image_list = args.image

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder, torch_dtype=torch.float16).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float16).to('cuda')

controlnet_list = [ControlNetModel.from_pretrained(x, torch_dtype=torch.float16).to('cuda') for x in args.controlnet]

image_list = [load_image(x) for x in args.image]
#control_image1 = load_image(image1)  # load_image always return RGB format image
#control_image2 = load_image(image2)  # refer to diffusers/src/diffusers/utils/testing_utils.py

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    controlnet=controlnet_list,
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

if args.prompt is not None and os.path.isfile(args.prompt):
    print(f'reading prompts from {args.prompt}')
    with open(args.prompt, 'r') as f:
        prompt_from_file = f.readlines()
        prompt_from_file = [x.strip() for x in prompt_from_file if x.strip() != '']
        prompt_from_file = ', '.join(prompt_from_file)
        prompt = f'{prompt_from_file}, best quality, extremely detailed'
else:
    prompt = 'best quality, extremely detailed'

negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'

print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

seed = args.seed

os.makedirs('results',exist_ok=True)

for i in range(args.n_samples):
    seed_i = seed + i * 1000
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image = image_list,
        generator = torch.manual_seed(seed_i),
        num_inference_steps=30,
    ).images[0]
    image.save(os.path.join('results', f"seed{seed_i}.png"))