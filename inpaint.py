import os
import torch
from stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline

from diffusers import ControlNetModel
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--controlnet',
    type=str,
    required=True,
    help='controlnet'
)
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='original image'
)
parser.add_argument(
    '--mask',
    type=str,
    required=True,
    help='mask image'
)
parser.add_argument(
    '--hint',
    type=str,
    required=True,
    help='controlnet hint image'
)
parser.add_argument(
    '--W',
    type=int,
    default=512,
    help='width'
)
parser.add_argument(
    '--H',
    type=int,
    default=512,
    help='height'
)
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--prompt',
    type=str,
    help='prompt'
)
parser.add_argument(
    '--scheduler',
    type=str,
    default='pndm',
    choices=['pndm', 'multistepdpm', 'eulera']
)
opt = parser.parse_args()

width = opt.W
height = opt.H

controlnet_model = opt.controlnet

n_samples = opt.n_samples

controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "model/stable-diffusion-inpainting", 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16).to('cuda')

scheduler = opt.scheduler
match scheduler:
    case 'pmdn':
        from diffusers import  PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    case 'multistepdpm':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    case 'eulera':
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    case _:
        None

pipe.enable_xformers_memory_efficient_attention()

image = load_image(opt.image).resize((width, height))
mask_image = load_image(opt.mask).resize((width, height))

if os.path.isdir(opt.hint):
    import glob
    hint_list = glob.glob(f'{opt.hint}/*.png')
    n_samples = 1
elif os.path.isfile(opt.hint):
    hint_list = [opt.hint]
print(f'n_samples: {n_samples}')

seed = opt.seed

if opt.prompt is not None and os.path.isfile(opt.prompt):
    print(f'reading prompts from {opt.prompt}')
    with open(opt.prompt, 'r') as f:
        prompt_from_file = f.readlines()
        prompt_from_file = [x.strip() for x in prompt_from_file if x.strip() != '']
        prompt_from_file = ', '.join(prompt_from_file)
        prompt = f'{prompt_from_file}, best quality, extremely detailed'
else:
    prompt = 'best quality, extremely detailed'

negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'

print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

os.makedirs('results', exist_ok=True)

for hint_image in hint_list:
    hint_fname = os.path.splitext(os.path.basename(hint_image))[0]
    controlnet_conditioning_image = load_image(hint_image).resize((width, height))
    for i in range(n_samples):
        seed_i = seed + i * 1000
        generator = torch.manual_seed(seed_i)
        image = pipe(
            prompt,
            image,
            mask_image,
            controlnet_conditioning_image,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        image.save(os.path.join('results', f"{hint_fname}_seed{seed_i}_{scheduler}.png"))