import os 
os.makedirs('results', exist_ok=True)

import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--controlnet',
    type=str,
    default='controlnet/sd-controlnet-canny'
)
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='model',
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
    '--scale',
    nargs='*',
    default=[9.0],    
    type=float,
    help='guidance_scale',
)
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
parser.add_argument(
    '--from_depth',
    action="store_true",
    help='if true, use depth image'
)
parser.add_argument(
    '--steps',
    default=30,
    type=int,
    help='num_inference_steps',
)
parser.add_argument(
    '--vae',
    type=str,
    help='vae'
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
args = parser.parse_args()

seed = args.seed
steps = args.steps
scale_list = args.scale

vae_folder =args.vae
control_model_id = args.controlnet
base_model_id = args.model

n_samples = args.n_samples

if args.from_depth:
    controlhint = load_image(args.image)
else:
    from depth import make_depth_image
    controlhint_fname = make_depth_image(args.image)
    controlhint = load_image(controlhint_fname)

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder, torch_dtype=torch.float16).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae', torch_dtype=torch.float16).to('cuda')

controlnet = ControlNetModel.from_pretrained(control_model_id, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')

scheduler = args.scheduler
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

print(f'n_samples: {n_samples}')
print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

for i in range(n_samples):
    seed_i = seed + i * 1000
    for scale in scale_list:
        generator = torch.manual_seed(seed_i)
        image = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image=controlhint,
            num_inference_steps=steps, 
            generator=generator,
            guidance_scale = scale,
            ).images[0]

        image.save(os.path.join('results', f'scale{scale}_seed{seed_i}.png'))
    
