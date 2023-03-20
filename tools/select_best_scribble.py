import os 
os.makedirs('results', exist_ok=True)

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, EulerAncestralDiscreteScheduler
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
    '--folder',
    type=str,
    required=True,
    help='scribble_results',
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
    default=9.0,    
    type=float,
    help='guidance_scale',
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
args = parser.parse_args()

seed = args.seed
steps = args.steps
scale = args.scale

width = args.W
height = args.H

vae_folder =args.vae
base_model_id = args.model

import glob
scribble_list = glob.glob(f'{args.folder}/*.png')

control_list = []
for canny in scribble_list:
    control_list.append(load_image(canny).resize((width, height)))
    
if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae').to('cuda')

controlnet = ControlNetModel.from_pretrained("basemodel/sd-controlnet-scribble")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None).to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

for i in range(args.n_samples):
    seed_i = seed + i * 1000
    for each_control in zip(control_list, scribble_list):    
        generator = torch.manual_seed(seed_i)
        image = pipe(
            prompt="no background, best quality, extremely detailed", 
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            image=each_control[0],
            width = width,
            height = height,
            num_inference_steps=steps, 
            generator=generator,
            guidance_scale = scale,
            ).images[0]
        image.save(os.path.join('results', f'seed{seed_i}_{os.path.basename(each_control[1])}'))
    
