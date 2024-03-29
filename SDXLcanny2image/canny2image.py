import os 
os.makedirs('results', exist_ok=True)

import glob
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--controlnet',
    type=str,
    default='controlnet/controlnet-sdxl-1.0'
)
parser.add_argument(
    '--model',
    type=str,
    default='model/stable-diffusion-xl-base-1.0',
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
    '--from_canny',
    action="store_true",
    help='if true, use canny image'
)
parser.add_argument(
    '--steps',
    default=30,
    type=int,
    help='num_inference_steps',
)
parser.add_argument(
    '--prompt',
    type=str,
    help='prompt'
)
parser.add_argument(
    '--scheduler',
    type=str,
    default='none',
    choices=['pndm', 'multistepdpm', 'eulera', 'none']
)
args = parser.parse_args()

seed = args.seed
steps = args.steps
scale_list = args.scale

control_model_id = args.controlnet
base_model_id = args.model

n_samples = args.n_samples

if args.from_canny:
    if os.path.isdir(args.image):
        controlhint_list = glob.glob(f'{args.image}/*.png')
        n_samples = 1
    else:
        controlhint_list = [args.image]
else:
    if os.path.isdir(args.image):
        assert False, 'image argument should be a image file name, not folder'
    else:
        from cv2_canny import canny_edge_detection
        controlhint_list = canny_edge_detection(args.image)
        n_samples = 1

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16)

controlnet = ControlNetModel.from_pretrained(
    control_model_id,
    torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    vae=vae,
    controlnet=controlnet,
    torch_dtype=torch.float16).to('cuda')

scheduler = args.scheduler
match scheduler:
    case 'pmdn':
        from diffusers import  PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    case 'multistepdpm':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    case 'eulera':
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    case _:
        None

if args.prompt is not None and os.path.isfile(args.prompt):
    print(f'reading prompts from {args.prompt}')
    with open(args.prompt, 'r') as f:
        prompt_from_file = f.readlines()
        prompt_from_file = [x.strip() for x in prompt_from_file if x.strip() != '']
        prompt_from_file = ', '.join(prompt_from_file)
        prompt = f'{prompt_from_file}, best quality, extremely detailed'
else:
    prompt = 'pretty japanese woman, 8k, extremely detailed'

negative_prompt = "paintings, sketches, worst quality, low quality"

print(f'n_samples: {n_samples}')
print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

for controlhint in controlhint_list:
    hint_fname = os.path.splitext(os.path.basename(controlhint))[0]
    hint_image = load_image(controlhint)
    for i in range(n_samples):
        seed_i = seed + i * 1000
        for scale in scale_list:
            generator = torch.manual_seed(seed_i)
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                image=hint_image,
                num_inference_steps=steps, 
                generator=generator,
                guidance_scale = scale,
                ).images[0]

            image.save(os.path.join('results', f'{hint_fname}_scale{scale}_seed{seed_i}.png'))
    
