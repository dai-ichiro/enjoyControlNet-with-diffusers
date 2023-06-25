import os 
os.makedirs('results', exist_ok=True)

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from compel import Compel, DiffusersTextualInversionManager

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--controlnet',
    type=str,
    default='controlnet/control_v11p_sd15_openpose'
)
parser.add_argument(
    '--model',
    type=str,
    default='model/OrangeMix-v2',
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
    '--from_pose',
    action="store_true",
    help='if true, use pose image'
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

if args.from_pose:
    controlhint_list = [args.image]
else:
    from aux_pose import pose_detection
    controlhint_list = pose_detection(args.image)

if vae_folder is not None:
    #vae = AutoencoderKL.from_pretrained(vae_folder, torch_dtype=torch.float16).to('cuda')
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    #vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae', torch_dtype=torch.float16).to('cuda')
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae').to('cuda')

#controlnet = ControlNetModel.from_pretrained(control_model_id, torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(control_model_id)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    #torch_dtype=torch.float16,
    )

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

pipe.load_textual_inversion("embeddings", weight_name="EasyNegative.safetensors", token="EasyNegative")
pipe.to("cuda")

#pipe.enable_xformers_memory_efficient_attention()

prompt = "(high resolution)++, 8k+, attractive, highly detailed, pretty 1girl, dancing, short hair, plain white t-shirt, white jeans"
negative_prompt = "EasyNegative, (Worst Quality)++, (low quality)+"

textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel_proc = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    textual_inversion_manager=textual_inversion_manager,
    truncate_long_prompts=False)

prompt_embeds = compel_proc([prompt])
negative_prompt_embeds = compel_proc([negative_prompt])

print(f'n_samples: {n_samples}')
print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

for controlhint in controlhint_list:
    hint_fname = os.path.splitext(os.path.basename(controlhint))[0]
    hint_image = load_image(controlhint).resize((768, 768))
    for i in range(n_samples):
        seed_i = seed + i * 1000
        for scale in scale_list:
            generator = torch.manual_seed(seed_i)
            image = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                #prompt=prompt, 
                #negative_prompt=negative_prompt,
                image=hint_image,
                num_inference_steps=steps, 
                generator=generator,
                guidance_scale = scale,
                ).images[0]

            image.save(os.path.join('results', f'{hint_fname}_scale{scale}_seed{seed_i}.png'))
    
