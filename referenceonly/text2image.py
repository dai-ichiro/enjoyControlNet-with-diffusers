from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
from compel import Compel, DiffusersTextualInversionManager
import os 

os.makedirs("results", exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=5,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--steps',
    type=int,
    default=25,
    help='num_inference_steps',
)
args = parser.parse_args()

seed = args.seed
steps = args.steps
scale = 7.0

model_id = "model/yayoi_mix"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    safety_checker=None)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_textual_inversion("embeddings", weight_name="EasyNegative.safetensors", token="EasyNegative")
pipe.to("cuda")

prompt = "(high resolution)++, 8k+, photorealistic+, attractive, highly detailed, pretty Japanese woman, dancing, short hair, plain white t-shirt, white long pants"
negative_prompt = "EasyNegative, (Worst Quality)++, (low quality)+"

textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel_proc = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    textual_inversion_manager=textual_inversion_manager,
    truncate_long_prompts=False)

prompt_embeds = compel_proc([prompt])
negative_prompt_embeds = compel_proc([negative_prompt])

for i in range(args.n_samples):
    temp_seed = seed + i * 100
    generator = torch.Generator(device="cuda").manual_seed(temp_seed)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds = negative_prompt_embeds,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=scale,
        width=768,
        height=768,
        ).images[0]
    image.save(os.path.join("results", f"step{steps}_seed{temp_seed}.png"))
