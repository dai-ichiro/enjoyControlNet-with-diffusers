import os
import torch
from diffusers import AutoPipelineForText2Image, ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

import json

config = json.load(open("config.json"))

width = config["width"]
height = config["height"]

start = config["seed_first"]
step = config["seed_step"]
end = start + step * config["n_samples"]

seeds = range(start, end, step)

controlnet_list = []
image_list = []
controlnet_scale = []
controlnet_name_list = []

for controlnet in config["controlnet"]:
    if controlnet["enable"]:
        controlnet_list.append(
            ControlNetModel.from_pretrained(controlnet["path"], torch_dtype=torch.float16).to('cuda')
        )
        image_list.append(
            load_image(controlnet["image"]).resize((width, height))
        )
        controlnet_scale.append(
            controlnet["conditioning_scale"]
        )
        controlnet_name_list.append(
            controlnet["name"]
        )

model_name = config["model"]
pipe = AutoPipelineForText2Image.from_pretrained(
    model_name,
    controlnet=controlnet_list,
    safety_checker=None,
    torch_dtype=torch.float16
).to('cuda')
print(f"model: {model_name}")

match config.get("scheduler"):
    case "pmdn":
        from diffusers import  PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        print("scheduler: pmdn")
    case "multistepdpm":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True
        )
        print("scheduler: multisteppdm")
    case "eulera":
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        print("scheduler: eulera")
    case _:
        None

for textual_inversion in config["textual_inversion"]:
    if textual_inversion["enable"]:
        file_path = textual_inversion["path"]
        pipe.load_textual_inversion(file_path, token=textual_inversion["token"])
        print(f"load {file_path}")

prompt = config["prompt"]
n_prompt = config["n_prompt"]

print(f'prompt: {prompt}')
print(f'negative prompt: {n_prompt}')

for lora in config["lora"]:
    if lora["enable"]:
        file_path = lora["path"]
        pipe.load_lora_weights(file_path)
        print(f"load {file_path}")

save_folder = "_".join(controlnet_name_list)
os.makedirs(save_folder, exist_ok=True)

for i in seeds:
    image = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image = image_list,
        generator = torch.manual_seed(i),
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=config["num_inference_steps"],
    ).images[0]
    image.save(os.path.join(save_folder, f"seed{i}.png"))
