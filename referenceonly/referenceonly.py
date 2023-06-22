import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from argparse import ArgumentParser
from diffusers import DiffusionPipeline
from compel import Compel, DiffusersTextualInversionManager

def main(args):

       input_image = load_image(args.image)

       pipe = DiffusionPipeline.from_pretrained(
              args.model,
              safety_checker=None,
              custom_pipeline="stable_diffusion_reference",
              )
       pipe.load_textual_inversion("embeddings", weight_name="EasyNegative.safetensors", token="EasyNegative")
       pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
       pipe.to("cuda")

       textual_inversion_manager = DiffusersTextualInversionManager(pipe)
       compel = Compel(
              tokenizer=pipe.tokenizer,
              text_encoder=pipe.text_encoder,
              textual_inversion_manager=textual_inversion_manager,
              truncate_long_prompts=False)

       prompt = "(high resolution)++, 8k+, photorealistic+, attractive, highly detailed, pretty Japanese woman, dancing, short hair, plain white t-shirt, white long pants"
       negative_prompt = "EasyNegative, (Worst Quality)++, (low quality)+"

       prompt_embeds = compel([prompt])
       negative_prompt_embeds = compel([negative_prompt])

       result_img = pipe(
              ref_image=input_image,
              prompt_embeds = prompt_embeds,
              negative_prompt_embeds=negative_prompt_embeds,
              num_inference_steps=25,
              reference_attn=True,
              reference_adain=False).images[0]

       result_img.save("result.png")
       
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
       '--model',
       required=True,
       type=str,
       help="model name"
    )
    parser.add_argument(
        '--image',
        required=True,
        type=str,
        help='original image'
    )
    args = parser.parse_args()
    main(args)