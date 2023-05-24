import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from argparse import ArgumentParser
from diffusers import DiffusionPipeline
from compel import Compel

def main(args):

       input_image = load_image(args.image)

       pipe = DiffusionPipeline.from_pretrained(
              args.model,
              safety_checker=None,
              torch_dtype=torch.float16,
              custom_pipeline="stable_diffusion_reference",
              ).to('cuda:0')
       pipe.load_textual_inversion("embeddings", weight_name="EasyNegative.safetensors", token="EasyNegative")
       pipe.load_textual_inversion("embeddings", weight_name="ng_deepnegative_v1_75t.pt", token="ng_deepnegative_v1_75t")
       pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
       compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

       prompt = "masterpiece+++, photorealistic+++, (best quality)+++, attractive, highly detailed, photo of pretty Japanese woman, short hair"
       negative_prompt = "EasyNegative, ng_deepnegative_v1_75t, (Worst Quality)+++"
       prompt_embeds = compel([prompt])
       negative_prompt_embeds = compel([negative_prompt])

       result_img = pipe(
              ref_image=input_image,
              prompt_embeds = prompt_embeds,
              negative_prompt_embeds=negative_prompt_embeds,
              num_inference_steps=40,
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