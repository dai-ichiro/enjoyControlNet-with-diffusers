from PIL import Image   
import os
from utils import colorize
from argparse import ArgumentParser
import torch

def make_zoedepth_image(image:str) -> str:

    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    model = model_zoe_nk.to('cuda')

    pil_image = Image.open(image).convert("RGB")
    depth = model.infer_pil(pil_image) # PIL.Image -> numpy.ndarray
    colored_depth = Image.fromarray(colorize(depth)).convert('L')

    os.makedirs('zoedepth_results', exist_ok=True)
    save_fname = os.path.join('zoedepth_results', 'zoedepth.png')
    colored_depth.save(save_fname)
    return save_fname

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--image',
        required=True,
        type=str,
        help='original image'
    )

    args = parser.parse_args()
    make_zoedepth_image(image=args.image)