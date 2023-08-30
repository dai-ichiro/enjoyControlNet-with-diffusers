import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import sys

torch.hub.help(
    "intel-isl/MiDaS",
    "DPT_BEiT_L_384",
    force_reload=True
    ) 
model_zoe_n = torch.hub.load(
    "isl-org/ZoeDepth",
    "ZoeD_NK",
    pretrained=True
    ).to("cuda")

image_file = sys.argv[1]

image = load_image(image_file)

depth_numpy = model_zoe_n.infer_pil(image)  # return: numpy.ndarray

from zoedepth.utils.misc import colorize
colored = colorize(depth_numpy) # numpy.ndarray => numpy.ndarray

# gamma correction
img = colored / 255
img = np.power(img, 2.2)
img = (img * 255).astype(np.uint8)

Image.fromarray(img).save("zoe_depth.png")