import os
import numpy as np
from PIL import Image

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--image', 
    type=str, 
    required=True,
    help='original image')
parser.add_argument(
    '--mask', 
    type=str, 
    required=True,
    help='mask image')
parser.add_argument(
    '--reverse', 
    action='store_true')
opt = parser.parse_args()

image = np.array(Image.open(opt.image))
mask = np.array(Image.open(opt.mask))

if image.ndim == 3 and mask.ndim ==2:
    mask = mask[:, :, None]
if image.ndim == 2 and mask.ndim ==3:
    image = image[:, :, None]

if opt.reverse:
    result = image * (mask == 255)
else:
    result = image * (mask != 255)

save_fname = os.path.splitext(os.path.basename(opt.image))[0]
Image.fromarray(result).save(f'mask_{save_fname}.png')    