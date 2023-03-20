from controlnet_aux import HEDdetector
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
parser.add_argument(
    '--resolution',
    type=int,
    help='resolution'
)
opt = parser.parse_args()

resolution = opt.resolution

if resolution is None:
    image = Image.open(opt.image)
    resolution = image.height
else:
    image = Image.open(opt.image).resize((resolution, resolution))

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
result = hed(image, detect_resolution=resolution, image_resolution=resolution)

result.save('hed.png')
