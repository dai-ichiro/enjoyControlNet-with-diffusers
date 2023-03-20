from controlnet_aux import OpenposeDetector
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
opt = parser.parse_args()

image = opt.image

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

image = Image.open(image)
result = openpose(image)

result.save('pose.png')
