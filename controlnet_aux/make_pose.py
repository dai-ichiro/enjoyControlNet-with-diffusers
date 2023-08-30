from controlnet_aux.processor import Processor
from diffusers.utils import load_image
import sys

image_file = sys.argv[1]

image = load_image(image_file)

processor = Processor("openpose")
openpose_image = processor(image, to_pil=True).resize((1024, 1024))
openpose_image.save("pose.png")