import os
from transformers import pipeline
from diffusers.utils import load_image
import argparse

def make_depth_image(image:str) -> str:
    depth_estimator = pipeline('depth-estimation')

    image = load_image(image)
    image = depth_estimator(image)['depth']
    
    os.makedirs('depth_results', exist_ok=True)
    save_fname = os.path.join('depth_results', 'depth.png')
    image.save(save_fname)
    return save_fname

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='path to original image'
    )
    args = parser.parse_args()

    make_depth_image(args.image)


