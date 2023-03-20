import os
from controlnet_aux import HEDdetector
import numpy as np
from PIL import Image
import argparse
from typing import List

def make_scribble_image(image:str, threshold:float=None) -> List[str]:
    
    if threshold is None:
        threshold_list = [x * 0.1 for x in range(1, 10)]
    else:
        threshold_list = [threshold]

    image = Image.open(image)
    resolution = image.height
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
    hed_array = np.array(hed(image, detect_resolution=resolution, image_resolution=resolution).convert('L'))
    
    os.makedirs('scribble_results', exist_ok=True)
    return_list = []
    for threshold in threshold_list:
        bool_array = hed_array > (255 * threshold)
        result_array = np.where(bool_array == False, 0, 255).astype(np.uint8)
        save_fname = os.path.join('scribble_results', f'sketch{threshold:.1f}.png')
        Image.fromarray(result_array, mode='L').save(save_fname)
        return_list.append(save_fname)
    return return_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold',
        type=float,
        help='threshold',
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='path to original image'
    )
    args = parser.parse_args()

    make_scribble_image(image=args.image, threshold=args.threshold)



    

