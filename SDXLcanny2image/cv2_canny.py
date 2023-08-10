import cv2              # type: ignore
from PIL import Image   # type: ignore
import os
import numpy as np
from argparse import ArgumentParser
from typing import List, Optional

def canny_edge_detection(image:str, width:Optional[int]=None, height:Optional[int]=None) -> List[str]:
    if width is None and height is None:
        original_image = np.array(Image.open(image))
    else:
        if width is None:
            original_image = np.array(Image.open(image).resize((height, height)))
        elif height is None:
            original_image = np.array(Image.open(image).resize((width, width)))
        else:
            original_image = np.array(Image.open(image).resize((width, height)))
        
    threshold1_list = [25, 50, 100, 150, 200]

    os.makedirs('canny_results', exist_ok=True)
    return_list = []
    for threshold1 in threshold1_list:
        threshold2_list = [x for x in threshold1_list if x >= threshold1]
        for threshold2 in threshold2_list:
            control = cv2.Canny(original_image, threshold1=threshold1, threshold2=threshold2)
            save_fname = os.path.join('canny_results', f'{threshold1}_{threshold2}.png')
            Image.fromarray(control).save(save_fname)
            return_list.append(save_fname)
    return return_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--image',
        required=True,
        type=str,
        help='original image'
    )
    parser.add_argument(
        '--W',
        type=int,
        help='width'
    )
    parser.add_argument(
        '--H',
        type=int,
        help='height'
    )
    args = parser.parse_args()
    canny_edge_detection(image=args.image, width=args.W, height=args.H)