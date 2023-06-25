from PIL import Image   # type: ignore
from controlnet_aux import OpenposeDetector   # type: ignore      
import os
import numpy as np
from argparse import ArgumentParser
from typing import List, Optional

def pose_detection(image:str, width:Optional[int]=None, height:Optional[int]=None) -> List[str]:
    if width is None and height is None:
        original_image = np.array(Image.open(image))
    else:
        if width is None:
            original_image = np.array(Image.open(image).resize((height, height)))
        elif height is None:
            original_image = np.array(Image.open(image).resize((width, width)))
        else:
            original_image = np.array(Image.open(image).resize((width, height)))

    os.makedirs('pose_results', exist_ok=True)
    
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    result = openpose(original_image)

    save_fname = os.path.join('pose_results', 'pose0.png')
    result.save(save_fname)
    return [save_fname]

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
    pose_detection(image=args.image, width=args.W, height=args.H)