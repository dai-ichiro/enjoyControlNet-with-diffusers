# refer to https://github.com/lllyasviel/ControlNet/blob/main/annotator/util.py

import numpy as np
import cv2
from argparse import ArgumentParser

def resize_image(input_image, resolution):
    original_image = cv2.imread(input_image)
    H, W, C = original_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(original_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--image',
        required=True,
        type=str,
        help='original image'
    )
    parser.add_argument(
        '--resolution',
        required=True,
        type=int,
        help='resolution'
    )
    args = parser.parse_args()
    
    img = resize_image(args.image, resolution=args.resolution)
    cv2.imwrite('resized_image.png', img)