import cv2
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    '--image',
    required=True,
    type=str,
    help='original image'
)
parser.add_argument(
    '--ksize',
    default=5,
    type=int,
    help='ksize'
)
parser.add_argument(
    '--threshold',
    default=200,
    type=int,
    help='threshold'
)
args = parser.parse_args()

original_image = cv2.imread(args.image)
ksize = args.ksize
threshold = args.threshold

img = cv2.Sobel(original_image, -1, 1, 0, ksize=ksize)

sobel_x_img = cv2.convertScaleAbs(img)
img_grey = cv2.cvtColor(sobel_x_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('result_grey_x.png', img_grey)

img_bool = np.where(img_grey > threshold, 255, 0)
cv2.imwrite(f'result_bool_{threshold}_x.png', img_bool)

img = cv2.Sobel(original_image, -1, 0, 1, ksize=ksize)

sobel_x_img = cv2.convertScaleAbs(img)
img_grey = cv2.cvtColor(sobel_x_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('result_grey_y.png', img_grey)

img_bool = np.where(img_grey > threshold, 255, 0)
cv2.imwrite(f'result_bool_{threshold}_y.png', img_bool)

img = cv2.Sobel(original_image, -1, 1, 1, ksize=ksize)

sobel_x_img = cv2.convertScaleAbs(img)
img_grey = cv2.cvtColor(sobel_x_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('result_grey_both.png', img_grey)

img_bool = np.where(img_grey > threshold, 255, 0)
cv2.imwrite(f'result_bool_{threshold}_both.png', img_bool)



