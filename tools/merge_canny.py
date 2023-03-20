from PIL import Image
import numpy as np
import cv2
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--image1',
    required=True,
    type=str,
    help='base image'
)
parser.add_argument(
    '--image2',
    required=True,
    type=str,
    help='second image'
)
args = parser.parse_args()
image1 = np.array(Image.open(args.image1))
image2 = np.array(Image.open(args.image2))

source_window = "draw_rectangle"
cv2.namedWindow(source_window)
rect = cv2.selectROI(source_window, image2, False, False)
# rect:(x1, y1, w, h)
# convert (x1, y1, w, h) to (x1, y1, x2, y2)
print(f'x_min = {rect[0]}')
print(f'x_max = {rect[0] + rect[2]}')
print(f'y_min = {rect[1]}')
print(f'y_max = {rect[1] + rect[3]}')

x_min = rect[0]
x_max = rect[0] + rect[2]
y_min = rect[1]
y_max = rect[1] + rect[3]

image1[y_min:y_max, x_min:x_max] = image2[y_min:y_max, x_min:x_max]

merge_image = Image.fromarray(image1)
merge_image.save('merged_image.png')