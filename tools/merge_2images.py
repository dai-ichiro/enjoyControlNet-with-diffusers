import os
import cv2
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--image1', type=str, help='primary image' )
parser.add_argument('--image2', type=str, help='secondary image' )
opt = parser.parse_args()

img1_path = opt.image1
img1_fname_no_ext = os.path.splitext(os.path.basename(img1_path))[0]

img2_path = opt.image2
img2_fname_no_ext = os.path.splitext(os.path.basename(img2_path))[0]

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

source_window = "select area"
cv2.namedWindow(source_window)

rect = cv2.selectROI(source_window, img2, False, False)
cv2.destroyAllWindows()

xmin, ymin, width, height = rect

cv2.rectangle(img1, (xmin, ymin), (xmin+width, ymin+height), 0, -1)

img2_after = np.zeros_like(img2)
img2_after[ymin:ymin+height, xmin:xmin+width] = img2[ymin:ymin+height, xmin:xmin+width]

cv2.imwrite(f'{img1_fname_no_ext}_mask.png', img1)
cv2.imwrite(f'{img2_fname_no_ext}_mask.png', img2_after)
