import os
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='original image'
)
parser.add_argument(
    '--size',
    type=int,
    default=8,
    help='rectangle size'
)
parser.add_argument(
    '--color',
    type=str,
    default='black',
    choices=['black', 'white'],
    help='line color'
)


def nothing(x):
    pass

opt = parser.parse_args()
color = opt.color
size = opt.size

img_path = opt.image
img_fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]

drawing = False 

def draw(event,x,y, flags, param):

    global drawing, mask_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing == True:
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
            xmin = x - size
            ymin = y - size
            xmax = x + size
            ymax = y + size

            if color=='white':
                cv2.rectangle(original_image, (xmin,ymin), (xmax, ymax), 255, -1)
            elif color=='black':
                cv2.rectangle(original_image, (xmin,ymin), (xmax, ymax), 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

width, height = original_image.shape[0:2]
position_x = 100
position_y = 100

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.moveWindow('image', position_x, position_y)
cv2.setMouseCallback('image', draw)

while True:  
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        cv2.imwrite(f'{img_fname_no_ext}_modified.png', original_image)
    elif k == ord('u'):
        position_y -= 200
        cv2.moveWindow('image', position_x, position_y)
    elif k == ord('d'):
        position_y += 200
        cv2.moveWindow('image', position_x, position_y)
    elif k & 0xFF == 27:
        break
    cv2.imshow('image',original_image)

cv2.destroyAllWindows()