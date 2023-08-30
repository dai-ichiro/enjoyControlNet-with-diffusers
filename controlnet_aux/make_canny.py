import cv2
from PIL import Image
import sys

image_file = sys.argv[1]

image = cv2.imread(image_file)
image = cv2.Canny(image, 50, 150)
canny_image = Image.fromarray(image)
canny_image.save("canny.png")