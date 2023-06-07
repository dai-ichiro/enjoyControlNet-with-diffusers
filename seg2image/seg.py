from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
import torch
from torch import nn
import numpy as np
import cv2

image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-xlarge")
model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-xlarge")

image = load_image('1.jpg')

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                                    size=image.size[::-1], # (height, width)
                                    mode='bilinear',
                                    align_corners=False)

predicted = (logits.argmax(1))[0].numpy()

seg = np.where(predicted==12, 255, 0)

cv2.imwrite('result.png', seg)
