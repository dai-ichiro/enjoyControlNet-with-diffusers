from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
import torch
import numpy

image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-xlarge")
model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-xlarge")

'''
filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
)
'''
image = load_image('1.jpg')

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

from torch import nn
logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                                    size=image.size[::-1], # (height, width)
                                    mode='bilinear',
                                    align_corners=False)

predicted = (logits.argmax(1))[0].numpy()

import numpy as np 

seg = np.where(predicted==12, 255, 0)

import cv2
cv2.imwrite('result.png', seg)



'''
logits = outputs.logits  # shape (batch_size, num_labels, height, width)
list(logits.shape)
[1, 150, 512, 512]
'''