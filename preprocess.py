import cv2              
from PIL import Image   
import os
from controlnet_aux.processor import Processor
from argparse import ArgumentParser

def controlnet_preprocess(video_file, processor_type):

    processor = Processor(processor_type)
    
    cap = cv2.VideoCapture(video_file)

    post_images = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        result = processor(img, to_pil=True)
        post_images.append(result)
    
    return post_images

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help='video file name'
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="preprocess type"
    )
    parser.add_argument(
        "--to_gif",
        action="store_true",
        help="output type, gif or not"
    )

    args = parser.parse_args()
    images = controlnet_preprocess(args.video, args.type)

    if args.to_gif:
        from diffusers.utils import export_to_gif
        export_to_gif(images, f"{args.type}.gif")
    else:
        os.makedirs(args.type, exist_ok=False)
        for i, image in enumerate(images):
            image.save(os.path.join(args.type, f"{i}.png"))