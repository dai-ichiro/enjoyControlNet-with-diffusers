# ControlNet with Diffusers
## Requirements

~~~
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors opencv-python
pip install xformers==0.0.17.dev466
~~~

## New Requirements (PyTorch 2.0)

~~~
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors opencv-python
pip install xformers
~~~

### option 1

~~~
pip install controlnet-aux
~~~

### option 2

~~~
pip install rembg[gpu]
~~~

### option 3 (when use zoedepth)

~~~
pip install matplotlib
pip install timm
~~~

## How to use canny2image.py

~~~
python canny2image.py ^
  --model model\Counterfeit-V2.5 ^
  --vae vae\counterfeit_vae ^
  --prompt prompt.txt ^
  --image sample.jpg ^
~~~

#### For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/02/23/181611

## How to use multi_controlnet.py

~~~
python multi_controlnet.py ^
  --controlnet controlnet/control_v11p_sd15_canny controlnet/control_v11f1p_sd15_depth ^
  --image canny.png depth.png ^
  --model model/anything-v4.0 ^
  --vae vae/anime2_vae ^
  --prompt prompt.txt ^
  --n_samples 20
~~~

#### For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/03/13/141954

## How to use inpaint.py

~~~
python inpaint.py ^
  --controlnet controlnet\sd-controlnet-openpose ^
  --image original_image.jpg ^
  --mask mask.png ^
  --hint pose.png ^
  --W 768 --H 768 ^
  --prompt prompt.txt ^
  --seed 40000 ^
  --n_samples 10
~~~

#### For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/03/10/135439

