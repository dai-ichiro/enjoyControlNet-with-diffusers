
## How to use
~~~
python cv2_canny.py ^
  --image sample.jpg
~~~

~~~
python canny2image_torch2.py ^
  --model model/Brav6 ^
  --controlnet controlnet/control_v11p_sd15_canny ^
  --vae vae/ft-mse-840000 ^
  --scheduler multistepdpm ^
  --prompt prompt.txt ^
  --image canny_results/50_50.png ^
  --from_canny ^
  --n_samples 30
~~~

## Link to my Blog
https://touch-sp.hatenablog.com/entry/2023/07/08/161506
