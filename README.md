## Accelerating Semantic Appearance Transfer with CLIP

<p align="center">
<img src="https://github.com/john19981016/Accelerating-Semantic-Appearance-Transfer-with-CLIP/blob/main/images/intro.png" width=100% height=100% 
class="center">
</p>

## Overview
<p align="center">
<img src="https://github.com/john19981016/Accelerating-Semantic-Appearance-Transfer-with-CLIP/blob/main/images/overview.png" width=100% height=100% 
class="center">
</p>

## Results
<p align="center">
<img src="https://github.com/john19981016/Accelerating-Semantic-Appearance-Transfer-with-CLIP/blob/main/images/qualitative.png" width=100% height=100% 
class="center">
</p>

<p align="center">
<img src="https://github.com/john19981016/Accelerating-Semantic-Appearance-Transfer-with-CLIP/blob/main/images/speed.png" width=100% height=100% 
class="center">
</p>



## Set up the environment
$ pip3 install -r requirements.txt
$ pip3 install git+https://github.com/openai/CLIP.git

## Usage
$ python3 splice.py

If you want to change the images and texts,
please change the images in datasets/curr_pair/A and datasets/curr_pair/B
and the text and local_text in splice.py

## Reference
https://github.com/omerbt/Splice

