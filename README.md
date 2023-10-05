# CLIP-Head
Official Implementation of the paper "CLIP-Head: Text-Guided Generation of Textured Neural Parametric 3D Head Models"

## Method
![Architecture](./images/Pipeline.png)

## Instructions

### Step 1
- Clone the repo
```
cd CLIP-Head
conda create -n clip_head python=3.9
conda activate clip_head
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
cd NPHM
mkdir checkpoints
pip install -e .
cd ..
```

### Step 2 
- Download the [NPHM](https://github.com/SimonGiebenhain/NPHM) pretrained checkpoints from [here](https://drive.google.com/drive/folders/1dajUVhnYgRxbmX9CpAXDw702YYb0VHm9) and place it in ./NPHM/checkpoints
- Download checkpoints for $ControlNet_{uv}$ from [here](https://drive.google.com/file/d/1ReBlV7BX6eIbrIjYj2MV7AeLAZeP3aft/view?usp=sharing).

### Step 3
```
bash run_ch.sh <prompt>
```



## Gradio Interface Coming Soon!

