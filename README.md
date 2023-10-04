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
pip install -e .
cd ..
```

### Step 2 
- Download the [NPHM](https://github.com/SimonGiebenhain/NPHM) pretrained checkpoints from [here](https://drive.google.com/drive/folders/1dajUVhnYgRxbmX9CpAXDw702YYb0VHm9) and place it in ./NPHM/checkpoints

### Step 3
```
bash run_ch.sh
```



## Gradio Interface Coming Soon!

