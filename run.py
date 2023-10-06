import argparse
import os
import subprocess
import json
import sys
import numpy as np
import argparse
import json, yaml
import os
import os.path as osp
import torch
import pyvista as pv
import trimesh
import time
from tqdm import tqdm
import natsort
import glob
import pymeshlab
from fitting_pcd import initialize, sample_shape_space
from mesh_preprocess_inference import mesh_preprocess
from parametrize_NPHM_dir import parameterize
from map2Clip import process_lcl as map2Clip_process
from normal_tex_dir import normal_tex
from diffusers import DiffusionPipeline
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline
import cv2
import mediapipe as mp
from nphm_2_staged_warp import process

parser = argparse.ArgumentParser(
    description='Run ClipHead')
parser.add_argument('-p','--prompt',help="prompt",type=str,required=True)

parser.add_argument('-resolution' , default=256, type=int)
parser.add_argument('-batch_points', default=20000, type=int)
parser.add_argument('-cfg_file', type=str, required=False,default='./NPHM/scripts/configs/fitting_nphm.yaml')
parser.add_argument('-exp_name', type=str, required=False,default='EXP_NAME')
parser.add_argument('-exp_tag', type=str, required=False,default='EXP_TAG')
parser.add_argument('-points', type=str, required=False)
parser.add_argument('-demo', required=False, action='store_true')
parser.set_defaults(demo=False)
parser.add_argument('-sample', required=False, action='store_true')
parser.set_defaults(sample=False)
parser.add_argument('-custom', required=False, action='store_true')
parser.add_argument('-hair', required=False, action='store_true')
parser.add_argument('-neutral', required=False, action='store_true')
parser.add_argument('-exp', required=False, action='store_true')
parser.add_argument('-tmp', required=False, action='store_true')

parser.add_argument('--input',type=str,required=False,default="./results/out/in/normal_map.png")
parser.add_argument('--a_prompt',type=str,default='best quality, photoreal hair, 8k')
parser.add_argument('--n_prompt',type=str,default='lowres, cropped, worst quality, low quality')
parser.add_argument('--num_samples',type=int,default=1)
parser.add_argument('--image_resolution',type=int,default=512)
parser.add_argument('--warp_resolution',type=int,default=768)
parser.add_argument('--detect_resolution',type=float,default=0.4)
parser.add_argument('--ddim_steps',type=int,default=25)
parser.add_argument('--guess_mode',type=bool,default=False)
parser.add_argument('--strength',type=float,default=0.85)
parser.add_argument('--scale',type=float,default=10.0)
parser.add_argument('--seed',type=int,default=-1)
parser.add_argument('--eta',type=float,default=0.0)
parser.add_argument('--bg_threshold',type=float,default=1.2)
parser.add_argument('--radius_threshold',type=float,default=1.0)
parser.add_argument('--warp_face',type=bool,default=True)

args=parser.parse_args()

class obj:
     
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
  
def dict2obj(dict1):
     
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

os.makedirs('./results/out/in',exist_ok=True)


#----------------------------------------------------------------#
# Map clip encoding to NPHM latent space, to get a latent vector
#----------------------------------------------------------------#
params=dict2obj({'prompt':args.prompt,'cont':True,'train':True})
map2Clip_process(params)

#----------------------------------------------------------------#
# Decode the latent vector, to get a mesh conditioned on text
#----------------------------------------------------------------#
pars=initialize(args)
print('HERE1!')

sample_shape_space(pars)

#----------------------------------------------------------------#
# The mesh obtained is not parameterized, preprocess it,
# follwed by parameterization into a common space
#----------------------------------------------------------------#
params_preprocess=dict2obj({'input':'./results'})

print('HERE2!')
mesh_preprocess(params_preprocess)
parameterize(params_preprocess)

#----------------------------------------------------------------#
# Find correspondeces between 2D points, and their 3D coordinates
#----------------------------------------------------------------#
subprocess.run(['python', 'UV_to_3D.py'])

#----------------------------------------------------------------#
# Generate the normal map in parameterized space
#----------------------------------------------------------------#
normal_tex(params_preprocess)


#----------------------------------------------------------------#
# Generate a texture map conditioned on the normal texture map
#----------------------------------------------------------------#

# NOTE: USE THIS BY DEFAULT
controlnet = ControlNetModel.from_pretrained("/ssd_data/common/sollid/staged_approach_low_train_more_epochs", torch_dtype=torch.float16)
#

# NOTE:
pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
)


# NOTE:
pipe2 = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe1.scheduler = UniPCMultistepScheduler.from_config(pipe1.scheduler.config)
pipe2.scheduler = UniPCMultistepScheduler.from_config(pipe2.scheduler.config)

pipe1.safety_checker = lambda images, clip_input: (images, [False])
pipe2.safety_checker = lambda images, clip_input: (images, [False])



# pipe_xl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()



pipe1.enable_xformers_memory_efficient_attention()
pipe2.enable_xformers_memory_efficient_attention()
pipe1.to('cuda')
pipe2.to('cuda')

face_mesh=mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.2)
json_path = "./MediaPipe/geometry.json" #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
uv_map = json.load(open(json_path))['export_const_UVS']
uv_map = np.array(uv_map)



input_img=cv2.imread(args.input)
input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
output=process(pipe1,pipe2,uv_map,face_mesh,input_img,args.prompt,args.a_prompt,args.n_prompt,args.num_samples,args.image_resolution,args.detect_resolution,args.ddim_steps,args.guess_mode,args.strength,args.scale,args.seed,args.eta,args.bg_threshold,args.radius_threshold, args.warp_resolution,args.warp_face)

prompt=args.prompt.replace(',','_').replace(' ','_')
cv2.imwrite(f'./results/out/in/{prompt}.png',np.array(output[-1])[...,::-1])



#----------------------------------------------------------------#
# Move around the files
#----------------------------------------------------------------#
os.makedirs('./outputs',exist_ok=True)
os.makedirs(f'./outputs/{args.prompt}',exist_ok=True)
# mkdir -p ./outputs
# mkdir ./outputs/$1 -p
os.rename('./results/out/in/',f'./outputs/{args.prompt}/')
os.remove(f'./outputs/{args.prompt}/scan_40k_nuv_surface_pos.npy')
os.rename(f'./outputs/{args.prompt}/scan_40k_nuv.obj',f'./outputs/{args.prompt}/head_mesh.obj')
# mv ./results/out/in/* ./outputs/$1/
# rm ./outputs/$1/scan_40k_nuv_surface_pos.npy

print('###################################################################')
print(f'Output saved at:\n ./outputs/{args.prompt}/')
print('###################################################################')
