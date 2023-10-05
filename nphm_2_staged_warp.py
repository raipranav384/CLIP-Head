
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline
# from src.pipeline_stable_diffusion_controlnet_inpaint import *

from diffusers import DiffusionPipeline
import torch

import open_clip
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
import skimage
import mediapipe as mp
import json
# import matplotlib.pyplot as plt
import time
# import kornia.geometry.transform as tf
import argparse

def fix_uv_misalignment(normal_cropped, texture_cropped, uv_res = (512, 512),uv_map=None,face_mesh=None):
    device=torch.device('cpu')
    H_uv, W_uv = uv_res
    keypoints_uv = np.array([(W_uv*x, H_uv*y) for x,y in uv_map*[1,1]]) # fixed for any image
    # calculate warp transform for normal cropped normal map
    H_normal,W_normal,_ = normal_cropped.shape

    results = face_mesh.process(normal_cropped)

    if results.multi_face_landmarks==None:
        return None
    assert len(results.multi_face_landmarks)==1 

    face_landmarks = results.multi_face_landmarks[0]
    keypoints = np.array([(W_normal*(point.x),H_normal*(1-point.y)) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else
    keypoints = np.array([(W_normal*point.x,H_normal*point.y) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else
    
    # breakpoint()
    # keypoints_uv_ten=torch.tensor(keypoints_uv,dtype=torch.float32,device=device).unsqueeze(0)/max(W_uv,H_uv)
    # keypoints_ten=torch.tensor(keypoints,dtype=torch.float32,device=device).unsqueeze(0)/(max(W_normal,H_normal))

    tform_img_normal = PiecewiseAffineTransform()
    tform_img_normal.estimate(keypoints,keypoints_uv)
    
    # calculate warp transform for cropped texture map
    H_tex,W_tex,_ = texture_cropped.shape
    
    results = face_mesh.process(texture_cropped)
    if results.multi_face_landmarks==None:
        return None
    assert len(results.multi_face_landmarks)==1 

    face_landmarks = results.multi_face_landmarks[0]
    keypoints = np.array([(W_tex*point.x,H_tex*point.y) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else

    keypoints_ten=torch.tensor(keypoints,dtype=torch.float32,device=device).unsqueeze(0)/(max(W_normal,H_normal))


    tform_uv_mis = PiecewiseAffineTransform()
    tform_uv_mis.estimate(keypoints_uv,keypoints)

    # calculate warp transform for cropped texture map

    tform_img_mis = PiecewiseAffineTransform()
    t1=time.time()
    tform_img_mis.estimate(keypoints,keypoints_uv)
    print("Estimateion time:",time.time()-t1)
    texture_cropped_ten=torch.tensor(texture_cropped,dtype=torch.float32,device=device).permute(2,0,1).unsqueeze(0)

    texture_cropped_ten=texture_cropped_ten/255.0


    mask_uv = warp(texture_cropped, tform_uv_mis, output_shape=uv_res)
    mask_uv_vis = (255*mask_uv).astype(np.uint8)
    
   
    warped_image = 255*warp(mask_uv_vis, tform_img_normal, output_shape=(H_normal,W_normal))
    warped_image_vis = (warped_image).astype(np.uint8)

    return warped_image_vis

    


def process(pipe1,pipe2,uv_map,face_mesh,input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, bg_threshold,radius_threshold,warp_resolution,warp_face):
    with torch.no_grad():

        img_h,img_w,_=input_image.shape
        h,w=int(detect_resolution*img_h),int(detect_resolution*img_w)
        # h=3*h//2
        x = img_w/2 - w/2
        y = img_h/2 - h/2 

        in_grow_y=h//4
        in_grow_x=w//4
        mask=input_image.sum(axis=-1)!=0
        mask=(mask.astype(np.uint8)*255)

        radius=(h//2)*radius_threshold
        x_ind,y_ind=np.meshgrid(np.arange(img_w),np.arange(img_h),indexing='ij')
        # mask[int(y):int(y+h), int(x):int(x+w)]=0
        cond_area=((x_ind-img_w//2)**2+(y_ind-img_h//2)**2<radius**2)|((x_ind-img_w//2)**2+(y_ind-img_h//2)**2>(1.5*radius)**2)
        # mask[(x_ind-img_w//2)**2+(y_ind-img_h//2)**2<radius**2]=0
        mask[cond_area]=0
        mask=np.stack([mask,mask,mask],axis=-1)

        mask=cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)
        input_crop=input_image[int(y):int(y+h), int(x):int(x+w)].copy()

        
        # input_image_res=cv2.resize(input_image,(512,512),interpolation=cv2.INTER_LINEAR)
        # input_image[int(y):int(y+h), int(x):int(x+w)]=0
        # input_crop=input_image

        input_crop_rs=cv2.resize(input_crop,(512,512),interpolation=cv2.INTER_LINEAR)
        input_image_copy=input_image.copy()
        input_image_copy=cv2.resize(input_image_copy,(512,512),interpolation=cv2.INTER_LINEAR)

        prompt+=", "+a_prompt

        if seed == -1:
            seed = random.randint(0, 65535)
        generator=torch.manual_seed(seed)
        print("!!SEED:",seed)
        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        print(prompt)
        init_img = pipe1(
            prompt=prompt,
            image=Image.fromarray(input_image_copy),
            negative_prompt=n_prompt,
            num_inference_steps=ddim_steps,
            controlnet_conditioning_scale=float(strength),
            guidance_scale=scale,
            guess_mode=guess_mode,
            generator=generator,
            eta=eta,
        ).images[0]

        # plt.imshow(np.array(init_img))
        # plt.show()
        # plt.imshow(input_image_copy)
        # plt.show()

        # linart=lines(np.array(init_img),coarse=False)
        # linart=np.stack([linart,linart,linart],axis=-1)

        # output.show()

        init_img_orig_size=cv2.resize(np.array(init_img),(img_w,img_h),interpolation=cv2.INTER_LINEAR)
        init_crop_orig=init_img_orig_size[int(y):int(y+h), int(x):int(x+w)].copy()

        # init_crop=cv2.resize(init_crop_orig,(512,512),interpolation=cv2.INTER_LINEAR)
        print("SHAPES:",input_crop.shape,init_crop_orig.shape)
        output_init = pipe1(
            prompt=prompt,
            # prompt="hres digital photograph of a person,realistic fatial features with bumps and skin pores, insanely detailed, 8k, film",
            image=Image.fromarray(input_crop_rs),
            negative_prompt=n_prompt,
            num_inference_steps=ddim_steps,
            controlnet_conditioning_scale=float(bg_threshold),
            guidance_scale=scale,
            guess_mode=guess_mode,
            generator=generator,
            # generator=torch.manual_seed(123456),
            eta=eta,
        ).images[0]

        # output_big=cv2.resize(np.array(output_init),(w,h),interpolation=cv2.INTER_LINEAR)
        # ini
        

        output_init_big=cv2.resize(np.array(output_init),(w,h),interpolation=cv2.INTER_LINEAR)
        init_img_orig_size2=init_img_orig_size.copy()
        init_img_orig_size2[int(y):int(y+h), int(x):int(x+w)]=output_init_big
        # Image.fromarray(init_img_orig_size2).save('/hdd_data/common/NPHM/nphm_test/test2/out/in/test_init1.png')
        
        # If resized:
        output_init_big=cv2.resize(np.array(output_init_big),(warp_resolution,warp_resolution),interpolation=cv2.INTER_LINEAR)
        init_crop_orig=cv2.resize(np.array(init_crop_orig),(warp_resolution,warp_resolution),interpolation=cv2.INTER_LINEAR)
        
        t1=time.time()
        print('TIME:',time.time()-t1)
        # warped_img=fix_uv_misalignment(init_img_orig_size2,init_img_orig_size,uv_res=(img_w,img_h))
        warped_img=fix_uv_misalignment(output_init_big,init_crop_orig,uv_res=(init_crop_orig.shape[0],init_crop_orig.shape[1]),uv_map=uv_map,face_mesh=face_mesh)
        if warp_face and  warped_img is not None:
            mask_crop=warped_img.sum(axis=-1)==0
            mask_crop=np.stack([mask_crop,mask_crop,mask_crop],axis=-1)
            mask_crop=mask_crop.astype(np.uint8)*255
            kernel = np.ones((25, 25), np.uint8)
  
            output=warped_img
            output=cv2.resize(output,(w,h),interpolation=cv2.INTER_LINEAR)
            mask_crop=cv2.resize(mask_crop,(w,h),interpolation=cv2.INTER_NEAREST)
            mask=np.zeros_like(input_image)
            mask[int(y-10):int(y+h+10), int(x-10):int(x+w+10)]=255
            mask[int(y):int(y+h), int(x):int(x+w)]=(mask_crop).astype(np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            # mask=cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)

            # input_image=cv2.resize(input_image,(512,512),interpolation=cv2.INTER_LINEAR)
            # input_image=Image.fromarray(input_image)
            init_img=np.array(init_img)
            init_img=cv2.resize(init_img,(img_w,img_h),interpolation=cv2.INTER_LINEAR)
            init_img[int(y):int(y+h), int(x):int(x+w)]=init_img[int(y):int(y+h), int(x):int(x+w)]*(mask_crop//255)
            init_img[int(y):int(y+h), int(x):int(x+w)]+=output
            print(init_img.shape)
        # input_image[int(y):int(y+h), int(x):int(x+w)]=output
        else:
            print("refiner failed!")
            # input_image=cv2.resize(input_image,(512,512),interpolation=cv2.INTER_LINEAR)
            output=output_init_big        
            output_init_big=cv2.resize(np.array(output_init),(w,h),interpolation=cv2.INTER_LINEAR)
            init_img=np.array(init_img)
            init_img=cv2.resize(init_img,(img_w,img_h),interpolation=cv2.INTER_LINEAR)
            init_img[int(y):int(y+h), int(x):int(x+w)]=output_init_big
        # input_image[int(y):int(y+h), int(x):int(x+w)]=output
        print(input_image.shape,init_img.shape)
        # Image.fromarray(init_img).save('/hdd_data/common/NPHM/nphm_test/test2/out/in/test_init2.png')
        input_image=cv2.resize(input_image,(1024,1024),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_NEAREST)
        init_img=cv2.resize(init_img,(1024,1024),interpolation=cv2.INTER_LINEAR)
        print("SHAPES::",input_image.shape,mask.shape,init_img.shape)
        output1 = pipe2(
            prompt,
            # added_prompt="best quality, photoreal hair, 8k",
            negative_prompt=n_prompt,
            num_inference_steps=ddim_steps,
            generator=generator,
            image=Image.fromarray(init_img),
            control_image=Image.fromarray(input_image),
            controlnet_conditioning_scale=float(0.8),
            guidance_scale=scale,
            mask_image=Image.fromarray(mask),
        ).images[0]
        # output1=output1.resize((img_w,img_h))
        output1=output1.resize((1024,1024))
        print(np.array(output1).shape,np.array(init_img_orig_size).shape)
        init_img_orig_size=Image.fromarray(init_img_orig_size).resize((1024,1024))
        print(np.array(output1).shape,np.array(init_img_orig_size).shape)

        results = [output1]
    return [init_img_orig_size] + results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=str,required=True)
    parser.add_argument('--prompt',type=str,required=True) 
    parser.add_argument('--a_prompt',type=str,default='best quality, photoreal hair, 8k')
    parser.add_argument('--n_prompt',type=str,default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
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

    
    # NOTE: USE THIS BY DEFAULT
    controlnet = ControlNetModel.from_pretrained("/ssd_data/common/sollid/staged_approach_low_train_more_epochs", torch_dtype=torch.float16)
    #

    # controlnet = ControlNetModel.from_pretrained("/hdd_data/common/sollid/sd_15_ckpt_diffuser", torch_dtype=torch.float16)
    # controlnet = ControlNetModel.from_pretrained("/hdd_data/common/sollid/sd_15_ckpt_diffuser_low_train", torch_dtype=torch.float16)


    # controlnet_lineart = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-lineart-diffusers", torch_dtype=torch.float16)

    # NOTE:
    pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
    )
    #
    # SPIDER-VERSE LORA
    # pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
    #     "nitrosocke/spider-verse-diffusion", controlnet=controlnet, torch_dtype=torch.float16
    # )
    # # Animated
    # pipe1 = StableDiffusionControlNetPipeline.from_single_file(
    #     "/hdd_data/common/sollid/nphm/hello25dvintageanime_hellorealvintageanime.safetensors", controlnet=controlnet, torch_dtype=torch.float16
    # )


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
    # pipe_xl.enable_xformers_memory_efficient_attention()
    pipe1.to('cuda')
    pipe2.to('cuda')
    # pipe_xl.to("cuda")
    # mode = model.cuda()
    # ddim_sampler = DDIMSampler(model)

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
