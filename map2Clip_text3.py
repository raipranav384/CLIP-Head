import torch
from torch import nn
import torch.nn.functional as F
import open_clip
from PIL import Image
# from map2Clip_data import Clip_data
# from map2Clip_data_text import Clip_data_exp
from tqdm import tqdm
import numpy as np
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--prompt',type=str,default=None)
parser.add_argument('--cont',action='store_true',default=None)
parser.add_argument('--train',action='store_true',default=None)
args=parser.parse_args()

class Clip2NPHM(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4]):
        """ 
        """
        super(Clip2NPHM, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W*2)]+[nn.Linear(W*2, W)] + [nn.Linear(W, W) if i+1 not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-2)])

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts=x
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
       
        outputs = self.output_linear(h)

        return outputs    



def process(args):
    if args.cont:
        print("!!CONTINUING!!")
        clip2nphm_model_id.load_state_dict(torch.load(f'./clip2nphm_model_id_more_exp_{1}.pth'))
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    clip2nphm_model_id=Clip2NPHM(input_ch=1024,output_ch=1344)
    clip2nphm_model_exp=Clip2NPHM(input_ch=1024,output_ch=200)
    nphm2clip_model_exp=Clip2NPHM(input_ch=200,output_ch=1024)
    clip2nphm_model_id.load_state_dict(torch.load(f'./checkpoints/clip2nphm_model_{0}.pth'))

    model=model.to('cuda')
    clip2nphm_model_id=clip2nphm_model_id.to('cuda')
    clip2nphm_model_exp=clip2nphm_model_exp.to('cuda')

    nphm2clip_model_exp=nphm2clip_model_exp.to('cuda')
    # clip2nphm_model=Clip2NPHM(input_ch)
    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).cuda()

    # with torch.no_grad(),torch.cuda.amp.autocast():
    #     image_features = model.encode_image(image)
    #     image_features/=image_features.norm(dim=-1,keepdim=True)


        # clip2nphm_model_exp.load_state_dict(torch.load(f'./clip2nphm_model_exp_more_exp_{1}.pth'))
        # clip2nphm_model_id.load_state_dict(torch.load(f'./clip2nphm_model_id_more_exp_{1}.pth'))
        # clip2nphm_model_id.load_state_dict(torch.load(f'./clip2nphm_model_plusShape{0}.pth'))

    clip2nphm_model_exp.load_state_dict(torch.load(f'./checkpoints/clip2nphm_model_exp_more_exp_L2_{134}.pth'))
    # clip2nphm_model_exp.load_state_dict(torch.load(f'./clip2nphm_model_exp_more_exp_{134}.pth'))
    # clip2nphm_model_id.load_state_dict(torch.load(f'./clip2nphm_model_id_{1}.pth'))

    clip2nphm_model_exp.eval()
    clip2nphm_model_id.eval()
    text = tokenizer([args.prompt]).cuda()
    # image=preprocess(Image.open(img_path)).unsqueeze(0).cuda()
    with torch.no_grad(),torch.cuda.amp.autocast():
        # image_features = model.encode_image(image)
        # image_features/=image_features.norm(dim=-1,keepdim=True)
        # nphm_vec_id=clip2nphm_model_id(image_features)
        # nphm_vec_exp=clip2nphm_model_exp(image_features)
        text_features = model.encode_text(text).cuda()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        nphm_vec_id=clip2nphm_model_id(text_features)
        nphm_vec_exp=clip2nphm_model_exp(text_features)
    nphm_vec_id=nphm_vec_id.detach().cpu()
    nphm_vec_exp=nphm_vec_exp.detach().cpu()
    # print(nphm_vec.shape)
    np.save("./results/Pred_nphm_id.npy",nphm_vec_id)
    np.save("./results/Pred_nphm_exp.npy",nphm_vec_exp)

if __name__=="__main__":


    process(args)
