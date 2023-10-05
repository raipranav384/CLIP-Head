from NPHM.models.deepSDF import DeepSDF, DeformationNetwork
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths
from NPHM.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from NPHM.models.reconstruction import deform_mesh, get_logits, get_logits_backward
from NPHM.models.fitting import inference_iterative_root_finding_joint, inference_identity_space
from NPHM.data.manager import DataManager

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
import pymeshlab
parser = argparse.ArgumentParser(
    description='Run generation'
)




def initialize(args):
    with open(args.cfg_file, 'r') as f:
        print('Loading config file from: ' + args.cfg_file)
        CFG = yaml.safe_load(f)

    print(json.dumps(CFG, sort_keys=True, indent=4))



    weight_dir_shape = env_paths.EXPERIMENT_DIR + '/{}/'.format(CFG['exp_name_shape'])
    if CFG['exp_name_expr'] is not None:
        weight_dir_expr = env_paths.EXPERIMENT_DIR + '/{}/'.format(CFG['exp_name_expr'])

    # load config files
    fname_shape = weight_dir_shape + 'configs.yaml'
    with open(fname_shape, 'r') as f:
        print('Loading config file from: ' + fname_shape)
        CFG_shape = yaml.safe_load(f)
    if CFG['exp_name_expr'] is not None:
        fname_expr = weight_dir_expr + 'configs.yaml'
        with open(fname_expr, 'r') as f:
            print('Loading config file from: ' + fname_expr)
            CFG_expr = yaml.safe_load(f)

    device = torch.device("cuda")

    print('###########################################################################')
    print('####################     Shape Model Configs     #############################')
    print('###########################################################################')
    print(json.dumps(CFG_shape, sort_keys=True, indent=4))

    if CFG['exp_name_expr'] is not None:
        print('###########################################################################')
        print('####################     Expression Model Configs     #############################')
        print('###########################################################################')
        print(json.dumps(CFG_expr, sort_keys=True, indent=4))


    if CFG['local_shape']:
        lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH)
        anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH)).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        lm_inds = None
        anchors = None


    if CFG['local_shape']:
        decoder_shape = FastEnsembleDeepSDFMirrored(
            lat_dim_glob=CFG_shape['decoder']['decoder_lat_dim_glob'],
            lat_dim_loc=CFG_shape['decoder']['decoder_lat_dim_loc'],
            hidden_dim=CFG_shape['decoder']['decoder_hidden_dim'],
            n_loc=CFG_shape['decoder']['decoder_nloc'],
            n_symm_pairs=CFG_shape['decoder']['decoder_nsymm_pairs'],
            anchors=anchors,
            n_layers=CFG_shape['decoder']['decoder_nlayers'],
            pos_mlp_dim=CFG_shape['decoder'].get('pos_mlp_dim', 256),
        )
    else:
        decoder_shape = DeepSDF(
            lat_dim=CFG_shape['decoder']['decoder_lat_dim'],
            hidden_dim=CFG_shape['decoder']['decoder_hidden_dim'],
            geometric_init=True
        )

    decoder_shape = decoder_shape.to(device)


    if CFG['exp_name_expr'] is not None:
        if CFG['local_shape']:
            decoder_expr = DeformationNetwork(mode=CFG_expr['ex_decoder']['mode'],
                                    lat_dim_expr=CFG_expr['ex_decoder']['decoder_lat_dim_expr'],
                                    lat_dim_id=CFG_expr['ex_decoder']['decoder_lat_dim_id'],
                                    lat_dim_glob_shape=CFG_expr['id_decoder']['decoder_lat_dim_glob'],
                                    lat_dim_loc_shape=CFG_expr['id_decoder']['decoder_lat_dim_loc'],
                                    n_loc=CFG_expr['id_decoder']['decoder_nloc'],
                                    anchors=anchors,
                                    hidden_dim=CFG_expr['ex_decoder']['decoder_hidden_dim'],
                                    nlayers=CFG_expr['ex_decoder']['decoder_nlayers'],
                                    input_dim=3, out_dim=3
                                    )
        else:
            decoder_expr = DeepSDF(lat_dim=512+200,
                                hidden_dim=1024,
                                out_dim=3)
        decoder_expr = decoder_expr.to(device)


    path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['checkpoint_shape']))
    print('Loaded checkpoint from: {}'.format(path))
    checkpoint = torch.load(path, map_location=device)
    decoder_shape.load_state_dict(checkpoint['decoder_state_dict'], strict=True)


    if 'latent_codes_state_dict' in checkpoint:
        n_train_subjects = checkpoint['latent_codes_state_dict']['weight'].shape[0]
        n_val_subjects = checkpoint['latent_codes_val_state_dict']['weight'].shape[0]
        if CFG['local_shape']:
            latent_codes_shape = torch.nn.Embedding(n_train_subjects, decoder_shape.lat_dim)
            latent_codes_shape_val = torch.nn.Embedding(n_val_subjects, decoder_shape.lat_dim)
        else:
            latent_codes_shape = torch.nn.Embedding(n_train_subjects, 512)
            latent_codes_shape_val = torch.nn.Embedding(n_val_subjects, 512)

        latent_codes_shape.load_state_dict(checkpoint['latent_codes_state_dict'])
        latent_codes_shape_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
    else:
        latent_codes_shape = None
        latent_codes_shape_val = None

    if CFG['exp_name_expr'] is not None:
        path = osp.join(weight_dir_expr, 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['checkpoint_expr']))
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path, map_location=device)
        decoder_expr.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
        if 'latent_codes_state_dict' in checkpoint:
            latent_codes_expr = torch.nn.Embedding(checkpoint['latent_codes_state_dict']['weight'].shape[0], 200)
            latent_codes_expr.load_state_dict(checkpoint['latent_codes_state_dict'])
            latent_codes_expr_val = torch.nn.Embedding(checkpoint['latent_codes_val_state_dict']['weight'].shape[0], 200)
            latent_codes_expr_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
        else:
            latent_codes_expr = None
            latent_codes_expr_val = None
    else:
        decoder_expr = None


    mini = [-.55, -.5, -.95]
    maxi = [0.55, 0.75, 0.4]
    grid_points = create_grid_points_from_bounds(mini, maxi, args.resolution)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)


    #TODO
    out_dir = env_paths.FITTING_DIR + '/forward_{}/{}/'.format(args.exp_name, args.exp_tag)


    os.makedirs(out_dir, exist_ok=True)
    fname = out_dir + 'configs.yaml'
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)

    params={
        'decoder_shape': decoder_shape,
        'decoder_expr': decoder_expr,
        'grid_points': grid_points,
        'mini': mini,
        'maxi': maxi,
        'resolution': args.resolution,
        'neutral': args.neutral,
    }
    return params
# exp_mean=torch.tensor(np.load('assets/exp_mean.npy'))[0]
# exp_std=torch.tensor(np.load('assets/exp_std.npy'))[0]
# exp_arr=torch.tensor(np.load('assets/exp_arr.npy'), dtype=torch.float32)
# exp_arr=exp_arr.reshape(exp_arr.shape[0],-1)
# print("EXP_ARR shape:",exp_arr.shape)
# exp_arr=torch.tensor(np.load('assets/coord_expanded.npy'), dtype=torch.float32)

def sample_shape_space(params):
    decoder_shape = params['decoder_shape']
    decoder_expr = params['decoder_expr']
    grid_points = params['grid_points']
    mini = params['mini']
    maxi = params['maxi']
    resolution = params['resolution']
    neutral=params['neutral']
    parent=os.path.dirname(__file__)
    if not os.path.exists(f"{parent}/results"):
        os.makedirs(f"{parent}/results",exist_ok=True)
    step = 0

    num_steps=1
    # lat_vectors=np.zeros((num_steps,1344))
    lat_vectors_shape=np.zeros((num_steps,1344))
    lat_vectors_exp=np.zeros((num_steps,200))
    for i in tqdm(range(num_steps)):
        lat_rep_shape=torch.tensor(np.load('./results/Pred_nphm_id.npy')).to(torch.float32).cuda()
        # rand_ind=torch.randint(low=0,high=exp_arr.shape[0],size=(1,))
        #NOTE: 
        lat_rep_exp=torch.tensor(np.load('./results/Pred_nphm_exp.npy')).to(torch.float32).cuda()

        logits,anchors_lcl = get_logits(decoder_shape, lat_rep_shape, grid_points, nbatch_points=25000*4,return_anchors=True)
        t=time.time()
        mesh = mesh_from_logits(logits, mini, maxi, resolution)
        #NOTE: 
        if not neutral:
            mesh = deform_mesh(mesh, decoder_expr, lat_rep_exp.unsqueeze(0), anchors_lcl, lat_rep_shape=lat_rep_shape.unsqueeze(0))
        t=time.time()-t
        t=time.time()
        mesh.export( './results/out/in/scan.ply')
        # NOTE: 
        lat_vectors_shape[i]=lat_rep_shape.cpu().detach().numpy()
        t=time.time()-t
        step += 1

if __name__ == '__main__':
    parser.add_argument('-resolution' , default=256, type=int)
    parser.add_argument('-batch_points', default=20000, type=int)
    parser.add_argument('-cfg_file', type=str, required=True)
    parser.add_argument('-exp_name', type=str, required=True)
    parser.add_argument('-exp_tag', type=str, required=True)
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

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]


    args = parser.parse_args() 
    initialize(args)
    if args.sample:
    # if False:
        params=initialize(args)
        sample_shape_space(params)

