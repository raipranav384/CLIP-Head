#! /bin/zsh
# conda init zsh
eval "$(conda shell.bash hook)"
mkdir -p ./results

conda activate clip_head
#----------------------------------------------------------------#
# Map clip encoding to NPHM latent space, to get a latent vector
#----------------------------------------------------------------#

python map2Clip.py --prompt $1 

#----------------------------------------------------------------#
# Decode the latent vector, to get a mesh conditioned on text
#----------------------------------------------------------------#
python ./NPHM/scripts/fitting/fitting_pointclouds.py -cfg_file ./NPHM/scripts/configs/fitting_nphm.yaml -exp_name EXP_NAME -exp_tag EXP_TAG -resolution 256 -sample


#----------------------------------------------------------------#
# The mesh obtained is not parameterized, preprocess it,
# follwed by parameterization into a common space
#----------------------------------------------------------------#
python mesh_preprocess_inference.py --input ./results
python parametrize_NPHM_dir.py --input ./results


#----------------------------------------------------------------#
# Find correspondeces between 2D points, and their 3D coordinates
#----------------------------------------------------------------#
python UV_to_3D.py
#----------------------------------------------------------------#
# Generate the normal map in parameterized space
#----------------------------------------------------------------#
python normal_tex_dir.py --input ./results

#----------------------------------------------------------------#
# Generate a texture map conditioned on the normal texture map
#----------------------------------------------------------------#
python nphm_2_staged_warp.py --input ./results/out/in/normal_map.png --prompt $1

#----------------------------------------------------------------#
# Move around the files
#----------------------------------------------------------------#
mkdir -p ./outputs
mkdir ./outputs/$1 -p
mv ./results/out/in/* ./outputs/$1/
rm ./outputs/$1/scan_40k_nuv_surface_pos.npy
