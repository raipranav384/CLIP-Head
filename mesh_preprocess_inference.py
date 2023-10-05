import pymeshlab
import numpy as np
import argparse
import os
import glob
import natsort
from tqdm import tqdm


# if args.output is None:
#     args.output=os.path.basedir(args.input)
def mesh_preprocess(args):
    mesh_list=glob.glob(os.path.join(args.input,'*'))
    mesh_list=natsort.natsorted(mesh_list)
    for mesh_id in tqdm(mesh_list):
        if not os.path.isdir(mesh_id):
            continue
        # print(mesh_id)
        mesh_exp=glob.glob(os.path.join(mesh_id,'*'))
        mesh_exp=natsort.natsorted(mesh_exp)
        for mesh_pth in tqdm(mesh_exp):
            if not os.path.isdir(mesh_id):
                continue
            # print(mesh_pth)
            if not os.path.exists(os.path.join(mesh_pth,'scan_40k.ply')):
                mesh=pymeshlab.MeshSet()
                mesh.load_new_mesh(os.path.join(mesh_pth,'scan.ply'))
                mesh.apply_filter('apply_coord_laplacian_smoothing',stepsmoothnum=2)
                mesh.apply_filter('meshing_decimation_quadric_edge_collapse',targetfacenum=40000)
                mesh.apply_filter('meshing_repair_non_manifold_vertices')
                mesh.apply_filter('meshing_remove_duplicate_faces')
                mesh.apply_filter('meshing_remove_duplicate_vertices')
                mesh.apply_filter('meshing_remove_null_faces')
                # mesh.apply_filter('meshing_merge_close_vertices')
                mesh.apply_filter('meshing_remove_connected_component_by_diameter',mincomponentdiag=pymeshlab.Percentage(30))
                mesh.apply_filter('compute_selection_by_condition_per_vertex',condselect='y<-0.7*z-0.55')
                mesh.apply_filter('meshing_remove_selected_vertices')
                mesh.apply_filter('compute_selection_from_mesh_border')
                mesh.apply_filter('meshing_remove_unreferenced_vertices')
                mesh.apply_filter('apply_coord_laplacian_smoothing',selected=True)

                mesh.save_current_mesh(os.path.join(mesh_pth,'scan_40k.ply'))
            if not os.path.exists(os.path.join(mesh_pth,'scan_200k.ply')):
                mesh=pymeshlab.MeshSet()
                mesh.load_new_mesh(os.path.join(mesh_pth,'scan.ply'))
                mesh.apply_filter('apply_coord_laplacian_smoothing',stepsmoothnum=2)
                mesh.apply_filter('meshing_decimation_quadric_edge_collapse',targetfacenum=2000000)
                mesh.apply_filter('meshing_repair_non_manifold_vertices')
                mesh.apply_filter('meshing_remove_duplicate_faces')
                mesh.apply_filter('meshing_remove_duplicate_vertices')
                mesh.apply_filter('meshing_remove_null_faces')
                # mesh.apply_filter('meshing_merge_close_vertices')
                mesh.apply_filter('meshing_remove_connected_component_by_diameter',mincomponentdiag=pymeshlab.Percentage(30))
                mesh.apply_filter('compute_selection_by_condition_per_vertex',condselect='y<-0.7*z-0.55')
                mesh.apply_filter('meshing_remove_selected_vertices')
                
                mesh.apply_filter('meshing_remove_unreferenced_vertices')
                mesh.apply_filter('compute_selection_from_mesh_border')
                mesh.apply_filter('apply_coord_laplacian_smoothing',selected=True)

                mesh.save_current_mesh(os.path.join(mesh_pth,'scan_200k.ply'))
                # mesh.apply_filter('compute_texcoord_transfer_wedge_to_vertex')
                # mesh.save_current_mesh(os.path.join(mesh_pth,'scan_40k_uv.ply'))
        #     break
        # break
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=str,required=True)
    parser.add_argument('--output', default=None,type=str,required=False)

    args=parser.parse_args()
    mesh_preprocess(args)