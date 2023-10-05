import trimesh
import pymeshlab as ps
import open3d as o3d
import numpy as np
import cv2
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import natsort
import glob
from tqdm import tqdm
import os
import argparse
import time

# orig_tex_pth='/hdd_data/common/NPHM/chunk_01/044/000/texture.png'
# orig_tex_pth='/ssd_data/common/Parameterization/v_test_temp.png'
# uv_pos_pth='/ssd_data/common/Parameterization/mesh_with_uvs_003_surface_pos.npy'
# new_mesh_pth='/ssd_data/common/Parameterization/mesh_with_uvs_003.obj'
# orig_mesh_pth='/ssd_data/common/NPHM/dataset/scan_40k.ply'


# orig_tex_pth='/hdd_data/common/NPHM/chunk_01/044/000/texture.png'
# uv_pos_pth='/ssd_data/common/Parameterization/mesh_with_uvs_003_surface_pos.npy'
# new_mesh_pth='/hdd_data/common/NPHM/chunk_01/044/000/scan_test.ply'
# orig_mesh_pth='/hdd_data/common/NPHM/chunk_01/044/000/scan_test.ply'

# orig_mesh_pth=new_mesh_pth
# seam_indices_pth='/ssd_data/common/NPHM/dataset/scan_40k_seam.npy'
def normal_tex(args):
    mesh_list=glob.glob(os.path.join(args.input,'*'))
    mesh_list=natsort.natsorted(mesh_list)
    for mesh_id in tqdm(mesh_list):
        if not os.path.isdir(mesh_id):
            continue
        # print(mesh_id)
        # breakpoint()
        mesh_exp=glob.glob(os.path.join(mesh_id,'*'))
        mesh_exp=natsort.natsorted(mesh_exp)
        for mesh_path in mesh_exp:
            if os.path.exists(os.path.join(mesh_path,'normal_map.png')):
                if os.path.exists(os.path.join(mesh_path,'scan_40k_nuv_surface_pos.npy')) and os.path.exists(os.path.join(mesh_path,'tex_map.png')):
                    # intput=input(f"Remove? {os.path.join(mesh_path,'scan_40k_nuv_surface_pos.npy')} (y/n)")
                    # if intput=='y':
                    os.remove(os.path.join(mesh_path,'scan_40k_nuv_surface_pos.npy'))
                continue
            orig_tex=cv2.imread(os.path.join(mesh_path,'texture.png'))
            while True:
                if not os.path.exists(os.path.join(mesh_path,'scan_40k_nuv_surface_pos.npy')):
                    print(f"Waiting for:{os.path.join(mesh_path,'scan_40k_nuv_surface_pos.npy')}")
                    time.sleep(5)
                else:
                    break
            uv_pos=np.load(os.path.join(mesh_path,"scan_40k_nuv_surface_pos.npy"))
            uv_mask=uv_pos.sum(axis=-1,keepdims=True)!=0
            if uv_mask.sum()==0:
                print(f"INVALID!!:",{mesh_path})
                continue
            # seam_indices=np.load(seam_indices_pth)
            uv_pos=uv_pos.reshape(-1,3)

            uv_pos_bool = uv_pos.sum(axis=-1)!=0
            # print(uv_pos_bool.shape)
            new_mesh_ms=ps.MeshSet()
            new_mesh_ms.load_new_mesh(os.path.join(mesh_path,'scan_40k_nuv.obj'))
            new_mesh=new_mesh_ms.current_mesh()

            orig_mesh_ms=ps.MeshSet()
            orig_mesh_ms.load_new_mesh(os.path.join(mesh_path,'scan_200k.ply'))
            orig_mesh=orig_mesh_ms.current_mesh()
            # orig_uv=orig_mesh.vertex_tex_coord_matrix()  # Vertex Index -> UV
            orig_verts=orig_mesh.vertex_matrix()
            orig_vert_count=orig_verts.shape[0]
            triangles=orig_mesh.face_matrix()

            face_centres=orig_verts[triangles].mean(axis=1)
            shape=int(np.sqrt(uv_pos.shape[0]))
            # orig_uv[:,0]=2*(orig_uv[:,0]-orig_uv[:,0].min())/(orig_uv[:,0].max()-orig_uv[:,0].min())
            # orig_uv[:,1]=2*(orig_uv[:,1]-orig_uv[:,1].min())/(orig_uv[:,1].max()-orig_uv[:,1].min())

            # orig_uv=(orig_uv-orig_uv.min(0))/orig_uv.max(0)-orig_uv.min(0)
            # orig_uv=2*(orig_uv-orig_uv.min())/(orig_uv.max()-orig_uv.min())

            # kd_query=KDTree(face_centres)
            # closest_dist,closest_uv=kd_query.query(uv_pos[uv_pos_bool])
            # closest_uv=closest_uv.reshape(-1,1)
            # orig_verts=orig_verts-orig_verts.mean(axis=0)
            anchor_idx=np.argmax(uv_pos[uv_pos_bool][...,2])
            anchor=uv_pos[uv_pos_bool][anchor_idx]
            anchor2_idx=np.argmax(orig_verts[...,2])
            anchor2=orig_verts[anchor2_idx]
            del_pos=anchor2-anchor
            uv_pos[uv_pos_bool]=uv_pos[uv_pos_bool]+del_pos
            uv_pos=uv_pos.astype(np.float32)
            # breakpoint()
            device = o3d.core.Device("CPU:0")
            dtype_f = o3d.core.float32
            dtype_i = o3d.core.int32
            orig_mesh_o3d = o3d.t.geometry.TriangleMesh(device)

            orig_mesh_o3d.vertex.positions = o3d.core.Tensor(orig_verts, dtype_f, device)
            orig_mesh_o3d.triangle.indices = o3d.core.Tensor(triangles, dtype_i, device)

            # print("Orig_UV shape:",orig_uv.shape,orig_uv.max(),orig_uv.min())
            # orig_mesh_o3d=o3d.t.io.read_triangle_mesh(orig_mesh_pth)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(orig_mesh_o3d)
            uv_query = o3d.core.Tensor(uv_pos) # search for the closest point in 
                                            # mesh, to the 3D mapping of each
                                            # texture pixel for the new_mesh
            close_q=scene.compute_closest_points(uv_query)
            tri_idx=close_q['primitive_ids'].numpy() # Index of closest triangle
            tri_uv=close_q['primitive_uvs'].numpy() # barycentric coordinate of closest point in closest triangle
            tri_uv3=np.zeros((tri_uv.shape[0],3))
            tri_uv3[:,1:]=tri_uv
            tri_uv3[:,0]=1-tri_uv.sum(axis=-1)

            corres_triangles=triangles[tri_idx]
            per_vertex_normal=orig_mesh.vertex_normal_matrix()
            # pixel2orig3d=orig_verts[corres_triangles]
            pixel2closest_origUV=per_vertex_normal[corres_triangles] # Mapping of 3 closest points to uv of orignal
            new2origUV=(pixel2closest_origUV*tri_uv3[...,np.newaxis]).sum(axis=-2) # Mapping of new UV to space of original UV

            new2origUV=new2origUV.reshape(shape,shape,3)
            new2origUV=(new2origUV+2*np.pi)/(4*np.pi)

            cv2.imwrite(os.path.join(mesh_path,'normal_map.png'),(uv_mask*new2origUV*255).astype(np.uint8))

            if os.path.exists(os.path.join(mesh_path,'normal_map.png')) and os.path.exists(os.path.join(mesh_path,'tex_map.png')):
                try:
                    os.remove(os.path.join(mesh_path,'scan_40k_nuv_surface_pos.npy'))
                except:
                    print("No npy file to delete")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input directory')
    args=parser.parse_args()
    normal_tex(args)