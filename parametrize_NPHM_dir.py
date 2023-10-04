import numpy as np 
import open3d as o3d
import os
from copy import deepcopy
import igl  
import cv2
from  tqdm import tqdm
import pymeshlab as ps
from scipy.spatial import KDTree
import natsort
import glob
import argparse
import time

def get_tex(uv, path):
    num_points = uv.shape[0]
    res = 511
    uv = (uv - uv.min()) /( uv.max() - uv.min())
    uv = (uv + 1) * (res/2)

    tex = np.full((res+1,res+1,3),0)
    
    for i in range(num_points):
        tex[int(uv[i,0]),int(uv[i,1])] =  [255,255,255]

    tex  = cv2.cvtColor(tex.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, tex)


def write_obj(verts, f_v, verts_tex, path, save_vert_color = False, vert_color=None):
    # path = os.path.join(self.result_dir, 'kurta_pred_tex.obj')
    # verts = (verts + 1) * (1/2)
    # verts_tex = (verts_tex + 1) * (1/2)
    f = open(path, "w")

    for i in range(len(verts)):
        v = verts[i]
        if(save_vert_color):
            vc = vert_color[i]
            f.write('v '+str(v[0])+' '+str(v[1])+' '+str(v[2])+' '+str(vc[0])+' '+str(vc[1])+' '+str(vc[2])+'\n')
        else:
            f.write('v '+str(v[0])+' '+str(v[1])+' '+str(v[2])+'\n')

    for vt in verts_tex:
        f.write('vt '+str(vt[0])+' '+str(vt[1])+'\n')

    for fidx in range(len(f_v)):
        v_f = f_v[fidx]
        vt_f = f_v[fidx]
        f.write('f '+str(v_f[0]+1)+'/'+str(vt_f[0]+1)+' '+str(v_f[1]+1)+'/'+str(vt_f[1]+1)+' '+str(v_f[2]+1)+'/'+str(vt_f[2]+1) + '\n')
    f.close()

def slice_mesh(mesh_path):

    mesh=o3d.io.read_triangle_mesh(mesh_path)
    vertices=np.asarray(mesh.vertices)
    mean=np.mean(vertices,axis=0)
    vertices-=mean
    faces=np.asarray(mesh.triangles)
    face_centre=np.mean(vertices[faces],axis=-2)
    # print(face_centre.shape,vertices[faces[0:1]].shape)
    #FOR NPHM
    z_face=np.where((face_centre[:,2]>0)*(face_centre[:,1]<0.25))[0]
    # z_face=np.where((face_centre[:,2]>0.01)*(face_centre[:,1]<0.29))[0]
    left_plane_faces=np.where(face_centre[:,0]<=0)[0]
    right_plane_faces=np.where(face_centre[:,0]>0)[0]

    colors=np.zeros((faces.shape[0])).astype(np.int32)
    colors[left_plane_faces]=1
    colors[right_plane_faces]= 2
    colors[z_face]=3

    seam_verts=[]


    # print(vertices.shape)

    t1=time.time()
    ## NOTE: Vectorized version is slower than for loop??
    faces_1=(faces[:,0,np.newaxis]==np.arange(vertices.shape[0])).astype(np.int8)
    faces_2=(faces[:,1,np.newaxis]==np.arange(vertices.shape[0])).astype(np.int8)
    faces_3=(faces[:,2,np.newaxis]==np.arange(vertices.shape[0])).astype(np.int8)
    idx_faces=(faces_1+faces_2+faces_3)>0
    idx_faces=idx_faces.astype(bool)
    print("Time taken:",time.time()-t1)
    # colors_faces=colors[idx_faces]
    colors_faces=np.zeros_like(idx_faces).astype(np.int16)
    colors_faces=np.repeat(colors[:,np.newaxis],idx_faces.shape[1],axis=1)
    colors_faces[~idx_faces]=0
    idx_3=colors_faces==3
    idx_2=colors_faces==2
    idx_1=colors_faces==1
    seam_verts_bool=(np.sum(idx_3,axis=0)==0)*(np.sum(idx_2,axis=0)>0)*(np.sum(idx_1,axis=0)>0)
    seam_verts=np.where(seam_verts_bool)[0]
    ## ENDS Heres
    # for i,vert in tqdm(enumerate(vertices),total=vertices.shape[0]):
    #     faces_1=faces[:,0]==i
    #     faces_2=faces[:,1]==i
    #     faces_3=faces[:,2]==i
    #     idx_faces=(faces_1+faces_2+faces_3)>0
    #     colors_faces=colors[idx_faces]
    #     unique_cols=np.unique(colors_faces)
    #     if unique_cols.shape[0]==2 and unique_cols.sum()==3:
    #         seam_verts.append(i)

    print("Time taken:",time.time()-t1)
    seam_verts=np.array(seam_verts,dtype=np.int32)

    colors[left_plane_faces]=1
    colors[right_plane_faces]= 2
    # np.save(args.output+"_faces.npy",colors)
    # np.save(args.output+"_seam.npy",seam_verts)
    return vertices,faces,colors,seam_verts

# mesh_path = '/ssd_data/common/NPHM/dataset/scan_40k_mesh.ply'
# seam_path = '/ssd_data/common/NPHM/dataset/scan_40k_seam.npy'
# face_col_path = '/ssd_data/common/NPHM/dataset/scan_40k_faces.npy'
# mesh_path = '/ssd_data/common/NPHM/dataset/0001_shape_mesh.ply'
# face_col_path = '/ssd_data/common/NPHM/dataset/0001_shape_faces.npy'

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=str,required=True)
parser.add_argument('--output', default=None,type=str,required=False)
args=parser.parse_args()
if args.output is None:
    args.output=args.input[:-4]

ref_seam_path = './checkpoints/0001_shape_seam.npy'
ref_mesh_path='./checkpoints/reference_uv_vt_scaled.obj'
# ref_bnd = igl.boundary_loop(ref_mesh.face_matrix())

ms=ps.MeshSet()
ms.load_new_mesh(ref_mesh_path)
# ms.load_new_mesh('/ssd_data/common/Parameterization/check_uv.obj')
ms.apply_filter('compute_selection_from_mesh_border')

ref_mesh=ms.current_mesh()
ref_bnd=np.arange(ref_mesh.vertex_matrix().shape[0])[ref_mesh.vertex_selection_array()]



ref_seam_indices=np.load(ref_seam_path)
ref_uv = ref_mesh.vertex_tex_coord_matrix()
ref_bnd_uv = ref_uv[ref_bnd]

tofit = deepcopy(ref_mesh.vertex_matrix())

ref_seam_right=(ref_bnd[...,np.newaxis]==ref_seam_indices).sum(axis=-1)
ref_seam_left=(ref_bnd[...,np.newaxis]==np.arange(tofit.shape[0]-ref_seam_indices.shape[0],tofit.shape[0])).sum(axis=-1)
ref_seam_right=ref_seam_right.astype(bool)
ref_seam_left=ref_seam_left.astype(bool)
ref_seam_right=(ref_seam_right+((~ref_seam_left)*(tofit[ref_bnd,1]>0)))>0

ref_seam_base_r=(~ref_seam_left)*(~ref_seam_right)*(tofit[ref_bnd,0]<0)
ref_seam_base_l=(~ref_seam_left)*(~ref_seam_right)*(tofit[ref_bnd,0]>=0)
ref_seam_base_l=ref_seam_base_l.astype(bool)
ref_seam_base_r=ref_seam_base_r.astype(bool)

print("Before:",tofit.shape,ref_mesh.vertex_matrix().shape)
ref_tree_r=KDTree(tofit[ref_bnd[ref_seam_right]])
ref_tree_l=KDTree(tofit[ref_bnd[ref_seam_left]])
ref_tree_br=KDTree(tofit[ref_bnd[ref_seam_base_r]]*100)
ref_tree_bl=KDTree(tofit[ref_bnd[ref_seam_base_l]])
print("After:",tofit.shape,ref_mesh.vertex_matrix().shape)



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
        if os.path.exists(os.path.join(mesh_path,'scan_40k_nuv.obj')):
            continue
        verts,faces,face_colors,seam_indices=slice_mesh(os.path.join(mesh_path,'scan_40k.ply'))
        face_mask = np.zeros((faces.shape))
        new_vertices=np.zeros((verts.shape[0]+seam_indices.shape[0],3))
        new_vertices[:verts.shape[0]]=verts
        # new_vertices[verts.shape[0]:]=verts[seam_indices]+np.array([0,0.05,0])
        new_vertices[verts.shape[0]:]=verts[seam_indices]

        seam_bool=np.zeros(verts.shape[0],dtype=bool)
        seam_bool[seam_indices]=True
        seam_faces=np.where(((seam_bool[faces[:,0]])+(seam_bool[faces[:,1]])+(seam_bool[faces[:,2]]))>0)[0]

        seam_faces=np.array(seam_faces)
        print(seam_faces.shape)
        bnd_orig = igl.boundary_loop(faces)
        counter=0
        ###############################################################################
        for i,seam_face in tqdm(enumerate(seam_faces),total=seam_faces.shape[0]):
            v0,v1,v2=faces[seam_face] 
            # print(face_colors[seam_face])
            if face_colors[seam_face]==1:
                counter+=1
                if seam_bool[v0]==True:
                    idx_seam=np.where(seam_indices==v0)[0]
                    faces[seam_face,0]=idx_seam[0]+verts.shape[0]
                    # print(v0,idx_seam[0]+verts.shape[0])
                if seam_bool[v1]==True:
                    idx_seam=np.where(seam_indices==v1)[0]
                    faces[seam_face,1]=idx_seam[0]+verts.shape[0]
                    # print(v1,idx_seam[0]+verts.shape[0])
                if seam_bool[v2]==True:
                    idx_seam=np.where(seam_indices==v2)[0]
                    faces[seam_face,2]=idx_seam[0]+verts.shape[0]
                    # print(v2,idx_seam[0]+verts.shape[0])
        # print(new_vertices[-seam_indices.shape[0]:])
        ##################################################################################


        seam_mesh_init=ps.Mesh(vertex_matrix=new_vertices,face_matrix=faces)
        ms=ps.MeshSet()
        ms.add_mesh(seam_mesh_init)
        # ms.load_new_mesh('/ssd_data/common/Parameterization/check_uv.obj')
        ms.apply_filter('compute_selection_from_mesh_border')
        seamed_mesh=ms.current_mesh()
        bnd=np.arange(seamed_mesh.vertex_matrix().shape[0])[seamed_mesh.vertex_selection_array()]


        # new_vertices[:,1]*=-1
        seam_right=((bnd[...,np.newaxis]==seam_indices).sum(axis=-1))
        seam_left=(bnd[...,np.newaxis]==(np.arange(verts.shape[0],verts.shape[0]+seam_indices.shape[0]))).sum(axis=-1)
        seam_right=seam_right.astype(bool)
        seam_left=seam_left.astype(bool)

        # breakpoint()
        seam_right=(seam_right+((~seam_left)*(new_vertices[bnd,1]>0)))>0

        seam_base_r=(~seam_right)*(~seam_left)*(new_vertices[bnd,0]<0)
        seam_base_l=(~seam_right)*(~seam_left)*(new_vertices[bnd,0]>=0)
        sean_base_r=seam_base_r.astype(bool)
        seam_base_l=seam_base_l.astype(bool)

        bnd_uv=np.zeros((bnd.shape[0],2))

        _, idx_r = ref_tree_r.query(new_vertices[bnd[seam_right]])

        # breakpoint()
        bnd_uv[seam_right]=ref_bnd_uv[ref_seam_right][idx_r]

        _, idx_l = ref_tree_l.query(new_vertices[bnd[seam_left]])

        bnd_uv[seam_left]=ref_bnd_uv[ref_seam_left][idx_l]


        _, idx_br = ref_tree_br.query(new_vertices[bnd[seam_base_r]]*100)

        bnd_uv[seam_base_r]=ref_bnd_uv[ref_seam_base_r][idx_br]

        _, idx_bl = ref_tree_bl.query(new_vertices[bnd[seam_base_l]])

        bnd_uv[seam_base_l]=ref_bnd_uv[ref_seam_base_l][idx_bl]

        uv = igl.harmonic_weights(new_vertices.astype(np.float32), faces.astype(np.int64), bnd.astype(np.int64), bnd_uv.astype(np.float32), 1)
        # _, uv = igl.lscm(new_vertices.astype(np.float32), faces.astype(np.int64), bnd.astype(np.int64), bnd_uv.astype(np.float32))
        # print(os.path.join(mesh_path,'scan_40k_uv.obj'))
        write_obj(new_vertices,faces,uv,os.path.join(mesh_path,'scan_40k_nuv.obj')) 
        # write_obj(new_vertices,faces,uv,os.path.join(mesh_path,'scan_40k_uv.obj')) 

        # breakpoint()