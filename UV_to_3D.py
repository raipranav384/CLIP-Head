# -*- coding: utf-8 -*-
# @Author  : thesigmaguy
# @File    : get_vertex_and_normal_texture.py

import os
import glm
import cv2
import math
import moderngl
import numpy as np
import moderngl_window
from pathlib import Path
import natsort
from tqdm import tqdm
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
from pyrr import Matrix44
from datetime import datetime
from moderngl_window import screenshot
import argparse
import glob


geometry_name = None
GEONAME = []
# window_size = [512, 512]
window_size = [4096, 4096]
geometry_filename = ''
save_texture_vertices, save_texture_vertex_normals = '', ''

class dummy(moderngl_window.WindowConfig):
    resource_dir = Path('.').absolute()
    gl_version = (4, 4)
    resizable = False
    window_size = window_size
    # vsync = False
    aspect_ratio = 1.
    title = 'Dummy'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.ctx.program(
            vertex_shader=open('./shaders/vertex_texture.vert.glsl').read(),
            fragment_shader=open('./shaders/vertex_texture.frag.glsl').read()
        )
        
        global geometry_name,window_size, GEONAME
        global save_texture_vertices, save_texture_vertex_normals, geometry_filename

        self.vao = []
        self.texnames = []
        for item in GEONAME:
            geometry_name = item
            geometry_filename = item

            end_part = geometry_name.split('/')[-1]
            save_texture_vertices = geometry_filename[:-len(end_part)] + end_part[:-4]+'_surface_pos.npy'
            
            model = trimesh.load(geometry_filename)

            # print(save_texture_vertices)


            vertices = np.array(model.vertices).astype('f4')
            faces = np.array(model.faces).astype('uint32')
            uv = np.array(model.visual.uv).astype('f4')
            # print(faces.dtype, faces.shape, vertices.shape, vertices.dtype)
            # print(uv.shape, uv.dtype)

            self.vbo = self.ctx.buffer(vertices.flatten().tobytes())
            self.ebo = self.ctx.buffer(faces.flatten().tobytes())

            uv = 2 * (uv - .5)
            # print(uv.min(), uv.max())
            self.uvo = self.ctx.buffer(uv.astype('f4').flatten().tobytes())

            self.vao_content = [
                (self.vbo, '3f', 'in_vert'),
                (self.uvo, '2f', 'in_uv')
            ]
            self.vao.append(self.ctx.vertex_array(self.program, self.vao_content, index_buffer=self.ebo, index_element_size=4))
            self.texnames.append(save_texture_vertices)

        self.gbuffer = self.ctx.framebuffer(
                                                color_attachments=(
                                                    self.ctx.texture(self.window_size, 4, dtype='f4'),
                                                ),
                                            )
        self.frame_no = 0

    def render(self, time: float, frame_time: float):
        self.ctx.clear(.0, .0, .0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.gbuffer.clear()
        self.gbuffer.use()
        # print(self.frame_no)
        # print(self.vao)
        self.vao[self.frame_no].render()

        vertex_texture = np.frombuffer(self.gbuffer.color_attachments[0].read(),
                                       dtype="f4").reshape(*self.window_size, 4)

        vertex_texture = np.flipud(vertex_texture)

        cv2.imwrite(f'v_test_temp.png', vertex_texture[:, :, :3] * 255)
        # print("Done")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",self.texnames[self.frame_no])
        np.save(self.texnames[self.frame_no], vertex_texture[:, :, :3])

        self.frame_no += 1 
        if self.frame_no > len(self.vao) - 1:
            exit()

    @classmethod
    def run(cls):
        moderngl_window.run_window_config(cls)

    @classmethod
    def add_arguments(cls, parser):
        # parser.add_argument(
        #     '--geometry',
        #     type=str
        # )
     
        parser.add_argument(
            '--width',
            type=int,
            default=1024,
            help='Width of render window'
        )

        parser.add_argument(
            '--height',
            type=int,
            default=1024,
            help='Height of render Window'
        )

       


# data_dir = os.environ['DATA_DIR']
# src = data_dir + '/'

# frames = natsort.natsorted(os.listdir(src))


# for idx in range(len(frames)):
#     # garment = ['top','bottom']
#     garment = ['top']
#     for g in garment:
#         folder_name = src + frames[idx] + '/out/' + g + '/'
#         for filename in os.listdir(folder_name):
#             if filename.endswith("_uv.obj"):
#                 file_path = folder_name + filename
#                 GEONAME.append(file_path)
input_path= "./results"
mesh_list=glob.glob(os.path.join(input_path,'*'))
mesh_list=natsort.natsorted(mesh_list)
for mesh_id in tqdm(mesh_list):
    if not os.path.isdir(mesh_id):
        continue
    # print(mesh_id)
    mesh_exp=glob.glob(os.path.join(mesh_id,'*'))
    mesh_exp=natsort.natsorted(mesh_exp)
    for mesh_pth in tqdm(mesh_exp):
        if os.path.exists(os.path.join(mesh_pth,'scan_40k_nuv_surface_pos.npy')):
            continue
        if os.path.exists(os.path.join(mesh_pth,'normal_map.png')) and os.path.exists(os.path.join(mesh_pth,'tex_map.png')):
            continue
        GEONAME.append(os.path.join(mesh_pth,'scan_40k_nuv.obj'))
        # break

# GEONAME.append("/hdd_data/common/NPHM/chunk_01/095/009/scan_40k_nuv.obj")
print(len(GEONAME))
print(GEONAME)
dummy.run()
