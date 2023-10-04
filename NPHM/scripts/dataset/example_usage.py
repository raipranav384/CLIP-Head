import pathlib
import pyvista as pv
import numpy as np

from NPHM.data.manager import DataManager
from NPHM.utils.mesh_operations import cut_trimesh_vertex_mask

def print_segment(segment_name):
    print('################')
    print(f'#### {segment_name} ####')
    print('################')



if __name__ == '__main__':

    cwd = pathlib.Path(__file__).parent.absolute().parents[1]
    dummy_data_path = str(cwd) + '/dataset/dummy_data/'


    manager = DataManager(dummy_path=dummy_data_path)

    ####################################################################################################################
    print_segment('Iterating over Subjects and Expressions')
    ####################################################################################################################

    all_subjects = manager.get_all_subjects()
    print(f'The dataset contains subjects with IDs {all_subjects}')

    for subject in all_subjects:

        expressions = manager.get_expressions(subject=subject)
        print(f'Subject {subject} has expressions {expressions}')

        for i, expression in enumerate(expressions[:6:3]):
            m_flame = manager.get_flame_mesh(subject=subject, expression=expression)
            m_registration = manager.get_registration_mesh(subject=subject, expression=expression)
            m_scan = manager.get_raw_mesh(subject=subject, expression=expression)

            print('Left: FLAME fitting, Middle: Registration (upsampled FLAME, refined with ARAP), Right: Raw Scan (slightly downsampled).')
            if i % 2 != 0:
                print('Here we show the texture that comes alongside the 3D scans.')
            pl = pv.Plotter(shape=(1, 3))

            pl.subplot(0, 0)
            pl.add_mesh(m_flame)

            pl.subplot(0, 1)
            pl.add_mesh(m_registration)

            pl.subplot(0, 2)
            if i % 2 == 0:
                pl.add_mesh(m_scan)
            else:
                pl.add_mesh(m_scan, texture=pv.numpy_to_texture(np.array(m_scan.visual.material.image)))


            pl.link_views()

            pl.camera_position = (0, 0.15, 6)
            pl.camera.zoom(4)
            pl.camera.roll = 0
            pl.camera.up = (0, 1, 0)
            pl.camera.focal_point = (0, 0.15, 0)
            pl.show()


    ####################################################################################################################
    print_segment('Facial Landmarks and Facial Anchor Points')
    ####################################################################################################################

    subject = 365
    expression = 0

    facial_landmarks = manager.get_landmarks(subject=subject, expression=expression)
    facial_anchors = manager.get_facial_anchors(subject=subject, expression=expression)
    m_scan = manager.get_raw_mesh(subject=subject, expression=expression)
    m_registration = manager.get_registration_mesh(subject=subject, expression=expression)

    print('Left Registered Mesh, Right Raw Scan. Bother overlayed with:')
    print('Facial Landmarks in white and')
    print('the facial anchors used in NPHM in red')

    pl = pv.Plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_mesh(m_registration)
    pl.add_points(facial_landmarks)
    pl.add_points(facial_anchors, color='red', point_size=10)

    pl.subplot(0, 1)
    pl.add_mesh(m_scan)
    pl.add_points(facial_landmarks)
    pl.add_points(facial_anchors, color='red', point_size=10)

    pl.link_views()

    pl.camera_position = (0, 0.15, 6)
    pl.camera.zoom(4)
    pl.camera.roll = 0
    pl.camera.up = (0, 1, 0)
    pl.camera.focal_point = (0, 0.15, 0)
    pl.show()

    ####################################################################################################################
    print_segment('Planar Segmentation of torso')
    ####################################################################################################################
    '''
    For simplicity we separated the torso from the head using a plane defined by 3 vertices on the FLAME fitting.
    For many people this removes a significant part of the hair.
    In the future more elaborate rays to perform this separation are necessary.
    '''


    head_mask = manager.cut_throat(m_scan.vertices, subject=subject, expression=expression)

    m_head = cut_trimesh_vertex_mask(m_scan.copy(), mask=head_mask)


    print('Red area is used for training. The white area is discarded in preprocessing.')
    pl = pv.Plotter()

    pl.add_mesh(m_scan)

    pl.add_mesh(m_head, color='red')

    pl.show()


    ####################################################################################################################
    print_segment('Single-View Observations')
    ####################################################################################################################
    '''
    Note: This is an example of our evaluation of the models' performance. 
    Time will tell whether the quality of observations is too high.
    It might be usefull to further make the observation sparse, or add artificial noise, 
    in order to make for a more challenging scenario.
    '''

    print('Fitting is based on multiple single view observations. One for each expression.')
    print('Additionally, we provide an observation from the back for the first expression only. This, in theory, enables more accurate reconstruction of hair.')

    for subject in all_subjects:
        expressions = manager.get_expressions(subject)
        pl = pv.Plotter(shape=(1, len(expressions)))
        for i, expression in enumerate(expressions):

            obs = manager.get_single_view_obs(subject, expression, include_back=i ==0)
            pl.subplot(0, i)
            pl.add_points(obs)

        pl.link_views()

        pl.camera_position = (0, 0.15, 6)
        pl.camera.roll = 0
        pl.camera.up = (0, 1, 0)
        pl.camera.focal_point = (0, 0.15, 0)
        pl.show()




