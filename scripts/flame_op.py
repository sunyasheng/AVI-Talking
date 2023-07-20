from meshio import Mesh
import numpy as np

import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, 'BlendshapeVisualizer/EMOCA'))
print(os.path.join(root, 'BlendshapeVisualizer/EMOCA'))
from gdl.models.DecaFLAME import FLAME, FLAMETex, FLAME_mediapipe

import pickle
from easydict import EasyDict


def meshtalk_eye():
    template_obj_path = '../meshtalk/assets/face_template.obj'
    template_obj = Mesh(template_obj_path)
    template_obj.colors = np.ones_like(template_obj.vertices) * 0.5
    red = np.array([1.0,1.0,0])
    
    eye_mask_path = '../meshtalk/assets/weighted_eye_mask.txt'
    eye_mask = np.loadtxt(eye_mask_path)
    template_obj.colors[eye_mask==1.0] = red
    template_obj.save('../meshtalk/assets/face_template_colored_eye.obj')

    mouth_mask_path = '../meshtalk/assets/weighted_mouth_mask.txt'
    mouth_mask = np.loadtxt(mouth_mask_path)
    template_obj.colors[mouth_mask==1.0] = red
    template_obj.save('../meshtalk/assets/face_template_colored_mouth.obj')


def color_eye_part():
    ref_obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template_eyes.obj'
    ref_mesh_obj = Mesh(ref_obj_path)
    
    left_eyeball_vertices = (ref_mesh_obj.colors[:,0]==1.) & (ref_mesh_obj.colors[:,1]==0.)& (ref_mesh_obj.colors[:,2]==0.)
    right_eyeball_vertices = (ref_mesh_obj.colors[:,0]==0.) & (ref_mesh_obj.colors[:,1]==1.)& (ref_mesh_obj.colors[:,2]==0.)

    obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template.obj'
    mesh_obj = Mesh(obj_path)
    # eye_vertices = (mesh_obj.vertices[:,2]>0.030) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]>1.52) & (mesh_obj.vertices[:,1]<1.57)
    eye_vertices = (mesh_obj.vertices[:,2]>0.030) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]>1.49) & (mesh_obj.vertices[:,1]<1.57)

    eye_vertices = (eye_vertices & (~left_eyeball_vertices))
    eye_vertices = (eye_vertices & (~right_eyeball_vertices))

    mesh_obj.colors = np.ones_like(mesh_obj.vertices) * 0.5
    red = np.array([1.0,1.0,0])
    mesh_obj.colors[eye_vertices] = red
    mesh_obj.save('./upper_face_0.obj')


def select_vert():
    obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template.obj'
    mesh_obj = Mesh(obj_path)
    # frontal_vertices = (mesh_obj.vertices[:,2]>0.035) & (mesh_obj.vertices[:,1]>1.4)
    
    # green = np.array([0,1.0,1.0])
    # mesh_obj.colors = np.ones_like(mesh_obj.vertices) * 0.5
    # mesh_obj.colors[frontal_vertices] = green

    # mouth_vertices = (mesh_obj.vertices[:,2]>0.035) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]<1.5)
    # red = np.array([1.0,1.0,0])
    # mesh_obj.colors = np.ones_like(mesh_obj.vertices) * 0.5
    # mesh_obj.colors[mouth_vertices] = red

    # mesh_obj.save('./head_template_w_color.obj')

    eye_vertices = (mesh_obj.vertices[:,2]>0.030) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]>1.52) & (mesh_obj.vertices[:,1]<1.57)
    red = np.array([1.0,1.0,0])
    mesh_obj.colors = np.ones_like(mesh_obj.vertices) * 0.5
    mesh_obj.colors[eye_vertices] = red
    mesh_obj.save('./head_template_w_color.obj')

    import pdb; pdb.set_trace();

def fix_cfg(flame_cfg, proj_root):
    new_flame_cfg = {}
    for k,v in flame_cfg.items():
        if 'path' in k and 'motion-latent-diffusion' in v:
            new_flame_cfg[k] = v.replace('/data/yashengsun/Proj/TalkingFace/motion-latent-diffusion',
                                         proj_root)
        else:
            new_flame_cfg[k] = v
    return new_flame_cfg

def scale_vert():
    obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template.obj'
    mesh_obj = Mesh(obj_path)
    
    with open('misc/flame_cfg.pkl', 'rb') as f:
        flame_cfg = pickle.load(f)
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        flame_cfg = fix_cfg(flame_cfg, proj_root)
        flame_cfg = EasyDict(flame_cfg)

    flame = FLAME_mediapipe(flame_cfg)
    # template = flame.v_template.reshape(1,1,args.vertice_dim)
    
    voca_path = 'texture_mesh.obj'
    voca_obj =Mesh(voca_path)
    voca_avg_path = 'templates.pkl'
    with open(voca_avg_path, 'rb') as f:
        voca_avg = pickle.load(f, encoding='latin1')

    for k, v in voca_avg.items():
        print(k)
        print(v.max(), v.min())


    mesh_obj.vertices = voca_avg['FaceTalk_170904_03276_TA']
    mesh_obj.save('FaceTalk_170904_03276_TA.obj')

    mesh_obj.vertices = flame.v_template
    mesh_obj.save('our_template.obj')
    
    import pdb; pdb.set_trace();


if __name__ == '__main__':
    # scale_vert()
    # select_vert()
    color_eye_part()
    # meshtalk_eye()
