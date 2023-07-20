import os
import pickle
import sys
import torch
import cv2
import torch.nn.functional as F
import torchvision
import numpy as np
from pytorch3d.io import load_obj

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BlendshapeVisualizer', 'EMOCA'))
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from BlendshapeVisualizer.EMOCA.blendshape_visualizer import Visualizer3DMM
import BlendshapeVisualizer.EMOCA.gdl.utils.DecaUtils as util
from BlendshapeVisualizer.EMOCA.gdl.models.DecaFLAME import FLAME, FLAMETex, FLAME_mediapipe
from BlendshapeVisualizer.EMOCA.gdl.models.Renderer import Pytorch3dRasterizer



def read_driven_folder(upper_face_folder, frame_num, resize):
    import glob
    import os
    from loop_utils import calc_loop_idx, loopback_frames
    img_paths = sorted(glob.glob(os.path.join(upper_face_folder, '*.png')))
    if (not os.path.exists(upper_face_folder)) or len(img_paths) == 0:
        from dataset.emoca_utils import get_detect_paths
        data_root = os.path.dirname(upper_face_folder)
        folder_name = os.path.basename(upper_face_folder)
        img_paths = get_detect_paths(data_root, folder_name)

    imgs = [cv2.imread(img_path) for img_path in img_paths]

    looped_imgs = []
    for i in range(frame_num):
        new_idx = calc_loop_idx(i, loop_num=len(imgs))
        looped_img = imgs[new_idx]
        looped_img = cv2.resize(looped_img, dsize=resize)
        looped_imgs.append(looped_img)
    return looped_imgs

def save_frames2video(frames, video_path, fps=25.0):
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]

    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        video_writer.write(frame)
        
class FlameVisualizer:
    def __init__(self, ):
        # with open('misc/flame_cfg.pkl', 'rb') as f:
        #     flame_cfg = pickle.load(f)
        # flame = FLAME_mediapipe(flame_cfg)
        # verts = flame.v_template
        # faces = flame.faces_tensor

        self.topology_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template.obj'
        _, faces, aux = load_obj(self.topology_path)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)

        self.faces, self.uvcoords, self.face_uvcoords = faces, uvcoords, face_uvcoords

    def plot(self, verts):

        faces, uvcoords, face_uvcoords = self.faces, self.uvcoords, self.face_uvcoords
        batch_size = verts.shape[0]
        cam = torch.Tensor([10., 0., 0.]).unsqueeze(0).unsqueeze(0).to(verts)
        cam = cam.expand(batch_size, -1, -1)
        faces, uvcoords, face_uvcoords = faces.to(verts), uvcoords.to(verts), face_uvcoords.to(verts)
        h, w = 256, 256
        albedos = torch.ones(size=[batch_size, 3, h, w]).to(verts)

        image_size = 256
        rasterizer = Pytorch3dRasterizer(image_size)

        # import pdb; pdb.set_trace();
        transformed_vertices = util.batch_orth_proj(verts, cam)

        # camera to image space
        transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # attributes
        face_vertices = util.face_vertices(transformed_vertices, faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(transformed_vertices, faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, faces.expand(batch_size, -1, -1))

        attributes = torch.cat([face_uvcoords.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                                -1)

        # import pdb; pdb.set_trace();
        # rasterize
        rendering = rasterizer(transformed_vertices, faces.expand(batch_size, -1, -1), attributes)
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        return normal_images

    def render_verts(self, verts):
        bs = verts.shape[0]
        per_batch_size = 16
        normal_images_all = []
        for b in range(bs):
            verts_b = verts[b]
            normal_images_b = []
            for i in range(0, verts_b.shape[0], per_batch_size):
                verts_i = verts_b[i:i+per_batch_size]
                normal_images_i = self.plot(verts_i.reshape(verts_i.shape[0],-1,3))
                normal_images_b.append(normal_images_i)
            normal_images_b = torch.concat(normal_images_b, dim=0)
            normal_images_all.append(normal_images_b)
        normal_images_all = torch.concat(normal_images_all, dim=0)
        return normal_images_all

    def visualize_verts(self, verts, save_root='./results_vis', save_name='render', driven_folder=None, audio_path=None):
        per_batch_size = 8
        normal_images = []
        for i in range(0, verts.shape[0], per_batch_size):
            verts_i = verts[i:i+per_batch_size]
            normal_images_i = self.plot(verts_i)
            normal_images.append(normal_images_i)
        normal_images_ts = torch.concat(normal_images, dim=0)
        normal_images_ts = 255*(normal_images_ts*0.5 + 0.5)
        normal_images_ts = normal_images_ts.permute(0,2,3,1)

        save_path = os.path.join(save_root, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        normal_imgs = normal_images_ts.detach().cpu().numpy().astype(np.uint8)
        normal_imgs = [normal_imgs[i] for i in range(len(normal_imgs))]
        if driven_folder is not None and os.path.exists(driven_folder):
            resize = (normal_images_ts.shape[-3], normal_images_ts.shape[-2])
            driven_imgs = read_driven_folder(driven_folder, verts.shape[0], resize)
            # import pdb; pdb.set_trace();
            cat_img_nps = [np.hstack([driven_img, normal_img]) for driven_img, normal_img in zip(driven_imgs, normal_imgs)]
        else:
            print('{} does not exist.'.format(driven_folder))
            cat_img_nps = normal_imgs

        # import pdb; pdb.set_trace();
        save_frames2video(cat_img_nps, save_path + '.mp4')
        print('3d results save to {}'.format(save_path + '.mp4'))

        if audio_path is not None:
            av_cmd = 'ffmpeg -i {} -i {} -shortest {} -y -loglevel error'.format(save_path+'.mp4', audio_path, save_path+'_av.mp4')
            print(av_cmd)
            os.system(av_cmd)
  

def main():
    flame_visualizer = FlameVisualizer()
    with open('misc/flame_cfg.pkl', 'rb') as f:
        flame_cfg = pickle.load(f)
    flame = FLAME_mediapipe(flame_cfg)
    verts = flame.v_template
    normal_image = flame_visualizer.plot(verts.unsqueeze(0).expand(2, -1, -1))
    torchvision.utils.save_image(normal_image*0.5+0.5, 'normal.jpg')


if __name__ == '__main__':
    main()
