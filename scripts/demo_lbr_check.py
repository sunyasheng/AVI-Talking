import os
import cv2
import numpy as np
import torch 
from torch import nn
from tqdm import tqdm
# from scipy.optimize import leastsq 
from scipy.optimize import leastsq 
from scipy.io import loadmat
import subprocess
import csv
import time

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    HardGouraudShader,
    SoftSilhouetteShader,
    SoftGouraudShader,
    SoftPhongShader,
    HardPhongShader,
    TexturesVertex,
    blending,
    TexturesUV
)

from scipy.optimize import leastsq 

from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.io import save_obj, load_obj



class ModelRenderer(nn.Module):
    def __init__(self, focal=2000, img_size=256, device='cuda:0', use_depth=False):
        super(ModelRenderer, self).__init__()
        self.img_size = img_size
        self.focal = focal
        self.device = device
        self.use_depth = use_depth
        self.alb_renderer = self._get_renderer(albedo=True)
        self.sha_renderer = self._get_renderer(albedo=False)
        

    def _get_renderer(self, albedo=True):
        R, T = look_at_view_transform(10, 0, 0)  # camera's position
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=200,
                                        fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)

        if albedo:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]],
                                ambient_color=[[1, 1, 1]],
                                specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])
        else:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]],
                                ambient_color=[[0.1, 0.1, 0.1]],
                                specular_color=[[0.0, 0.0, 0.0]], diffuse_color=[[0.95, 0.95, 0.95]])
            raster_settings = RasterizationSettings(
                image_size=self.img_size,
                blur_radius=0.0,
                # blur_radius=1,
                faces_per_pixel=1,
                perspective_correct=True
            )
            raster_settings.perspective_correct = True
            # blend_params = blending.BlendParams(background_color=[1, 1, 1])
            blend_params = blending.BlendParams(background_color=[0, 0, 0])
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                    shader= SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=lights,
                    blend_params=blend_params
                )
            )
            return renderer

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True
        )
        raster_settings.perspective_correct = True
        blend_params = blending.BlendParams(background_color=[1, 1, 1])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            # shader = SimpleShader(
            #     device=self.device,
            #     blend_params=blend_params
            # ),
            shader= HardFlatShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
            # shader= HardPhongShader(
            #     device=self.device,
            #     cameras=cameras,
            #     lights=lights,
            #     blend_params=blend_params
            # )
        )
        return renderer

def test(wav_path, npy_path):
    save_root = 'temps'

    cmd = "rm -rf temps/*"
    subprocess.call(cmd, shell=True)

    renderer = ModelRenderer(2000, 256)
    # renderer = ModelRenderer(4000, 512)
    move = torch.tensor([[0, 0, 160]]).cuda().view(1, 1, 3)
    
    hstd = torch.tensor(np.array([0.01453, 0.01309, 0.03104]).reshape([1, 3])).cuda()
    hmean = torch.tensor(np.array([-4.236e-05, -5.696e-03, -1.239e-01])).cuda().view([1, 3])
    # id_mean = torch.tensor(np.load(os.path.join(data_root, 'lmz_vertice.npy'))).cuda().view(-1, 3)
    face_buf =load_obj(os.path.join("HiFi", "hifi_9518_new.obj"))[1].verts_idx.cuda()

    vertice = torch.tensor(np.load(npy_path)).cuda()
    print('vertice: ', vertice.shape)

    import pdb; pdb.set_trace()
    for i in tqdm(range(vertice.shape[0])):

        # cur_vertice = (vertice[i].view(-1, 3) - hmean) / hstd
        cur_vertice = (vertice[i].view(-1, 3))
        cur_vertice = cur_vertice.float()
        # save_obj('temps/%s.obj'%i, cur_vertice, face_buf[0])
        cur_vertice = cur_vertice.view(1, -1, 3).float() - move
        vertice_render = renderer.sha_renderer(Meshes(cur_vertice, face_buf.view(1, -1, 3), TexturesVertex(torch.ones_like(cur_vertice))))
        
        # cur_bs = torch.mm(bs[i].view(1, 147), exp_base).view(-1, 3) + id_mean
        # cur_vertice = cur_bs.view(1, -1, 3).float() - move
        # bs_render = renderer.sha_renderer(Meshes(cur_vertice, face_buf, TexturesVertex(torch.ones_like(cur_vertice))))
        # res = torch.cat((vertice_render, bs_render), dim=2)
        res = vertice_render.cpu().numpy() * 255
        cv2.imwrite(os.path.join(save_root,  "%05d"%i + '.png'), res[0][:, :, :3])
    cmd = "ffmpeg -r 25  -pattern_type glob  -i  temps/'*.png' -r 25 -i  %s  -pix_fmt yuv420p    -qscale:v 2  -y -r 25  -shortest results/%s.mp4"%(wav_path, wav_path.split('/')[-1][:-4])
    subprocess.call(cmd, shell=True)
    
    # B = 1
    # for i in range(10):
    #     cur_vertice = (vertice[0:B, :].view(-1, 3))
    #     cur_vertice = cur_vertice.float()
    #     cur_vertice = cur_vertice.view(B, -1, 3).float() - move
    #     # print('time0: ', time.time())
    #     print('cur_vertice: ', cur_vertice.shape)
    #     #mesh = Meshes(self.pred_vertex, self.facemodel.face_buf.view(1, -1, 3).repeat(self.pred_vertex.shape[0], 1, 1), face_color_tv)
    #     time0 = time.time()
    #     vertice_render = renderer.sha_renderer(Meshes(cur_vertice, face_buf.view(1, -1, 3).repeat(cur_vertice.shape[0], 1, 1), TexturesVertex(torch.ones_like(cur_vertice))))
    #     time1 = time.time()
    #     print('time0: ', time1 - time0)
    #     print('vertice_render: ', vertice_render.shape)
    #     res = vertice_render.cpu().numpy() * 255
    #     cv2.imwrite('debug/debug.png', res[0][:, :, :3])


if __name__ == '__main__':
    # python scripts/demo_lbr_check.py ./yuwei_ffm/wav/shortvideo_00001.wav ./yuwei_ffm/vertices_npy_om/shortvideo_00001.npy
    import sys
    wav_path = sys.argv[1]
    npy_path = sys.argv[2]
    test(wav_path, npy_path)
