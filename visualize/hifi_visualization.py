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


class HIFIVisualizer:
    def __init__(self, ):
        super().__init__()
        self.renderer = ModelRenderer(2000, 256)
        self.hstd = torch.tensor(np.array([0.01453, 0.01309, 0.03104]).reshape([1, 3])).cuda()
        self.hmean = torch.tensor(np.array([-4.236e-05, -5.696e-03, -1.239e-01])).cuda().view([1, 3])
        self.move = torch.tensor([[0, 0, 160]]).view(1, 1, 3)

        self.face_buf =load_obj(os.path.join("HiFi", "hifi_9518_new.obj"))[1].verts_idx

    @torch.no_grad()
    def visualize_verts(self, vertice, save_path):
        os.makedirs('temps', exist_ok=True)
        self.face_buf = self.face_buf.to(vertice)
        self.hmean, self.hstd, self.move = self.hmean.to(vertice), self.hstd.to(vertice), self.move.to(vertice)
        # import pdb; pdb.set_trace();
        for i in tqdm(range(vertice.shape[0])):
            cur_vertice = (vertice[i].view(-1, 3) - self.hmean) / self.hstd
            # cur_vertice = (vertice[i].view(-1, 3))
            cur_vertice = cur_vertice.float()
            cur_vertice = cur_vertice.view(1, -1, 3).float() - self.move
            vertice_render = self.renderer.sha_renderer(Meshes(cur_vertice, self.face_buf.view(1, -1, 3), TexturesVertex(torch.ones_like(cur_vertice))))            
            res = vertice_render.cpu().numpy() * 255
            cv2.imwrite(os.path.join('temps',  "%05d"%i + '.png'), res[0][:, :, :3])
        
        cmd = "ffmpeg -r 25  -pattern_type glob  -i  temps/'*.png' -r 25 -pix_fmt yuv420p -qscale:v 2  -y -r 25  -shortest {} -loglevel error".format(save_path)
        subprocess.call(cmd, shell=True)
        cmd = "rm -rf temps"
        subprocess.call(cmd, shell=True)
