import numpy as np
import torch
from .bfm import ParametricFaceModel
from .nvdiffrast import MeshRenderer


class Visualizer3DMM:
    def __init__(self, bfm_folder):
        super(Visualizer3DMM, self).__init__()
        # bfm_folder = 'BFM'
        camera_d = 10.
        z_near = 5.
        z_far = 15.
        isTrain = False
        focal = 1015.
        center = 112.
        bfm_model = 'BFM_model_front.mat'
        self.facemodel = ParametricFaceModel(
            bfm_folder=bfm_folder, camera_distance=camera_d, focal=focal, center=center,
            is_train=isTrain, default_name=bfm_model
        )
        self.facemodel.to('cuda:0')
        fov = 2 * np.arctan(center / focal) * 180 / np.pi

        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center),
            use_opengl=False
        )

    def __call__(self, output_coeff):
        # import pdb; pdb.set_trace()
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        # import pdb; pdb.set_trace()
        self.pred_face = torch.clamp(self.pred_face, min=0, max=1.0)
        return self.pred_face
