import os
import torch
from scipy.io import loadmat
from kornia.geometry import warp_affine
from .d3dfr_pytorch import ReconNetWrapper, ResNet50_nofc
from .BFM09Model import BFM09ReconModel
from .preprocess import D3DFR_DEFAULT_CROP_SIZE


class Visualizer3DMM(torch.nn.Module):
    def __init__(self, checkpoint_path, device='cuda'):
        super(Visualizer3DMM, self).__init__()

        model_path = os.path.join(checkpoint_path, 'BFM09_model_info.mat')
        model_dict = loadmat(model_path)
        self.recon_model = BFM09ReconModel(model_dict, device=device, img_size=256, focal=1015*256/224)

    def forward(self, D3D_coeff):
        pred_dict = self.recon_model(D3D_coeff, render=True)

        # warp back to original image
        rendered_imgs = pred_dict['rendered_img']

        return rendered_imgs
