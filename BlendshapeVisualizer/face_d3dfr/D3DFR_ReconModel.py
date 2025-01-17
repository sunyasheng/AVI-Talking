import os
import torch
from scipy.io import loadmat
from kornia.geometry import warp_affine
from .d3dfr_pytorch import ReconNetWrapper, ResNet50_nofc
from .BFM09Model import BFM09ReconModel
from .preprocess import D3DFR_DEFAULT_CROP_SIZE


class model_FR3D_new_wrapper(torch.nn.Module):
    def __init__(self, pretrained="./checkpoints/netG_epoch_25.pth", tag_finetune=False):
        super(model_FR3D_new_wrapper, self).__init__()

        # initial model
        #self.model_FR3d = ReconNetWrapper()
        self.model_FR3d = ResNet50_nofc([256,256], 257+4, use_last_fc=False)
        if pretrained is not None and os.path.exists(pretrained):
            checkpoint_no_module = {}
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
            for k, v in checkpoint.items():
                if k.startswith('module'):
                    k = k[7:]
                checkpoint_no_module[k] = v
            info = self.model_FR3d.load_state_dict(checkpoint_no_module, strict=False)
        print(pretrained, info)       
        self.model_FR3d.eval()

        if not tag_finetune:
            for param in self.model_FR3d.parameters():
                param.requires_grad = False

    def forward(self, im, warpmat, dsize = [256,256]):
        """
        :param im: range [-1,1], B*C*512*512,
        :param warp_mat: B*2*3
        :return:
             coeff: B*257
        """
        im_crop_tensor = warp_affine(im, warpmat, dsize=[256,256])
        #pred_coeff = self.model_FR3d(im_crop_tensor * 0.5 + 0.5)
        pred_coeff = self.model_FR3d(im_crop_tensor)
        return pred_coeff

class model_FR3D_wrapper(torch.nn.Module):
    def __init__(self, tag_finetune=False):
        super(model_FR3D_wrapper, self).__init__()

        # initial model
        self.model_FR3d = ReconNetWrapper()
        #self.model_FR3d = ResNet50_nofc()
        self.model_FR3d.eval()

        if not tag_finetune:
            for param in self.model_FR3d.parameters():
                param.requires_grad = False

    def forward(self, im, warpmat, dsize=D3DFR_DEFAULT_CROP_SIZE):
        """
        :param im: range [-1,1], B*C*512*512,
        :param warp_mat: B*2*3
        :return:
             coeff: B*257
        """
        im_crop_tensor = warp_affine(im, warpmat, dsize=dsize)
        pred_coeff = self.model_FR3d(im_crop_tensor * 0.5 + 0.5)
        return pred_coeff


class model_3DMM_wrapper(torch.nn.Module):
    def __init__(self, checkpoint_path, device='cuda'):
        #import pdb;pdb.set_trace()
        #device = 'cpu'
        super(model_3DMM_wrapper, self).__init__()

        model_path = os.path.join(checkpoint_path, 'BFM09_model_info.mat')
        model_dict = loadmat(model_path)
        self.recon_model = BFM09ReconModel(model_dict, device=device, img_size=224)

    def forward(self, D3D_coeff, inverse_warpmat):
        """
        :param D3D_coeff: B*257
        :param inverse_warpmat: B*2*3
        :return:
             result_dict:
                'rendered_img': rendered_img, #[B H W 4] , range[-1,1]
                'vs': vs_t,
        """
        pred_dict = self.recon_model(D3D_coeff, render=True)

        # warp back to original image
        rendered_imgs = pred_dict['rendered_img']
        out_img_224 = (rendered_imgs[:, :, :, :3]/255.0).permute(0, 3, 1, 2)
        out_mask_224 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)

        out_img_512 = warp_affine(out_img_224, inverse_warpmat, dsize=(512, 512))
        out_mask_512 = warp_affine(out_mask_224, inverse_warpmat, dsize=(512, 512)).detach()
        im_3DRec = (out_img_512 * out_mask_512)*2-1

        # warp back to original image
        rendered_imgs = pred_dict['rendered_img_gray']
        out_img_224 = (rendered_imgs[:, :, :, :3] / 255.0).permute(0, 3, 1, 2)
        out_mask_224 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)
        # inverse_warp_mat = invert_affine_transform(warp_mat)
        out_img_512 = warp_affine(out_img_224, inverse_warpmat, dsize=(512, 512))
        out_mask_512 = warp_affine(out_mask_224, inverse_warpmat, dsize=(512, 512)).detach()
        im_3DRec_gray = (out_img_512 * out_mask_512) * 2 - 1

        VertexPosition = pred_dict['vs']
        VertexColor = pred_dict['color']

        return {'im_3DRec': im_3DRec, 'im_3DRec_gray': im_3DRec_gray,
                'VertexPosition': VertexPosition, 'VertexColor': VertexColor,
                'out_mask_512': out_mask_512}


class model_3DMM_new_wrapper(torch.nn.Module):
    def __init__(self, checkpoint_path, device='cuda'):
        #import pdb;pdb.set_trace()
        #device = 'cpu'
        super(model_3DMM_new_wrapper, self).__init__()

        model_path = os.path.join(checkpoint_path, 'BFM09_model_info.mat')
        model_dict = loadmat(model_path)
        self.recon_model = BFM09ReconModel(model_dict, device=device, img_size=256, focal=1015*256/224)

    def forward(self, D3D_coeff, inverse_warpmat):
        """
        :param D3D_coeff: B*257
        :param inverse_warpmat: B*2*3
        :return:
             result_dict:
                'rendered_img': rendered_img, #[B H W 4] , range[-1,1]
                'vs': vs_t,
        """
        pred_dict = self.recon_model(D3D_coeff, render=True)

        # warp back to original image
        rendered_imgs = pred_dict['rendered_img']
        out_img_224 = (rendered_imgs[:, :, :, :3]/255.0).permute(0, 3, 1, 2)
        out_mask_224 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)

        out_img_512 = warp_affine(out_img_224, inverse_warpmat, dsize=(512, 512))
        out_mask_512 = warp_affine(out_mask_224, inverse_warpmat, dsize=(512, 512)).detach()
        im_3DRec = (out_img_512 * out_mask_512)*2-1

        # warp back to original image
        rendered_imgs = pred_dict['rendered_img_gray']
        out_img_224 = (rendered_imgs[:, :, :, :3] / 255.0).permute(0, 3, 1, 2)
        out_mask_224 = (rendered_imgs[:, :, :, 3:4] > 0).float().permute(0, 3, 1, 2)
        # inverse_warp_mat = invert_affine_transform(warp_mat)
        out_img_512 = warp_affine(out_img_224, inverse_warpmat, dsize=(512, 512))
        out_mask_512 = warp_affine(out_mask_224, inverse_warpmat, dsize=(512, 512)).detach()
        im_3DRec_gray = (out_img_512 * out_mask_512) * 2 - 1

        VertexPosition = pred_dict['vs']
        VertexColor = pred_dict['color']

        return {'im_3DRec': im_3DRec, 'im_3DRec_gray': im_3DRec_gray,
                'VertexPosition': VertexPosition, 'VertexColor': VertexColor,
                'out_mask_512': out_mask_512}
