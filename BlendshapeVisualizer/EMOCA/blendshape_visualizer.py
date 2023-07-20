import os.path

import numpy as np
import os
import copy
from PIL import Image
import sys
import torchvision.transforms as transforms
import torch
sys.path.append(os.path.dirname(__file__))
from .gdl_apps.EMOCA.utils.load import load_model
from .gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode


class Visualizer3DMM:
    def __init__(self, path=None):
        super(Visualizer3DMM, self).__init__()
        root = os.path.dirname(os.path.abspath(__file__))
        path_to_models = os.path.join(root, 'assets/EMOCA/models')
        model_name = 'EMOCA_v2_lr_mse_20'
        mode = 'detail'
        # mode = 'coarse'
        # 1) Load the model
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()
        self.emoca = emoca

        # templ_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templ.png')
        templ_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templ_closed_mouth.jpg')
        templ_pil = Image.open(templ_path)
        templ_pil = templ_pil.resize((224, 224))
        transform = transforms.ToTensor()

        # Apply the transformation to the image
        tensor_image = transform(templ_pil)
        img = {}
        img['image'] = tensor_image.cuda()
        if len(img["image"].shape) == 3:
            img["image"] = img["image"].view(1, 3, 224, 224)

        with torch.no_grad():
            self.templ_vals = emoca.encode(img, training=False)
        # import pdb; pdb.set_trace()

    def __call__(self, exp_coeff, pose_coeff, shape_coeff=None):
        exp_coeff = exp_coeff.cuda()
        pose_coeff = pose_coeff.cuda()
        rendered_imgs_stack = []
        for i in range(len(exp_coeff)):
            vals = copy.deepcopy(self.templ_vals)
            # vals = {}
            # for k,v in self.templ_vals.items():
            #     import pdb; pdb.set_trace()
            #     vals[k] = v.clone()
            # import pdb; pdb.set_trace()
            if shape_coeff is not None:
                vals['shapecode'][0] = shape_coeff[i,:]
            vals['expcode'][0] = exp_coeff[i,:]
            vals['posecode'][0, 3:] = pose_coeff[i,3:] if pose_coeff.shape[1] == 6 else pose_coeff[i,:]
            if pose_coeff.shape[1] == 6:
                vals['posecode'][0, :3] = pose_coeff[i,:3]
            vals['original_code']['exp'][0] = exp_coeff[i,:]
            vals['original_code']['pose'][0, 3:] = pose_coeff[i,3:] if pose_coeff.shape[1] == 6 else pose_coeff[i,:]
            # print(vals['posecode'])
            # vals['expcode'][0] = exp_coeff[i, :]
            # vals['posecode'][0, :] = pose_coeff[i, :]
            # vals['original_code']['exp'][0] = exp_coeff[i, :]
            # vals['original_code']['pose'][0, :] = pose_coeff[i, :]

            rendered_imgs_dict, mesh_imgs_dict = decode(self.emoca, vals, training=False)
            # rendered_imgs_stack.append(rendered_imgs_dict['predicted_images'])
            rendered_imgs_stack.append(mesh_imgs_dict['geometry_coarse'])
            # import pdb; pdb.set_trace()
        rendered_imgs_stack = torch.concat(rendered_imgs_stack, dim=0)
        # import torchvision
        # torchvision.utils.save_image(rendered_imgs_stack, '/home/v-yashengsun/emoca/rendered_imgs_stack.jpg')
        # import pdb; pdb.set_trace()
        return rendered_imgs_stack

