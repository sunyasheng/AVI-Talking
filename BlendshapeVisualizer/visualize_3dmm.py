import os
from scipy.io import loadmat
from face_d3dfr.blendshape_visualizer import Visualizer3DMM
import torch
import torchvision
import argparse
import numpy as np
import glob
from tqdm import tqdm


def visualize_exp(exp_path, coeff_dict, save_root='./', save_name='render'):
    exp = np.load(exp_path)
    add_id_pose = coeff_dict['coeff'][:exp.shape[0]]
    add_id_pose[:, 80:144] = exp
    rendered_img = visualizer_3dmm(torch.from_numpy(add_id_pose[::3, :]).cuda())
    rendered_img = rendered_img.permute(0, 3, 1, 2)[:, :3]
    save_path = os.path.join(save_root, save_name)
    torchvision.utils.save_image(rendered_img / 255., '{}.jpg'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='')
    args = parser.parse_args()

    checkpoint_path = 'checkpoints/BFM/'
    visualizer_3dmm = Visualizer3DMM(checkpoint_path=checkpoint_path)

    coeff_path = '/data/yashengsun/Proj/TalkingFace/CelebV-Text/downloaded_celebvtext/fps25_coef/fps25_aligned_av/kLS1WvRnQtA_5_0.mp4.mat'
    coeff_dict = loadmat(coeff_path)
    # print(coeff_dict.keys())
    # print(coeff_dict['coeff'].shape)

    proj_root = '/data/yashengsun/Proj/TalkingFace/motion-latent-diffusion'
    exp_root = 'results/mld/3DMM_PELearn_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2023-06-26-04-53-24'
    exp_paths = glob.glob(os.path.join(proj_root, exp_root, '*_ref.npy'))

    save_root = os.path.join(proj_root, exp_root.replace('results/', 'results_vis/'))
    os.makedirs(save_root, exist_ok=True)
    for i, exp_path in tqdm(enumerate(exp_paths)):
        visualize_exp(exp_path, coeff_dict, save_root=save_root, save_name='render_ref_{:05d}_'.format(i))
        pred_exp_path = exp_path.replace('_ref.npy', '.npy')
        visualize_exp(pred_exp_path, coeff_dict, save_root=save_root, save_name='render_{:05d}_'.format(i))
        # import pdb; pdb.set_trace()
        # pass

    # rendered_img = visualizer_3dmm(torch.from_numpy(coeff_dict['coeff'][::3,:]).cuda())
    # rendered_img = rendered_img.permute(0, 3, 1,2)[:,:3]
    # import pdb; pdb.set_trace()
    # torchvision.utils.save_image(rendered_img/255., 'rendered.jpg')
    # print(rendered_img.shape)
