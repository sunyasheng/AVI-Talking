import copy
import os
import pickle

import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .emoca_utils import read_exp, get_data
import scipy.signal as signal


black_videotoken_list = [
    'M022_front_happy_level1_027',
    'M013_front_surprised_level1_028',
    'M012_front_disgusted_level2_027',
    'M013_front_fear_level2_028',
    'W014_front_angry_level1_017',
    'W014_front_fear_level1_008',
    'W014_front_disgusted_level2_024'
]

def butter_lowpass_filter(data, cutoff_freq, fs=100, order=4):
    """
    Apply a Butterworth low-pass filter to the input signal.

    Parameters:
        data (numpy array): Input signal.
        cutoff_freq (float): Cutoff frequency of the low-pass filter.
        fs (float): Sampling frequency of the input signal.
        order (int, optional): Order of the Butterworth filter. Default is 4.

    Returns:
        numpy array: Filtered signal.
    """
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def smooth_pose(pred_pose):
    res = np.zeros_like(pred_pose)
    res[:, 0] = butter_lowpass_filter(pred_pose[:, 0], cutoff_freq=1)
    res[:, 1] = butter_lowpass_filter(pred_pose[:, 1], cutoff_freq=1)
    res[:, 2] = butter_lowpass_filter(pred_pose[:, 2], cutoff_freq=1)
    return res

class RecordDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        self.semantic_radius = opt.semantic_radius

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

        # self.data_root = ['/data/yashengsun/local_storage/paishe_w_cam/proc_emoca',
        #                 '/data/yashengsun/local_storage/cn_hd/cn_fps25_0_output']

        self.data_root = []
        all_data_root = {
            'paishe': '/data/yashengsun/local_storage/paishe_w_cam/proc_emoca',
            'Mead_M': '/data/yashengsun/local_storage/Mead_emoca/Mead_M',
            'Mead_W': '/data/yashengsun/local_storage/Mead_emoca/Mead_W',
            'tfhd': '/data/yashengsun/local_storage/tfhd_data_output',
            'cn': '/data/yashengsun/local_storage/cn_hd/cn_fps25_0_output'
        }

        self.data_root = {}
        for dataset_name in opt.dataset_names.split(','):
            self.data_root[dataset_name] = all_data_root[dataset_name]

        infer_tag = 'test' if is_inference else 'train'
        data_names = '_'.join(self.data_root.keys())
        cached_path = 'datadict_{}_{}.pkl'.format(infer_tag, data_names)
        # import pdb; pdb.set_trace()

        if os.path.exists(cached_path):
            with open(cached_path, 'rb') as f:
                self.data_dict = pickle.load(f)
        else:
            res_data_dict = {}
            for k, data_root in self.data_root.items():
                # import pdb; pdb.set_trace()
                data_dict = get_data(data_root, is_inference=is_inference)
                res_data_dict.update(data_dict)
            self.data_dict = res_data_dict
            with open(cached_path, 'wb') as f:
                pickle.dump(self.data_dict, f)

        # print(data_dict.keys())
        self.video_names = list(self.data_dict.keys())
        self.video_names = list(self.data_dict.keys())
        if 'transpose_crop_MVI_0031_002' in self.video_names:
            self.video_names.remove('transpose_crop_MVI_0031_002')
        for video_name in black_videotoken_list:
            if video_name in self.video_names:
                self.video_names.remove(video_name)
        self.neutral_video_names = [name for name in self.video_names if 'neutral' in name]
        # import pdb; pdb.set_trace()

        self.neutral_dict = {}
        for nvn in self.neutral_video_names:
            key = nvn[:4]
            if key not in self.neutral_dict:
                self.neutral_dict[key] = []
            self.neutral_dict[key].append(nvn)

    def __len__(self):
        return len(self.video_names)

    def random_select_frames(self, num_frame):
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def get_pair_dict(self, video_name):
        data = {}
        img_paths = self.data_dict[video_name]['paths']
        frame_source, frame_target = self.random_select_frames(len(img_paths))
        img_path_1, img_path_2 = img_paths[frame_source], img_paths[frame_target]

        img1 = Image.open(img_path_1)
        data['source_image'] = self.transform(img1)

        img2 = Image.open(img_path_2)
        data['target_image'] = self.transform(img2)

        data['source_semantics'] = self.transform_semantic(self.data_dict[video_name], frame_source)
        data['target_semantics'] = self.transform_semantic(self.data_dict[video_name], frame_target)
        return data

    def __getitem__(self, index):
        data = {}
        video_name = self.video_names[index]
        data1 = self.get_pair_dict(video_name)
        # print(img_paths[:10])

        data = copy.deepcopy(data1)
        if video_name[:4] in self.neutral_dict:
            neutral_video_name_candidates = self.neutral_dict[video_name[:4]]
            neutral_video_name = random.choice(neutral_video_name_candidates)
            data2 = self.get_pair_dict(neutral_video_name)
            data['target_semantics'] = data2['target_semantics']
            data['target_image'] = data2['target_image']
            # print(video_name, neutral_video_name)
        else:
            pass

        return data


    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq

    def transform_semantic(self, semantic, frame_index, crop_norm_ratio=None, is_smooth_cam=False):
        # print(semantic.keys())
        index = self.obtain_seq_index(frame_index, semantic['exp'].shape[0])
        # coeff_3dmm = semantic[frame_index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        # ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        # angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        # translation = coeff_3dmm[:,254:257] #translation
        # crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = semantic
        ex_coeff = coeff_3dmm['exp'][index,...]
        if is_smooth_cam:
            coeff_3dmm_pose_smooth = smooth_pose(coeff_3dmm['pose'])
            coeff_3dmm_cam_smooth = smooth_pose(coeff_3dmm['cam'])
            # coeff_3dmm_cam_smooth = coeff_3dmm['cam']
            angles = coeff_3dmm_pose_smooth[index,...]
            cam = coeff_3dmm_cam_smooth[index,...]
            # print(len(index), self.semantic_radius)
            # cam = coeff_3dmm_cam_smooth[list(range(self.semantic_radius*2+1)),...]
        else:
            cam = coeff_3dmm['cam'][index,...]
            angles = coeff_3dmm['pose'][index,...]

        # print(ex_coeff.shape, angles.shape, cam.shape)
        coeff_3dmm = np.concatenate([ex_coeff, angles, cam], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)
