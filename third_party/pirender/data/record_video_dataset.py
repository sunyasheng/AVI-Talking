import os
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
from .record_dataset import RecordDataset


class RecordVideoDataset(RecordDataset):
    def __init__(self, opt, is_inference):
        super(RecordVideoDataset, self).__init__(opt, is_inference)
        self.video_index = -1
        self.cross_id = opt.cross_id
        # whether normalize the crop parameters when performing cross_id reenactments
        # set it as "True" always brings better performance
        self.norm_crop_param = False # In future, please make it True

    def __len__(self):
        return len(self.data_dict.keys())

    def load_next_video(self):
        data = {}
        if self.cross_id is True:
            self.video_index += 1

            # self.video_index = 0
            video_name = self.video_names[0+1]
            # video_name = self.video_names[self.video_index]
            img_paths = self.data_dict[video_name]['paths']
            print('motion video name: ', video_name)

            source_video_name = self.video_names[self.video_index+1]
            source_img_paths = self.data_dict[source_video_name]['paths']

            img0 = Image.open(source_img_paths[0])
            # img0 = Image.open(img_paths[0])
            data['source_image'] = self.transform(img0)
            crop_norm_ratio = None
            print('ref video name: ', source_video_name)

            data['target_image'], data['target_semantics'] = [], []
            for frame_index in range(len(img_paths)):
                img1 = Image.open(img_paths[frame_index])
                # data['target_image'].append(self.transform(img1))
                data['target_image'].append(self.transform(img0))
                data['target_semantics'].append(
                    self.transform_semantic(self.data_dict[video_name], frame_index, crop_norm_ratio, is_smooth_cam=True)
                )
            # data['video_name'] = video_name
            data['video_name'] = source_video_name
            return data
        else:
            self.video_index += 1

            video_name = self.video_names[self.video_index]
            img_paths = self.data_dict[video_name]['paths']

            img0 = Image.open(img_paths[0])
            data['source_image'] = self.transform(img0)
            crop_norm_ratio = None
            data['target_image'], data['target_semantics'] = [], []
            for frame_index in range(len(img_paths)):
                img1 = Image.open(img_paths[frame_index])
                data['target_image'].append(self.transform(img1))
                data['target_semantics'].append(
                    self.transform_semantic(self.data_dict[video_name], frame_index, crop_norm_ratio)
                )
            data['video_name'] = video_name
            return data

    def __getitem__(self, index):
        data = {}
        video_name = self.video_names[index]
        img_paths = self.data_dict[video_name]['paths']
        # print(img_paths[:10])

        frame_source, frame_target = self.random_select_frames(len(img_paths))
        img_path_1, img_path_2 = img_paths[frame_source], img_paths[frame_target]

        img1 = Image.open(img_path_1)
        data['source_image'] = self.transform(img1)

        img2 = Image.open(img_path_2)
        data['target_image'] = self.transform(img2)

        data['target_semantics'] = self.transform_semantic(self.data_dict[video_name], frame_target)
        data['source_semantics'] = self.transform_semantic(self.data_dict[video_name], frame_source)

        return data
