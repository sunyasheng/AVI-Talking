import os
import torch
import numpy as np
import pickle
import cv2
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data
from os.path import join as pjoin
from torchvision import transforms
from .emoca_utils import read_exp, get_data
import scipy.signal as signal
import sys
import json

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)
from talkclip_text_generation.text_gen import TalkClipDatabase, black_videotoken_list
from scripts.celev_info import get_vid_name2action, action_dict, get_duration, get_actions


def save_head_plot(head_np, pic_path, length=75):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # import pdb; pdb.set_trace()
    x = list(range(min(head_np.shape[1], length)))
    ax.plot(x, head_np[0,:length,0])
    ax.plot(x, head_np[0,:length,1])
    ax.plot(x, head_np[0,:length,2])
    os.makedirs(os.path.dirname(pic_path), exist_ok=True)
    plt.savefig(pic_path)


def butter_lowpass_filter(data, cutoff_freq, fs=25, order=4):
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
    res[:, 0] = butter_lowpass_filter(pred_pose[:, 0], cutoff_freq=2.5)
    res[:, 1] = butter_lowpass_filter(pred_pose[:, 1], cutoff_freq=2.5)
    res[:, 2] = butter_lowpass_filter(pred_pose[:, 2], cutoff_freq=2.5)
    return res




class TalkDataset(data.Dataset):
    ### keep similar format with PIRender for convenient merge
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, opt, is_inference=False, is4diffusion=False, only_load_caption=False):
        self.opt = opt
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

        all_data_root = {
            'paishe': '/data/yashengsun/local_storage/paishe_w_cam/proc_emoca',
            'Mead_M': '/data/yashengsun/local_storage/Mead_emoca/Mead_M',
            'Mead_W': '/data/yashengsun/local_storage/Mead_emoca/Mead_W',
            'head_dynamics': '/data/yashengsun/local_storage/instruct_data/head_dynamics'
        }

        self.data_root = {}
        for dataset_name in opt.dataset_names.split(','):
            self.data_root[dataset_name] = all_data_root[dataset_name]

        self.is4diffusion = is4diffusion
        self.only_load_caption = opt.only_load_caption
        if is4diffusion:
            is_wo_audio = True
            temporal_annotation_path = 'annotations.pkl'
            clip_annotation_path = 'celebvtext_info.json'

            self.temporal_annotation = pickle.load(open(temporal_annotation_path, 'rb'))
            self.clip_annotation = json.load(open(clip_annotation_path))
            self.vid_name2action = get_vid_name2action(action_dict)

        else: is_wo_audio = False

        infer_tag = 'test' if is_inference else 'train'
        data_names = '_'.join(self.data_root.keys())
        cached_path = 'datadict_{}_{}.pkl'.format(infer_tag, data_names)
        if os.path.exists(cached_path):
            with open(cached_path, 'rb') as f:
                self.data_dict = pickle.load(f)
        else:
            res_data_dict = {}
            for k, data_root in self.data_root.items():
                data_dict = get_data(data_root, is_inference=is_inference, is_wo_audio=is_wo_audio)
                res_data_dict.update(data_dict)
            self.data_dict = res_data_dict
            with open(cached_path, 'wb') as f:
                pickle.dump(self.data_dict, f)

        # print(data_dict.keys())
        self.video_names = list(self.data_dict.keys())
        # if 'transpose_crop_MVI_0031_002' in self.video_names:
        #     self.video_names.remove('transpose_crop_MVI_0031_002')
        for video_name in black_videotoken_list:
            if video_name in self.video_names:
                self.video_names.remove(video_name)

        self.video_names = self.filter_by_seqlen(self.video_names, self.data_dict, self.opt.seq_length)

        # if is_inference is False:
        #     manual_id_name = 'transpose_crop_MVI_0013'
        #     self.video_names = [name for name in self.video_names if manual_id_name in name]
        #     # import pdb; pdb.set_trace()

        wav2vec2model_path = 'facebook/wav2vec2-base-960h'
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2model_path)
        self.coeff_std = np.load('misc/coeff_std.npy')
        self.coeff_mean = np.load('misc/coeff_mean.npy')

        # import pdb; pdb.set_trace()
        if opt.vertice_dim > len(self.coeff_mean): # fill up std distribution for pose and cam dimension
            zeros = np.zeros(6)
            ones = np.ones(6)
            self.coeff_mean = np.concatenate([self.coeff_mean, zeros])
            self.coeff_std = np.concatenate([self.coeff_std, ones])

        if opt.load_talkclip_dataset:
            self.talkclip_generator = TalkClipDatabase()

        # print(self.data_dict.keys())
        # del self.data_dict['transpose_crop_MVI_0031_002']
        self.neutral_video_names = [name for name in self.video_names if 'neutral' in name]
        self.neutral_dict = {}
        for nvn in self.neutral_video_names:
            key = nvn[:4]
            if key not in self.neutral_dict:
                self.neutral_dict[key] = []
            self.neutral_dict[key].append(nvn)

        poses = np.concatenate([self.data_dict[video_name]["pose"] for video_name in self.video_names])
        cams = np.concatenate([self.data_dict[video_name]["cam"] for video_name in self.video_names])
        # import pdb; pdb.set_trace();

        if 'Mead' in opt.dataset_names:
            # self.coeff_mean, self.coeff_std = self.get_exp_stats()
            # import pdb; pdb.set_trace()
            # self.coeff_std = np.load('misc/coeff_std.npy')
            # self.coeff_mean = np.load('misc/coeff_mean.npy')
            # np.save('misc/coeff_mean_Mead.npy', self.coeff_mean)
            # np.save('misc/coeff_std_Mead.npy', self.coeff_std)
            self.coeff_mean = np.load('misc/coeff_mean_Mead.npy')
            self.coeff_std = np.load('misc/coeff_std_Mead.npy')

        self.pose_std, self.pose_mean = None, None
        if 'head' in opt.dataset_names:
            self.pose_mean, self.pose_std = self.get_pose_stats()
            np.save('misc/pose_std_celebv.npy', self.pose_std)
            np.save('misc/pose_mean_celebv.npy', self.pose_mean)
            self.pose_std = np.load('misc/pose_std_celebv.npy')
            self.pose_mean = np.load('misc/pose_mean_celebv.npy')
            # import pdb; pdb.set_trace()

    def get_pose_stats(self,):
        poses = []
        for video_name in self.video_names:
            # if 'neutral' not in video_name: continue
            pose = self.data_dict[video_name]["pose"]
            poses.append(pose)
        poses = np.concatenate(poses)
        poses_mean = poses.mean(axis=0)
        poses_std = poses.std(axis=0)
        # import pdb; pdb.set_trace()
        return poses_mean, poses_std

    def get_exp_stats(self, ):
        exps, poses = [], []
        for video_name in self.video_names:
            if 'neutral' not in video_name: continue
            exp = self.data_dict[video_name]["exp"]
            pose = self.data_dict[video_name]["pose"]
            exps.append(exp)
            poses.append(pose)
        exps = np.concatenate(exps)
        poses = np.concatenate(poses)
        coefs = np.concatenate([exps, poses[:,3:]], axis=1)
        coefs_mean = coefs.mean(axis=0)
        coefs_std = coefs.std(axis=0)

        coeff_mean, coeff_std = coefs_mean, coefs_std
        if self.opt.vertice_dim > len(coefs_mean): # fill up std distribution for pose and cam dimension
            zeros = np.zeros(6)
            ones = np.ones(6)
            coeff_mean = np.concatenate([coefs_mean, zeros])
            coeff_std = np.concatenate([coefs_std, ones])

        return coeff_mean, coeff_std

    def filter_by_seqlen(self, video_names, data_dict, seq_length):
        selected_video_names = []
        for video_name in video_names:
            # import pdb; pdb.set_trace()
            try:
                if data_dict[video_name]["exp"].shape[0] > seq_length+10:
                    selected_video_names.append(video_name)
            except:
                import traceback
                traceback.print_exc()
                print(video_name, data_dict[video_name].keys())
        print('get {} items out of {}'.format(len(selected_video_names), len(video_names)))
        return selected_video_names

    def to_Tensor(self, img):
        if img.ndim == 3:
            wrapped_img = img.transpose(2, 0, 1) / 255.0
        elif img.ndim == 4:
            wrapped_img = img.transpose(0, 3, 1, 2) / 255.0
        else:
            wrapped_img = img / 255.0
        wrapped_img = torch.from_numpy(wrapped_img).float()

        return wrapped_img * 2 - 1

    def transform_semantic(self, semantic, is_smooth_cam=False):
        exp = semantic['exp']
        if is_smooth_cam:
            cam = smooth_pose(semantic['cam'])
            angles = smooth_pose(semantic['pose'])
        else:
            cam = semantic['cam']
            angles = semantic['pose']
        # print(exp.shape, angles.shape, cam.shape)
        coeff = np.concatenate([exp, angles, cam], 1)
        return torch.Tensor(coeff)#.permute(1,0)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        video_name = self.video_names[index]
        img_paths_full = self.data_dict[video_name]['paths']

        if video_name[:4] in self.neutral_dict:
            neutral_video_name_candidates = self.neutral_dict[video_name[:4]]
            neutral_video_name = random.choice(neutral_video_name_candidates)
            ref_img_paths_full = self.data_dict[neutral_video_name]['paths']
        else:
            ref_img_paths_full = img_paths_full

        # print(len(img_paths), self.data_dict[video_name].keys())
        # seq_len, fea_dim
        file_name = self.data_dict[video_name]["name"]
        text_description = ''
        # audio, motion_descriptor = None, None
        # coeff, pose_norm, shape, cam, img_tss, ref_img_tss = None,None,None,None,None,None
        audio, motion_descriptor = {}, {}
        coeff, pose_norm, shape, cam, img_tss, ref_img_tss = {},{},{},{},{},{}

        if self.opt.load_talkclip_dataset:
            if self.opt.wo_dataset_aug: random.seed(42)
            text_description = self.talkclip_generator.query(file_name)
            # print('text: ', text_description)

        if not self.only_load_caption:
            exp_full = self.data_dict[video_name]["exp"]
            pose_full = self.data_dict[video_name]["pose"]
            shape_full = self.data_dict[video_name]["shape"]
            cam_full = self.data_dict[video_name]["cam"]
            motion_descriptor_full = self.transform_semantic(self.data_dict[video_name])
            # print('name: ', video_name, self.data_dict[video_name]["wav"], )

            if (self.is4diffusion is False) or self.opt.load_talkclip_dataset:
                offset = 5
                audio_path = self.data_dict[video_name]["wav"]
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                audio_full = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)

                pose = pose_full[offset: -offset]
                shape = shape_full[offset: -offset]
                audio = audio_full[offset*640: ]
                cam = cam_full[offset: -offset]
                exp = exp_full[offset: -offset]
                motion_descriptor = motion_descriptor_full[offset:-offset]
                img_paths = img_paths_full[offset: -offset]
                ref_img_paths = ref_img_paths_full[offset: -offset]
            else:
                # wo audio and find the duration annotated by meta file.
                action_name = self.vid_name2action[video_name]
                start_sec, end_sec = get_duration(action_name, video_name, self.temporal_annotation, self.clip_annotation)
                shape = shape_full[start_sec*25:end_sec*25]
                is_smooth_cam = True
                if is_smooth_cam:
                    # print(pose_full, cam_full)
                    # print(pose_full.shape)
                    # save_head_plot(pose_full[np.newaxis,...], 'demo/result/before_smooth.jpg')
                    pose_full = smooth_pose(pose_full)
                    # save_head_plot(pose_full[np.newaxis,...], 'demo/result/after_smooth.jpg')
                    # exit(-1)
                    cam_full = smooth_pose(cam_full)
                    # print(pose_full, cam_full)
                pose = pose_full[start_sec*25:end_sec*25]
                cam = cam_full[start_sec*25:end_sec*25]
                exp = exp_full[start_sec*25:end_sec*25]
                motion_descriptor = motion_descriptor_full[start_sec*25:end_sec*25]
                img_paths = img_paths_full[start_sec*25:end_sec*25]
                ref_img_paths = ref_img_paths_full[start_sec*25:end_sec*25]

            seq_length = 200 if self.opt is None else self.opt.seq_length
            if self.is4diffusion is True:
                start_idx = 0
                if len(cam) > seq_length:
                    start_idx = np.random.randint(low=0, high=len(cam) - seq_length)
                # print('start_idx: ', start_idx, 'cam length: ', len(cam))
            else:
                start_idx = np.random.randint(low=0, high=len(cam)-seq_length)
            # coeff = coeff[start_idx:start_idx+seq_length]
            pose = pose[start_idx:start_idx+seq_length]
            shape = shape[start_idx:start_idx+seq_length]
            cam = cam[start_idx:start_idx+seq_length]
            exp = exp[start_idx:start_idx+seq_length]
            motion_descriptor = motion_descriptor[start_idx:start_idx+seq_length]
            img_paths = img_paths[start_idx:start_idx+seq_length]
            ref_start_idx = 0
            ref_paths = ref_img_paths[ref_start_idx:ref_start_idx+seq_length]
            if self.is4diffusion is False:
                audio = audio[start_idx*640:(start_idx+seq_length)*640+80]

            img_nps = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in img_paths]
            img_nps = np.stack(img_nps, axis=0)
            img_tss = self.to_Tensor(img_nps)

            ref_img_nps = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in ref_paths]
            ref_img_nps = np.stack(ref_img_nps, axis=0)
            ref_img_tss = self.to_Tensor(ref_img_nps)
            # print('ref: ', ref_paths[:10])
            # import pdb; pdb.set_trace()
            ## normalize
            # print(np.mean(coeff, axis=0), np.std(coeff, axis=0))
            # coeff = np.concatenate([exp, pose[:,3:]], axis=1)
            coeff = np.concatenate([exp, pose[:,3:], pose[:,:3], cam], axis=1)
            coeff = (coeff - self.coeff_mean[np.newaxis,:coeff.shape[1]]) / self.coeff_std[np.newaxis,:coeff.shape[1]]
            # print(np.mean(coeff, axis=0), np.std(coeff, axis=0))
            # import pdb;pdb.set_trace()
            # frame_cnt = 5
            # target_frame_idxes = self.get_frame_idxes(len(coeff), frame_cnt=frame_cnt)

            # coeff, coeff_pre, audio = self.get_by_frame(target_frame_idxes, coeff, audio=audio)
            # import pdb;pdb.set_trace()
            # print('audio shape: ', audio.shape, 'video name: ', video_name, 'audio full ', audio_full.shape)

            audio = torch.FloatTensor(audio) if audio is not None else None

            pose_norm = pose
            if self.pose_std is not None:
                pose_norm = (pose - self.pose_mean) * 1. / self.pose_std
                # import pdb; pdb.set_trace()
            # print('pose norm: ', pose_norm)
            coeff, pose_norm, shape, cam = torch.FloatTensor(coeff),torch.FloatTensor(pose_norm),\
                                    torch.FloatTensor(shape),torch.FloatTensor(cam)
        return audio,coeff,pose_norm,shape,cam,motion_descriptor,img_tss,ref_img_tss,file_name,text_description

    def __len__(self):
        return len(self.video_names)


def get_dataloaders(args):
    dataset = {}
    # train_data, valid_data, test_data, subjects_dict = read_data(args)
    # train_data = TalkDataset(train_data,subjects_dict,"train",args.read_audio)
    # dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # valid_data = TalkDataset(valid_data,subjects_dict,"val",args.read_audio)
    # dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.workers)
    # test_data = TalkDataset(test_data,subjects_dict,"test",args.read_audio)
    # dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=args.workers)

    train_data = TalkDataset(args, is_inference=False)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    valid_data = TalkDataset(args, is_inference=True)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=2, shuffle=False, num_workers=args.workers, drop_last=True)
    test_data = TalkDataset(args, is_inference=True)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=2, shuffle=False, num_workers=args.workers, drop_last=True)

    return dataset


if __name__ == "__main__":
    from easydict import EasyDict
    args = EasyDict()
    args.data_root = '/data/yashengsun/local_storage/paishe/paise'
    args.wav2vec2model_path = 'facebook/wav2vec2-base-960h'
    args.batch_size = 1
    args.workers = 0
    args.read_audio = True
    dataset_dict = get_dataloaders(args)
    print(dataset_dict['test'].dataset[0])
