import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
# from util import util
# from lib.models.networks.audio_network import ResNetSE, SEBasicBlock
# from lib.models.networks.FAN_feature_extractor import FAN_use
# from lib.models.networks.vision_network import ResNeXt50
from .audio_network import ResNetSE, SEBasicBlock
from .FAN_feature_extractor import FAN_use
from .vision_network import ResNeXt50


import omegaconf as omega


class ResSEAudioEncoder(nn.Module):
    def __init__(self, opt, nOut=2048, n_mel_T=None):
        super(ResSEAudioEncoder, self).__init__()
        self.nOut = nOut
        self.opt = opt
        pose_dim = self.opt.model.net_motion.pose_dim
        eye_dim = self.opt.model.net_motion.eye_dim
        motion_dim = self.opt.model.net_motion.motion_dim
        # Number of filters
        num_filters = [32, 64, 128, 256]
        if n_mel_T is None: # use it when use audio identity
            n_mel_T = opt.model.net_audio.n_mel_T
        self.model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, self.nOut, n_mel_T=n_mel_T)
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
            

    def forward(self, x, _type=None):
        input_size = x.size()
        if len(input_size) == 5:
            bz, clip_len, c, f, t = input_size
            x = x.view(bz * clip_len, c, f, t)
        out = self.model(x)
        
        out = out.view(-1, out.shape[-1])
        mouth_embed = self.mouth_embed(out)
        return out, mouth_embed



class ResSESyncEncoder(ResSEAudioEncoder):
    def __init__(self, opt):
        super(ResSESyncEncoder, self).__init__(opt, nOut=512, n_mel_T=1)


class ResNeXtEncoder(ResNeXt50):
    def __init__(self, opt):
        super(ResNeXtEncoder, self).__init__(opt)


def load_checkpoints(net_motion, cfg):
    ########################## initialize model ###############################################################
    # load pretrained or random initialize net_appearance model
    return_list = []

    # load pretrained or random initialize net_motion model
    if not net_motion is None:
        if cfg.model.net_motion.resume:
            # load pretrained model
            model_file = cfg.model.net_motion.pretrained_model
            model_dict = torch.load(model_file)
            init_dict = net_motion.state_dict()
            key_list = list(set(model_dict.keys()).intersection(set(init_dict.keys())))
            for k in key_list:

                if "mouth_fc" in k or "headpose_fc" in k or "classifier" in k or "to_feature" in k or "to_embed" in k:
                    continue

                init_dict[k] = model_dict[k]
            net_motion.load_state_dict(init_dict)
        else:
            print("ERROR: non-identity model not be loaded!")
        return_list.append(net_motion)
        print("non-identity model loaded!")
    else:
        print("ERROR: non-identity model needed!")

    print("all models load sucessfully!")


class FanEncoder(nn.Module):
    def __init__(self, opt=None):
        super(FanEncoder, self).__init__()

        # Load the YAML file
        cur_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # print(cur_root)
        opt = omega.OmegaConf.load(os.path.join(cur_root, "configs/inference.yaml"))

        self.opt = opt
        pose_dim = self.opt.model.net_motion.pose_dim
        eye_dim = self.opt.model.net_motion.eye_dim
        self.model = FAN_use()

        self.to_mouth = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512-pose_dim-eye_dim))
        
        self.to_headpose = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))

        self.to_eye = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.eye_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, eye_dim))

        self.to_emo = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.emo_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 30))

    def forward_feature(self, x):
        net = self.model(x)
        return net

    def forward(self, x):
        x = self.model(x)
        mouth_feat = self.to_mouth(x)
        headpose_feat = self.to_headpose(x)
        headpose_emb = self.headpose_embed(headpose_feat)
        eye_feat = self.to_eye(x)
        eye_embed = self.eye_embed(eye_feat)
        emo_feat = self.to_emo(x)
        emo_embed = self.emo_embed(emo_feat)
        return headpose_emb, eye_embed, emo_embed, mouth_feat
