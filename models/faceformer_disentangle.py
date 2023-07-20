import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import pickle
import math
import torchvision
import random
import cv2
from easydict import EasyDict
from omegaconf import OmegaConf
from loop_utils import loopback_frames, calc_loop_idx
from scripts.meshio import Mesh
from .lib.wav2vec import Wav2Vec2Model
from .network_utils import freeze_params

from visualize.flame_visualization import FlameVisualizer

import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, 'BlendshapeVisualizer/EMOCA'))
print(os.path.join(root, 'BlendshapeVisualizer/EMOCA'))

from gdl.models.DecaFLAME import FLAME, FLAMETex, FLAME_mediapipe
import gdl.layers.losses.DecaLosses as lossfunc
# import gdl.layers.losses.MediaPipeLandmarkLosses as lossfunc_mp
import gdl.utils.DecaUtils as util
from gdl.models.DECA import create_emo_loss

sys.path.append(os.path.join(root, 'third_party', 'pirender'))
from third_party.pirender.generators.face_model import FaceGenerator
from third_party.pirender.config import Config
from third_party.pirender.loss.perceptual import PerceptualLoss
from third_party.pirender.util.meters import Meter, set_summary_writer


def plot_lmk(lmk, image_path='ldmk.jpg'):
    lmk = lmk.detach().cpu().numpy()
    ldmk_int = 256*((lmk+1)*0.5)
    ldmk_int = ldmk_int.astype(np.uint8)
    height, width = 256, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    color = (0, 0, 255)  # Red in BGR

    # Draw the point on the image
    for i in range(len(ldmk_int)):
        x, y = ldmk_int[i][0], ldmk_int[i][1]
        cv2.circle(image, (x, y), 5, color, -1)

    cv2.imwrite(image_path, image)


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def fix_cfg(flame_cfg, proj_root):
    new_flame_cfg = {}
    for k,v in flame_cfg.items():
        if 'path' in k and 'motion-latent-diffusion' in v:
            new_flame_cfg[k] = v.replace('/data/yashengsun/Proj/TalkingFace/motion-latent-diffusion',
                                         proj_root)
        else:
            new_flame_cfg[k] = v
    return new_flame_cfg

def mask_lip(img):
    # h_ratio = [100./224., 210./224.]
    # w_ratio = [ 40./224., 185./224.]
    h_ratio = [100./224., 224./224.]
    w_ratio = [  0./224., 224./224.]
    h_range = [h_ratio[0]*img.shape[3],h_ratio[1]*img.shape[3]]
    w_range = [w_ratio[0]*img.shape[2],w_ratio[1]*img.shape[2]]
    mask_wo_lip = torch.ones_like(img)
    mask_wo_lip[:,:,int(h_range[0]):int(h_range[1]),int(w_range[0]):int(w_range[1])] = 0
    img_wo_lip = mask_wo_lip * img

    # import torchvision
    # torchvision.utils.save_image(torch.cat([img,img_wo_lip],dim=2), 'img_stack.jpg')
    # import pdb; pdb.set_trace()
    return img_wo_lip


class FLAMESelector:
    def __init__(self):
        # obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template.obj'
        obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template_eyes.obj'
        mesh_obj = Mesh(obj_path)
        self.frontal_vertices = (mesh_obj.vertices[:,2]>0.035) & (mesh_obj.vertices[:,1]>1.4)
        self.mouth_vertices = (mesh_obj.vertices[:,2]>0.035) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]<1.5)
        # self.eye_vertices = (mesh_obj.vertices[:,2]>0.030) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]>1.52) & (mesh_obj.vertices[:,1]<1.57)
        self.eye_vertices = (mesh_obj.vertices[:,2]>0.030) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]>1.49) & (mesh_obj.vertices[:,1]<1.57)

        # import pdb; pdb.set_trace()
        self.left_eyeball_vertices = (mesh_obj.colors[:,0]==1.) & (mesh_obj.colors[:,1]==0.)& (mesh_obj.colors[:,2]==0.)
        self.right_eyeball_vertices = (mesh_obj.colors[:,0]==0.) & (mesh_obj.colors[:,1]==1.)& (mesh_obj.colors[:,2]==0.)

        self.eye_vertices = (self.eye_vertices & (~self.left_eyeball_vertices))
        self.eye_vertices = (self.eye_vertices & (~self.right_eyeball_vertices))

        self.frontal_vertices_unfold =  np.stack([self.frontal_vertices,]*3,axis=-1).reshape(-1)
        self.mouth_vertices_unfold = np.stack([self.mouth_vertices,]*3,axis=-1).reshape(-1)
        self.eye_vertices_unfold = np.stack([self.eye_vertices,]*3,axis=-1).reshape(-1)

class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.vertice_scale = 1.0
        self.args = args
        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)

        if args.is_concat_mode == 0:
            # motion encoder
            self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
            # motion decoder
            self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
            # style embedding
            self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
            # periodic positional encoding
            self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        else:
            # motion encoder
            self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim+6+30)
            # motion decoder
            self.vertice_map_r = nn.Linear(args.feature_dim+6+30, args.vertice_dim)
            # style embedding
            self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim+6+30, bias=False)
            # periodic positional encoding
            self.PPE = PeriodicPositionalEncoding(args.feature_dim+6+30, period = args.period)

        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        d_model = args.feature_dim if args.is_concat_mode == 0 else 6+30+args.feature_dim
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model+args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # self.obj_vector = nn.Linear(1, args.feature_dim, bias=False)
        self.obj_embedding = nn.Parameter(torch.zeros(size=(1, args.feature_dim)))
        self.device = args.device
        self.emo2idx = {"neutral": 0, "angry": 1, "contempt": 2, "disgusted": 3, "fear": 4, "happy": 5, "sad": 6, "surprised": 7}
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

        with open('misc/flame_cfg.pkl', 'rb') as f:
            flame_cfg = pickle.load(f)
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # print(flame_cfg)
        flame_cfg = fix_cfg(flame_cfg, proj_root)
        # print(flame_cfg)
        flame_cfg = EasyDict(flame_cfg)

        self.flame = FLAME_mediapipe(flame_cfg)
        self.template = self.flame.v_template.reshape(1,1,args.vertice_dim) * self.vertice_scale
        # import pdb; pdb.set_trace()

        # modify mean and std to
        coeff_mean = np.load('misc/coeff_mean_Mead.npy')
        coeff_std = np.load('misc/coeff_std_Mead.npy')
        self.coeff_std = torch.from_numpy(coeff_std).unsqueeze(0).unsqueeze(0)
        self.coeff_mean = torch.from_numpy(coeff_mean).unsqueeze(0).unsqueeze(0)

        self.fan_net = None
        if args.w_fan == 1:
            from third_party.pd_fgc_inference.lib.models.networks.encoder import FanEncoder, load_checkpoints
            self.fan_net = FanEncoder()
            load_checkpoints(self.fan_net, self.fan_net.opt)
            if args.tune_emo_branch:
                freeze_params(self.fan_net.model)
            else:
                freeze_params(self.fan_net)
            # import pdb; pdb.set_trace()
            # self.v_mouth2hidden = nn.Linear(512, args.feature_dim)
            self.coeff2style = nn.Linear(args.vertice_dim, args.feature_dim)

        if args.w_vert_emo == 1:
            from third_party.meshtalk.models.encoders import ExpressionEncoder
            ## TODO: change the mean and stddev parameters
            self.vert_emo_net = ExpressionEncoder(30, args.vertice_dim//3, mean=None, stddev=None)
            # import pdb; pdb.set_trace()
        if args.is_concat_mode == 0:
            self.v_merge2hidden = nn.Linear(6+30+args.feature_dim, args.feature_dim)

        if args.is_emonet_pretrain == 1 or args.w_emo_cls_loss == 1:
            self.verts_visualizer = FlameVisualizer()
            if False:
                emoloss_trainable = False
                # emoloss_dual = True # also try False
                emoloss_dual = False # also try False
                normalize_features = False
                emo_feat_loss = 'mse_loss'
                emonet_model_path = '/data/yashengsun/Proj/TalkingFace/InstructedFFTalker/BlendshapeVisualizer/EMOCA/assets/EmotionRecognition/image_based_networks/ResNet50'
                self.custom_emonet = create_emo_loss(device='cpu',
                                            emoloss=emonet_model_path,
                                            trainable=emoloss_trainable,
                                            dual=emoloss_dual,
                                            normalize_features=normalize_features,
                                            emo_feat_loss=emo_feat_loss)
            
            if True:
                from third_party.pd_fgc_inference.lib.models.networks.encoder import FanEncoder, load_checkpoints
                self.fan_net = FanEncoder()
                load_checkpoints(self.fan_net, self.fan_net.opt)

            # self.custom_emonet_head = nn.Sequential(nn.Linear(2048,128), nn.ReLU(),
            self.custom_emonet_head = nn.Sequential(nn.Linear(512,128), nn.ReLU(),
                                              nn.BatchNorm1d(128), nn.Linear(128,8))
            self.emo2idx = {"neutral": 0, "angry": 1, "contempt": 2, "disgusted": 3, "fear": 4, "happy": 5, "sad": 6, "surprised": 7}
            if args.w_emo_cls_loss:
                freeze_params(self.fan_net)
                freeze_params(self.custom_emonet_head)

        if (args.w_render_loss == 1 or args.w_emo_loss == 1):
            file_path = os.path.join(root, 'third_party', 'pirender', 'config', 'flame_wo_crop.yaml')
            pirender_cfg = Config(filename=file_path, is_train=False)
            self.pirender = FaceGenerator(**pirender_cfg.gen.param)
            # ckpt_path = os.path.join(root, 'third_party', 'pirender', 'checkpoints/epoch_16000_iteration_000096000_checkpoint.pt')
            # ckpt_path = os.path.join(root, 'third_party', 'pirender', 'checkpoints/epoch_20000_iteration_000120000_checkpoint.pt')
            ckpt_dict = torch.load(args.stg2_ckpt_path, map_location='cpu')
            new_state_dict = {k.replace('module.', ''): v for k, v in ckpt_dict['net_G'].items()}
            # import pdb; pdb.set_trace()
            self.pirender.load_state_dict(new_state_dict)
            self.pirender.eval()
            for param in self.pirender.parameters():
                param.requires_grad = False

            print('model {} loaded.'.format(args.stg2_ckpt_path))

            if (not args.isTest):
                self._init_perceptual_loss(pirender_cfg)

        if args.w_emo_loss == 1:
            emoloss_trainable = False
            # emoloss_dual = True # also try False
            emoloss_dual = False # also try False
            normalize_features = False
            emo_feat_loss = 'mse_loss'
            emonet_model_path = '/data/yashengsun/Proj/TalkingFace/InstructedFFTalker/BlendshapeVisualizer/EMOCA/assets/EmotionRecognition/image_based_networks/ResNet50'
            self.emonet_loss = create_emo_loss(device='cpu',
                                           emoloss=emonet_model_path,
                                           trainable=emoloss_trainable,
                                           dual=emoloss_dual,
                                           normalize_features=normalize_features,
                                           emo_feat_loss=emo_feat_loss)

            self.emo_cls_head = nn.Sequential(nn.Linear(2048,128), nn.ReLU(),
                                              nn.BatchNorm1d(128), nn.Linear(128,8))
            # import pdb; pdb.set_trace()
        if args.load_mld == 1:
            from third_party.motion_latent_diffusion.mld.models.modeltype.mld import MLD
            from third_party.motion_latent_diffusion.mld.config import get_module_config
            cfg_base = OmegaConf.load('./config/mld/base.yaml')
            cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load('config/mld/action_celebvtext.yaml'))
            cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
            cfg_assets = OmegaConf.load('config/mld/assets.yaml')
            cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

            self.mld_model = MLD(cfg, None)
            state_dict = torch.load(args.mld_ckpt_path, map_location="cpu")["state_dict"]
            self.mld_model.load_state_dict(state_dict, strict=True)
            # import pdb; pdb.set_trace()

        if not args.isTest:
            tensorboard_dir = os.path.join(args.dataset, args.save_path)
            os.makedirs(tensorboard_dir, exist_ok=True)
            set_summary_writer(tensorboard_dir)

        self.learnable_eye_embed = nn.Parameter(torch.zeros(size=(1,1,6)))
        self.learnable_emo_embed = nn.Parameter(torch.zeros(size=(1,1,30)))
        self.learnable_pose_embed = nn.Parameter(torch.zeros(size=(1,1,6)))

        self.meters = {}
        self.losses_dict = {}

        self.flame_selector = FLAMESelector()
        self.eye_mask = torch.from_numpy(self.flame_selector.eye_vertices_unfold).unsqueeze(0).unsqueeze(0).float()
        self.mouth_mask = torch.from_numpy(self.flame_selector.mouth_vertices_unfold).unsqueeze(0).unsqueeze(0).float()
        # import pdb; pdb.set_trace();

    def _write_loss_meters(self):
        r"""Write all loss values to tensorboard."""
        for loss_name, loss in self.losses_dict.items():
            full_loss_name = 'gen_update' + '/' + loss_name
            if full_loss_name not in self.meters.keys():
                # Create a new meter if it doesn't exist.
                self.meters[full_loss_name] = Meter(full_loss_name)
            self.meters[full_loss_name].write(loss.item())

    def _flush_meters(self, meters, current_iteration):
        r"""Flush all meters using the current iteration."""
        for meter in meters.values():
            meter.flush(current_iteration)

    def _init_perceptual_loss(self, opt):
        self.perceptual_criteria = {}
        self.perceptual_weights = {}
        self._assign_perceptual_criteria(
            'perceptual_warp',
            PerceptualLoss(
                network=opt.trainer.vgg_param_warp.network,
                layers=opt.trainer.vgg_param_warp.layers,
                num_scales=getattr(opt.trainer.vgg_param_warp, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_warp, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_warp, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_warp)

        self._assign_perceptual_criteria(
            'perceptual_final',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final)

        # import pdb; pdb.set_trace()

    def _assign_perceptual_criteria(self, name, criterion, weight):
        self.perceptual_criteria[name] = criterion
        self.perceptual_weights[name] = weight

    def deca_emo_loss(self, pred_image, target_image, use_feat_1=False, use_feat_2=True, use_valence=True, use_arousal=True, use_expression=False):
        input_emotion = self.emonet_loss(pred_image)
        # import pdb; pdb.set_trace()
        target_emotion = self.emonet_loss(target_image)
        emo_feat_loss_2 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat_2'],
                                                         target_emotion['emo_feat_2'])
        valence_loss = self.emonet_loss.valence_loss(input_emotion['valence'], target_emotion['valence'])
        arousal_loss = self.emonet_loss.arousal_loss(input_emotion['arousal'], target_emotion['arousal'])

        total_loss = torch.zeros_like(emo_feat_loss_2)
        if use_feat_1:
            emo_feat_loss_1 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat'],
                                                             target_emotion['emo_feat'])
            total_loss = total_loss + emo_feat_loss_1

        if use_feat_2:
            total_loss = total_loss + emo_feat_loss_2

        if use_valence:
            total_loss = total_loss + valence_loss

        if use_arousal:
            total_loss = total_loss + arousal_loss

        return total_loss


    def forward(self, audio, coeff, pose, shape, cam=None, motion_des=None, img=None, ref_img=None, criterion=None,
                file_name=None, text_desc=None, teacher_forcing=True):
        if self.args.is_emo2emo:
            return self.forward_emo2emo(audio, coeff, pose, shape, cam, motion_des, img, ref_img, criterion,
                file_name, text_desc, teacher_forcing)
        elif self.args.is_emonet_pretrain:
            return self.forward_emonet(coeff, pose, shape, cam, motion_des, img, ref_img, file_name)
        elif self.args.is_only_emo:
            return self.forward_switch_frame(audio, coeff, pose, shape, cam, motion_des, img, ref_img, criterion,
                file_name, text_desc, teacher_forcing)
        else:
            return self.forward_switch_batch(audio, coeff, pose, shape, cam, motion_des, img, ref_img, criterion,
                file_name, text_desc, teacher_forcing)

    def convert_coeff2verts(self, gt_coeff, gt_pose, gt_shape): # TODO: Logically, gt_pose is not supposed to be here.
        # import pdb; pdb.set_trace();
        self.coeff_mean, self.coeff_std = self.coeff_mean.to(gt_coeff), self.coeff_std.to(gt_coeff)
        gt_coeff_unnorm = gt_coeff*self.coeff_std[0,:,:gt_coeff.shape[-1]] + self.coeff_mean[0,:,:gt_coeff.shape[-1]]
        ## set pose to [0,0,0]
        gt_pose[...,:3] = 0.0
        # import pdb; pdb.set_trace();
        gt_verts, _, _, _ = self.flame(shape_params=gt_shape, expression_params=gt_coeff_unnorm[:,:50], pose_params=gt_pose)
        return gt_verts

    def forward_ff(self, gt_verts, hidden_states, obj_embedding, frame_num, teacher_forcing):
        if self.args.is_concat_mode == 0:
            hidden_states_mix = self.v_merge2hidden(hidden_states)
        else:
            hidden_states_mix = hidden_states
        vertice_outs = []
        for j in range(len(hidden_states_mix)):
            hidden_states = hidden_states_mix[j:j+1]
            if teacher_forcing:
                vertice = gt_verts[j:j+1]
                vertice_emb = obj_embedding.unsqueeze(1)[j:j+1] # (1,1,feature_dim)
                style_emb = vertice_emb

                vertice_input = torch.cat([self.template, vertice[:, :-1]], 1)
                vertice_input = (vertice_input - self.template)
                vertice_input = self.vertice_map(vertice_input)
                vertice_input = vertice_input + style_emb #+ ref_style_embed
                vertice_input = self.PPE(vertice_input)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(
                    device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                vertice_outs.append(vertice_out)
            else:
                # import pdb; pdb.set_trace();
                for i in range(frame_num):
                    # print('i: ', i)
                    if i==0:
                        vertice_emb = obj_embedding.unsqueeze(1)[j:j+1] # (1,1,feature_dim)
                        style_emb = vertice_emb
                        vertice_input = self.PPE(style_emb)
                    else:
                        vertice_input = self.PPE(vertice_emb)
                    tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                    memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                    # print(vertice_input.shape, hidden_states.shape, tgt_mask.shape, memory_mask.shape)
                    vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                    vertice_out = self.vertice_map_r(vertice_out)
                    new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                    new_output = new_output + style_emb
                    vertice_emb = torch.cat((vertice_emb, new_output), 1)

                vertice_outs.append(vertice_out)

        vertice_outs = torch.concat(vertice_outs)
        vertice_outs = vertice_outs + self.template
        return vertice_outs

    def forward_emonet(self, coeff_b, pose, shape, cam, motion_des, img, ref_img, file_name):
        emo_label = torch.tensor([self.emo2idx[fn.split('_')[2]] for fn in file_name])
        emo_label = emo_label.to(coeff_b.device)

        self.template = self.template.to(coeff_b)
        
        frame_num = coeff_b.shape[1]
        coeffs = coeff_b
        gt_coeffs, gt_poses, gt_shapes = coeffs[:,:,:53].reshape(-1, 53), pose.reshape(-1, pose.shape[-1]), shape.reshape(-1, shape.shape[-1])
        gt_shapes_zeros = torch.zeros_like(gt_shapes)
        gt_verts_unfold = self.convert_coeff2verts(gt_coeffs, gt_poses, gt_shapes_zeros) * self.vertice_scale
        gt_verts = gt_verts_unfold.reshape(-1, frame_num, self.args.vertice_dim)
        
        with torch.no_grad():
            normal_images_all = self.verts_visualizer.render_verts(gt_verts)
            noraml_images_all_resize = F.interpolate(normal_images_all, (224,224))
            pred_emotion_feat = self.fan_net.model(noraml_images_all_resize)
 
        pred_emotion_feat = pred_emotion_feat.detach().requires_grad_(True)
        pred_emotion = self.custom_emonet_head(pred_emotion_feat)
        cls_loss = self.emo_cls_loss(pred_emotion, emo_label, frame_num)
        # import pdb; pdb.set_trace();
        loss_dict = {'emo_cls': cls_loss}
        res_loss = sum(loss_dict.values())
        
        self.losses_dict.update(loss_dict)
        self._write_loss_meters()

        return res_loss

    def emo_cls_loss(self, pred_emotion, emo_label, frame_num):
        emo_label_n_samples = emo_label.unsqueeze(-1).expand(emo_label.shape[0], frame_num).reshape(-1)
        cls_loss = nn.CrossEntropyLoss()(pred_emotion, emo_label_n_samples)
        return cls_loss

    def forward_emo2emo(self, audio, coeff_b, pose, shape, cam, motion_des, img, ref_img, criterion,
                file_name, text_desc, teacher_forcing):
        
        self.eye_mask, self.mouth_mask  = self.eye_mask.to(self.device), self.mouth_mask.to(self.device)

        teacher_forcing = False if self.args.w_cross_modal_disen else teacher_forcing
        self.template = self.template.to(coeff_b)

        teacher_forcing = False if self.args.w_cross_modal_disen else teacher_forcing
        self.template = self.template.to(coeff_b)
        
        one_hot = torch.zeros(size=(audio.shape[0], len(self.args.train_subjects.split()))).to(audio.device)
        one_hot[:,0] = 1
        obj_embedding = self.obj_vector(one_hot) #(1, feature_dim)

        frame_num = coeff_b.shape[1]
        hidden_states_a = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state 
        hidden_states_a = self.audio_feature_map(hidden_states_a)

        # import pdb;pdb.set_trace()
        coeffs = coeff_b
        gt_coeffs, gt_poses, gt_shapes = coeffs[:,:,:53].reshape(-1, 53), pose.reshape(-1, pose.shape[-1]), shape.reshape(-1, shape.shape[-1])
        gt_shapes_zeros = torch.zeros_like(gt_shapes)
        gt_verts_unfold = self.convert_coeff2verts(gt_coeffs, gt_poses, gt_shapes_zeros) * self.vertice_scale
        gt_verts = gt_verts_unfold.reshape(-1, frame_num, self.args.vertice_dim)

        with torch.no_grad():
            if self.args.w_vert_emo == 0:
                eye_embed, emo_embed, head_embed = [], [], []
                for i in range(len(img)):
                    head_embed_i, eye_embed_i, emo_embed_i, _ = self.fan_net(img[i])
                    eye_embed.append(eye_embed_i)
                    emo_embed.append(emo_embed_i)
                    head_embed.append(head_embed_i)

                emo_wolip_embed = []
                emo_wolip_img = []
                for b in range(len(img)):
                    emo_wolip_img_b = []
                    for i in range(len(img[b])):
                        if self.args.use_cross_frame_emotion:
                            offset = np.random.randint(4,8)
                            j = (i+offset) if i+offset < len(img[b]) else i-offset
                        else:
                            # import torchvision
                            # import pdb; pdb.set_trace();
                            j = i
                        mask_img_i = mask_lip(img[b,j:j+1])
                        emo_wolip_img_b.append(mask_img_i)

                    emo_wolip_img_b = torch.cat(emo_wolip_img_b)
                    _, _, emo_embed_b, _ = self.fan_net(emo_wolip_img_b)
                    emo_wolip_embed.append(emo_embed_b)
                    emo_wolip_img.append(emo_wolip_img_b)

                eye_embed = torch.stack(eye_embed, dim=0)
                emo_embed = torch.stack(emo_embed, dim=0)
                head_embed = torch.stack(head_embed, dim=0)
                emo_wolip_embed = torch.stack(emo_wolip_embed, dim=0)
                emo_wolip_img = torch.stack(emo_wolip_img, dim=0)
            else:
                # import pdb; pdb.set_trace();
                gt_verts_wo_lip = gt_verts
                emo_wolip_embed = self.vert_emo_net(gt_verts_wo_lip.reshape(gt_verts_wo_lip.shape[0],
                                                                            gt_verts_wo_lip.shape[1],
                                                                            -1,
                                                                            3))['code']

        emo_wolip_embed = emo_wolip_embed.detach().requires_grad_(True)

        bs,t = hidden_states_a.shape[0], hidden_states_a.shape[1]
        hidden_states = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), emo_wolip_embed, hidden_states_a], dim=-1)
        vertice_outs = self.forward_ff(gt_verts, hidden_states, obj_embedding, frame_num, teacher_forcing)

        # print(vertice_outs.device, self.eye_mask.device, gt_verts.device)
        loss = criterion(vertice_outs*self.eye_mask, gt_verts*self.eye_mask).mean()
        loss_dict = {'verts': loss}

        loss = sum(loss_dict.values())

        self.losses_dict.update(loss_dict)
        self._write_loss_meters()

        vis_intermediate = True
        # vis_intermediate = False
        if vis_intermediate:
            from visualize.flame_visualization import FlameVisualizer
            from visualize.flame_visualization import save_frames2video
            visualizer = FlameVisualizer()
            intermediate_res_dir = 'intermediate_res'
            os.makedirs(intermediate_res_dir, exist_ok=True)
            # import pdb; pdb.set_trace();
            for batch_idx in range(gt_verts.shape[0]):
                driven_name = 'batch_idx_{}'.format(batch_idx)
                # import pdb; pdb.set_trace();
                vertice_outs[...,~self.eye_mask[0,0].bool()] = gt_verts[...,~self.eye_mask[0,0].bool()]
                visualizer.visualize_verts(vertice_outs[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_pred'), driven_folder=None)
                visualizer.visualize_verts(gt_verts[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_gt'), driven_folder=None)

                img_i = (255.*(img[batch_idx].permute(0,2,3,1)*0.5+0.5)).detach().cpu().numpy().astype(np.uint8)
                img_i = [cv2.cvtColor(img_i[i], cv2.COLOR_BGR2RGB) for i in range(len(img_i))]
                save_frames2video(img_i, 'intermediate_res/input_video_{}.mp4'.format(batch_idx))

            import pdb; pdb.set_trace();

        return loss

    ## we assume the whole video bears the identical emotion, thus swapping frames is a practical operation.
    def forward_switch_frame(self, audio, coeff_b, pose, shape, cam=None, motion_des=None, img=None, ref_img=None, criterion=None,
                file_name=None, text_desc=None, teacher_forcing=True, vis_intermediate=False):

        teacher_forcing = False if self.args.w_cross_modal_disen else teacher_forcing
        self.template = self.template.to(coeff_b)
        
        one_hot = torch.zeros(size=(audio.shape[0], len(self.args.train_subjects.split()))).to(audio.device)
        one_hot[:,0] = 1
        obj_embedding = self.obj_vector(one_hot) #(1, feature_dim)

        frame_num = coeff_b.shape[1]
        hidden_states_a = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state 
        hidden_states_a = self.audio_feature_map(hidden_states_a)

        # import pdb;pdb.set_trace()
        coeffs = coeff_b
        gt_coeffs, gt_poses, gt_shapes = coeffs[:,:,:53].reshape(-1, 53), pose.reshape(-1, pose.shape[-1]), shape.reshape(-1, shape.shape[-1])
        gt_shapes_zeros = torch.zeros_like(gt_shapes)
        gt_verts_unfold = self.convert_coeff2verts(gt_coeffs, gt_poses, gt_shapes_zeros) * self.vertice_scale
        gt_verts = gt_verts_unfold.reshape(-1, frame_num, self.args.vertice_dim)

        with torch.no_grad():
            if self.args.w_vert_emo == 0:
                eye_embed, emo_embed, head_embed = [], [], []
                for i in range(len(img)):
                    head_embed_i, eye_embed_i, emo_embed_i, _ = self.fan_net(img[i])
                    eye_embed.append(eye_embed_i)
                    emo_embed.append(emo_embed_i)
                    head_embed.append(head_embed_i)

                emo_wolip_embed = []
                emo_wolip_img = []
                for b in range(len(img)):
                    emo_wolip_img_b = []
                    for i in range(len(img[b])):
                        if self.args.use_cross_frame_emotion:
                            offset = np.random.randint(4,8)
                            j = (i+offset) if i+offset < len(img[b]) else i-offset
                        else:
                            # import torchvision
                            # import pdb; pdb.set_trace();
                            j = i
                        mask_img_i = mask_lip(img[b,j:j+1])
                        emo_wolip_img_b.append(mask_img_i)

                    emo_wolip_img_b = torch.cat(emo_wolip_img_b)
                    _, _, emo_embed_b, _ = self.fan_net(emo_wolip_img_b)
                    emo_wolip_embed.append(emo_embed_b)
                    emo_wolip_img.append(emo_wolip_img_b)

                eye_embed = torch.stack(eye_embed, dim=0)
                emo_embed = torch.stack(emo_embed, dim=0)
                head_embed = torch.stack(head_embed, dim=0)
                emo_wolip_embed = torch.stack(emo_wolip_embed, dim=0)
                emo_wolip_img = torch.stack(emo_wolip_img, dim=0)
            else:
                # import pdb; pdb.set_trace();
                gt_verts_wo_lip = gt_verts
                emo_wolip_embed = self.vert_emo_net(gt_verts_wo_lip.reshape(gt_verts_wo_lip.shape[0],
                                                                            gt_verts_wo_lip.shape[1],
                                                                            -1,
                                                                            3))['code']

        emo_wolip_embed = emo_wolip_embed.detach().requires_grad_(True)

        bs,t = hidden_states_a.shape[0], hidden_states_a.shape[1]
        hidden_states = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), emo_wolip_embed, hidden_states_a], dim=-1)
        vertice_outs = self.forward_ff(gt_verts, hidden_states, obj_embedding, frame_num, teacher_forcing)

        loss = criterion(vertice_outs, gt_verts) #*self.args.lip_coeff_weight  # (batch, seq_len, V*3)
        loss_coeff_dict = {'verts': loss}

        self.eye_mask, self.mouth_mask  = self.eye_mask.to(self.device), self.mouth_mask.to(self.device)

        if self.args.w_cross_modal_disen: # cross-madality disentanglement, similar to MeshTalk
            # import pdb; pdb.set_trace();
            emo_wolip_embed_shuffle = emo_wolip_embed.index_select(0, torch.randperm(emo_wolip_embed.size(0)).to(emo_wolip_embed.device))
            hidden_states_a_shuffle = hidden_states_a.index_select(0, torch.randperm(hidden_states_a.size(0)).to(hidden_states_a.device))

            hidden_states_shuffle_emo = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), emo_wolip_embed_shuffle, hidden_states_a], dim=-1)
            hidden_states_shuffle_audio = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), emo_wolip_embed, hidden_states_a_shuffle], dim=-1)
            
            vertice_outs_shuffle_emo = self.forward_ff(gt_verts, hidden_states_shuffle_emo, obj_embedding, frame_num, teacher_forcing)
            vertice_outs_shuffle_audio = self.forward_ff(gt_verts, hidden_states_shuffle_audio, obj_embedding, frame_num, teacher_forcing)
            
            # import pdb; pdb.set_trace()
            loss_shuffle_emo = criterion(vertice_outs_shuffle_audio*self.eye_mask, gt_verts*self.eye_mask).mean()
            loss_shuffle_audio = criterion(vertice_outs_shuffle_emo*self.mouth_mask, gt_verts*self.mouth_mask).mean()

            loss_coeff_dict['verts_eye_area'] = loss_shuffle_emo
            loss_coeff_dict['verts_mouth_area'] = loss_shuffle_audio

        if self.args.w_emo_cls_loss:
            emo_label = torch.tensor([self.emo2idx[fn.split('_')[2]] for fn in file_name])
            emo_label = emo_label.to(coeff_b.device)
            
            # import pdb; pdb.set_trace();
            sample_index = list(range(0,vertice_outs.shape[1],20))
            normal_images_all = self.verts_visualizer.render_verts(vertice_outs[:,sample_index,:])
            noraml_images_all_resize = F.interpolate(normal_images_all, (224,224))
            pred_emotion_feat = self.fan_net.model(noraml_images_all_resize)    
            pred_emotion = self.custom_emonet_head(pred_emotion_feat)
            loss_emo_cls = self.emo_cls_loss(pred_emotion, emo_label, len(sample_index))

            loss_coeff_dict['emo_cls'] = loss_emo_cls * 0.1

        loss = sum(loss_coeff_dict.values())

        self.losses_dict.update(loss_coeff_dict)
        self._write_loss_meters()

        # vis_intermediate = True
        vis_intermediate = False
        if vis_intermediate:
            from visualize.flame_visualization import FlameVisualizer
            from visualize.flame_visualization import save_frames2video
            visualizer = FlameVisualizer()
            intermediate_res_dir = 'intermediate_res'
            os.makedirs(intermediate_res_dir, exist_ok=True)
            # import pdb; pdb.set_trace();
            for batch_idx in range(gt_verts.shape[0]):
                driven_name = 'batch_idx_{}'.format(batch_idx)
                # vertice_outs[batch_idx][...,~self.flame_selector.frontal_vertices_unfold] = \
                #             gt_verts[batch_idx][...,~self.flame_selector.frontal_vertices_unfold]
                visualizer.visualize_verts(vertice_outs[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_pred'), driven_folder=None)
                visualizer.visualize_verts(gt_verts[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_gt'), driven_folder=None)

                img_i = (255.*(img[batch_idx].permute(0,2,3,1)*0.5+0.5)).detach().cpu().numpy().astype(np.uint8)
                img_i = [cv2.cvtColor(img_i[i], cv2.COLOR_BGR2RGB) for i in range(len(img_i))]
                save_frames2video(img_i, 'intermediate_res/input_video_{}.mp4'.format(batch_idx))

            import pdb; pdb.set_trace();
        return loss

    def save_vis_image(self, image_path):
        if hasattr(self, 'vis_dict'):
            for k,v in self.vis_dict.items():
                torchvision.utils.save_image(v, image_path+k+'.jpg')
            # import pdb; pdb.set_trace()

    @torch.no_grad()
    def predict(self, audio, head_img, eye_img, emotion_img, text=None):
        self.template = self.template.to(audio)
        
        one_hot = torch.zeros(size=(audio.shape[0], len(self.args.train_subjects.split()))).to(audio.device)
        one_hot[:,0] = 1
        obj_embedding = self.obj_vector(one_hot) #(1, feature_dim)

        hidden_states_a = self.audio_encoder(audio, self.dataset).last_hidden_state 
        hidden_states_a = self.audio_feature_map(hidden_states_a)
        frame_num = hidden_states_a.shape[1]

        head_img = loopback_frames(head_img, frame_num)
        eye_img = loopback_frames(eye_img, frame_num)
        emotion_img = loopback_frames(emotion_img, frame_num)

        eye_embed, emo_embed, head_embed = [], [], []
        for i in range(len(head_img)):
            head_embed_i, _, _, _ = self.fan_net(head_img[i:i+1])
            head_embed.append(head_embed_i)
        for i in range(len(eye_img)):
            _, eye_embed_i, _, _ = self.fan_net(eye_img[i:i + 1])
            eye_embed.append(eye_embed_i)
        for i in range(len(emotion_img)):
            emotion_img_wo_lip = mask_lip(emotion_img[i:i+1])
            _, _, emo_embed_i, _ = self.fan_net(emotion_img_wo_lip)
            emo_embed.append(emo_embed_i)
        
        eye_embed = torch.concat(eye_embed, dim=0).unsqueeze(0)
        emo_embed = torch.concat(emo_embed, dim=0).unsqueeze(0)
        head_embed = torch.concat(head_embed, dim=0).unsqueeze(0)

        bs,t = hidden_states_a.shape[0], hidden_states_a.shape[1]
        ## if using text-based emotion, replace emo_embed with diffusion prior
        if self.args.load_mld:
            input_dict = {'length': [emo_embed.shape[1],], 'motion': emo_embed, 'text': [text]}
            # print(emo_embed.shape)
            emo_embed_np = self.mld_model.generate_long_expressions(input_dict)["m_rst"]
            emo_embed = torch.from_numpy(emo_embed_np)[:,:hidden_states_a.shape[1]].to(hidden_states_a.device)
            hidden_states = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), emo_embed, hidden_states_a], dim=-1)
        else:
            hidden_states = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), emo_embed, hidden_states_a], dim=-1)
        
        vertice_out = self.forward_ff(gt_verts=None, hidden_states=hidden_states, \
                          obj_embedding=obj_embedding, frame_num=frame_num, teacher_forcing=False)
        return vertice_out