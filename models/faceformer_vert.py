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
    h_ratio = [100./224., 210./224.]
    w_ratio = [ 40./224., 185./224.]
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
        obj_path = 'BlendshapeVisualizer/EMOCA/assets/FLAME/geometry/head_template.obj'
        mesh_obj = Mesh(obj_path)
        self.frontal_vertices = (mesh_obj.vertices[:,2]>0.035) & (mesh_obj.vertices[:,1]>1.4)
        self.mouth_vertices = (mesh_obj.vertices[:,2]>0.035) & (mesh_obj.vertices[:,1]>1.4) & (mesh_obj.vertices[:,1]<1.5)

        self.frontal_vertices_unfold =  np.stack([self.frontal_vertices,]*3,axis=-1).reshape(-1)
        self.mouth_vertices_unfold = np.stack([self.mouth_vertices,]*3,axis=-1).reshape(-1)


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
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
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
        # import pdb; pdb.set_trace()

        self.fan_net = None
        if args.w_fan == 1:
            from third_party.pd_fgc_inference.lib.models.networks.encoder import FanEncoder, load_checkpoints
            self.fan_net = FanEncoder()
            load_checkpoints(self.fan_net, self.fan_net.opt)
            freeze_params(self.fan_net)
            # self.v_mouth2hidden = nn.Linear(512, args.feature_dim)
            self.coeff2style = nn.Linear(args.vertice_dim, args.feature_dim)
            self.v_merge2hidden = nn.Linear(6+6+30+args.feature_dim, args.feature_dim)

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
        if self.args.is_only_emo:
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


    ## we assume the whole video bears the identical emotion, thus swapping frames is a practical operation.
    def forward_switch_frame(self, audio, coeff_b, pose, shape, cam=None, motion_des=None, img=None, ref_img=None, criterion=None,
                file_name=None, text_desc=None, teacher_forcing=True, vis_intermediate=False):

        self.template = self.template.to(coeff_b)
        
        one_hot = torch.zeros(size=(audio.shape[0], len(self.args.train_subjects.split()))).to(audio.device)
        one_hot[:,0] = 1
        obj_embedding = self.obj_vector(one_hot) #(1, feature_dim)

        frame_num = coeff_b.shape[1]
        hidden_states_a = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state 
        hidden_states_a = self.audio_feature_map(hidden_states_a)
        # import pdb;pdb.set_trace()

        with torch.no_grad():
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
                    offset = np.random.randint(4,8)
                    j = (i+offset) if i+offset < len(img[b]) else i-offset
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

        # hidden_states = torch.cat([eye_embed, emo_embed, hidden_states_a, head_embed], dim=-1)
        # self.coeff_mean, self.coeff_std = self.coeff_mean.to(gt_coeff), self.coeff_std.to(gt_coeff)

        coeffs = coeff_b
        ## what happend here.
        gt_coeffs, gt_poses, gt_shapes = coeffs[:,:,:53].reshape(-1, 53), pose.reshape(-1, pose.shape[-1]), shape.reshape(-1, shape.shape[-1])
        gt_shapes_zeros = torch.zeros_like(gt_shapes)
        gt_verts_unfold = self.convert_coeff2verts(gt_coeffs, gt_poses, gt_shapes_zeros) * self.vertice_scale
        # gt_verts_unfold = self.convert_coeff2verts(gt_coeffs, gt_poses, gt_shapes) * self.vertice_scale
        gt_verts = gt_verts_unfold.reshape(-1, frame_num, self.args.vertice_dim)

        # random_indices = torch.randint(frame_num, size=(1,)).to(coeff_b.device)
        # ref_gt_verts = torch.index_select(gt_verts, 1, random_indices)
        # ref_style_embeds = self.coeff2style(ref_gt_verts)
        # ref_style_embeds = ref_style_embeds.repeat(1,frame_num,1)
        # print(coeff_b.shape, ref_style_embed.shape)
        # import pdb; pdb.set_trace();
        
        # eye_embed_zeros = torch.zeros_like(eye_embed)
        # pose_embed_zeros = torch.zeros_like(head_embed)
        # emo_embed_zeros = torch.zeros_like(emo_embed)

        # hidden_states = torch.cat([eye_embed_zeros, emo_embed_zeros, hidden_states_a, pose_embed_zeros], dim=-1)
        # hidden_states = torch.cat([eye_embed_zeros, emo_embed, hidden_states_a, pose_embed_zeros], dim=-1)
        bs,t = hidden_states_a.shape[0], hidden_states_a.shape[1]
        # hidden_states = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), 
        #                             self.learnable_emo_embed.expand(bs,t,-1), 
        #                             hidden_states_a, 
        #                             self.learnable_pose_embed.expand(bs,t,-1)], dim=-1)
        # hidden_states_mix = self.v_merge2hidden(hidden_states)
        
        hidden_states_mix = hidden_states_a
        # import pdb; pdb.set_trace();
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
        # import pdb; pdb.set_trace();
        # loss = criterion(vertice_outs[:,:,self.flame_selector.frontal_vertices_unfold], 
        #                      gt_verts[:,:,self.flame_selector.frontal_vertices_unfold]) #*self.args.lip_coeff_weight  # (batch, seq_len, V*3)
        
        loss = criterion(vertice_outs, gt_verts) #*self.args.lip_coeff_weight  # (batch, seq_len, V*3)
        loss = torch.mean(loss) * 10.
        loss_coeff_dict = {'verts': loss}
        self.losses_dict.update(loss_coeff_dict)

        # if self.args.w_render_loss or self.args.w_emo_loss:
        #     poses, shapes, cams, motion_dess, imgs, ref_imgs, emo_wolip_imgs = \
        #         pose, shape, cam, motion_des, img, ref_img, emo_wolip_img
        #     select_input_image, select_ref_image, select_emo_image, output = self.render2image(vertice_outs, coeffs, poses, shapes, cams,
        #                                                                      motion_dess, imgs, ref_imgs, emo_wolip_imgs)

        # if self.args.w_render_loss:
        #     loss_render, loss_render_dict = self.compute_render_loss(output, select_input_image, select_ref_image, select_emo_image)
        #     self.losses_dict.update(loss_render_dict)
        #     loss = loss_render * 0.015 + loss

        # if self.args.w_lip_ldmk_loss:
        #     poses, shapes, cams = pose, shape, cam
        #     loss_lip_ldmk, lip_ldmk_dict = self.compute_ldmk_loss(vertice_outs[:,:,:53],
        #                                                           coeffs[:,:,:53],
        #                                                           poses, shapes, gt_cam=cams,
        #                                                           lmk_weight=0, lipd_weight=1,
        #                                                           eyed_weight=0)
        #     self.losses_dict.update(lip_ldmk_dict)
        #     loss = torch.mean(loss_lip_ldmk) * 10. + loss

        # if self.args.w_emo_loss:
        #     emo_label = torch.tensor([self.emo2idx[fn.split('_')[2]] for fn in file_name])
        #     emo_label = emo_label.to(audio.device)

        #     batch_size = pose.shape[0]
        #     loss_emo, loss_emo_dict = self.compute_emo_loss(gt_image=select_input_image,
        #                                                     emo_labels=emo_label, output_dict=output,
        #                                                     batch_size=batch_size)
        #     # print('emo dict: ', loss_emo_dict)
        #     self.losses_dict.update(loss_emo_dict)
        #     loss = torch.mean(loss_emo) * 0.15 + loss

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
                # vertice_outs[batch_idx][...,~self.flame_selector.frontal_vertices_unfold] = \
                #             gt_verts[batch_idx][...,~self.flame_selector.frontal_vertices_unfold]
                visualizer.visualize_verts(vertice_outs[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_pred'), driven_folder=None)
                visualizer.visualize_verts(gt_verts[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_gt'), driven_folder=None)

                img_i = (255.*(img[batch_idx].permute(0,2,3,1)*0.5+0.5)).detach().cpu().numpy().astype(np.uint8)
                img_i = [cv2.cvtColor(img_i[i], cv2.COLOR_BGR2RGB) for i in range(len(img_i))]
                save_frames2video(img_i, 'intermediate_res/input_video_{}.mp4'.format(batch_idx))

            import pdb; pdb.set_trace();
        return loss

    def compute_emo_loss(self, gt_image, emo_labels, output_dict, batch_size=8):
        loss_dict = {}
        pi_key = 'fake_image'
        pred_image = output_dict[pi_key]
        loss_emo = self.deca_emo_loss(pred_image*0.5+0.5, gt_image*0.5+0.5)
        # import torchvision
        # torchvision.utils.save_image(pred_image*0.5+0.5, 'pred_image.jpg')
        # torchvision.utils.save_image(gt_image*0.5+0.5, 'gt_image.jpg')
        # import pdb; pdb.set_trace()
        ## when they are the same, no need to deal
        # loss_ecls = self.emo_class_loss(pred_image*0.5+0.5, emo_labels, batch_size=batch_size)
        loss_dict['emo'] = loss_emo
        # loss_dict['ecls'] = loss_ecls * 0.5
        loss_emo = sum(loss_dict.values())
        return loss_emo, loss_dict

    def emo_class_loss(self, emo_img, emo_label, batch_size=8):
        emo_feat_2 = self.emonet_loss(emo_img)['emo_feat_2']
        pred_label = self.emo_cls_head(emo_feat_2.squeeze(-1).squeeze(-1))

        # exapand along temporal dimension
        num_samples = pred_label.shape[0] // batch_size
        emo_label_n_samples = emo_label.unsqueeze(-1).expand(batch_size, num_samples).reshape(-1)
        emo_cls_loss = nn.CrossEntropyLoss()(pred_label,emo_label_n_samples)
        # import pdb; pdb.set_trace()
        return emo_cls_loss

    def save_vis_image(self, image_path):
        if hasattr(self, 'vis_dict'):
            for k,v in self.vis_dict.items():
                torchvision.utils.save_image(v, image_path+k+'.jpg')
            # import pdb; pdb.set_trace()

    def render2image(self, pred_coeff, gt_coeff, gt_pose, gt_shape, gt_cam, gt_motion_des, input_image, ref_image,
                     emo_wolip_img, n_samples=4):
        self.coeff_mean, self.coeff_std = self.coeff_mean.to(pred_coeff), self.coeff_std.to(pred_coeff)
        batch_size = pred_coeff.shape[0]
        pred_coeff, gt_coeff, gt_pose, gt_shape, gt_cam, gt_motion_des  = \
            pred_coeff.view(-1, pred_coeff.shape[-1]), gt_coeff.view(-1, gt_coeff.shape[-1]), \
            gt_pose.view(-1, gt_pose.shape[-1]), gt_shape.view(-1, gt_shape.shape[-1]), \
            gt_cam.view(-1, gt_cam.shape[-1]), gt_motion_des.view(-1, gt_motion_des.shape[-1]), \

        pred_coeff_unnorm = pred_coeff * self.coeff_std[0] + self.coeff_mean[0]
        # import pdb; pdb.set_trace()
        pred_coeff_unnorm = pred_coeff_unnorm[:, :53]
        pred_posecode = torch.concat([gt_pose[:, :3], pred_coeff_unnorm[:, -3:]], dim=-1)
        pred_expcode = pred_coeff_unnorm[:, :-3]
        pred_motion_des = torch.concat([pred_expcode, pred_posecode, gt_cam], -1)
        pred_motion_des = pred_motion_des.view(batch_size, -1, *pred_motion_des.shape[1:])

        select_frame_idxs = random.choices(list(range(pred_motion_des.shape[1])), k=n_samples)
        select_pred_motion_des, select_input_image, select_ref_image, select_emo_image = [], [], [], []
        for select_frame_idx in select_frame_idxs:
            seg_select_idxs = self.obtain_seq_index(select_frame_idx, pred_motion_des.shape[1])
            select_pred_motion_des_i = pred_motion_des[:, seg_select_idxs, :]
            select_pred_motion_des.append(select_pred_motion_des_i)
            select_input_image.append(input_image[:, select_frame_idx])
            select_ref_image.append(ref_image[:, select_frame_idx])
            select_emo_image.append(emo_wolip_img[:, select_frame_idx])
            # self.pirender(input_image, )
        select_pred_motion_des = torch.stack(select_pred_motion_des, dim=1)
        select_input_image = torch.stack(select_input_image, dim=1)
        select_ref_image = torch.stack(select_ref_image, dim=1)
        select_emo_image = torch.stack(select_emo_image, dim=1)
        select_pred_motion_des = select_pred_motion_des.view(-1, *select_pred_motion_des.shape[2:])

        select_input_image = select_input_image.view(-1, *select_input_image.shape[2:])
        select_emo_image = select_emo_image.view(-1, *select_emo_image.shape[2:])
        select_pred_motion_des = select_pred_motion_des.permute(0, 2, 1)
        select_ref_image = select_ref_image.reshape(-1,*select_ref_image.shape[2:])
        output = self.pirender(select_ref_image, select_pred_motion_des)

        return select_input_image, select_ref_image, select_emo_image, output

    def obtain_seq_index(self, index, num_frames):
        self.semantic_radius = 13
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq

    # TODO: gt_shape is useless
    def compute_render_loss(self, output, select_input_image, select_ref_image, select_emo_image):
        loss_dict = {}

        # import torchvision
        # torchvision.utils.save_image(torch.cat([select_ref_image, output['warp_image'], select_input_image], dim=-2)*0.5+0.5, 'cat.jpg')
        # import pdb; pdb.set_trace()
        self.vis_dict = {}
        self.vis_dict['gt_image'] = select_input_image*0.5 + 0.5
        self.vis_dict['emo_image'] = select_emo_image*0.5 + 0.5
        self.vis_dict['fake_image'] = output['fake_image']*0.5 + 0.5
        self.vis_dict['warp_image'] = output['warp_image']*0.5 + 0.5
        self.vis_dict['ref_image'] = select_ref_image*0.5 + 0.5

        upper_face_mask = torch.ones_like(select_input_image)
        half_h = upper_face_mask.shape[-2]//2
        upper_face_mask[...,half_h:, :] = 0

        # import torchvision
        # torchvision.utils.save_image(torch.cat([select_ref_image, output['warp_image'], select_input_image], dim=-2)*0.5+0.5, 'cat.jpg')
        # torchvision.utils.save_image(upper_face_mask, 'upper_face_mask.jpg')
        # import pdb; pdb.set_trace()

        # loss_dict['render_l1'] = torch.nn.L1Loss()(output['fake_image'], select_input_image)
        ## Note that we only compute the upper face
        loss_dict['render_per_warp'] = self.perceptual_criteria['perceptual_warp'](output['warp_image']*upper_face_mask, select_input_image*upper_face_mask) * self.perceptual_weights['perceptual_warp']
        loss_dict['render_per_final'] = self.perceptual_criteria['perceptual_final'](output['fake_image']*upper_face_mask, select_input_image*upper_face_mask) * self.perceptual_weights['perceptual_final']

        render_loss = sum(loss_dict.values())

        # import pdb; pdb.set_trace()
        return render_loss, loss_dict

    def compute_ldmk_loss(self, pred_coeff, gt_coeff, gt_pose, gt_shape, gt_cam=None, lmk_weight=1, lipd_weight=1, eyed_weight=1):
        self.coeff_mean, self.coeff_std = self.coeff_mean.to(pred_coeff), self.coeff_std.to(pred_coeff)
        pred_coeff, gt_coeff, gt_pose, gt_shape, gt_cam = \
            pred_coeff.reshape(-1,pred_coeff.shape[-1]), gt_coeff.reshape(-1,gt_coeff.shape[-1]), \
            gt_pose.reshape(-1,gt_pose.shape[-1]), gt_shape.reshape(-1,gt_shape.shape[-1]), \
            gt_cam.reshape(-1,gt_cam.shape[-1])

        coeff_unnorm = gt_coeff*self.coeff_std[0,:,:gt_coeff.shape[-1]] + self.coeff_mean[0,:,:gt_coeff.shape[-1]]
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            shapecode = gt_shape
            gt_expcode = coeff_unnorm[:,:-3]
            gt_posecode = gt_pose
            _, landmarks2d_gt, landmarks3d_gt, landmarks2d_mediapipe_gt = self.flame(shape_params=shapecode,
                                                                                     expression_params=gt_expcode,
                                                                                     pose_params=gt_posecode)
            lmk = util.batch_orth_proj(landmarks2d_gt, gt_cam)[:, :, :2]
            lmk[:, :, 1:] = - lmk[:, :, 1:]
            ## TODO: Here I do not leverage the image landmarks. Instead, I utilize .

        vertice_out_unnorm = pred_coeff*self.coeff_std[0,:,:pred_coeff.shape[-1]] + self.coeff_mean[0,:,:pred_coeff.shape[-1]]

        pred_posecode = torch.concat([gt_pose[:,:3], vertice_out_unnorm[:,-3:]], dim=-1)
        pred_expcode = vertice_out_unnorm[:,:-3]
        _, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.flame(shape_params=shapecode, expression_params=pred_expcode, pose_params=pred_posecode)
        # ldmk_loss = criterion(landmarks2d_mediapipe, landmarks2d_mediapipe_gt.detach())*1e6 + criterion(landmarks3d, landmarks3d_gt.detach())*1e6

        # lmk_weight, lipd_weight, eyed_weight = 0.01, 0.05, 0.02 # 1 and 5 is too large
        # lmk_weight, lipd_weight, eyed_weight = 1, 1, 1 # 1 and 5 is too large
        geom_losses_idxs = pred_coeff.shape[0]
        predicted_landmarks = util.batch_orth_proj(landmarks2d, gt_cam)[:, :, :2]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        loss_dict = {}
        suffix = ''
        if eyed_weight > 0:
            suffix += 'eye_'
        if lipd_weight > 0:
            suffix += 'lip_'

        loss_dict[suffix+'landmark'] = lossfunc.landmark_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                   lmk[:geom_losses_idxs, ...]) * lmk_weight

        # import pdb; pdb.set_trace();
        # plot_lmk(predicted_landmarks[:geom_losses_idxs, ...][0], image_path='pred.jpg')
        # plot_lmk(lmk[:geom_losses_idxs, ...][0], image_path='gt.jpg')

        loss_dict[suffix+'lip_distance'] = lossfunc.lipd_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                               lmk[:geom_losses_idxs, ...]) * lipd_weight
        loss_dict[suffix+'mouth_corner_distance'] = lossfunc.mouth_corner_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                                                lmk[:geom_losses_idxs, ...]) * lipd_weight
        loss_dict[suffix + 'eye_distance'] = lossfunc.eyed_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                                                lmk[:geom_losses_idxs, ...]) * eyed_weight
        # eye switch distance calculation
        # loss_dict[suffix+'eye_distance'] = lossfunc.eyed_loss(predicted_landmarks[:geom_losses_idxs//2, ...],
        #                                        lmk[geom_losses_idxs//2:, ...]) * eyed_weight + \
        #                             lossfunc.eyed_loss(predicted_landmarks[geom_losses_idxs//2:, ...],
        #                                                lmk[:geom_losses_idxs//2, ...]) * eyed_weight
        ldmk_loss = sum(loss_dict.values())

        # import pdb; pdb.set_trace()
        return ldmk_loss, loss_dict

    @torch.no_grad()
    def drive_by_coeff(self, pred_coeff, ref_image, driven_data, is_use_pred_pose=False):
        seq_len = pred_coeff.shape[0]

        gt_pose, gt_cam = driven_data['pose'][:seq_len], driven_data['cam'][:seq_len]
        if len(gt_pose) < seq_len:
            gt_pose = np.pad(gt_pose, pad_width=((0, seq_len-len(gt_pose)),(0,0)), mode='reflect')
            gt_cam = np.pad(gt_cam, pad_width=((0, seq_len - len(gt_cam)), (0, 0)), mode='reflect')

        gt_pose, gt_cam = torch.from_numpy(gt_pose).to(pred_coeff), torch.from_numpy(gt_cam).to(pred_coeff)
        gt_pose, gt_cam = gt_pose[:1].expand(seq_len, *gt_pose.shape[1:]), gt_cam[:1].expand(seq_len, *gt_cam.shape[1:])
        ref_image = ref_image.to(pred_coeff)

        if is_use_pred_pose is True:
            # import pdb; pdb.set_trace()
            pred_posecode = torch.concat([pred_coeff[:, -6:-3], pred_coeff[:, 50:53]], dim=-1)
            pred_expcode = pred_coeff[:, :-3]
            pred_motion_des = torch.concat([pred_expcode, pred_posecode, pred_coeff[:, -3:]], -1)
        else:
            pred_posecode = torch.concat([gt_pose[:, :3], pred_coeff[:, -3:]], dim=-1)
            pred_expcode = pred_coeff[:, :-3]
            pred_motion_des = torch.concat([pred_expcode, pred_posecode, gt_cam], -1)

        pred_motion_des_tctx = []
        select_frame_idxs = list(range(seq_len))
        # import pdb;pdb.set_trace()

        for select_frame_idx in select_frame_idxs:
            seg_select_idxs = self.obtain_seq_index(select_frame_idx, pred_motion_des.shape[0])
            select_pred_motion_des_i = pred_motion_des[seg_select_idxs,:]
            # print(select_frame_idx, seg_select_idxs)
            pred_motion_des_tctx.append(select_pred_motion_des_i)
        pred_motion_des_tctx = torch.stack(pred_motion_des_tctx)
        pred_motion_des_tctx = pred_motion_des_tctx.permute(0,2,1)

        output = self.pirender(ref_image.expand(pred_motion_des_tctx.size(0), *ref_image.shape[1:]), pred_motion_des_tctx)
        return output

    @torch.no_grad()
    def predict(self, audio, head_img, eye_img, emotion_img, text=None):
        self.template = self.template.to(audio)

        hidden_states_a = self.audio_encoder(audio, self.dataset).last_hidden_state
        hidden_states_a = self.audio_feature_map(hidden_states_a)
        frame_num = hidden_states_a.shape[1]
        # import pdb; pdb.set_trace()

        head_img = loopback_frames(head_img, frame_num)
        eye_img = loopback_frames(eye_img, frame_num)
        emotion_img = loopback_frames(emotion_img, frame_num)
        # import pdb; pdb.set_trace()
        eye_embed, emo_embed, head_embed = [], [], []
        for i in range(len(head_img)):
            head_embed_i, _, _, _ = self.fan_net(head_img[i:i+1])
            head_embed.append(head_embed_i)
        for i in range(len(eye_img)):
            _, eye_embed_i, _, _ = self.fan_net(eye_img[i:i + 1])
            eye_embed.append(eye_embed_i)
        for i in range(len(emotion_img)):
            emotion_img_wo_lip = mask_lip(emotion_img[i:i+1])
            # import torchvision
            # import pdb; pdb.set_trace()
            _, _, emo_embed_i, _ = self.fan_net(emotion_img_wo_lip)
            emo_embed.append(emo_embed_i)

        eye_embed = torch.concat(eye_embed, dim=0).unsqueeze(0)
        emo_embed = torch.concat(emo_embed, dim=0).unsqueeze(0)
        head_embed = torch.concat(head_embed, dim=0).unsqueeze(0)

        # import pdb;pdb.set_trace()
        if self.args.load_mld:
            input_dict = {'length': [200,], 'motion': emo_embed, 'text': [text]}
            # print(emo_embed.shape)
            emo_embed_np = self.mld_model.generate_long_expressions(input_dict)["m_rst"]
            emo_embed = torch.from_numpy(emo_embed_np)[:,:head_embed.shape[1]].to(head_embed.device)
            # import pdb; pdb.set_trace()

        eye_embed_zeros = torch.zeros_like(eye_embed)
        pose_embed_zeros = torch.zeros_like(head_embed)
        emo_embed_zeros = torch.zeros_like(emo_embed)

        bs,t = hidden_states_a.shape[0], hidden_states_a.shape[1]
        # hidden_states = torch.cat([self.learnable_eye_embed.expand(bs,t,-1), 
        #                     self.learnable_emo_embed.expand(bs,t,-1), 
        #                     hidden_states_a, 
        #                     self.learnable_pose_embed.expand(bs,t,-1)], dim=-1)
        # # hidden_states = torch.cat([eye_embed_zeros, emo_embed_zeros, hidden_states_a, pose_embed_zeros], dim=-1)
        # hidden_states = self.v_merge2hidden(hidden_states)
        hidden_states = hidden_states_a
        # ref_style_emb = self.coeff2style(self.template) ## should use gt-injection, we have no idea it is id or emo
        # ref_style_embeds = ref_style_embeds.repeat(1,frame_num,1)

        one_hot = torch.ones(size=(audio.shape[0],1)).to(audio.device)
        obj_embedding = self.obj_vector(one_hot) #(1, feature_dim)

        for i in range(frame_num):
            # print('i: ', i)
            if i == 0:
                # vertice_emb = self.vertice_map(torch.zeros_like(self.template))
                # vertice_emb = vertice_emb #+ ref_style_emb
                # vertice_input = self.PPE(vertice_emb)
                vertice_emb = obj_embedding.unsqueeze(1)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(
                device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            # print(vertice_input.shape, hidden_states.shape, tgt_mask.shape, memory_mask.shape)
            # import pdb;pdb.set_trace();
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask,
                                                   memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb#+ ref_style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + self.template
        return vertice_out*1.0/self.vertice_scale


    # def compute_voca_loss(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True):
    #     self.template = self.template.to(vertice)
    #     # import pdb; pdb.set_trace()

    #     frame_num = vertice.shape[1]
    #     obj_embedding = self.obj_vector(one_hot) #(1, feature_dim)

    #     hidden_states_a = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state 
    #     hidden_states_a = self.audio_feature_map(hidden_states_a)
    #     bs,t = hidden_states_a.shape[0], hidden_states_a.shape[1]

    #     hidden_states_mix = hidden_states_a

    #     vertice_outs = []
    #     for j in range(len(hidden_states_mix)):
    #         hidden_states = hidden_states_mix[j:j+1]

    #         for i in range(frame_num):
    #             # import pdb; pdb.set_trace();
    #             if i==0:
    #                 vertice_emb = obj_embedding.unsqueeze(1)[j:j+1] # (1,1,feature_dim)
    #                 style_emb = vertice_emb
    #                 vertice_input = self.PPE(style_emb)
    #             else:
    #                 vertice_input = self.PPE(vertice_emb)
    #             tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
    #             memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
    #             vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
    #             vertice_out = self.vertice_map_r(vertice_out)
    #             new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
    #             new_output = new_output + style_emb
    #             vertice_emb = torch.cat((vertice_emb, new_output), 1)

    #         vertice_outs.append(vertice_out)

    #     vertice_outs = torch.concat(vertice_outs)
    #     vertice_outs = vertice_outs + self.template

    #     loss = criterion(vertice_outs, vertice) #*self.args.lip_coeff_weight  # (batch, seq_len, V*3)
    #     loss = torch.mean(loss)
    #     loss_coeff_dict = {'verts': loss}
    #     self.losses_dict.update(loss_coeff_dict)

    #     self._write_loss_meters()

    #     gt_verts = vertice
    #     vis_intermediate = True
    #     # vis_intermediate = False
    #     if vis_intermediate:
    #         from visualize.flame_visualization import FlameVisualizer
    #         from visualize.flame_visualization import save_frames2video
    #         visualizer = FlameVisualizer()
    #         intermediate_res_dir = 'intermediate_res'
    #         os.makedirs(intermediate_res_dir, exist_ok=True)
    #         # import pdb; pdb.set_trace();
    #         for batch_idx in range(gt_verts.shape[0]):
    #             driven_name = 'batch_idx_{}'.format(batch_idx)
    #             # vertice_outs[batch_idx][...,~self.flame_selector.frontal_vertices_unfold] = \
    #             #             gt_verts[batch_idx][...,~self.flame_selector.frontal_vertices_unfold]
    #             visualizer.visualize_verts(vertice_outs[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_pred'), driven_folder=None)
    #             visualizer.visualize_verts(gt_verts[batch_idx].reshape(frame_num,-1,3)/self.vertice_scale, save_root=intermediate_res_dir, save_name='{}'.format(driven_name+'_3d_gt'), driven_folder=None)

    #             # img_i = (255.*(img[batch_idx].permute(0,2,3,1)*0.5+0.5)).detach().cpu().numpy().astype(np.uint8)
    #             # img_i = [cv2.cvtColor(img_i[i], cv2.COLOR_BGR2RGB) for i in range(len(img_i))]
    #             # save_frames2video(img_i, 'intermediate_res/input_video_{}.mp4'.format(batch_idx))

    #         import pdb; pdb.set_trace();

    #     return loss