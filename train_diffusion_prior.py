import argparse
import sys
import os

proj_root = os.path.join('third_party', 'inferno')
sys.path.append(proj_root)

from tqdm import tqdm
import time
from models.diffusion_prior import InstructDiffusionPrior, VersatileDiffusionPriorNetwork, BrainNetwork, FrozenCLIPEmbedder
from inferno_apps.TalkingHead.evaluation.TalkingHeadWrapper import TalkingHeadWrapper
from inferno_apps.TalkingHead.evaluation.evaluation_functions import *
from inferno.datasets.FaceVideoDataModule import dict_to_device

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join('third_party', 'pd_fgc_inference'))
from dataset.data_loader import get_dataloaders
from pathlib import Path
import glob

sys.path.append(os.path.join('third_party', 'pirender'))
from third_party.pirender.util.meters import Meter, set_summary_writer
from talkclip_text_generation.text_gen import TalkClipDatabase, black_videotoken_list
from emoca_utils import get_data

def read_list(path):
    with open(path, 'r') as file:
        lines = file.read().splitlines()
    return lines

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class ScreenedMeadAudio:
    def __init__(self):
        all_data_root = {
            'Mead_M': '/data/yashengsun/local_storage/Mead_emoca/Mead_M',
            'Mead_W': '/data/yashengsun/local_storage/Mead_emoca/Mead_W',
        }
        good_audio_meta_path = '/data/yashengsun/Proj/TalkingFace/SECap/scripts/meta_audio.txt'
        good_audio_paths = read_list(good_audio_meta_path)

        self.talkclip_generator = TalkClipDatabase()

        both_data_root = {}
        dataset_names = ['Mead_M', 'Mead_W']
        for dataset_name in dataset_names:
            both_data_root[dataset_name] = all_data_root[dataset_name]
        
        res_data_dict = {}
        for k,data_root in both_data_root.items():
            data_dict = get_data(data_root, is_inference=False, is_wo_audio=False)
            res_data_dict.update(data_dict)

        self.wav_paths = []
        for key, value in res_data_dict.items():
            try:
                text_description = self.talkclip_generator.query(key)
                wav_path = res_data_dict[key]['wav']
            except Exception:
                print(key, value)
                continue

            if wav_path not in good_audio_paths: continue
            self.wav_paths.append(wav_path)

        self.wav_paths = sorted(self.wav_paths)
        # import pdb; pdb.set_trace()

class FpParser:
    def __init__(self,):
        self.training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                    'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                    'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                    'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                    ]
        self._emotions = {'neutral':0, 'happy':1, 'sad':2, 'surprised':3, 'fear':4, 'disgusted':5, 'angry':6, 'contempt':7, 'none':8}

    def get_emotion_idx(self, emotion_name):
        # emotion_name = e[0].upper() + e[1:].lower()
        # print(emotion_name)
        # emotion_index = AffectNetExpressions.from_str(emotion_name)
        emotion_index = self._emotions[emotion_name]
        return emotion_index

    def get_identity_idx(self, id_name):
        id_index = self.training_ids.index(id_name)
        return id_index

    def get_intensity_idx(self, intensity_name):
        intensity_index = int(intensity_name.replace('level','')) - 1
        return intensity_index

    def parse_fn(self, fn):
        id_name, _, emotion_name, intensity_name, _ = fn.split('_')
        id_index = self.get_identity_idx(id_name)
        emo_index = self.get_emotion_idx(emotion_name)
        intensity_index = self.get_intensity_idx(intensity_name)
        return id_index, emo_index, intensity_index

    @staticmethod
    def recursive_collate(batch):
        """
        Recursive collate function for handling nested dictionaries with NumPy arrays.
        """
        if isinstance(batch[0], dict):
            collated_batch = {}
            for key in batch[0].keys():
                collated_batch[key] = FpParser.recursive_collate([item[key] for item in batch])
            return collated_batch
        elif isinstance(batch[0], np.ndarray):
            return torch.from_numpy(np.stack(batch)).cuda()
        else:
            return batch
            
def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def save_ckpt(tag, outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs):
    ckpt_path = outdir+f'/{tag}.pth'
    os.makedirs(outdir, exist_ok=True)
    print(f'saving {ckpt_path}',flush=True)
    # try:
    torch.save({
        'epoch': epoch,
        'model_state_dict': diffusion_prior.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'lrs': lrs,
        }, ckpt_path)
    # except:
    #     print("Couldn't save... moving on to prevent crashing.")
    
def prepare_train_data(fp_parser, file_name, talking_head, base_sample, silent_frames_start, silent_frames_end):
    silent_emotion_start, silent_emotion_end = 0,0
    id_emo_intensity = [fp_parser.parse_fn(file_name_i) for file_name_i in file_name]
    identity_list = [kk[0] for kk in id_emo_intensity]
    emotion_index_list = [kk[1] for kk in id_emo_intensity]
    intensity_list = [kk[2] for kk in id_emo_intensity]
    # import pdb; pdb.set_trace()

    samples = create_high_intensity_emotions(talking_head, 
                                            base_sample, 
                                            identity_list=identity_list,
                                            emotion_index_list=emotion_index_list,
                                            intensity_list=intensity_list,
                                            silent_frames_start=silent_frames_start,
                                            silent_frames_end=silent_frames_end, 
                                            silent_emotion_start = silent_emotion_start,
                                            silent_emotion_end = silent_emotion_end)
    # print(samples.keys())
    samples = FpParser.recursive_collate(samples)
    # print(samples.keys())
    # import pdb; pdb.set_trace()
    # samples = dict_to_device(samples, device=torch.device('cuda'))
    with torch.no_grad():
        clip_target = talking_head(samples, only_style_emb=True)[:,:1].float()
    clip_target = clip_target.requires_grad_(True)
    return samples, clip_target
    
def prepare_test_data(fp_parser, talking_head, base_sample, silent_frames_start, silent_frames_end, batch_size=1):
    silent_emotion_start, silent_emotion_end = 0,0
    identity_list = [0 for i in range(batch_size)]
    emotion_index_list = [0 for i in range(batch_size)]
    intensity_list = [0 for i in range(batch_size)]

    samples = create_high_intensity_emotions(talking_head, 
                                            base_sample, 
                                            identity_list=identity_list,
                                            emotion_index_list=emotion_index_list,
                                            intensity_list=intensity_list,
                                            silent_frames_start=silent_frames_start,
                                            silent_frames_end=silent_frames_end, 
                                            silent_emotion_start = silent_emotion_start,
                                            silent_emotion_end = silent_emotion_end)
    samples = FpParser.recursive_collate(samples)
    # print(samples.keys())
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        clip_target = talking_head(samples, only_style_emb=True)[:,:1].float()
    clip_target = clip_target.requires_grad_(True)
    return samples, clip_target

def write_loss_meters(meters, losses_dict):
    r"""Write all loss values to tensorboard."""
    for loss_name, loss in losses_dict.items():
        full_loss_name = 'diffusion' + '/' + loss_name
        if full_loss_name not in meters.keys():
            # Create a new meter if it doesn't exist.
            meters[full_loss_name] = Meter(full_loss_name)
        # meters[full_loss_name].write(loss.item())
        meters[full_loss_name].write(loss)

def flush_meters(meters, current_iteration):
    r"""Flush all meters using the current iteration."""
    for meter in meters.values():
        meter.flush(current_iteration)

# if resume_from_ckpt:
def resume_ckpt(ckpt_path, optimizer, lr_scheduler, diffusion_prior):
    print("\n---resuming from last.pth ckpt---\n")
    # try:
    # checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # except:
    #     print('last.pth failed... trying last_backup.pth')
    #     checkpoint = torch.load(outdir+'/last_backup.pth', map_location='cpu')
    epoch = checkpoint['epoch']
    print("Epoch",epoch)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    diffusion_prior.load_state_dict(checkpoint['model_state_dict'])
    return epoch

def get_gt_data_rvd():
    from dataset.emoca_utils import get_data
    import pickle
    all_data_root = {
        'Actor_01': '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote/Actor_01',
        'Actor_02': '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote/Actor_02',
        'Actor_03': '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote/Actor_03',
        'Actor_04': '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote/Actor_04',
        'Actor_05': '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote/Actor_05',
        'Actor_06': '/data/yashengsun/local_storage/Video_Speech_Actor_fps25_emote/Actor_06',
    }
    data_root = {}
    dataset_names = 'Actor_01,Actor_02,Actor_03,Actor_04,Actor_04,Actor_05,Actor_06'
    for dataset_name in dataset_names.split(','):
        data_root[dataset_name] = all_data_root[dataset_name]

    is_inference = False
    infer_tag = 'test' if is_inference else 'train'
    data_names = '_'.join(data_root.keys())
    cached_path = 'datadict_{}_{}.pkl'.format(infer_tag, data_names)
    is_wo_audio = False

    if os.path.exists(cached_path):
        with open(cached_path, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        res_data_dict = {}
        for k, data_root in data_root.items():
            data_dict = get_data(data_root, is_inference=is_inference, is_wo_audio=is_wo_audio)
            res_data_dict.update(data_dict)
        data_dict = res_data_dict
        with open(cached_path, 'wb') as f:
            pickle.dump(data_dict, f)

    # import pdb; pdb.set_trace()

    return data_dict


def get_gt_data():
    from dataset.emoca_utils import get_data
    import pickle
    # import pdb; pdb.set_trace()
    all_data_root = {
        'paishe': '/data/yashengsun/local_storage/paishe_w_cam/proc_emoca',
        'Mead_M': '/data/yashengsun/local_storage/Mead_emoca/Mead_M',
        'Mead_W': '/data/yashengsun/local_storage/Mead_emoca/Mead_W',
        'head_dynamics': '/data/yashengsun/local_storage/instruct_data/head_dynamics'
    }
    data_root = {}
    dataset_names = 'Mead_M,Mead_W'
    for dataset_name in dataset_names.split(','):
        data_root[dataset_name] = all_data_root[dataset_name]

    is_inference = False
    infer_tag = 'test' if is_inference else 'train'
    data_names = '_'.join(data_root.keys())
    cached_path = 'datadict_{}_{}.pkl'.format(infer_tag, data_names)
    # import pdb; pdb.set_trace()
    with open(cached_path, 'rb') as f:
        res_data_dict = pickle.load(f)
    # import pdb; pdb.set_trace()
    return res_data_dict


def is_audio_in_whitelist(audio_path):
    whitelist = [
        'M024_front_sad_level3_027',
        'M024_front_sad_level2_016',
        'M024_front_happy_level3_013',
        'W009_front_happy_level1_020',
        'W009_front_happy_level2_013',
        'M023_front_fear_level3_005',
        'W009_front_fear_level1_009',
        'M023_front_angry_level3_008',
        'M023_front_angry_level3_019',
        'W018_front_surprised_level1_026',
        'W018_front_surprised_level3_015',
        'W014_front_happy_level2_015',
    ]
    for name in whitelist:
        if name in audio_path: return True
    return False


def trainer(args, train_dl, val_dl, diffusion_prior, talking_head, optimizer, distributed=False):
    lr_scheduler_type = 'cycle'
    num_train = len(train_dl)
    max_lr, num_epochs, local_rank, clip_size = args.max_lr, args.max_epoch, args.local_rank, args.clip_size

    total_steps=int(num_epochs*(num_train))*5
    if lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps,
            last_epoch=-1
        )
    elif lr_scheduler_type == 'cycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )

    outdir = os.path.abspath(f'train_logs/{args.jobname}')

    epoch = args.epoch
    if args.resume_from_ckpt:
        if os.path.exists(args.ckpt_path):
            epoch = resume_ckpt(args.ckpt_path, optimizer, lr_scheduler, diffusion_prior)
        else:
            print('{} does not exist.'.format(args.ckpt_path))
            import pdb; pdb.set_trace();

    if args.is_tensorboard_log:
        tensorboard_dir = os.path.join(outdir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        set_summary_writer(tensorboard_dir)
        losses_dict, meters, loss_dict = {}, {}, {}
        
    fp_parser = FpParser()
    # import pdb; pdb.set_trace();
    print(f"starting with epoch {epoch} / {num_epochs}")
    progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))

    silent_frames_start, silent_frames_end = 0,0
    audio_path = args.test_audio_path
    base_sample = create_base_sample(talking_head, audio_path=audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
    
    clip_text_embdder = FrozenCLIPEmbedder().to(torch.device("cuda"))
    print('clip_text_params: ', count_parameters(clip_text_embdder))
    print('diffusion_prior voxel2clip params: ', count_parameters(diffusion_prior.voxel2clip))
    print('diffusion_prior net params: ', count_parameters(diffusion_prior.net))
    print('talking head params: ', count_parameters(talking_head))
    # import pdb; pdb.set_trace();
    hidden, prior, v2c = True, True, True

    if args.unset_prior: prior = False
    if args.unset_v2c: v2c = False

    mixup_pct = 0.
    soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
    if hidden:
        prior_mult = 30
    else:
        prior_mult = .03
    losses, val_losses, lrs = [], [], []
    nce_losses, val_nce_losses = [], []
    sim_losses, val_sim_losses = [], []
    best_val_loss = 1e9

    if not args.is_test:
        save_at_end, ckpt_saving = False, True
        for epoch in progress_bar:
            diffusion_prior.train()

            sims_base = 0.
            val_sims_base = 0.
            fwd_percent_correct = 0.
            bwd_percent_correct = 0.
            val_fwd_percent_correct = 0.
            val_bwd_percent_correct = 0.
            loss_nce_sum = 0.
            loss_prior_sum = 0.
            val_loss_nce_sum = 0.
            val_loss_prior_sum = 0.

            for train_i, (_, _, _, _, _, _, _, _, file_name, text_descr) in tqdm(enumerate(train_dl)):
                train_iter = train_i + len(train_dl)*epoch
                # torch.cuda.synchronize()
                t = time.time()
                
                samples, clip_target = prepare_train_data(fp_parser, file_name, talking_head, base_sample, silent_frames_start, silent_frames_end)

                # torch.cuda.synchronize()
                data_t = time.time() -t
                t = time.time()
                
                # import pdb; pdb.set_trace();
                with torch.cuda.amp.autocast():
                    optimizer.zero_grad()

                    with torch.no_grad():
                        voxel = clip_text_embdder(text_descr)
                        voxel = torch.mean(voxel, dim=1) #### TODO, we use mean 77 tokens to obtain semantic info
                    voxel = voxel.requires_grad_(True)
                    
                    clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
                    # import pdb; pdb.set_trace()
                    
                    if hidden:
                        clip_voxels = clip_voxels.view(len(voxel),-1,clip_size)
                    
                    if prior:
                        loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                        aligned_clip_voxels /= diffusion_prior.module.image_embed_scale if distributed else diffusion_prior.image_embed_scale
                    else:
                        aligned_clip_voxels = clip_voxels

                    clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                    # import pdb; pdb.set_trace()

                    if epoch < int(mixup_pct * num_epochs):
                        loss_nce = utils.mixco_nce(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006, 
                            perm=perm, betas=betas, select=select)
                    else:
                        epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                        loss_nce = soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=epoch_temp)

                    if prior and v2c:
                        loss_nce_sum += loss_nce.item()
                        loss_prior_sum += loss_prior.item()
                        loss = loss_nce + (prior_mult * loss_prior)
                    elif v2c:
                        loss_nce_sum += loss_nce.item()
                        loss = loss_nce
                    elif prior:
                        loss_prior_sum += loss_prior.item()
                        loss = prior_mult * loss_prior
                    check_loss(loss)
                    # utils.check_loss(loss)
                    
                    # accelerator.backward(loss)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    lrs.append(optimizer.param_groups[0]['lr'])

                    sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item()

                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_target_norm)).to(torch.device("cuda")) 
                    fwd_percent_correct += topk(batchwise_cosine_similarity(clip_voxels_norm,clip_target_norm), labels, k=1)
                    bwd_percent_correct += topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)

                    if lr_scheduler_type is not None:
                        lr_scheduler.step()
                forward_t = time.time() - t

                if args.is_tensorboard_log:
                    loss_dict['train_sims_base'] = sims_base / (train_i + 1)
                    loss_dict['train_fwd_percent_correct'] = fwd_percent_correct / (train_i + 1)
                    loss_dict['train_bwd_percent_correct'] = bwd_percent_correct / (train_i + 1)
                    loss_dict['train_loss_nce'] = loss_nce_sum / (train_i + 1)
                    loss_dict['train_loss_prior'] = loss_prior_sum / (train_i + 1)
                    loss_dict['train_loss'] = np.mean(losses[-(train_i+1):])
                    losses_dict.update(loss_dict)
                    write_loss_meters(meters, losses_dict)

                    if train_iter % args.log_loss_steps == 0:
                        flush_meters(meters, train_iter)
                # print('data: ', data_t, ' forward: ', forward_t, ' loss:', loss.item())
                # import pdb; pdb.set_trace()

            diffusion_prior.eval()
            for val_i, (_, _, _, _, _, _, _, _, file_name, text_descr) in enumerate(val_dl): 
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # repeat_index = val_i % 3
                        val_iter = val_i + len(val_dl)*epoch

                        samples, clip_target = prepare_train_data(fp_parser, file_name, talking_head, base_sample, silent_frames_start, silent_frames_end)

                        with torch.no_grad():
                            voxel = clip_text_embdder(text_descr)
                            voxel = torch.mean(voxel, dim=1) #### TODO, we use mean 77 tokens to obtain semantic info
                        voxel = voxel.requires_grad_(True)

                        clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)
                        if hidden:
                            clip_voxels = clip_voxels.view(len(voxel),-1,clip_size)
                        
                        if prior:
                            val_loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                            aligned_clip_voxels /= diffusion_prior.module.image_embed_scale if distributed else diffusion_prior.image_embed_scale
                        else:
                            aligned_clip_voxels = clip_voxels

                        clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                        if epoch < int(mixup_pct * num_epochs):
                            val_loss_nce = utils.mixco_nce(
                                clip_voxels_norm,
                                clip_target_norm,
                                temp=.006, 
                                perm=None, betas=None, select=None)
                        else:
                            val_loss_nce = soft_clip_loss(
                                clip_voxels_norm,
                                clip_target_norm,
                                temp=epoch_temp)

                        if prior and v2c:
                            val_loss_nce_sum += val_loss_nce.item()
                            val_loss_prior_sum += val_loss_prior.item()
                            val_loss = val_loss_nce + (prior_mult * val_loss_prior)
                        elif v2c:
                            val_loss_nce_sum += val_loss_nce.item()
                            val_loss = val_loss_nce
                        elif prior:
                            val_loss_prior_sum += val_loss_prior.item()
                            val_loss = prior_mult * val_loss_prior
                        check_loss(val_loss)
                        
                        val_losses.append(val_loss.item())

                        # clip_voxel_gather = accelerator.gather(clip_voxels_norm.view(len(voxel),-1).contiguous())
                        # clip_target_gather = accelerator.gather(clip_target_norm.view(len(voxel),-1).contiguous())

                        val_sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item()
                        
                        labels = torch.arange(len(clip_target_norm)).to(torch.device("cuda"))
                        val_fwd_percent_correct += topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1)
                        val_bwd_percent_correct += topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)

                        if args.is_tensorboard_log:
                            loss_dict['val_sims_base'] = val_sims_base / (val_i + 1)
                            loss_dict['val_fwd_percent_correct'] = val_fwd_percent_correct / (val_i + 1)
                            loss_dict['val_bwd_percent_correct'] = val_bwd_percent_correct / (val_i + 1)
                            loss_dict['val_loss_nce'] = val_loss_nce_sum / (val_i + 1)
                            loss_dict['val_loss_prior'] = val_loss_prior_sum / (val_i + 1)
                            loss_dict['val_loss'] = np.mean(val_losses[-(val_i+1):])
                            losses_dict.update(loss_dict)
                            write_loss_meters(meters, losses_dict)

                            # if val_iter % args.log_loss_steps == 0:
                            #     flush_meters(meters, val_iter)

            if local_rank==0:        
                ckpt_saving = (train_iter % 100 == 0)
                if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
                    # save best model
                    val_loss = np.mean(val_losses[-(val_i+1):])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_ckpt('best', outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs)
                    else:
                        print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
                        
                # if utils.is_interactive():
                #     clear_output(wait=True)
                    
                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "val/loss": np.mean(val_losses[-(val_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "val/num_steps": len(val_losses),
                    "train/cosine_sim_base": sims_base / (train_i + 1),
                    "val/cosine_sim_base": val_sims_base / (val_i + 1),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                    "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                    "train/loss_nce": loss_nce_sum / (train_i + 1),
                    "train/loss_prior": loss_prior_sum / (train_i + 1),
                    "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                    "val/loss_prior": val_loss_prior_sum / (val_i + 1)}
                progress_bar.set_postfix(**logs)

                # Save model checkpoint and reconstruct
                save_ckpt(f'last', outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs)
                # if epoch % ckpt_interval == 0:
                #     save_ckpt(f'last_backup', outdir, epoch, diffusion_prior, optimizer, lr_scheduler, losses, val_losses, lrs)
    
    if args.is_test and False:
        print('inferring and visualizing...')
        # import pdb; pdb.set_trace()
        for val_i, (_, _, _, _, _, _, _, _, file_name, text_descr) in enumerate(val_dl):
            # text_descr = ['The person face lit up with joy, displaying a radiant and happy expression.',
            #     'A look of sadness weighed heavily on the person face, marked by downturned lips and sorrowful eyes.']
            with torch.no_grad():
                with torch.cuda.amp.autocast():

                    samples, clip_target = prepare_train_data(fp_parser, file_name, talking_head, base_sample, silent_frames_start, silent_frames_end)

                    with torch.no_grad():
                        voxel = clip_text_embdder(text_descr)
                        voxel = torch.mean(voxel, dim=1) #### TODO, we use mean 77 tokens to obtain semantic info
                    voxel = voxel.requires_grad_(True)

                    # style_emb = clip_target
                    print(text_descr)
                    style_emb = voxel2style_emb(voxel, diffusion_priors=diffusion_prior)
                    # import pdb; pdb.set_trace();

                    # style_emb = torch.zeros_like(clip_target)
                    # print('style_emb.shape: ', style_emb.shape, 'clip_target.shape: ', clip_target.shape)
                    eval_talking_head_on_audio(talking_head, Path(audio_path), samples, 
                                    style_emb=style_emb, is_external_style_emb=True)
            exit(-1)

    if args.is_talking_instruct:
        print('final usage of talking face instruction...')
        # import pdb; pdb.set_trace()
        def read_json_file(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        if os.path.isdir(args.test_json_path):
            test_json_paths = glob.glob(os.path.join(args.test_json_path, '*.json'))
            text_descrs, audio_paths = [], []
            for test_json_path in tqdm(test_json_paths):
                test_data = read_json_file(test_json_path)
                text_descrs.append(test_data['caption'][0].split('\n#')[0])
                audio_paths.append(test_data['mm_paths'])
            # import pdb; pdb.set_trace()
        else:
            pair_dict = read_json_file(args.test_json_path)
            text_descrs, audio_paths = pair_dict['text_descs'], pair_dict['audio_paths']

        screen_mead_audio = ScreenedMeadAudio()
        wav_paths = screen_mead_audio.wav_paths
        wav_paths = wav_paths[::10]

        if args.is_output_gt:
            if args.is_use_rvd:
                res_data_dict = get_gt_data_rvd()
                audio_paths = [res_data_dict[key]['wav'] for key in res_data_dict.keys()]
                text_descrs = ['dummy' for _ in range(len(audio_paths))]
            else:
                res_data_dict = get_gt_data()

        # import pdb; pdb.set_trace()
        all_diversity_score = []
        t = time.time()
        for val_i, (text_descr, audio_path) in enumerate(zip(text_descrs, audio_paths)):
            # import pdb; pdb.set_trace()
            if (not args.is_use_rvd) and (audio_path not in wav_paths): continue
            # if (not args.is_use_rvd) and args.is_output_gt and not is_audio_in_whitelist(audio_path): continue
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    base_sample = create_base_sample(talking_head, audio_path=audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
                    samples, _ = prepare_test_data(fp_parser, talking_head, base_sample, silent_frames_start, silent_frames_end, batch_size=1)
                    # import pdb; pdb.set_trace()

                    audio_name = os.path.basename(audio_path).split('.')[0]

                    # if False:
                    if args.is_output_gt:
                        gt_exp = res_data_dict[audio_name]['exp']
                        gt_pose = res_data_dict[audio_name]['pose']
                    else:
                        gt_exp = None
                        gt_pose = None
                    # import pdb; pdb.set_trace()

                    voxel = clip_text_embdder(text_descr)
                    voxel = torch.mean(voxel, dim=1) #### TODO, we use mean 77 tokens to obtain semantic info

                    image_embed = None
                    if args.is_cal_diversity:
                        device = torch.device('cuda')
                        shape= (10,1,128)
                        image_embeds = torch.randn(shape, device=device)
                        style_embeds = []
                        for kkk in range(image_embeds.shape[0]):
                            image_embed = image_embeds[kkk]
                            style_emb_kkk = voxel2style_emb(voxel, diffusion_priors=diffusion_prior, image_embed=image_embed)
                            style_embeds.append(style_emb_kkk)
                        style_embeds = torch.cat(style_embeds, dim=0)
                        diversity_score = torch.norm(style_embeds.unsqueeze(0) - style_embeds.unsqueeze(1), dim=3, p=2).sum()/(10*9)
                        diversity_score = diversity_score.item()
                        all_diversity_score.append(diversity_score)
                        print('mean of all diversity score: ', sum(all_diversity_score)*1.0/len(all_diversity_score))
                        continue
                        # import pdb; pdb.set_trace()
                    elif args.is_vis_diversity:
                        device = torch.device('cuda')
                        sample_num = 5
                        shape = (sample_num,1,128)
                        image_embeds = torch.randn(shape, device=device)
                        style_embeds = []
                        for kkk in range(image_embeds.shape[0]):
                            image_embed = image_embeds[kkk]
                            style_emb_kkk = voxel2style_emb(voxel, diffusion_priors=diffusion_prior,
                                                            image_embed=image_embed)
                            style_embeds.append(style_emb_kkk)
                            style_emb = style_emb_kkk
                            out_folder = Path(talking_head.cfg.inout.full_run_dir)/'test_videos_{}'.format(args.save_subdir)
                            save_name = audio_path.split('/')[-3] + '/' + audio_path.split('/')[-2]
                            out_folder = os.path.join(out_folder, save_name, 'samples_{}'.format(kkk))
                            out_instruction_path = os.path.join(out_folder, 'instruction.txt')
                            save_text(text_descr, out_instruction_path)
                            eval_talking_head_on_audio(talking_head, Path(audio_path), samples,
                                            style_emb=style_emb, is_external_style_emb=True, out_folder=out_folder,
                                                gt_exp=gt_exp, gt_pose=gt_pose)
                        continue
                    else:
                        if args.is_no_diffusion:
                            style_emb = voxel2style_emb(voxel, diffusion_priors=diffusion_prior, image_embed=image_embed, no_diffusion=True)
                        else:
                            style_emb = voxel2style_emb(voxel, diffusion_priors=diffusion_prior, image_embed=image_embed)

                    # out_folder = Path(talking_head.cfg.inout.full_run_dir)/'test_videos'/'{:06d}'.format(val_i)
                    out_folder = Path(talking_head.cfg.inout.full_run_dir)/'test_videos_{}'.format(args.save_subdir)
                    save_name = audio_path.split('/')[-3] + '/' + audio_path.split('/')[-2]
                    out_folder = os.path.join(out_folder, save_name)
                    out_instruction_path = os.path.join(out_folder, 'instruction.txt')
                    save_text(text_descr, out_instruction_path)
                    # import pdb; pdb.set_trace()
                    eval_talking_head_on_audio(talking_head, Path(audio_path), samples,
                                    style_emb=style_emb, is_external_style_emb=True, out_folder=out_folder,
                                    gt_exp=gt_exp, gt_pose=gt_pose)

            # exit(-1)
            print(val_i, text_descr, audio_path)
            end_time = time.time()
            print('{:04d} cost {:.3f} s, ave {:.3f} s'.format(val_i, end_time - t, (end_time - t)/(val_i+1)))

            # if val_i > 20: break
            # import pdb; pdb.set_trace()


def save_text(text_descr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text_descr)


def voxel2style_emb(
    voxel, 
    diffusion_priors=None,
    recons_per_sample = 1,
    plotting=True,
    verbose=False,
    img_variations=False,
    seed = 0,
    retrieve = False,
    timesteps_prior = 100,
    n_samples_save=1,
    image_embed=None,
    no_diffusion = False):
    # assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    
    device = voxel.device
    brain_recons = None
    
    # voxel=voxel[:n_samples_save]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if diffusion_priors is not None:
        if not isinstance(diffusion_priors, list):
            diffusion_priors = [diffusion_priors]
        brain_clip_embeddings_sum = None
        for diffusion_prior in diffusion_priors:
            brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(voxel.to(device).float())
            if retrieve:
                continue
            # brain_clip_embeddings0 = brain_clip_embeddings0.view(len(voxel),-1,768) if isinstance(clip_extractor,Clipper) else brain_clip_embeddings0.view(len(voxel),-1,1024)
            brain_clip_embeddings0 = brain_clip_embeddings0.view(len(voxel),-1,128) 
            # import pdb; pdb.set_trace()
            
            if recons_per_sample>0:
                # import pdb; pdb.set_trace()
                if no_diffusion:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    # brain_clip_embeddings = copy.deepcopy(proj_embeddings)
                    brain_clip_embeddings = F.normalize(proj_embeddings, p=2, dim=-1) * 2.0
                    # import pdb; pdb.set_trace()
                elif not img_variations:
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1, 1)
                    try:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior,
                                                generator=generator, image_embed=image_embed)
                    except:
                        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps_prior, image_embed=image_embed)
                    # import pdb; pdb.set_trace()
                else:
                    brain_clip_embeddings0 = brain_clip_embeddings0.view(-1,768)
                    brain_clip_embeddings0 = brain_clip_embeddings0.repeat(recons_per_sample, 1)
                    brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = 1000, #1000 timesteps used from nousr pretraining
                                                generator=generator, image_embed=image_embed)
                if brain_clip_embeddings_sum is None:
                    brain_clip_embeddings_sum = brain_clip_embeddings
                else:
                    brain_clip_embeddings_sum += brain_clip_embeddings

        # average embeddings for all diffusion priors
        if recons_per_sample>0:
            brain_clip_embeddings = brain_clip_embeddings_sum / len(diffusion_priors)
    # import pdb; pdb.set_trace()
    return brain_clip_embeddings


def eval_talking_head_on_audio(
            talking_head, 
            audio_path, 
            samples,
            silent_frames_start=0, 
            silent_frames_end=0, 
            silent_emotion_start = 0, 
            silent_emotion_end = 0, 
            outfolder=None,
            identity_idx=0,
            emotion_index_list=None,
            intensity_list=None,
            save_flame=True,
            save_meshes=False,
            save_videos=False,
            neutral_mesh_path=None,
            style_emb=None,
            is_external_style_emb=True,
            out_folder=None,
            gt_exp=None,
            gt_pose=None
            ):
    silent_intervals = []
    if silent_frames_start > 0:
        num_frames_to_open_mouth = 5
        silent_intervals += [(0,silent_frames_start-num_frames_to_open_mouth)]
        manual_mouth_opening_intervals = [(silent_frames_start-num_frames_to_open_mouth, silent_frames_start)]
    else: 
        num_frames_to_open_mouth = 0
        manual_mouth_opening_intervals = []
    if silent_frames_end > 0:
        num_frames_to_close_mouth = 5
        silent_intervals += [(-silent_frames_end+num_frames_to_close_mouth, -1)]    
        manual_mouth_closure_intervals = [(-silent_frames_end, -silent_frames_end+num_frames_to_close_mouth)]
    else:
        num_frames_to_close_mouth = 0
        manual_mouth_closure_intervals = []
    
    orig_audio, sr = librosa.load(audio_path) 
    ## prepend the silent frames
    if silent_frames_start > 0:
        orig_audio = np.concatenate([np.zeros(int(silent_frames_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
    if silent_frames_end > 0:
        orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_frames_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)
    
    orig_audios = [(orig_audio, sr)]*len(samples)
    run_evalutation(talking_head, samples, audio_path, style_emb=style_emb, is_external_style_emb=is_external_style_emb, 
                out_folder=out_folder, gt_exp=gt_exp, gt_pose=gt_pose, save_flame=save_flame)


def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')

    parser.add_argument("--max_epoch", type=int, default=5000, help='number of epochs')
    parser.add_argument("--epoch", type=int, default=0, help='number of epochs')
    parser.add_argument("--local_rank", type=int, default=0, help='number of epochs')
    parser.add_argument("--clip_size", type=int, default=128, help='number of epochs')
    parser.add_argument('--model_name', type=str, default='EMOTE', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=str(get_path_to_assets() / "TalkingHead/models"))

    parser.add_argument("--use_projector",action=argparse.BooleanOptionalAction,default=True,)

    parser.add_argument('--jobname', type=str, default='text2emo', help='Name of the model to use.')
    parser.add_argument('--save_subdir', type=str, default='', help='')
    parser.add_argument('--is_tensorboard_log', type=int, default=1, help='Name of the model to use.')
    parser.add_argument("--is_test", type=int, default=0)
    parser.add_argument("--is_talking_instruct", type=int, default=0)
    parser.add_argument("--log_loss_steps", type=int, default=5)
    parser.add_argument("--resume_from_ckpt", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--test_audio_path", type=str, default='/data/yashengsun/local_storage/Mead_emoca/Mead_W/W019_front_angry_level2_007/W019_front_angry_level2_007.wav')
    parser.add_argument("--test_json_path", type=str, default='')

    ######  dataset config  #######
    parser.add_argument("--is_output_gt", type=int, default=0)
    parser.add_argument("--is_use_rvd", type=int, default=0)
    parser.add_argument("--is_cal_diversity", type=int, default=0)
    parser.add_argument("--is_vis_diversity", type=int, default=0)
    parser.add_argument("--is_no_diffusion", type=int, default=0)
    parser.add_argument("--unset_prior", type=int, default=0)
    parser.add_argument("--unset_v2c", type=int, default=0)
    parser.add_argument("--load_talkclip_dataset", type=int, default=1)
    parser.add_argument("--wo_dataset_aug", type=int, default=0)
    parser.add_argument("--dataset_names", type=str, default='')
    parser.add_argument("--seq_length", type=int, default=25)
    parser.add_argument("--vertice_dim", type=int, default=53, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only_load_caption", type=int, default=1)

    ###### optimizer       ########
    parser.add_argument("--max_lr", type=float, default=3e-4,)

    args = parser.parse_args()

    # load data
    dataset = get_dataloaders(args)

    # talking head generator
    model_path = Path(args.path_to_models) / args.model_name  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)
    talking_head = talking_head.to(torch.device("cuda"))
    talking_head.eval()

    # clip text to emotion latent model
    clip_size = args.clip_size
    out_dim = clip_size
    voxel2clip_kwargs = dict(in_dim=768,out_dim=clip_size,clip_size=clip_size,use_projector=args.use_projector)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)

    # prior model
    guidance_scale = 3.5
    timesteps = 100
    depth = 6
    dim_head = 64
    heads = clip_size//16
    prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = 1,
            learned_query_mode="pos_emb"
        ).to(torch.device("cuda"))
    # import pdb; pdb.set_trace();
    # use dalle interface to include prior model and clip text-to-emotion models
    timesteps = 100
    diffusion_prior = InstructDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,)
    assert torch.cuda.is_available()
    diffusion_prior = diffusion_prior.to(torch.device("cuda"))

    ## optimizer
    max_lr = args.max_lr
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    trainer(args, dataset["train"], dataset["valid"], diffusion_prior, talking_head, optimizer)


if __name__ == "__main__":
    main()
