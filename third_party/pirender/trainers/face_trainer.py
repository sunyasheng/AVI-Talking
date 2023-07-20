import math

import torch

from trainers.base import BaseTrainer
from util.trainer import accumulate, get_optimizer
from loss.perceptual import PerceptualLoss
from loss.id_loss import VGGFace2Loss

class FaceTrainer(BaseTrainer):
    r"""Initialize lambda model trainer.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, opt, net_G, opt_G, sch_G,
                 train_data_loader, val_data_loader=None):
        super(FaceTrainer, self).__init__(opt, net_G, opt_G, sch_G, train_data_loader, val_data_loader)
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        # import pdb; pdb.set_trace()
        self.is_cross_id_loss = opt.is_cross_id_loss
        if self.is_cross_id_loss:
            self._assign_criteria(
                'id_loss',
                VGGFace2Loss().to('cuda'),
                opt.trainer.loss_weight.weight_id_loss
            )

    def _init_loss(self, opt):
        self._assign_criteria(
            'perceptual_warp',
            PerceptualLoss(
                network=opt.trainer.vgg_param_warp.network,
                layers=opt.trainer.vgg_param_warp.layers,
                num_scales=getattr(opt.trainer.vgg_param_warp, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_warp, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_warp, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_warp)

        self._assign_criteria(
            'perceptual_final',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight


    def mix_semantic(self, semantic):
        # keep the camera and head pose untouched and mix exp coeff
        # import pdb; pdb.set_trace()
        mixed_semantic = semantic.clone()

        batch_columns = semantic.size(0)
        shuffled_indices = torch.randperm(batch_columns)
        shuffled_semantic = semantic[shuffled_indices, :]

        mixed_semantic[:, :50] = shuffled_semantic[:, :50]
        mixed_semantic[:, 53:56] = shuffled_semantic[:, 53:56]
        return mixed_semantic

    def optimize_parameters(self, data):
        self.gen_losses = {}
        source_image, target_image = data['source_image'], data['target_image']
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)
        gt_image = torch.cat((target_image, source_image), 0) 

        output_dict = self.net_G(input_image, input_semantic, self.training_stage)

        # print('training_stage: ', self.training_stage)
        if self.training_stage == 'gen':
            fake_img = output_dict['fake_image']
            warp_img = output_dict['warp_image']
            self.gen_losses["perceptual_final"] = self.criteria['perceptual_final'](fake_img, gt_image)
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_img, gt_image)

            if self.is_cross_id_loss:
                mix_input_semantic = self.mix_semantic(input_semantic)
                mix_output_dict = self.net_G(input_image, mix_input_semantic, self.training_stage)
                mix_fake_img = mix_output_dict['fake_image']
                # import pdb; pdb.set_trace()
                # import torchvision
                # torchvision.utils.save_image(0.5+0.5*torch.cat([mix_fake_img, fake_img, gt_image], dim=-1), 'cat.jpg')
                ## ideally, the mix fake image is supposed to enjoy identical pose and cam with gt but different expression
                self.gen_losses["id_loss"] = self.criteria['id_loss'](mix_fake_img*0.5+0.5, gt_image*0.5+0.5)
        else:
            warp_img = output_dict['warp_image']
            # print(warp_img.shape, gt_image.shape)
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_img, gt_image)

        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]
            # print(key, self.gen_losses[key])

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)

    def _start_of_iteration(self, data, current_iteration):
        self.training_stage = 'gen' if current_iteration >= self.opt.trainer.pretrain_warp_iteration else 'warp'
        if current_iteration == self.opt.trainer.pretrain_warp_iteration:
            self.reset_trainer()
        return data

    def reset_trainer(self):
        self.opt_G = get_optimizer(self.opt.gen_optimizer, self.net_G.module)

    def _get_visualizations(self, data):
        source_image, target_image = data['source_image'], data['target_image']
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)

        with torch.no_grad():
            self.net_G_ema.eval()
            output_dict = self.net_G_ema(
                input_image, input_semantic, self.training_stage
                )
            if self.training_stage == 'gen':
                fake_img = torch.cat([output_dict['warp_image'], output_dict['fake_image']], 3)
                if self.is_cross_id_loss:
                    mix_input_semantic = self.mix_semantic(input_semantic)
                    mix_output_dict = self.net_G_ema(input_image, mix_input_semantic, self.training_stage)
                    fake_img = torch.cat([fake_img, mix_output_dict['fake_image']], 3)
            else:
                fake_img = output_dict['warp_image']

            fake_source, fake_target = torch.chunk(fake_img, 2, dim=0)
            sample_source = torch.cat([source_image, fake_source, target_image], 3)
            sample_target = torch.cat([target_image, fake_target, source_image], 3)                    
            sample = torch.cat([sample_source, sample_target], 2)
            sample = torch.cat(torch.chunk(sample, sample.size(0), 0)[:3], 2)
        return sample

    def test(self, data_loader, output_dir, current_iteration=-1):
        pass

    def _compute_metrics(self, data, current_iteration):
        if self.training_stage == 'gen':
            source_image, target_image = data['source_image'], data['target_image']
            source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

            input_image = torch.cat((source_image, target_image), 0)
            input_semantic = torch.cat((target_semantic, source_semantic), 0)        
            gt_image = torch.cat((target_image, source_image), 0)        
            metrics = {}
            with torch.no_grad():
                self.net_G_ema.eval()
                output_dict = self.net_G_ema(
                    input_image, input_semantic, self.training_stage
                    )
                fake_image = output_dict['fake_image']
                metrics['lpips'] = self.lpips(fake_image, gt_image).mean()
            return metrics