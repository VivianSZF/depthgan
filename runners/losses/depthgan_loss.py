# python3.7
"""Defines loss functions for StyleGAN2 training."""

import numpy as np

import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch.nn.functional import grid_sample

from third_party.stylegan2_official_ops import conv2d_gradfix
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['DepthGANLoss']


class DepthGANLoss(BaseLoss):
    """Contains the class to compute losses for training StyleGAN2.

    Basically, this class contains the computation of adversarial loss for both
    generator and discriminator, perceptual path length regularization for
    generator, and gradient penalty as the regularization for discriminator.
    """

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        if runner.enable_amp:
            raise NotImplementedError('DepthGAN loss does not support '
                                      'automatic mixed precision training yet.')

        # Setting for discriminator loss.
        self.d_loss_kwargs = d_loss_kwargs or dict()
        # Loss weight for gradient penalty on real images.
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        # How often to perform gradient penalty regularization.
        self.r1_interval = int(self.d_loss_kwargs.get('r1_interval', 16))

        self.num_classes = int(self.d_loss_kwargs.get('num_classes', 10))

        self.gdloss = self.d_loss_kwargs.get('gdloss', 0.001)
        self.ddloss = self.d_loss_kwargs.get('ddloss', 0.8)

        self.g_loss_kwargs = g_loss_kwargs or dict()
        self.drotloss = self.g_loss_kwargs.get('drotloss', 50.0)
        self.rgbrotloss = self.g_loss_kwargs.get('rgbrotloss', 10.0)

        if self.r1_interval <= 0:
            self.r1_interval = 1
            self.r1_gamma = 0.0
        assert self.r1_gamma >= 0.0
        runner.running_stats.add('Loss/D Fake',
                                 log_name='loss_d_fake',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D Real',
                                 log_name='loss_d_real',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D depth',
                                 log_name='loss_d_depth',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.r1_gamma > 0.0:
            runner.running_stats.add('Loss/Real Gradient Penalty',
                                     log_name='loss_gp',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')

        # Settings for generator loss.
        self.g_loss_kwargs = g_loss_kwargs or dict()
        # Factor to shrink the batch size for path length regularization.
        self.pl_batch_shrink = int(self.g_loss_kwargs.get('pl_batch_shrink', 2))
        # Loss weight for perceptual path length regularization.
        self.pl_weight = self.g_loss_kwargs.get('pl_weight', 2.0)
        # Decay factor for perceptual path length regularization.
        self.pl_decay = self.g_loss_kwargs.get('pl_decay', 0.01)
        # How often to perform perceptual path length regularization.
        self.pl_interval = int(self.g_loss_kwargs.get('pl_interval', 4))

        if self.pl_interval <= 0:
            self.pl_interval = 1
            self.pl_weight = 0.0
        assert self.pl_batch_shrink >= 1
        assert self.pl_weight >= 0.0
        assert 0.0 <= self.pl_decay <= 1.0
        runner.running_stats.add('Loss/G',
                                 log_name='loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/G depth',
                                 log_name='loss_g_depth',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/G rot depth',
                                 log_name='loss_g_rot_d',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/G rot rgb',
                                 log_name='loss_g_rot_rgb',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.pl_weight > 0.0:
            runner.running_stats.add('Loss/Path Length Penalty RGB',
                                     log_name='loss_pl_rgb',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')
            self.pl_mean_rgb = torch.zeros((), device=runner.device)
            runner.running_stats.add('Loss/Path Length Penalty Depth',
                                     log_name='loss_pl_d',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')
            self.pl_mean_d = torch.zeros(()).cuda()

    @staticmethod
    def run_G(runner, batch_size=None, repeat=False, return_angles=False, sync=True):
        """Forwards generator."""

        G_rgb = runner.ddp_models['generator_rgb']
        G_depth = runner.ddp_models['generator_depth']
        G_rgb_kwargs = runner.model_kwargs_train['generator_rgb']
        G_depth_kwargs = runner.model_kwargs_train['generator_depth']

        # Prepare latent codes and labels.
        batch_size = batch_size or runner.batch_size
        latent_dim_rgb = runner.models['generator_rgb'].latent_dim
        latent_dim_depth = runner.models['generator_depth'].latent_dim
        label_size = runner.models['generator_rgb'].label_size
        angles = runner.a_dis.sample([batch_size, 1]).to(runner.device)
        if repeat:
            latents_d = torch.randn((batch_size//2, *latent_dim_depth), device=runner.device).repeat(2,1)
            latents_rgb = torch.randn((batch_size//2, *latent_dim_rgb), device=runner.device).repeat(2,1)
            labels = None
            if label_size > 0:
                rnd_labels = torch.randint(
                    0, label_size, (batch_size//2,), device=runner.device).repeat(2)
                labels = F.one_hot(rnd_labels, num_classes=label_size)
        else:
            latents_d = torch.randn((batch_size, *latent_dim_depth), device=runner.device)
            latents_rgb = torch.randn((batch_size, *latent_dim_rgb), device=runner.device)
            labels = None
            if label_size > 0:
                rnd_labels = torch.randint(
                    0, label_size, (batch_size,), device=runner.device)
                labels = F.one_hot(rnd_labels, num_classes=label_size)
        with ddp_sync(G_depth, sync=sync):
            results_d = G_depth(latents_d, angles, labels, **G_depth_kwargs)
        with ddp_sync(G_rgb, sync=sync):
            results_rgb = G_rgb(latents_rgb, results_d, labels, **G_rgb_kwargs)

        if return_angles == True:
            return results_d, results_rgb, angles
        else:
            return results_d, results_rgb

    @staticmethod
    def run_D(runner, images, labels, depths=None, sync=True):
        """Forwards discriminator."""
        D = runner.ddp_models['discriminator']
        D_kwargs = runner.model_kwargs_train['discriminator']
        images = runner.augment(images)
        if depths is not None:
            output_images = torch.cat((images, depths), dim=1)
            with ddp_sync(D, sync=sync):
                D_result = D(output_images, labels, **D_kwargs)
        else:
            with ddp_sync(D, sync=sync):
                D_result = D(images, labels, **D_kwargs)
        
        return D_result['score'], D_result['depthimage']

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        with conv2d_gradfix.no_weight_gradients():
            image_grad = torch.autograd.grad(
                outputs=[scores.sum()],
                inputs=[images],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        grad_penalty = image_grad.square().sum((1, 2, 3))
        return grad_penalty

    def compute_pl_penalty(self, images, latents, depth=False):
        """Computes perceptual path length penalty."""
        res_h, res_w = images.shape[2:4]
        pl_noise = torch.randn_like(images) / np.sqrt(res_h * res_w)
        with conv2d_gradfix.no_weight_gradients():
            code_grad = torch.autograd.grad(
                outputs=[(images * pl_noise).sum()],
                inputs=[latents],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        pl_length = code_grad.square().sum(2).mean(1).sqrt()
        if depth:
            pl_mean_d = (self.pl_mean_d * (1 - self.pl_decay) +
                    pl_length.mean() * self.pl_decay)
            self.pl_mean_d.copy_(pl_mean_d.detach())
            pl_penalty = (pl_length - pl_mean_d).pow(2)
        else:
            pl_mean_rgb = (self.pl_mean_rgb * (1 - self.pl_decay) +
                   pl_length.mean() * self.pl_decay)
            self.pl_mean_rgb.copy_(pl_mean_rgb.detach())
            pl_penalty = (pl_length - pl_mean_rgb).pow(2)
        return pl_penalty

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""
        results_d, results_rgb = self.run_G(runner, sync=sync)
        fake_depths, fake_images, labels = results_d['image'], results_rgb['image'], results_rgb['label']
        fake_scores, depthimage = self.run_D(runner, 
                                             images=fake_images,
                                             labels=labels,
                                             depths=fake_depths,
                                             sync=sync)
        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})

        return (g_loss + 0 * depthimage[:,0,0,0]).mean()
    
    def g_depth_loss(self, runner, _data, sync=True):
        results_d, results_rgb = self.run_G(runner, sync=sync)
        fake_depths, fake_images, labels = results_d['image'], results_rgb['image'], results_rgb['label']
        fake_scores, depthimage = self.run_D(runner,
                                             images=fake_images,
                                             labels=labels,
                                             sync=sync)
        loss_Gdepth = torch.nn.functional.cross_entropy(depthimage, torch.floor((fake_depths[:,0]+1)/2*self.num_classes).clamp(0,self.num_classes-1).long())
        runner.running_stats.update({'Loss/G depth': loss_Gdepth})
        return (fake_scores * 0+ self.gdloss * loss_Gdepth).mean()  #(fake_scores * 0).mean() + self.gdloss * loss_Gdepth

    def d_rot_loss(self, runner, _data, sync=True, occlusion_aware=False):
        if occlusion_aware:
            batch_size = runner.batch_size
            results_d, results_rgb, angles = self.run_G(runner, repeat=True, return_angles=True, sync=sync)
            fake_depths, fake_images = results_d['image'], results_rgb['image']
            output_images = torch.cat((fake_images, fake_depths), dim=1)

            new_depth_1, mask_1, new_d_1 = self.change_a(output_images[:(batch_size//2)], angles[:(batch_size//2)], angles[(batch_size//2):], output_images[(batch_size//2):], depth_only=True)
            mask_closer_1 = torch.lt(new_d_1, new_depth_1) 
            loss_rotdepth_1 = torch.nn.functional.l1_loss(mask_closer_1*mask_1*new_depth_1, mask_closer_1*mask_1*new_d_1)

            new_depth_2, mask_2, new_d_2 = self.change_a(output_images[(batch_size//2):], angles[(batch_size//2):], angles[:(batch_size//2)], output_images[:(batch_size//2)], depth_only=True)
            mask_closer_2 = torch.lt(new_d_2, new_depth_2) 
            loss_rotdepth_2 = torch.nn.functional.l1_loss(mask_closer_2*mask_2*new_depth_2, mask_closer_2*mask_2*new_d_2)
            loss_rotdepth = loss_rotdepth_1 + loss_rotdepth_2
            runner.running_stats.update({'Loss/G rot depth': loss_rotdepth})
            return (self.drotloss * loss_rotdepth + fake_images[:,0,0,0]*0).mean() #self.drotloss * loss_rotdepth + (fake_images[:,0,0,0]*0).mean()
        else:
            batch_size = runner.batch_size
            results_d, results_rgb, angles = self.run_G(runner, repeat=True, return_angles=True, sync=sync)
            fake_depths, fake_images = results_d['image'], results_rgb['image']
            # fake_images = runner.augment(fake_images)
            output_images = torch.cat((fake_images, fake_depths), dim=1)
            new_depth, mask, new_d = self.change_a(output_images[:(batch_size//2)], angles[:(batch_size//2)], angles[(batch_size//2):], output_images[(batch_size//2):], depth_only=True)
            loss_rotdepth = torch.nn.functional.l1_loss(mask*new_depth, mask*new_d)
            runner.running_stats.update({'Loss/G rot depth': loss_rotdepth})
            return (self.drotloss * loss_rotdepth + fake_images[:,0,0,0]*0).mean() #self.drotloss * loss_rotdepth + (fake_images[:,0,0,0]*0).mean()

        


    def rgb_rot_loss(self, runner, _data, sync=True, occlusion_aware=False):
        if occlusion_aware:
            batch_size = runner.batch_size
            results_d, results_rgb, angles = self.run_G(runner, repeat=True, return_angles=True, sync=sync)
            fake_depths, fake_images = results_d['image'], results_rgb['image']

            # cannot use augmentation to rgb images!
            output_images = torch.cat((fake_images, fake_depths), dim=1)

            new_depth_1, new_rgb_1, mask_1, new_d_1 = self.change_a(output_images[:(batch_size//2)], angles[:(batch_size//2)], angles[(batch_size//2):], output_images[(batch_size//2):], depth_only=False)
            mask_closer_1 = torch.lt(new_d_1, new_depth_1)
            loss_rotrgb_1 = torch.nn.functional.l1_loss(mask_closer_1*mask_1*new_rgb_1, mask_closer_1*mask_1*output_images[:(batch_size//2),:3])

            new_depth_2, new_rgb_2, mask_2, new_d_2 = self.change_a(output_images[(batch_size//2):], angles[(batch_size//2):], angles[:(batch_size//2)], output_images[:(batch_size//2)], depth_only=False)
            mask_closer_2 = torch.lt(new_d_2, new_depth_2)
            loss_rotrgb_2 = torch.nn.functional.l1_loss(mask_closer_2*mask_2*new_rgb_2, mask_closer_2*mask_2*output_images[(batch_size//2):,:3])

            loss_rotrgb = loss_rotrgb_1 + loss_rotrgb_2

            runner.running_stats.update({'Loss/G rot rgb': loss_rotrgb})
            return (self.rgbrotloss * loss_rotrgb + fake_depths[:,0,0,0]*0).mean() #self.rgbrotloss * loss_rotrgb + (fake_depths[:,0,0,0]*0).mean()
        else:


            batch_size = runner.batch_size
            results_d, results_rgb, angles = self.run_G(runner, repeat=True, return_angles=True, sync=sync)
            fake_depths, fake_images = results_d['image'], results_rgb['image']

            # cannot use augmentation to rgb images!
            output_images = torch.cat((fake_images, fake_depths), dim=1)

            new_depth, new_rgb, mask, new_d = self.change_a(output_images[:(batch_size//2)], angles[:(batch_size//2)], angles[(batch_size//2):], output_images[(batch_size//2):], depth_only=False)
            loss_rotrgb = torch.nn.functional.l1_loss(mask*new_rgb, mask*output_images[:(batch_size//2),:3])

            runner.running_stats.update({'Loss/G rot rgb': loss_rotrgb})
            return (self.rgbrotloss * loss_rotrgb + fake_depths[:,0,0,0]*0).mean() #self.rgbrotloss * loss_rotrgb + (fake_depths[:,0,0,0]*0).mean()
        

    def g_reg(self, runner, _data, sync=True):
        """Computes the regularization loss for generator."""
        if runner.iter % self.pl_interval != 1 or self.pl_weight == 0.0:
            return None

        batch_size = max(runner.batch_size // self.pl_batch_shrink, 1)
        results_d, results_rgb = self.run_G(runner, batch_size=batch_size, sync=sync)
        
        pl_penalty_d = self.compute_pl_penalty(
            results_d['image'], results_d['wp'], depth=True)
        runner.running_stats.update(
            {'Loss/Path Length Penalty Depth': pl_penalty_d})
        pl_penalty_d = pl_penalty_d * self.pl_weight * self.pl_interval

        pl_penalty_rgb = self.compute_pl_penalty(
            results_rgb['image'], results_rgb['wp'], depth=False)
        runner.running_stats.update(
            {'Loss/Path Length Penalty RGB': pl_penalty_rgb})
        pl_penalty_rgb = pl_penalty_rgb * self.pl_weight * self.pl_interval

        return (results_rgb['image'][:, 0, 0, 0] * 0 + results_d['image'][:,0,0,0]*0+ pl_penalty_d + pl_penalty_rgb).mean()

    def d_fake_loss(self, runner, _data, sync=True):
        """Computes discriminator loss on generated images."""
        results_d, results_rgb = self.run_G(runner, sync=False)
        fake_depths, fake_images, labels = results_d['image'], results_rgb['image'], results_rgb['label']
        fake_scores, depthimage = self.run_D(
            runner, fake_images, labels, depths=fake_depths, sync=sync)
        d_fake_loss = F.softplus(fake_scores)
        runner.running_stats.update({'Loss/D Fake': d_fake_loss})

        return (d_fake_loss + depthimage[:,0,0,0]*0).mean()

    def d_real_loss(self, runner, data, sync=True):
        """Computes discriminator loss on real images."""
        real_images = data['image'].detach()
        real_rgbs = real_images[:,:3]
        depths = real_images[:,3].unsqueeze(1)

        real_labels = data.get('label', None)
        real_scores, depthimage = self.run_D(runner, real_rgbs, real_labels, depths=depths, sync=sync)
        d_real_loss = F.softplus(-real_scores)
        runner.running_stats.update({'Loss/D Real': d_real_loss})

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return (d_real_loss + depthimage[:,0,0,0]* 0).mean()

    def d_depth_loss(self, runner, data, sync=True):
        real_images = data['image'].detach()
        real_rgbs = real_images[:,:3]
        real_labels = data.get('label', None)
        real_scores, depthimage = self.run_D(runner, real_rgbs, real_labels, sync=sync)
        loss_Ddepth = torch.nn.functional.cross_entropy(depthimage, torch.floor((real_images[:,3]+1)/2*self.num_classes).clamp(0,self.num_classes-1).long())

        runner.running_stats.update({'Loss/D depth': loss_Ddepth})

        return (self.ddloss * loss_Ddepth + real_scores * 0).mean() #self.ddloss * loss_Ddepth + (real_scores * 0).mean()

    def d_reg(self, runner, data, sync=True):
        """Computes the regularization loss for discriminator."""
        if runner.iter % self.r1_interval != 1 or self.r1_gamma == 0.0:
            return None

        real_images = data['image'].detach().requires_grad_(True)
        real_labels = data.get('label', None)

        real_scores, depthimage = self.run_D(runner, real_images[:,:3], real_labels, depths=real_images[:,3].unsqueeze(1), sync=sync)
        r1_penalty = self.compute_grad_penalty(images=real_images,
                                               scores=real_scores)
        runner.running_stats.update({'Loss/Real Gradient Penalty': r1_penalty})
        r1_penalty = r1_penalty * (self.r1_gamma * 0.5) * self.r1_interval
        
        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return (real_scores * 0 + r1_penalty+ depthimage[:,0,0,0]*0).mean()

    def unproject(self,depth_map, K):
        """
        depth_map: h, w
        K: 3, 3
        """
        N, H, W = depth_map.shape
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
        y = y.cuda().unsqueeze(0).repeat(N,1,1)
        x = x.cuda().unsqueeze(0).repeat(N,1,1)
        xy_map = torch.stack([x, y], axis=3) * depth_map[..., None]
        xyz_map = torch.cat([xy_map, depth_map[..., None]], axis=-1)
        xyz = xyz_map.view(-1, 3)
        xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 0, 1))
        xyz_map = xyz.view(N, H, W, 3)
        return xyz_map

    def calculate_rotation(self,theta, axis):
        rotdir = torch.tensor(axis).cuda().unsqueeze(0).repeat(theta.size(0),1) * theta
        rotmat = tgm.angle_axis_to_rotation_matrix(rotdir)[:,:3,:3]
        return rotmat

    def change_a(self, img, a1, a2, img2, depth_only=False):
        s=-0.13
        depth = img[:,3]
        depth = (depth+1)/2
        depth2 = img2[:,3].unsqueeze(1)
        depth2 = (depth2+1)/2
        depth = depth - s
        K = torch.tensor([[260, 0., depth.size(2) / 2], [0., 260, depth.size(1) / 2],[0., 0., 1.]]).cuda()
        xyz_map = self.unproject(depth, K)

        axis = [0, -1, 0]
        theta = a2-a1
        rot = self.calculate_rotation(theta, axis)

        N, H, W, C = xyz_map.size()
        xyz = xyz_map.view(N, -1, 3)

        #translation
        ori_x = ((torch.max(xyz[:,:,0],dim=1)[0]+torch.min(xyz[:,:,0],dim=1)[0])/2).unsqueeze(1)
        ori_z = ((torch.max(xyz[:,:,2],dim=1)[0]+torch.min(xyz[:,:,2],dim=1)[0])/2).unsqueeze(1)

        xyz[:,:,0] = xyz[:,:,0]-ori_x
        xyz[:,:,2] = xyz[:,:,2]-ori_z
        xyz = torch.bmm(xyz, rot)
        xyz[:,:,0] = xyz[:,:,0]+ori_x
        xyz[:,:,2] = xyz[:,:,2]+ori_z

        # project
        xyz = torch.matmul(xyz, torch.transpose(K,0,1))
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-8)
        xy = xy.view(N, H, W, 2)
        z_r = xyz[:, :, 2].view(N,H,W)

        grid_ = 2* xy / depth.size(2) - 1
        mask = 1 - ((grid_[:,:,:,0] < -1) | (grid_[:,:,:,0] > 1) | (grid_[:,:,:,1] < -1) | (grid_[:,:,:,1] > 1)).float()
        
        new_depth = grid_sample(depth2-s, grid_, align_corners=False)
        if not depth_only:
            new_rgb = grid_sample(img2[:,:3], grid_, align_corners=False)
            return new_depth, new_rgb, mask.unsqueeze(1), z_r.unsqueeze(1)
        else:
            return new_depth, mask.unsqueeze(1), z_r.unsqueeze(1)
