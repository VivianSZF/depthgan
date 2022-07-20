# python3.7
"""Contains the class to evaluate GANs with rotation metrics.

FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch.nn.functional import grid_sample

from models import build_model
from utils.misc import Infix
from .base_gan_metric_rgbd import BaseGANMetric

__all__ = ['RotEval', 'RotEval50K']


class RotEval(BaseGANMetric):
    """Defines the class for rotation metric computation."""

    def __init__(self,
                 name='RotEval',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim_rgb=512,
                 latent_dim_depth=512,
                 a_dis=None,
                 latent_angles=None,
                 latent_codes_depth=None,
                 latent_codes_rgb=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 real_num=-1,
                 fake_num=-1):
        """Initializes the class with number of real/fakes samples.

        Args:
            real_num: Number of real images used for FID evaluation. If not set,
                all images from the given evaluation dataset will be used.
                (default: -1)
            fake_num: Number of fake images used for FID evaluation.
                (default: -1)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=fake_num,
                         latent_dim_depth=latent_dim_depth,
                         latent_dim_rgb=latent_dim_rgb,
                         a_dis=a_dis,
                         latent_angles=latent_angles,
                         latent_codes_depth=latent_codes_depth,
                         latent_codes_rgb=latent_codes_rgb,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.real_num = real_num
        self.fake_num = fake_num


    def compute_depth_diff(self, generator, generator_kwargs, generator_depth, generator_depth_kwargs):
        """Extracts inception features from fake data."""
        fake_num = self.fake_num
        batch_size = self.batch_size
        if self.random_latents:
            g1 = torch.Generator(device=self.device)
            g1.manual_seed(self.seed)
            g2 = torch.Generator(device=self.device)
            g2.manual_seed(self.seed)
        else:
            latent_codes_rgb = np.load(self.latent_file_rgb)[self.replica_indices]
            latent_codes_depth = np.load(self.latent_file_depth)[self.replica_indices]
            latent_codes_angles = np.load(self.latent_file_angles)[self.replica_indices]
        if self.random_labels:
            g3 = torch.Generator(device=self.device)
            g3.manual_seed(self.seed)
        else:
            labels = np.load(self.label_file)[self.replica_indices]

        G_rgb = generator
        G_d = generator_depth
        G_d_mode = G_d.training
        G_rgb_mode = G_rgb.training
        G_rgb.eval()
        G_d.eval()
        G_rgb_kwargs = generator_kwargs
        G_d_kwargs = generator_depth_kwargs

        self.logger.info(f'Extracting inception features from fake data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('RotEval', total=fake_num)
        all_loss = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = start + batch_size
            with torch.no_grad():
                if self.random_latents:
                    batch_codes_d = torch.randn((batch_size//2, *self.latent_dim_depth),
                                              generator=g1,
                                              device=self.device).repeat(2,1)
                    batch_codes_rgb = torch.randn((batch_size//2, *self.latent_dim_rgb),
                                              generator=g2,
                                              device=self.device).repeat(2,1)
                    batch_codes_angles = self.a_dis.sample([batch_size, 1]).to(self.device)
                else:
                    batch_codes_d = latent_codes_depth[start:end].cuda().detach()
                    batch_codes_rgb = latent_codes_rgb[start:end].cuda().detach()
                    batch_codes_angles = latent_codes_angles[start:end].cuda().detach()
                if self.random_labels:
                    if self.label_dim == 0:
                        batch_labels = torch.zeros((batch_size, 0),
                                                   device=self.device)
                    else:
                        rnd_labels = torch.randint(
                            0, self.label_dim, (batch_size//2,),
                            generator=g3,
                            device=self.device).repeat(2)
                        batch_labels = F.one_hot(
                            rnd_labels, num_classes=self.label_dim)
                else:
                    batch_labels = labels[start:end].cuda().detach()
                batch_results = torch.zeros((batch_codes_d.shape[0]//2, 3), dtype=torch.float64, device=self.device)
                results_d = G_d(batch_codes_d, batch_codes_angles, batch_labels, **G_d_kwargs)
                generated_depth = results_d['image']
                batch_images = G_rgb(batch_codes_rgb, results_d, batch_labels, **G_rgb_kwargs)['image']
                output_images = torch.cat((batch_images, generated_depth), dim=1)
                new_rgb, new_depth, mask, new_d = self.change_a(output_images[:(batch_size//2)], batch_codes_angles[:(batch_size//2)], batch_codes_angles[(batch_size//2):], output_images[(batch_size//2):])
                loss_rotdepth = torch.abs(new_depth-new_d) * mask
                loss_rotrgb = torch.abs(new_rgb - output_images[:(batch_size//2),:3]) * mask
                batch_results[:, 0] += loss_rotdepth.to(torch.float64).sum(dim=(1,2,3))
                batch_results[:, 1] += loss_rotrgb.to(torch.float64).sum(dim=(1,2,3))
                batch_results[:, 2] += mask.to(torch.float64).sum(dim=(1,2,3))
                gathered_loss_rot = self.gather_batch_results(batch_results)
                self.append_batch_results(gathered_loss_rot, all_loss)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        all_losses = self.gather_all_results(all_loss)[:fake_num]
        if self.is_chief:
            all_losses_depth = float(np.sum(all_losses[:,0])/np.sum(all_losses[:,2]))
            all_losses_rgb = float(np.sum(all_losses[:,1])/np.sum(all_losses[:,2]))
        else:
            assert len(all_losses) == 0
            all_losses = None
            all_losses_depth = None
            all_losses_rgb = None

        if G_d_mode:
            G_d.train()
        if G_rgb_mode:
            G_rgb.train()
        self.sync()
        return all_losses_depth, all_losses_rgb

    def evaluate(self, data_loader, generator, generator_kwargs, generator_d, generator_d_kwargs):
        losses_depth, losses_rgb = self.compute_depth_diff(generator, generator_kwargs, generator_d, generator_d_kwargs)
        if self.is_chief:
            result = {self.name+'_depth': losses_depth}
            result[self.name+'_rgb'] = losses_rgb
        else:
            assert losses_depth is None and losses_rgb is None
            result = None
        self.sync()
        return result
    
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

    def change_a(self, img, a1, a2, img2):
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
        
        # occlusion??
        new_rgb = grid_sample(img2[:,:3], grid_, align_corners=False)
        new_depth = grid_sample(depth2-s, grid_, align_corners=False)
        return new_rgb, new_depth, mask.unsqueeze(1), z_r.unsqueeze(1)

    def _is_better_than(self, metric_name, new, ref):
        """A lower rotation metric is better.

        Example:

        FID = FIDMetric()
        5.3 << FID.is_better_than >> 6.8  # gives `True`
        """
        return ref is None or new < ref

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        loss_depth = float(result[self.name+'_depth'])
        assert isinstance(loss_depth, float)
        loss_rgb = float(result[self.name+'_rgb'])
        assert isinstance(loss_rgb, float)
        prefix_depth = f'Evaluating `{self.name}_depth`: '
        prefix_rgb = f'Evaluating `{self.name}_rgb`: '
        if log_suffix is None:
            msg = f'{prefix_depth}{loss_depth:.3f}. {prefix_rgb}{loss_rgb:.3f}.'
        else:
            msg = f'{prefix_depth}{loss_depth:.3f}, {prefix_rgb}{loss_rgb:.3f}, {log_suffix}.'
        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning(f'`Tag` is missing when writing data to '
                                    f'TensorBoard, hence, the data may be '
                                    f'mixed up!')
            self.tb_writer.add_scalar(f'Metrics/{self.name}_depth', loss_depth, tag)
            self.tb_writer.add_scalar(f'Metrics/{self.name}_rgb', loss_rgb, tag)
            self.tb_writer.flush()
        self.sync()


class RotEval50K(RotEval):
    """Defines the class for RotEval50K metric computation.

    50_000 real/fake samples will be used for feature extraction.
    """

    def __init__(self,
                 name='RotEval50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim_rgb=512,
                 latent_dim_depth=512,
                 a_dis=None,
                 latent_angles=None,
                 latent_codes_depth=None,
                 latent_codes_rgb=None,
                 label_dim=0,
                 labels=None,
                 seed=0):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim_rgb=latent_dim_rgb,
                         latent_dim_depth=latent_dim_depth,
                         a_dis=a_dis,
                         latent_angles=latent_angles,
                         latent_codes_depth=latent_codes_depth,
                         latent_codes_rgb=latent_codes_rgb,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         real_num=50_000,
                         fake_num=50_000)

