# python3.7
"""Contains the class to evaluate DepthGAN by saving snapshots.

Basically, this class traces the quality of images synthesized by DepthGAN.
"""

import os.path
import numpy as np

import torch
import torch.nn.functional as F
import math
import cv2

from utils.misc import Infix
from utils.visualizers import GridVisualizer
from utils.image_utils import postprocess_image
from utils.visualizers import VideoVisualizer
from .base_gan_metric_rgbd import BaseGANMetric

__all__ = ['GANSnapshotRGBD']


class GANSnapshotRGBD(BaseGANMetric):
    """Defines the class for saving snapshots synthesized by DepthGAN."""

    def __init__(self,
                 name='snapshot',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim_rgb=512,
                 latent_dim_depth=512,
                 a_dis=None,
                 latent_angles=None,
                 latent_codes_depth=None,
                 latent_codes_rgb=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 equal_interval=False,
                 keep_same_depth=False,
                 keep_same_rgb=False,
                 fix_depth=False,
                 fix_rgb=False,
                 fix_angle=False,
                 fix_all=False,
                 interpolate_depth=False,
                 interpolate_rgb=False,
                 test=False,
                 image_save=True,
                 save_separate=False,
                 video_save=False,
                 frame_size=(224,224)):
        """Initializes the class with number of samples for each snapshot.

        Args:
            latent_num: Number of latent codes used for each snapshot.
                (default: -1)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim_rgb=latent_dim_rgb,
                         latent_dim_depth=latent_dim_depth,
                         a_dis=a_dis,
                         latent_angles=latent_angles,
                         latent_codes_depth=latent_codes_depth,
                         latent_codes_rgb=latent_codes_rgb,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         equal_interval=equal_interval,
                         keep_same_depth=keep_same_depth,
                         keep_same_rgb=keep_same_rgb,
                         test=test)
        self.keep_same_depth = keep_same_depth
        self.keep_same_rgb = keep_same_rgb
        self.fix_depth = fix_depth
        self.fix_rgb = fix_rgb
        self.fix_angle = fix_angle
        self.fix_all = fix_all
        self.interpolate_depth = interpolate_depth
        self.interpolate_rgb = interpolate_rgb
        self.image_save = image_save
        self.save_separate = save_separate
        if self.image_save:
            self.visualizer_rgb = GridVisualizer(grid_size=self.latent_num)
            self.visualizer_depth = GridVisualizer(grid_size=self.latent_num)
        self.video_save = video_save
        if self.video_save:
            self.visualizer_video = VideoVisualizer(frame_size=frame_size, fps=10)

    def synthesize(self, generator, generator_kwargs, generator_d, generator_d_kwargs):
        """Synthesizes image with the generator."""
        latent_num = self.latent_num
        batch_size = self.batch_size
        if self.random_latents:
            g1 = torch.Generator(device=self.device)
            g1.manual_seed(self.seed)
            g2 = torch.Generator(device=self.device)
            g2.manual_seed(self.seed)
            g4 = torch.Generator(device=self.device)
            g4.manual_seed(self.seed)
        else:
            latent_codes_rgb = np.load(self.latent_file_rgb)[self.replica_indices]
            latent_codes_depth = np.load(self.latent_file_depth)[self.replica_indices]
            latent_codes_angles = np.load(self.latent_file_angle)[self.replica_indices]
        if self.random_labels:
            g3 = torch.Generator(device=self.device)
            g3.manual_seed(self.seed)
        else:
            labels = np.load(self.label_file)[self.replica_indices]

        G_rgb = generator
        G_d = generator_d
        G_rgb_mode = G_rgb.training
        G_d_mode = G_d.training
        G_rgb.eval()
        G_d.eval()
        G_rgb_kwargs = generator_kwargs
        G_d_kwargs = generator_d_kwargs

        self.logger.info(f'Synthesizing {latent_num} images {self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Synthesis', total=latent_num)
        all_images = []
        if self.test:
            codes_d = []
            codes_rgb = []
            codes_angles = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = start + batch_size
            with torch.no_grad():
                if self.test:
                    if self.fix_depth:
                        batch_codes_d = torch.randn((1, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_rgb = torch.randn((batch_size, *self.latent_dim_rgb),
                                                generator=g2,
                                                device=self.device)
                        batch_codes_angles = ((torch.rand((1,1), 
                                                generator=g4,
                                                device=self.device).repeat(batch_size,1)) * 2 - 1) * (math.pi/12)
                    elif self.fix_rgb:
                        batch_codes_d = torch.randn((batch_size, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device)
                        batch_codes_rgb = torch.randn((1, *self.latent_dim_rgb),
                                                generator=g2,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_angles = ((torch.rand((1,1), 
                                                generator=g4,
                                                device=self.device).repeat(batch_size,1)) * 2 - 1) * (math.pi/12)
                    elif self.fix_angle:
                        batch_codes_d = torch.randn((batch_size, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device)
                        batch_codes_rgb = torch.randn((batch_size, *self.latent_dim_rgb),
                                                generator=g2,
                                                device=self.device)
                        batch_codes_angles = ((torch.rand((1,1), 
                                                generator=g4,
                                                device=self.device).repeat(batch_size,1)) * 2 - 1) * (math.pi/12)
                    elif self.fix_all:
                        batch_codes_d = torch.randn((1, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_rgb = torch.randn((1, *self.latent_dim_rgb),
                                                generator=g2,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_angles = ((torch.rand((1,1), 
                                                generator=g4,
                                                device=self.device).repeat(batch_size,1)) * 2 - 1) * (math.pi/12)
                    elif self.interpolate_depth:
                        random_codes_d = torch.randn((2, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device)
                        t = torch.linspace(0, 1, steps=batch_size).unsqueeze(1).to(self.device)
                        batch_codes_d = self.slerp(random_codes_d[0:1], random_codes_d[1:], t)
                        batch_codes_rgb = torch.randn((1, *self.latent_dim_rgb),
                                                generator=g2,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_angles = ((torch.rand((1,1), 
                                                generator=g4,
                                                device=self.device).repeat(batch_size,1)) * 2 - 1) * (math.pi/12)
                    elif self.interpolate_rgb:
                        batch_codes_d = torch.randn((1, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device).repeat(batch_size,1)
                        random_codes_rgb = torch.randn((2, *self.latent_dim_depth),
                                                generator=g2,
                                                device=self.device)
                        t = torch.linspace(0, 1, steps=batch_size).unsqueeze(1).to(self.device)
                        batch_codes_rgb = self.slerp(random_codes_rgb[0:1], random_codes_rgb[1:], t)
                        batch_codes_angles = ((torch.rand((1,1), 
                                                generator=g4,
                                                device=self.device).repeat(batch_size,1)) * 2 - 1) * (math.pi/12)
                    else:
                        batch_codes_d = torch.randn((1, *self.latent_dim_depth),
                                                generator=g1,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_rgb = torch.randn((1, *self.latent_dim_rgb),
                                                generator=g2,
                                                device=self.device).repeat(batch_size,1)
                        batch_codes_angles = torch.linspace(-math.pi/12, math.pi/12, steps=batch_size).unsqueeze(1).to(self.device)
                    codes_d.append(batch_codes_d.detach().cpu().numpy())
                    codes_rgb.append(batch_codes_rgb.detach().cpu().numpy())
                    codes_angles.append(batch_codes_angles.detach().cpu().numpy())
                elif self.random_latents:
                    batch_codes_d = torch.randn((batch_size, *self.latent_dim_depth),
                                              generator=g1,
                                              device=self.device)
                    batch_codes_rgb = torch.randn((batch_size, *self.latent_dim_rgb),
                                              generator=g2,
                                              device=self.device)
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
                            0, self.label_dim, (batch_size,),
                            generator=g3,
                            device=self.device)
                        batch_labels = F.one_hot(
                            rnd_labels, num_classes=self.label_dim)
                else:
                    batch_labels = labels[start:end].cuda().detach()
                results_d = G_d(batch_codes_d, batch_codes_angles, batch_labels, **G_d_kwargs)
                fake_depths = results_d['image']
                fake_rgbs = G_rgb(batch_codes_rgb, results_d, batch_labels, **G_rgb_kwargs)['image']
                batch_images = torch.cat((fake_rgbs, fake_depths), dim=1)
                gathered_images = self.gather_batch_results(batch_images)
                self.append_batch_results(gathered_images, all_images)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        if self.test:
            np.save(self.latent_file_depth, np.concatenate(codes_d, axis=0))
            np.save(self.latent_file_rgb, np.concatenate(codes_rgb, axis=0))
            np.save(self.latent_file_angle, np.concatenate(codes_angles, axis=0))

        all_images = self.gather_all_results(all_images)[:latent_num]
        if self.is_chief:
            assert all_images.shape[0] == latent_num
        else:
            assert len(all_images) == 0
            all_images = None

        if G_rgb_mode:
            G_rgb.train()
        if G_d_mode:
            G_d.train()
        self.sync()
        return all_images

    def evaluate(self, _data_loader, generator, generator_kwargs, generator_d, generator_d_kwargs):
        images = self.synthesize(generator, generator_kwargs, generator_d, generator_d_kwargs)
        if self.is_chief:
            result = {self.name: images}
        else:
            assert images is None
            result = None
        self.sync()
        return result
    
    def slerp(self, a, b, t):
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        d = (a * b).sum(dim=-1, keepdim=True)
        p = t * torch.acos(d)
        c = b - d * a
        c = c / c.norm(dim=-1, keepdim=True)
        d = a * torch.cos(p) + c * torch.sin(p)
        d = d / d.norm(dim=-1, keepdim=True)
        return d


    def _is_better_than(self, metric_name, new, ref):
        """GAN snapshot is not supposed to judge quality."""
        return False

    def save_separate_func(self, images, file_name):
        if images.shape[3] == 3:
            save_dir = os.path.join(self.work_dir, 'rgb')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            for idx in range(images.shape[0]):
                img_path = os.path.join(save_dir, f'{file_name}_{idx:05d}.png')
                cv2.imwrite(img_path, cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR))
        else:
            save_dir = os.path.join(self.work_dir, 'depth')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            for idx in range(images.shape[0]):
                img_path = os.path.join(save_dir, f'{file_name}_{idx:05d}.png')
                cv2.imwrite(img_path, images[idx])

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        images = result[self.name]
        assert isinstance(images, np.ndarray)
        rgbs, depths = postprocess_image(images)
        filename = target_filename or self.name
        if self.image_save:
            if self.save_separate:
                self.save_separate_func(rgbs, filename)
                self.save_separate_func(depths, filename)
            save_path_rgb = os.path.join(self.work_dir, f'{filename}_rgb.png')
            save_path_depth = os.path.join(self.work_dir, f'{filename}_depth.png')
            self.visualizer_rgb.visualize_collection(rgbs, save_path_rgb)
            self.visualizer_depth.visualize_collection(depths, save_path_depth)

        if self.video_save:
            idx = 0
            for start in range(0, self.replica_latent_num, self.batch_size):
                end = start + self.batch_size
                save_path_video_rgb = os.path.join(self.work_dir, f'{filename}_{idx}_rgb.mp4')
                save_path_video_depth = os.path.join(self.work_dir, f'{filename}_{idx}_depth.mp4')
                self.visualizer_video.visualize_collection(rgbs[start:end], save_path_video_rgb)
                self.visualizer_video.visualize_collection(depths[start:end], save_path_video_depth)
                idx += 1

        prefix = f'Evaluating `{self.name}` with {self.latent_num} samples'
        if log_suffix is None:
            msg = f'{prefix}.'
        else:
            msg = f'{prefix}, {log_suffix}.'
        self.logger.info(msg)

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning(f'`Tag` is missing when writing data to '
                                    f'TensorBoard, hence, the data may be '
                                    f'mixed up!')
            self.tb_writer.add_image(self.name+'_rgb', self.visualizer_rgb.grid, tag,
                                     dataformats='HWC')
            self.tb_writer.add_image(self.name+'_depth', self.visualizer_depth.grid, tag,
                                     dataformats='HWC')
            self.tb_writer.flush()
        self.sync()
