# python3.7
"""Contains the class to evaluate GANs with generated depth compared with depth estimated from pre-trained network.

FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from models import build_model
from utils.misc import Infix
from .base_gan_metric_rgbd import BaseGANMetric

__all__ = ['GeneratedDepthEval', 'GeneratedDepthEval50K']


class GeneratedDepthEval(BaseGANMetric):
    """Defines the class for generated depth evaluation."""

    def __init__(self,
                 name='generateddeptheval',
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
        """Initializes the class with number of real/fakes samples for FID.

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

        # Build inception model for feature extraction.
        self.depthpred_model = build_model('DepthPredModel').eval()

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
        pbar_task = self.logger.add_pbar_task('GeneratedDepthEval', total=fake_num)
        all_loss = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = start + batch_size
            with torch.no_grad():
                if self.random_latents:
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
                generated_depth = results_d['image']
                batch_images = G_rgb(batch_codes_rgb, results_d, batch_labels, **G_rgb_kwargs)['image']
                predicted_depth = self.depthpred_model(batch_images)
                loss_depth = F.l1_loss((generated_depth+1)*0.5, predicted_depth/predicted_depth.max(), reduction='none') # ??
                gathered_loss_depth = self.gather_batch_results(loss_depth)
                self.append_batch_results(gathered_loss_depth, all_loss)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        all_losses = self.gather_all_results(all_loss)[:fake_num]
        if self.is_chief:
            all_losses = all_losses.mean()
            # assert all_losses.shape[0] == fake_num
        else:
            assert len(all_losses) == 0
            all_losses = None

        if G_d_mode:
            G_d.train()
        if G_rgb_mode:
            G_rgb.train()
        self.sync()
        return all_losses

    def evaluate(self, data_loader, generator, generator_kwargs, generator_d, generator_d_kwargs):
        losses = self.compute_depth_diff(generator, generator_kwargs, generator_d, generator_d_kwargs)
        if self.is_chief:
            result = {self.name: losses}
        else:
            assert losses is None
            result = None
        self.sync()
        return result


    def _is_better_than(self, metric_name, new, ref):
        """A lower FID is better.

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
        fid = float(result[self.name])
        assert isinstance(fid, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}{fid:.3f}.'
        else:
            msg = f'{prefix}{fid:.3f}, {log_suffix}.'
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
            self.tb_writer.add_scalar(f'Metrics/{self.name}', fid, tag)
            self.tb_writer.flush()
        self.sync()


class GeneratedDepthEval50K(GeneratedDepthEval):
    """Defines the class for FID50K metric computation.

    50_000 real/fake samples will be used for feature extraction.
    """

    def __init__(self,
                 name='GeneratedDepthEval50K',
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

