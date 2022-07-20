# python3.7
"""Contains the class to evaluate discriminator on real depth images.

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

__all__ = ['RealDepthEval', 'RealDepthEval50K']



class RealDepthEval(BaseGANMetric):
    """Defines the class for discriminator evaluation on real depth images."""

    def __init__(self,
                 name='realdeptheval',
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

    def compute_depth_diff(self, data_loader, discriminator, discriminator_kwargs):
        """Extracts inception features from real data."""
        if self.real_num < 0:
            real_num = len(data_loader.dataset)
        else:
            real_num = min(self.real_num, len(data_loader.dataset))

        D = discriminator
        D_mode = D.training
        D.eval()
        D_kwargs = discriminator_kwargs

        self.logger.info(f'Extracting inception features from real data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('RealDepthEval', total=real_num)
        batch_size = data_loader.batch_size
        replica_num = self.get_replica_num(real_num)
        all_loss = []
        for batch_idx in range(len(data_loader)):
            if batch_idx * batch_size >= replica_num:
                # NOTE: Here, we always go through the entire dataset to make
                # sure the next evaluator can visit the data loader from the
                # beginning.
                _batch_data = next(data_loader)
                continue
            with torch.no_grad():
                batch_data = next(data_loader)['image'].cuda().detach()
                predicted_depth = D(batch_data[:,:3])['depthimage']
                num_classes=predicted_depth.shape[1]
                loss_depth = F.cross_entropy(predicted_depth, torch.floor((batch_data[:,3]+1)/2*num_classes).clamp(0,num_classes-1).long(), reduction='none')
                gathered_loss_depth = self.gather_batch_results(loss_depth)
                self.append_batch_results(gathered_loss_depth, all_loss)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        all_losses = self.gather_all_results(all_loss)[:real_num]
        if self.is_chief:
            all_losses = all_losses.mean()
        else:
            assert len(all_losses) == 0
            all_losses = None

        if D_mode:
            D.train()
        self.sync()
        return all_losses


    def evaluate(self, data_loader, disciminator, discriminator_kwargs):
        losses = self.compute_depth_diff(data_loader, disciminator, discriminator_kwargs)
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


class RealDepthEval50K(RealDepthEval):
    """Defines the class for FID50K metric computation.

    50_000 real/fake samples will be used for feature extraction.
    """

    def __init__(self,
                 name='RealDepthEval50K',
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

