# python3.7
"""Contains the runner for DepthGAN."""

from copy import deepcopy
import torch
import math
from metrics import build_metric

from .base_runner import BaseRunner

__all__ = ['DepthGANRunner']


class DepthGANRunner(BaseRunner):
    """Defines the runner for DepthGAN."""

    def __init__(self, config):
        self._a_dis = torch.distributions.uniform.Uniform(-math.pi/12, math.pi/12)
        super().__init__(config)

    
    @property
    def a_dis(self):
        """Returns the rank of the current runner."""
        return self._a_dis
    
    def build_metrics(self):
        """Builds metrics used for model evaluation."""

        self.logger.info('Building metrics ...')
        for metric_type, metric_config in self.config.metrics.items():
            metric = dict()

            # Settings for metric computation.
            init_kwargs = metric_config.get('init_kwargs', dict())
            init_kwargs['work_dir'] = self.result_dir
            init_kwargs['logger'] = self.logger
            init_kwargs['tb_writer'] = self.tb_writer
            init_kwargs['a_dis'] = self._a_dis
            if 'batch_size' not in init_kwargs:
                init_kwargs['batch_size'] = self.val_batch_size
            metric['fn'] = build_metric(metric_type, **init_kwargs)

            # Evaluation kwargs should be a dictionary, where each key stands
            # for a model name in `self.models`, specifying the model to test,
            # while each value contains the runtime kwargs for model forward,
            # specifying the model behavior when testing.
            eval_kwargs = metric_config.get('eval_kwargs', dict())
            for key, val in eval_kwargs.items():
                assert isinstance(key, str)
                assert isinstance(val, dict)
            metric['kwargs'] = eval_kwargs

            # Evaluation interval.
            metric['interval'] = metric_config.get('interval', None)
            metric['first_iter'] = metric_config.get('first_iter', None)
            interval = metric['interval']
            first_iter = metric['first_iter']

            # Settings for saving best checkpoint.
            metric['save_best'] = metric_config.get('save_best', None)
            metric['save_running_metadata'] = metric_config.get(
                'save_running_metadata', None)
            metric['save_optimizer'] = metric_config.get('save_optimizer', None)
            metric['save_learning_rate'] = metric_config.get(
                'save_learning_rate', None)
            metric['save_loss'] = metric_config.get('save_loss', None)
            metric['save_augment'] = metric_config.get('save_augment', None)
            metric['save_running_stats'] = metric_config.get(
                'save_running_stats', None)
            save_best = metric['save_best']
            save_running_metadata = metric['save_running_metadata']
            save_optimizer = metric['save_optimizer']
            save_learning_rate = metric['save_learning_rate']
            save_loss = metric['save_loss']
            save_augment = metric['save_augment']
            save_running_stats = metric['save_running_stats']

            self.metrics[metric['fn'].name] = metric

            self.logger.info(metric_type, indent_level=1)
            for key, val in metric['fn'].info().items():
                self.logger.info(f'{key}: {val}', indent_level=2)
            self.logger.info(f'Evaluation interval: {interval}', indent_level=2)
            self.logger.info(f'Evaluate on first iter: {first_iter}',
                             indent_level=2)
            if save_best:
                self.logger.info('Save the best checkpoint:', indent_level=2)
                self.logger.info(
                    f'Saving running metadata: {save_running_metadata}',
                    indent_level=3)
                self.logger.info(f'Saving optimizer state: {save_optimizer}',
                                 indent_level=3)
                self.logger.info(
                    f'Saving learning rate scheduler: {save_learning_rate}',
                    indent_level=3)
                self.logger.info(f'Saving loss: {save_loss}', indent_level=3)
                self.logger.info(f'Saving augment: {save_augment}',
                                 indent_level=3)
                self.logger.info(f'Saving running stats: {save_running_stats}',
                                 indent_level=3)
            else:
                self.logger.info('Do not save the best checkpoint.',
                                 indent_level=2)
        self.logger.info('Finish building metrics.\n')

    def build_models(self):
        super().build_models()

        self.g_ema_img = self.config.models['generator_depth'].get(
            'g_ema_img', 10000)
        self.g_ema_rampup = self.config.models['generator_depth'].get(
            'g_ema_rampup', None)
        if 'generator_rgb_smooth' not in self.models:
            self.models['generator_rgb_smooth'] = deepcopy(self.models['generator_rgb'])
            self.model_kwargs_init['generator_rgb_smooth'] = deepcopy(
                self.model_kwargs_init['generator_rgb'])
        if 'generator_depth_smooth' not in self.models:
            self.models['generator_depth_smooth'] = deepcopy(self.models['generator_depth'])
            self.model_kwargs_init['generator_depth_smooth'] = deepcopy(
                self.model_kwargs_init['generator_depth'])
        if 'generator_rgb_smooth' not in self.model_kwargs_val:
            self.model_kwargs_val['generator_rgb_smooth'] = deepcopy(
                self.model_kwargs_val['generator_rgb'])
        if 'generator_depth_smooth' not in self.model_kwargs_val:
            self.model_kwargs_val['generator_depth_smooth'] = deepcopy(
                self.model_kwargs_val['generator_depth'])

    def build_loss(self):
        super().build_loss()
        self.running_stats.add('Misc/Gs Beta',
                               log_name='Gs_beta',
                               log_format='.4f',
                               log_strategy='CURRENT')

    def train_step(self, data, **train_kwargs):
        # Update generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator_depth'].requires_grad_(True)
        self.models['generator_rgb'].requires_grad_(True)
        # Update with adversarial loss.
        g_loss = self.loss.g_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator_depth')
        self.zero_grad_optimizer('generator_rgb')
        g_loss.backward()
        self.step_optimizer('generator_depth')
        self.step_optimizer('generator_rgb')
        # Update with perceptual path length regularization if needed.
        pl_penalty = self.loss.g_reg(self, data, sync=True)
        if pl_penalty is not None:
            self.zero_grad_optimizer('generator_depth')
            self.zero_grad_optimizer('generator_rgb')
            pl_penalty.backward()
            self.step_optimizer('generator_depth')
            self.step_optimizer('generator_rgb')
        
        # Update generator_depth with Depth Rotation loss
        self.models['discriminator'].requires_grad_(False)
        self.models['generator_depth'].requires_grad_(True)
        self.models['generator_rgb'].requires_grad_(False)
        d_rotation_loss = self.loss.d_rot_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator_depth')
        d_rotation_loss.backward()
        self.step_optimizer('generator_depth')

        # Update the rgb generator
        self.models['discriminator'].requires_grad_(False)
        self.models['generator_depth'].requires_grad_(False)
        self.models['generator_rgb'].requires_grad_(True)
        # Update with rgb rotation loss
        rgb_rotation_loss = self.loss.rgb_rot_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator_rgb')
        rgb_rotation_loss.backward()
        self.step_optimizer('generator_rgb')
        # Update with depth estimation loss
        g_depth_loss = self.loss.g_depth_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator_rgb')
        g_depth_loss.backward()
        self.step_optimizer('generator_rgb')

        # Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator_depth'].requires_grad_(False)
        self.models['generator_rgb'].requires_grad_(False)
        self.zero_grad_optimizer('discriminator')
        # Update with fake images (get synchronized together with real loss).
        d_fake_loss = self.loss.d_fake_loss(self, data, sync=False)
        d_fake_loss.backward()
        # Update with real images.
        d_real_loss = self.loss.d_real_loss(self, data, sync=True)
        d_real_loss.backward()
        self.step_optimizer('discriminator')
        # Update with depth estimation.
        d_depth_loss = self.loss.d_depth_loss(self, data, sync=True)
        self.zero_grad_optimizer('discriminator')
        d_depth_loss.backward()
        self.step_optimizer('discriminator')
        # Update with gradient penalty.
        r1_penalty = self.loss.d_reg(self, data, sync=True)
        if r1_penalty is not None:
            self.zero_grad_optimizer('discriminator')
            r1_penalty.backward()
            self.step_optimizer('discriminator')

        # Life-long update generator.
        if self.g_ema_rampup is not None and self.g_ema_rampup > 0:
            g_ema_img = min(self.g_ema_img, self.seen_img * self.g_ema_rampup)
        else:
            g_ema_img = self.g_ema_img
        beta = 0.5 ** (self.minibatch / max(g_ema_img, 1e-8))
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator_rgb'],
                          avg=self.models['generator_rgb_smooth'],
                          beta=beta)
        self.smooth_model(src=self.models['generator_depth'],
                          avg=self.models['generator_depth_smooth'],
                          beta=beta)
