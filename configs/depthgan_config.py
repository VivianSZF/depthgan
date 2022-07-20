# python3.7
"""Configuration for training DepthGAN."""

from .base_config import BaseConfig

__all__ = ['DepthGANConfig']

RUNNER = 'DepthGANRunner'
DATASET = 'RGBDDataset'
DISCRIMINATOR = 'DepthGANDiscriminator'
GENERATOR_RGB = 'DepthGANGenerator_rgb'
GENERATOR_DEPTH = 'DepthGANGenerator_depth'
LOSS = 'DepthGANLoss'


class DepthGANConfig(BaseConfig):
    """Defines the configuration for training DepthGAN."""

    name = 'depthgan'
    hint = 'Train a DepthGAN model.'
    info = '''
To train a DepthGAN model, the recommended settings are as follows:

\b
- batch_size: 4 (for FF-HQ dataset, 8 GPU)
- val_batch_size: 16 (for FF-HQ dataset, 8 GPU)
- data_repeat: 200 (for FF-HQ dataset)
- total_img: 25_000_000 (for FF-HQ dataset)
- train_data_mirror: True (for FF-HQ dataset)
'''

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.config.runner_type = RUNNER

    @classmethod
    def get_options(cls):
        options = super().get_options()

        options['Data transformation settings'].extend([
            cls.command_option(
                '--resolution', type=cls.int_type, default=128,
                help='Resolution of the training images.'),
        ])

        options['Network settings'].extend([
            cls.command_option(
                '--g_init_res', type=cls.int_type, default=4,
                help='The initial resolution to start convolution with in '
                     'generator.'),
            cls.command_option(
                '--latent_dim_rgb', type=cls.int_type, default=512,
                help='The dimension of the RGB latent space.'),
            cls.command_option(
                '--latent_dim_depth', type=cls.int_type, default=512,
                help='The dimension of the depth latent space.'),
            cls.command_option(
                '--label_dim', type=cls.int_type, default=0,
                help='Number of classes in conditioning training. Set to `0` '
                     'to disable conditional training.'),
            cls.command_option(
                '--d_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'discriminator, which will be `factor * 32768`.'),
            cls.command_option(
                '--d_mbstd_groups', type=cls.int_type, default=8,
                help='Number of groups for MiniBatchSTD layer of '
                     'discriminator.'),
            cls.command_option(
                '--g_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'generator, which will be `factor * 32768`.'),
            cls.command_option(
                '--g_num_mappings', type=cls.int_type, default=8,
                help='Number of mapping layers of generator.'),
            cls.command_option(
                '--impl', type=str, default='cuda',
                help='Control the implementation of some neural operations.'),
            cls.command_option(
                '--num_fp16_res', type=cls.int_type, default=0,
                help='Number of (highest) resolutions that use `float16` '
                     'precision for training, which speeds up the training yet '
                     'barely affects the performance. The official '
                     'StyleGAN-ADA uses 4 by default.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--d_lr', type=cls.float_type, default=0.002,
                help='The learning rate of discriminator.'),
            cls.command_option(
                '--d_beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--d_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--g_lr', type=cls.float_type, default=0.002,
                help='The learning rate of generator.'),
            cls.command_option(
                '--g_beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1` for generator '
                     'optimizer.'),
            cls.command_option(
                '--g_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for generator '
                     'optimizer.'),
            cls.command_option(
                '--w_moving_decay', type=cls.float_type, default=0.995,
                help='Decay factor for updating `w_avg`.'),
            cls.command_option(
                '--sync_w_avg', type=cls.bool_type, default=False,
                help='Synchronizing the update of `w_avg` across replicas.'),
            cls.command_option(
                '--style_mixing_prob', type=cls.float_type, default=0.9,
                help='Probability to perform style mixing as a training '
                     'regularization.'),
            cls.command_option(
                '--r1_gamma', type=cls.float_type, default=1.0,
                help='Factor to control the strength of gradient penalty.'),
            cls.command_option(
                '--r1_interval', type=cls.int_type, default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--pl_batch_shrink', type=cls.int_type, default=2,
                help='Factor to reduce the batch size for perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--pl_weight', type=cls.float_type, default=2.0,
                help='Factor to control the strength of perceptual path length '
                     'regularization.'),
            cls.command_option(
                '--pl_decay', type=cls.float_type, default=0.01,
                help='Decay factor for perceptual path length regularization.'),
            cls.command_option(
                '--pl_interval', type=cls.int_type, default=4,
                help='Interval (in iterations) to perform perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--g_ema_img', type=cls.int_type, default=10_000,
                help='Factor for updating the smoothed generator, which is '
                     'particularly used for inference.'),
            cls.command_option(
                '--g_ema_rampup', type=cls.float_type, default=0.0,
                help='Rampup factor for updating the smoothed generator, which '
                     'is particularly used for inference. Set as `0` to '
                     'disable warming up.'),
            cls.command_option(
                '--use_ada', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--d_num_classes', type=cls.int_type, default=10,
                help='Number of classes for depth prediction.'),
            cls.command_option(
                '--gdloss', type=cls.float_type, default=0.001,
                help='Loss weight for generated depth supervision.'),
            cls.command_option(
                '--ddloss', type=cls.float_type, default=0.8,
                help='Loss weight for real depth supervision.'),
            cls.command_option(
                '--drotloss', type=cls.float_type, default=50,
                help='Loss weight for depth rotation loss.'),
            cls.command_option(
                '--rgbrotloss', type=cls.float_type, default=0.3,
                help='Loss weight for rgb rotation loss.'),
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'resolution', 'num_fp16_res', 'latent_dim', 'label_dim', 'd_lr',
            'g_lr', 'd_fmaps_factor', 'd_mbstd_groups', 'g_fmaps_factor',
            'g_num_mappings', 'g_ema_img', 'style_mixing_prob', 'use_ada',
            'r1_gamma', 'r1_interval', 'pl_weight', 'pl_interval'
        ])
        return recommended_opts

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')

        # Parse data transformation settings.
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.update(
            resolution=resolution, 
            crop_resize_resolution=resolution
        )
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.update(
            resolution=resolution,
            crop_resize_resolution=resolution
        )

        g_init_res = self.args.pop('g_init_res')
        d_init_res = 4  # This should be fixed as 4.
        latent_dim_rgb = self.args.pop('latent_dim_rgb')
        latent_dim_depth = self.args.pop('latent_dim_depth')
        label_dim = self.args.pop('label_dim')
        d_fmaps_base = int(self.args.pop('d_fmaps_factor') * (32 << 10))
        g_fmaps_base = int(self.args.pop('g_fmaps_factor') * (32 << 10))
        impl = self.args.pop('impl')
        num_fp16_res = self.args.pop('num_fp16_res')

        # Parse network settings and training settings.
        if not isinstance(num_fp16_res, int) or num_fp16_res <= 0:
            d_fp16_res = None
            g_fp16_res = None
            conv_clamp = None
        else:
            d_fp16_res = max(resolution // (2 ** (num_fp16_res - 1)),
                             d_init_res * 2)
            g_fp16_res = max(resolution // (2 ** (num_fp16_res - 1)),
                             g_init_res * 2)
            conv_clamp = 256

        d_lr = self.args.pop('d_lr')
        d_beta_1 = self.args.pop('d_beta_1')
        d_beta_2 = self.args.pop('d_beta_2')
        g_lr = self.args.pop('g_lr')
        g_beta_1 = self.args.pop('g_beta_1')
        g_beta_2 = self.args.pop('g_beta_2')
        r1_interval = self.args.pop('r1_interval')
        pl_interval = self.args.pop('pl_interval')

        if r1_interval is not None and r1_interval > 0:
            d_mb_ratio = r1_interval / (r1_interval + 1)
            d_lr = d_lr * d_mb_ratio
            d_beta_1 = d_beta_1 ** d_mb_ratio
            d_beta_2 = d_beta_2 ** d_mb_ratio
        if pl_interval is not None and pl_interval > 0:
            g_mb_ratio = pl_interval / (pl_interval + 1)
            g_lr = g_lr * g_mb_ratio
            g_beta_1 = g_beta_1 ** g_mb_ratio
            g_beta_2 = g_beta_2 ** g_mb_ratio
        
        g_num_mappings = self.args.pop('g_num_mappings')
        w_moving_decay = self.args.pop('w_moving_decay')
        sync_w_avg=self.args.pop('sync_w_avg')
        style_mixing_prob=self.args.pop('style_mixing_prob')
        g_ema_img=self.args.pop('g_ema_img')
        g_ema_rampup=self.args.pop('g_ema_rampup')
        num_classes = self.args.pop('d_num_classes')

        self.config.models.update(
            discriminator=dict(
                model=dict(model_type=DISCRIMINATOR,
                           resolution=resolution,
                           init_res=d_init_res,
                           fmaps_base=d_fmaps_base,
                           conv_clamp=conv_clamp,
                           mbstd_group=self.args.pop('d_mbstd_groups'),
                           num_classes=num_classes),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=d_lr,
                         betas=(d_beta_1, d_beta_2)),
                kwargs_train=dict(fp16_res=d_fp16_res, impl=impl),
                kwargs_val=dict(fp16_res=None, impl=impl)
            ),
            generator_rgb=dict(
                model=dict(model_type=GENERATOR_RGB,
                           resolution=resolution,
                           init_res=g_init_res,
                           z_dim=latent_dim_rgb,
                           mapping_layers=g_num_mappings,
                           fmaps_base=g_fmaps_base,
                           conv_clamp=conv_clamp),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                # Please turn off `fused_modulate` during training, which is
                # because the customized gradient computation omits weights, and
                # the fused operation will introduce division by 0.
                kwargs_train=dict(
                    w_moving_decay=w_moving_decay,
                    sync_w_avg=sync_w_avg,
                    style_mixing_prob=style_mixing_prob,
                    noise_mode='none',
                    fused_modulate=False,
                    fp16_res=g_fp16_res,
                    impl=impl),
                kwargs_val=dict(noise_mode='none',
                                fused_modulate=False,
                                fp16_res=None,
                                impl=impl),
                g_ema_img=g_ema_img,
                g_ema_rampup=g_ema_rampup
            ),
            generator_depth=dict(
                model=dict(model_type=GENERATOR_DEPTH,
                           resolution=resolution,
                           init_res=g_init_res,
                           z_dim=latent_dim_depth,
                           mapping_layers=g_num_mappings,
                           fmaps_base=g_fmaps_base,
                           conv_clamp=conv_clamp),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                # Please turn off `fused_modulate` during training, which is
                # because the customized gradient computation omits weights, and
                # the fused operation will introduce division by 0.
                kwargs_train=dict(
                    w_moving_decay=w_moving_decay,
                    sync_w_avg=sync_w_avg,
                    style_mixing_prob=style_mixing_prob,
                    noise_mode='none',
                    fused_modulate=False,
                    fp16_res=g_fp16_res,
                    impl=impl),
                kwargs_val=dict(noise_mode='none',
                                fused_modulate=False,
                                fp16_res=None,
                                impl=impl),
                g_ema_img=g_ema_img,
                g_ema_rampup=g_ema_rampup
            )
        )

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma'),
                               r1_interval=r1_interval,
                               num_classes=num_classes,
                               gdloss=self.args.pop('gdloss'),
                               ddloss=self.args.pop('ddloss')),
            g_loss_kwargs=dict(pl_batch_shrink=self.args.pop('pl_batch_shrink'),
                               pl_weight=self.args.pop('pl_weight'),
                               pl_decay=self.args.pop('pl_decay'),
                               pl_interval=pl_interval,
                               drotloss=self.args.pop('drotloss'),
                               rgbrotloss=self.args.pop('rgbrotloss'))
        )


        if self.args.pop('use_ada'):
            self.config.aug.update(
                aug_type='AdaAug',
                # Default augmentation strategy adopted by StyleGAN2-ADA.
                xflip=1,
                rotate90=1,
                xint=1,
                scale=1,
                rotate=1,
                aniso=1,
                xfrac=1,
                brightness=1,
                contrast=1,
                lumaflip=1,
                hue=1,
                saturation=1,
                imgfilter=0,
                noise=0,
                cutout=0
            )
            self.config.aug_kwargs.update(impl='cuda')
            self.config.controllers.update(
                AdaAugController=dict(
                    every_n_iters=4,
                    init_p=0.0,
                    target_p=0.6,
                    speed_img=500_000,
                    strategy='adaptive'
                )
            )

        self.config.metrics.update(
            RotEval50K=dict(
                init_kwargs=dict(name='RotEval50K', latent_dim_rgb=latent_dim_rgb, latent_dim_depth=latent_dim_depth),
                eval_kwargs=dict(
                    generator_rgb_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                    generator_depth_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            ),
            RealDepthEval50K=dict(
                init_kwargs=dict(name='RealDepthEval50K', latent_dim_rgb=latent_dim_rgb, latent_dim_depth=latent_dim_depth),
                eval_kwargs=dict(
                    discriminator=dict(impl=impl,
                                       fp16_res=None),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            ),
            GeneratedDepthEval50K=dict(
                init_kwargs=dict(name='GeneratedDepthEval50K', latent_dim_rgb=latent_dim_rgb, latent_dim_depth=latent_dim_depth),
                eval_kwargs=dict(
                    generator_rgb_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                    generator_depth_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            ),
            FID50KRGB=dict(
                init_kwargs=dict(name='fid50k_rgb', latent_dim_rgb=latent_dim_rgb, latent_dim_depth=latent_dim_depth),
                eval_kwargs=dict(
                    generator_rgb_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                    generator_depth_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                ),
                interval=None,
                first_iter=None,
                save_best=True
            ),
            FID50KDepth=dict(
                init_kwargs=dict(name='fid50k_depth', latent_dim_rgb=latent_dim_rgb, latent_dim_depth=latent_dim_depth),
                eval_kwargs=dict(
                    generator_rgb_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                    generator_depth_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            ),
            GANSnapshotRGBD=dict(
                init_kwargs=dict(name='snapshot', latent_dim_rgb=latent_dim_rgb, latent_dim_depth=latent_dim_depth,
                                 latent_num=32),
                eval_kwargs=dict(
                    generator_rgb_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                    generator_depth_smooth=dict(noise_mode='none',
                                          fused_modulate=False,
                                          impl=impl,
                                          fp16_res=None),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            ),
        )
