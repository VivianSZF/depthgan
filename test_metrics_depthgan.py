# python3.7
"""Test metrics.

NOTE: This file can be used as an example for distributed inference/evaluation.
This file only supports testing GAN related metrics (including FID, IS, KID,
GAN precision-recall, saving snapshot, and equivariance) by loading a
pre-trained generator. To test more metrics, please customize your own script.
"""

import argparse
from datasets.rgbd_dataset import crop_resize_image

import torch
import math

from datasets import build_dataset
from models import build_model
from metrics import build_metric
from utils.loggers import build_logger
from utils.parsing_utils import parse_bool
from utils.parsing_utils import parse_json
from utils.dist_utils import init_dist
from utils.dist_utils import exit_dist


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run metric test.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset used for metric computation.')
    parser.add_argument('--annotation_path', type=str, default=None,
                        help='Path to annotations of datasets.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--work_dir', type=str,
                        default='work_dirs/metric_tests',
                        help='Working directory for metric test. (default: '
                             '%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed. (default: %(default)s)')
    parser.add_argument('--real_num', type=int, default=-1,
                        help='Number of real data used for testing. Negative '
                             'means using all data. (default: %(default)s)')
    parser.add_argument('--fake_num', type=int, default=1000,
                        help='Number of fake data used for testing. (default: '
                             '%(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size used for metric computation. '
                             '(default: %(default)s)')
    parser.add_argument('--test_fid_rgb', type=parse_bool, default=False,
                        help='Whether to test FID on RGB images. (default: %(default)s)')
    parser.add_argument('--test_fid_depth', type=parse_bool, default=False,
                        help='Whether to test FID on depth images. (default: %(default)s)')
    parser.add_argument('--test_kid', type=parse_bool, default=False,
                        help='Whether to test KID. (default: %(default)s)')
    parser.add_argument('--test_snapshot', type=parse_bool, default=False,
                        help='Whether to test saving snapshot. '
                             '(default: %(default)s)')
    parser.add_argument('--test_rotation', type=parse_bool, default=False,
                        help='Whether to test rotation-related metrics. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_depth', type=float, default=1,
                         help='truncation for depth')
    parser.add_argument('--trunc_rgb', type=float, default=1,
                         help='truncation for rgb')
    parser.add_argument('--fix_depth', type=parse_bool, default=False,
                        help='Whether to fix the latent code for depth. '
                             '(default: %(default)s)')
    parser.add_argument('--fix_rgb', type=parse_bool, default=False,
                        help='Whether to fix the latent code for rgb. '
                             '(default: %(default)s)')
    parser.add_argument('--fix_angle', type=parse_bool, default=False,
                        help='Whether to fix the angles. '
                             '(default: %(default)s)')
    parser.add_argument('--fix_all', type=parse_bool, default=False,
                        help='Whether to fix all codes(to test stylemixing). '
                             '(default: %(default)s)')
    parser.add_argument('--interpolate_depth', type=parse_bool, default=False,
                        help='Whether to interpolate in the latent space for depth. '
                             '(default: %(default)s)')
    parser.add_argument('--interpolate_rgb', type=parse_bool, default=False,
                        help='Whether to interpolate in the latent space for rgb. '
                             '(default: %(default)s)')
    parser.add_argument('--frame_size', type=int, default=224,
                        help='Frame size of the video to save. (default: '
                             '%(default)s)')
    parser.add_argument('--video_save', type=parse_bool, default=False,
                        help='Whether to save videos. '
                             '(default: %(default)s)')
    parser.add_argument('--image_save', type=parse_bool, default=True,
                        help='Whether to save images. '
                             '(default: %(default)s)')
    parser.add_argument('--save_separate', type=parse_bool, default=False,
                        help='Whether to save images separately. '
                             '(default: %(default)s)')
    parser.add_argument('--launcher', type=str, default='pytorch',
                        choices=['pytorch', 'slurm'],
                        help='Distributed launcher. (default: %(default)s)')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo', 'mpi'],
                        help='Distributed backend. (default: %(default)s)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Replica rank on the current node. This field is '
                             'required by `torch.distributed.launch`. '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize distributed environment.
    init_dist(launcher=args.launcher, backend=args.backend)

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    state = torch.load(args.model)
    G_depth = build_model(**state['model_kwargs_init']['generator_depth_smooth'])
    G_depth.load_state_dict(state['models']['generator_depth_smooth'])
    G_depth.eval().cuda()
    G_depth_kwargs = dict(trunc_psi=args.trunc_depth)
    G_rgb = build_model(**state['model_kwargs_init']['generator_rgb_smooth'])
    G_rgb.load_state_dict(state['models']['generator_rgb_smooth'])
    G_rgb.eval().cuda()
    G_rgb_kwargs = dict(trunc_psi=args.trunc_rgb)

    dataset_kwargs = dict(dataset_type='RGBDDataset',
                          root_dir=args.dataset,
                          resolution=G_depth.resolution,
                          annotation_path=args.annotation_path,
                          annotation_meta=None,
                          max_samples=args.real_num,
                          mirror=False,
                          crop_resize_resolution=G_depth.resolution,
                          transform_kwargs=None)
    data_loader_kwargs = dict(data_loader_type='iter',
                              repeat=1,
                              num_workers=4,
                              prefetch_factor=2,
                              pin_memory=True)
    data_loader = build_dataset(for_training=False,
                                batch_size=args.batch_size,
                                dataset_kwargs=dataset_kwargs,
                                data_loader_kwargs=data_loader_kwargs)

    if torch.distributed.get_rank() == 0:
        logger = build_logger('normal', logfile=None, verbose_log=True)
    else:
        logger = build_logger('dummy')

    real_num = (len(data_loader.dataset)
                if args.real_num < 0 else args.real_num)
    
    angle = math.pi/12
    if args.test_fid_rgb:
        logger.info('========== Test FID RGB ==========')
        metric = build_metric('FIDMetricRGB',
                              name=f'fid_rgb{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              a_dis=torch.distributions.uniform.Uniform(-angle, angle),
                              latent_dim_rgb=G_rgb.latent_dim,
                              latent_dim_depth=G_depth.latent_dim,
                              label_dim=G_rgb.label_size,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G_rgb, G_rgb_kwargs, G_depth, G_depth_kwargs)
        metric.save(result)
    if args.test_fid_depth:
        logger.info('========== Test FID Depth==========')
        metric = build_metric('FIDMetricDepth',
                              name=f'fid_depth{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              a_dis=torch.distributions.uniform.Uniform(-angle, angle),
                              latent_dim_rgb=G_rgb.latent_dim,
                              latent_dim_depth=G_depth.latent_dim,
                              label_dim=G_rgb.label_size,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G_rgb, G_rgb_kwargs, G_depth, G_depth_kwargs)
        metric.save(result)
    
    if args.test_snapshot:
        logger.info('========== Test GAN Snapshot ==========')
        metric = build_metric('GANSnapshotRGBD',
                              name='snapshot',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              a_dis=torch.distributions.uniform.Uniform(-angle, angle),
                              latent_dim_rgb=G_rgb.latent_dim,
                              latent_dim_depth=G_depth.latent_dim,
                              label_dim=G_rgb.label_size,
                              latent_num=args.fake_num,
                              equal_interval=True,
                              keep_same_depth=True,
                              keep_same_rgb=True,
                              fix_depth=args.fix_depth,
                              fix_rgb=args.fix_rgb,
                              fix_angle=args.fix_angle,
                              fix_all=args.fix_all,
                              interpolate_depth=args.interpolate_depth,
                              interpolate_rgb=args.interpolate_rgb,
                              seed=args.seed,
                              test=True,
                              image_save=args.image_save,
                              save_separate=args.save_separate,
                              video_save=args.video_save,
                              frame_size=(args.frame_size, args.frame_size)
                              )
        result = metric.evaluate(data_loader, G_rgb, G_rgb_kwargs, G_depth, G_depth_kwargs)
        metric.save(result)
    
    if args.test_rotation:
        logger.info('========== Test Rotation ==========')
        metric = build_metric('RotEval',
                              name=f'rotation{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              a_dis=torch.distributions.uniform.Uniform(-angle, angle),
                              latent_dim_rgb=G_rgb.latent_dim,
                              latent_dim_depth=G_depth.latent_dim,
                              label_dim=G_rgb.label_size,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G_rgb, G_rgb_kwargs, G_depth, G_depth_kwargs)
        metric.save(result)
    

    # Exit distributed environment.
    exit_dist()


if __name__ == '__main__':
    main()
