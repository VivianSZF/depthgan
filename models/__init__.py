# python3.7
"""Collects all models."""

from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .stylegan3_generator import StyleGAN3Generator
from .ghfeat_encoder import GHFeatEncoder
from .depthgan_generator import DepthGANGenerator_depth, DepthGANGenerator_rgb
from .depthgan_discriminator import DepthGANDiscriminator
from .perceptual_model import PerceptualModel
from .inception_model import InceptionModel
from .depthpred_model import DepthPredModel

__all__ = ['build_model']

_MODELS = {
    'PGGANGenerator': PGGANGenerator,
    'PGGANDiscriminator': PGGANDiscriminator,
    'StyleGANGenerator': StyleGANGenerator,
    'StyleGANDiscriminator': StyleGANDiscriminator,
    'StyleGAN2Generator': StyleGAN2Generator,
    'StyleGAN2Discriminator': StyleGAN2Discriminator,
    'StyleGAN3Generator': StyleGAN3Generator,
    'GHFeatEncoder': GHFeatEncoder,
    'DepthGANGenerator_depth': DepthGANGenerator_depth,
    'DepthGANGenerator_rgb': DepthGANGenerator_rgb, 
    'DepthGANDiscriminator': DepthGANDiscriminator,
    'PerceptualModel': PerceptualModel.build_model,
    'InceptionModel': InceptionModel.build_model,
    'DepthPredModel': DepthPredModel.build_model
}


def build_model(model_type, **kwargs):
    """Builds a model based on its class type.

    Args:
        model_type: Class type to which the model belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the model.

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[model_type](**kwargs)
