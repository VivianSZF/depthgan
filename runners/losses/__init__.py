# python3.7
"""Collects all loss functions."""

from .stylegan_loss import StyleGANLoss
from .stylegan2_loss import StyleGAN2Loss
from .stylegan3_loss import StyleGAN3Loss
from .depthgan_loss import DepthGANLoss

__all__ = ['build_loss']

_LOSSES = {
    'StyleGANLoss': StyleGANLoss,
    'StyleGAN2Loss': StyleGAN2Loss,
    'StyleGAN3Loss': StyleGAN3Loss,
    'DepthGANLoss': DepthGANLoss
}


def build_loss(runner, loss_type, **kwargs):
    """Builds a loss based on its class type.

    Args:
        runner: The runner on which the loss is built.
        loss_type: Class type to which the loss belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the loss.

    Raises:
        ValueError: If the `loss_type` is not supported.
    """
    if loss_type not in _LOSSES:
        raise ValueError(f'Invalid loss type: `{loss_type}`!\n'
                         f'Types allowed: {list(_LOSSES)}.')
    return _LOSSES[loss_type](runner, **kwargs)
