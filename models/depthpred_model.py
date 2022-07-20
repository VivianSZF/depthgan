"""
    Depth estimation model from 'Learning to Recover 3D Scene Shape from a Single Image'
"""
import os.path
import warnings
import hashlib

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnext import resnext101_32x8d

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist
from collections import OrderedDict

from utils.misc import download_url

__all__ = ['DepthPredModel']

_MODEL_URL_SHA256 = {
    # This model is provided by `torchvision`, which is ported from TensorFlow.
    'onedrive': (
        'https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download',
        '1d696b2ef3e8336b057d0c15bc82d2fecef821bfebe5ef9d7671a5ec5dde520b'  # hash sha256
    )
}


class DepthPredModel(object):
    """Defines the RelDepth prediction model.

    This is a static class, which is used to avoid this model to be built
    repeatedly. Consequently, this model is particularly used for inference,
    like computing FID. If training is required, please implement by yourself.

    NOTE: The pre-trained model assumes the inputs to be with `RGB` channel
    order and pixel range [-1, 1].
    """
    models = dict()

    @staticmethod
    def build_model():
        """Builds the model and load pre-trained weights.

        The built model supports following arguments when forwarding:

        - transform_input: Whether to transform the input back to pixel range
            (-1, 1). Please disable this argument if your input is already with
            pixel range (-1, 1). (default: False)
        - output_logits: Whether to output the categorical logits instead of
            features. (default: False)
        - remove_logits_bias: Whether to remove the bias when computing the
            logits. The official implementation removes the bias by default.
            Please refer to
            `https://github.com/openai/improved-gan/blob/master/inception_score/model.py`.
            (default: False)
        - output_predictions: Whether to output the final predictions, i.e.,
            `softmax(logits)`. (default: False)
        """
        model_source = 'onedrive'

        fingerprint = model_source

        if fingerprint not in DepthPredModel.models:
            if dist.is_initialized() and dist.get_rank() != 0:
                dist.barrier()  # Make sure the weight is downloaded only once.

            # Build model.
            depth_model = RelDepthModel(backbone='resnext101')
            # Get (download if needed) pre-trained weights.
            url, sha256= _MODEL_URL_SHA256[fingerprint]
            filename = f'depthpref_model_{model_source}_{sha256}.pth'
            model_path, hash_check = download_url(url,
                                                  filename=filename,
                                                  sha256=sha256)
            state_dict = torch.load(model_path, map_location='cpu')
            if hash_check is False:
                warnings.warn(f'Hash check failed! The remote file from URL '
                              f'`{url}` may be changed, or the downloading is '
                              f'interrupted. The loaded inception model may '
                              f'have unexpected behavior.')
            
            if dist.is_initialized() and dist.get_rank() == 0:
                dist.barrier()  # Wait for other replicas to build.

            # Load weights.
            depth_model.load_state_dict(strip_prefix_if_present(state_dict['depth_model'], 'module.'), strict=True)
            del state_dict
            # Move model to GPU, set `eval` mode and ignores gradient.
            depth_model.eval().requires_grad_(False).cuda()

            DepthPredModel.models[fingerprint] = depth_model
            

        return DepthPredModel.models[fingerprint]

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class RelDepthModel(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(RelDepthModel, self).__init__()
        if backbone == 'resnet50':
            encoder = 'resnet50_stride32'
        elif backbone == 'resnext101':
            encoder = 'resnext101_stride32x8d'
        self.depth_model = DepthModel(encoder)

    def forward(self, rgb):
        rgb = (rgb + 1.0) * 0.5 # Change the pixel range from [-1, 1] to [0, 1]
        with torch.no_grad():
            input = rgb.cuda()
            depth = self.depth_model(input)
            pred_depth_out = depth - depth.min() + 0.01
            return pred_depth_out


class DepthModel(nn.Module):
    def __init__(self, encoder):
        super(DepthModel, self).__init__()
        if encoder == 'resnet50_stride32':
            self.encoder_modules = resnet50_stride32()
        elif encoder == 'resnext101_stride32x8d':
            self.encoder_modules = resnext101_stride32x8d()
        self.decoder_modules = Decoder()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit = self.decoder_modules(lateral_out)
        return out_logit


def resnet50_stride32():
    return DepthNet(backbone='resnet', depth=50, upfactors=[2, 2, 2, 2])

def resnext101_stride32x8d():
    return DepthNet(backbone='resnext101_32x8d', depth=101, upfactors=[2, 2, 2, 2])


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.inchannels =  [256, 512, 1024, 2048]
        self.midchannels = [256, 256, 256, 512]
        self.upfactors = [2,2,2,2]
        self.outchannels = 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)
        
        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])
        
        self.outconv = AO(inchannels=self.midchannels[0], outchannels=self.outchannels, upfactor=2)
        self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, features):
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16

        x_8 = self.ffm2(features[2], x_16)  # 1/8
        x_4 = self.ffm1(features[1], x_8)  # 1/4
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        #-----------------------------------------
        x = self.outconv(x_2)  # original size
        return x

class DepthNet(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152
    }
    def __init__(self,
                 backbone='resnet',
                 depth=50,
                 upfactors=[2, 2, 2, 2]):
        super(DepthNet, self).__init__()
        self.backbone = backbone
        self.depth = depth
        self.pretrained = False
        self.inchannels = [256, 512, 1024, 2048]
        self.midchannels = [256, 256, 256, 512]
        self.upfactors = upfactors
        self.outchannels = 1

        # Build model
        if self.backbone == 'resnet':
            if self.depth not in DepthNet.__factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = DepthNet.__factory[depth](pretrained=self.pretrained)
        elif self.backbone == 'resnext101_32x8d':
            self.encoder = resnext101_32x8d(pretrained=self.pretrained)

    def forward(self, x):
        x = self.encoder(x)  # 1/32, 1/16, 1/8, 1/4
        return x


class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels
        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1,
                               bias=True)
        # NN.BatchNorm2d
        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True), \
                                         nn.BatchNorm2d(num_features=self.mid), \
                                         nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True))
        self.relu = nn.ReLU(inplace=True)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ATA(nn.Module):
    def __init__(self, inchannels, reduction=8):
        super(ATA, self).__init__()
        self.inchannels = inchannels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.inchannels * 2, self.inchannels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.inchannels // reduction, self.inchannels),
                                nn.Sigmoid())
        self.init_params()

    def forward(self, low_x, high_x):
        n, c, _, _ = low_x.size()
        x = torch.cat([low_x, high_x], 1)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.fc(x).view(n, c, 1, 1)
        x = low_x * x + high_x

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.normal(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        # self.ata = ATA(inchannels = self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

        self.init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ReLU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

        self.init_params()

    def forward(self, x):
        x = self.adapt_conv(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



# ==============================================================================================================


class ResidualConv(nn.Module):
    def __init__(self, inchannels):
        super(ResidualConv, self).__init__()
        # NN.BatchNorm2d
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(num_features=inchannels),
            nn.ReLU(inplace=False),
            # nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1, stride=1, groups=inchannels,bias=True),
            # nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels / 2, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=inchannels / 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=inchannels / 2, out_channels=inchannels, kernel_size=3, padding=1, stride=1,
                      bias=False)
        )
        self.init_params()

    def forward(self, x):
        x = self.conv(x) + x
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FeatureFusion(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(FeatureFusion, self).__init__()
        self.conv = ResidualConv(inchannels=inchannels)
        # NN.BatchNorm2d
        self.up = nn.Sequential(ResidualConv(inchannels=inchannels),
                                nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                                   stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(num_features=outchannels),
                                nn.ReLU(inplace=True))

    def forward(self, lowfeat, highfeat):
        return self.up(highfeat + self.conv(lowfeat))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class SenceUnderstand(nn.Module):
    def __init__(self, channels):
        super(SenceUnderstand, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(nn.Linear(512 * 8 * 8, self.channels),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True))
        self.initial_params()

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = x.view(n, self.channels, 1, 1)
        x = self.conv2(x)
        x = x.repeat(1, 1, h, w)
        return x

    def initial_params(self, dev=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, dev)
