import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
import torchvision.models as models
import numpy as np

from .utils import export, parameter_count
from .CycledViewProjection import CycledViewProjection
from .CrossViewTransformer import CrossViewTransformer


@export
def cityscapes_vae(pretrained=False, **kwargs):
    assert not pretrained
    model = vae_mapping()
    return model



def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class upsample(nn.Module):

    def __init__(self, if_deconv, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)

        return x

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class encoder_after_vgg(nn.Module):

    def __init__(self):
        super(encoder_after_vgg, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.mu_dec = nn.Linear(4096, 512)
        self.logvar_dec = nn.Linear(4096, 512)

    def forward(self, x):  #x输入（-1, 512, 8, 16）
        x = self.conv(x)  #（-1, 128, 4, 8）
        x = x.view(-1, 4096)
        mu = self.mu_dec(x)  #（-1, 512）
        logvar = self.logvar_dec(x)  #（-1, 512）

        return mu, logvar   #生成均值和方差

class decoder_conv(nn.Module):
    def __init__(self, if_deconv):
        super(decoder_conv, self).__init__()

        self.up1 = upsample(if_deconv=if_deconv, channels=128)
        self.conv1 = double_conv(128, 256)
        self.up2 = upsample(if_deconv=if_deconv, channels=256)
        self.conv2 = double_conv(256, 256)
        self.up3 = upsample(if_deconv=if_deconv, channels=256)
        self.conv3 = double_conv(256, 256)
        self.up4 = upsample(if_deconv=if_deconv, channels=256)
        self.conv4 = double_conv(256, 256)
        self.up5 = upsample(if_deconv=if_deconv, channels=256)
        self.conv5 = double_conv(256, 256)
        self.conv_out = nn.Conv2d(256, 4, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):  #输入x(-1, 512)
        x = x.view(-1, 128, 2, 2)
        x = self.up1(x)    #(-1, 128, 4, 4)
        x = self.conv1(x)  #(-1, 256, 4, 4)

        x = self.up2(x)  #(-1, 256, 8, 8)
        x = self.conv2(x)  #(-1, 256, 8, 8)

        x = self.up3(x)  #(-1, 256, 16, 16)
        x = self.conv3(x)  #(-1, 256, 16, 16)

        x = self.up4(x)   #(-1, 256, 32, 32)
        x = self.conv4(x)   #(-1, 256, 32, 32)

        x = self.up5(x)   #(-1, 256, 64, 64)
        x = self.conv5(x)  #(-1, 256, 64, 64)

        #x = self.conv_out(x)   #(-1, 4, 64, 64)

        return x

class vae_mapping(nn.Module):

    def __init__(self):
        super(vae_mapping, self).__init__()

        self.vgg16 = models.vgg16_bn(pretrained=True)
        self.vgg16_feature = nn.Sequential(*list(self.vgg16.features.children())[:])
        self.encoder_afterv_vgg = encoder_after_vgg()
        self.decoder = decoder_conv(if_deconv=True)
        self.cnn1 = nn.Conv2d(256, 4, 3, padding=1)
        self.cnn2 = nn.Conv2d(256, 4, 3, padding=1)
        

    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # 用一个标准正态分布乘标准差，再加上均值，使隐含向量变为正太分布
        else:
            return mu

    def forward(self, x, is_training, defined_mu=None):
        x = self.vgg16_feature(x)   #编码 vgg输出（-1, 512, 8, 16）
        #print("in vae_nets vgg16_feature size: ", x.size())
        mu, logvar = self.encoder_afterv_vgg(x)  #编码  均值mu(-1, 512)  方差logvar(-1, 512)
        z = self.reparameterize(is_training, mu, logvar)   #重新参数化成正态分布  z(-1, 512)
        #print("in vae_nets z size: ", z.size())
        if defined_mu is not None:
            z = defined_mu
        z = self.decoder(z)   #pred_map(-1, 4, 64, 64)
        pred_map1 = self.cnn1(z)
        pred_map2 = self.cnn2(z)

        return pred_map1, pred_map2, mu, logvar


###################################################################################################################################

# seg_hrnet_ocr
@export
def hrnet_ocr(cfg, pretrained=False, **kwargs):
    assert not pretrained
    model = get_hrnet_model(cfg)
    return model


##################################################################################################################################

# EfficientNet_decoder

@export
def EfficientNet(pretrained=False, **kwargs):
    model = Efficient_decoder(num_class=4, phi=5, load_weights=True)
    return model

class upsample(nn.Module):

    def __init__(self, if_deconv=True, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        return x

class change_channel(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(change_channel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Efficient_decoder(nn.Module):
    def __init__(self, num_class, phi=0, load_weights=False):
        super(Efficient_decoder, self).__init__()
        self.backbone = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        self.ratio_convert = nn.Conv2d(in_channels=2048, out_channels=640, kernel_size=(1,4), stride=(1,2), padding=(0,1))
        
        self.up1 = upsample(if_deconv=True, channels=640)  #recover the image size,8->16
        self.change_channel1 = change_channel(640, 256)
        self.up2 = upsample(if_deconv=True, channels=256)  #recover the image size,16->32
        self.change_channel2 = change_channel(256, 128)
        self.up3 = upsample(if_deconv=True, channels=128)  #recover the image size,32->64
        self.change_channel3 = change_channel(128, 64)
        self.conv_out_1 = nn.Conv2d(64, num_class, 3, padding=1)
        self.conv_out_2 = nn.Conv2d(64, num_class, 3, padding=1)

    def decoder(self, x):
        x = self.up1(x)   #([N, 640, 16, 16])
        x = self.change_channel1(x)   #([N, 256, 16, 16])
        
        x = self.up2(x)   #([N, 256, 32, 32])
        x = self.change_channel2(x)   #([N, 128, 32, 32])
        
        x = self.up3(x)   #([N, 128, 64, 64])
        x = self.change_channel3(x)   #([N, 64, 64, 64])
        
        x1 = self.conv_out_1(x)
        x2 = self.conv_out_2(x)
        return x1, x2

    def forward(self, x):
        x = self.backbone.extract_features(x)  # （N，1280, H/32, W/32）
        x = self.ratio_convert(x)   #（N，640, H/32, W/64） 
        out1, out2 = self.decoder(x)
        return out1, out2


##################################################################################################################################################################
# EfficientNet_decoder_view_trans

@export
def EfficientNet_view(pretrained=False, **kwargs):
    model = Efficient_decoder_view(num_class=4, phi=7, load_weights=True)
    return model

class Efficient_decoder_view(nn.Module):
    def __init__(self, num_class, phi=0, load_weights=False):
        super(Efficient_decoder_view, self).__init__()
        self.backbone = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        self.ratio_convert = nn.Conv2d(in_channels=2560, out_channels=640, kernel_size=(1,4), stride=(1,2), padding=(0,1))

        self.CVP = CycledViewProjection(8)
        self.CVT = CrossViewTransformer(640)
        
        self.up1 = upsample(if_deconv=True, channels=640)  #恢复图像大小,8->16
        self.change_channel1 = change_channel(640, 256)
        self.up2 = upsample(if_deconv=True, channels=256)  #恢复图像大小,16->32
        self.change_channel2 = change_channel(256, 128)
        self.up3 = upsample(if_deconv=True, channels=128)  #恢复图像大小,32->64
        self.change_channel3 = change_channel(128, 64)
        self.conv_out_1 = nn.Conv2d(64, num_class, 3, padding=1)
        self.conv_out_2 = nn.Conv2d(64, num_class, 3, padding=1)

    def decoder(self, x):
        x = self.up1(x)   #([N, 640, 16, 16])
        x = self.change_channel1(x)   #([N, 256, 16, 16])
        
        x = self.up2(x)   #([N, 256, 32, 32])
        x = self.change_channel2(x)   #([N, 128, 32, 32])
        
        x = self.up3(x)   #([N, 128, 64, 64])
        x = self.change_channel3(x)   #([N, 64, 64, 64])
        
        x1 = self.conv_out_1(x)
        x2 = self.conv_out_2(x)
        return x1, x2

    def forward(self, x):
        x = self.backbone.extract_features(x)  # （N，1280, H/32, W/32）
        x = self.ratio_convert(x)   #（N，640, H/32, W/64） 
        x_feature = x
        x_trans, x_retrans = self.CVP(x)
        x_bird = self.CVT(x, x_trans, x_retrans)
        out1, out2 = self.decoder(x_bird)
        return out1, out2, x_feature, x_retrans



##########################################################################################################33
@export
def cityscape_BotNet(pretrained=False, **kwargs):
    model = BotNet(n_class=4)
    return model

##########################################################################################################
@export
def ResNet(pretrained=False, **kwargs):
    print("we are using ResNet")

##########################################################################################################
@export
def EfficientNet(pretrained=False, **kwargs):
    print("we are using EfficientNet")
