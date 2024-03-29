from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .ResnetEncoder import ResnetEncoder
from .EfficientNet import EfficientNet as EffNet
import matplotlib.pyplot as PLT

verbose = False
IF_RESNET = False
print("***now we are using the modle:", "ResNet***" if IF_RESNET else "EfficientNet***")

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()

        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class Seconde_Encoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Seconde_Encoder, self).__init__()

        if_front_encoder = False
        self.resnet_encoder = ResnetEncoder(num_layers, pretrained, if_front_encoder=if_front_encoder)
        num_ch_enc = self.resnet_encoder.num_ch_enc
        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)

        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """

        batch_size, c, h, w = x.shape
        x = self.resnet_encoder(x)[-1]
        if verbose: print("#Sec_Encoder# out after resnet_encoder: ", x.size())  #([N, 128, 32, 64])  only use layer_1 and layer_2 
        x = self.conv1(x)
        if verbose: print("#Sec_Encoder# out after conv1: ", x.size())  #([N, 128, 8, 16])
        x = self.pool(x)  
        if verbose: print("#Sec_Encoder# out after pool1: ", x.size())  #([N, 128, 4, 8])
        x = self.conv2(x)
        if verbose: print("#Sec_Encoder# out after conv2: ", x.size())  #([N, 128, 4, 8])
        x = self.pool(x)
        if verbose: print("#Sec_Encoder# out after pool2: ", x.size())  #([N, 128, 2, 4])
        return x 

class Decoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_ch_enc : list
        channels used by the ResNet Encoder at different layers

    Methods
    -------
    forward(x, ):
        Processes input image features into output occupancy maps/layouts
    """

    def __init__(self, num_ch_enc, num_class=2, type=''):
        super(Decoder, self).__init__()
        self.num_output_channels = num_class
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64])  # for resNet self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        if IF_RESNET: self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        self.dec_layers = len(self.num_ch_dec)
        for i in range(self.dec_layers-1, -1, -1):
            # upconv_0
            if type == 'transform_decoder':
                num_ch_in = 128 if i == (self.dec_layers-1) else self.num_ch_dec[i + 1]
            else:
                num_ch_in = 128 if i == (self.dec_layers-1) else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)
        
        self.convs["topview"] = Conv3x3(
            self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, x, is_training=True):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of encoded feature tensors
            | Shape: (batch_size, 128, occ_map_size/2^5, occ_map_size/2^5)
        is_training : bool
            whether its training or testing phase

        Returns
        -------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)
        """

        h_x = 0
        for i in range(self.dec_layers-1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if verbose: print("in MeanTeacher, in decoder after upconv{}_0: {}".format(i, x.shape))
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            if verbose: print("in MeanTeacher, in decoder after upsample: {}".format(x.shape))
            x = self.convs[("upconv", i, 1)](x)
            if verbose: print("in MeanTeacher, in decoder after upconv{}_1: {}".format(i, x.shape))
            x = self.convs[("norm", i, 1)](x)

        if is_training:
            x = self.convs["topview"](x)
            if verbose: print("in MeanTeacher, in decoder after convs[topview]: {}".format( x.shape))
        else:
            softmax = nn.Softmax2d()
            x = softmax(self.convs["topview"](x))

        return x

def ratio_convert(in_channels, out_channels):
    """change the ratio between the img_width and img_height"""

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,4), stride=(1,2), padding=(0,1))

class Stu_TeaNet(nn.Module):
    def __init__(self, num_layers, img_ht, img_wt, pretrained=True, num_class=4):
        super(Stu_TeaNet, self).__init__()
        if IF_RESNET: 
            self.encoder = Seconde_Encoder(num_layers, img_ht, img_wt, pretrained)  # for ResNet
            self.ratio_convert = ratio_convert(in_channels=128, out_channels=128)
            self.decoder = Decoder(self.encoder.resnet_encoder.num_ch_enc, num_class)   #for resNet
        else:
            in_channels = num_layers
            self.ratio_convert = ratio_convert(in_channels=in_channels, out_channels=128)   # for resNet in_channels=128, out_channels=128
            self.decoder = Decoder(None, num_class)  #for efficientNet
        

    def forward(self, x):
        if IF_RESNET: 
            x = self.encoder(x)  # for ResNet  torch.Size([2, 128, 2, 4])
        else:
            pass
        if verbose: print("in MeanTeacher, after encode: ", x.shape)
        x = self.ratio_convert(x)  #efficientNet:torch.Size([2, 128, 8, 8])   ResNet:torch.Size([2, 128, 2, 2])
        if verbose: print("in MeanTeacher, after ratio_convert: ", x.shape)
        x = self.decoder(x)   #efficientNet:torch.Size([2, 4, 256, 256])   ResNet:torch.Size([2, 4, 64, 64])
        if verbose: print("in MeanTeacher, after decode: ", x.shape)

        return x 

        


if __name__ == "__main__":
    num_minibatch = 2
    if IF_RESNET:
        input_img = torch.randn(num_minibatch, 128, 32, 64).cuda(0)  #for resNet
    else:
        input_img = torch.randn(num_minibatch, 224, 16, 32).cuda(0)  #for efficientNet
    net = Stu_TeaNet(18, 32, 64, True, 4).cuda(0)
    out = net(input_img)  
    print(out.shape)