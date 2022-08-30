import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.nn.modules import padding

verbose = False

class CycledViewProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CycledViewProjection, self).__init__()
        self.transform_module = TransformModule(in_channels=in_channels, out_channels=out_channels)  #in_dim为输入图像的尺寸
        self.retransform_module = TransformModule(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        if verbose: print("input size: ", x.shape)
        B, C, H, W = x.view([-1, int(x.size()[1])] + list(x.size()[2:])).size()
        transform_feature = self.transform_module(x)
        if verbose: print("feature size after 1st transform_module: ", transform_feature.shape)
        retransform_features = self.retransform_module(transform_feature)
        if verbose: print("feature size after 2nd transform_module: ", retransform_features.shape)
        return transform_feature, retransform_features


class TransformModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(TransformModule, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        return self.double_conv(x)


if __name__ == '__main__':
    features = torch.randn(4, 320, 8, 16)
    CVP = CycledViewProjection(320, 320)
    print("output 1: ", CVP(features)[0].shape)
    print("output 2: ", CVP(features)[1].shape)