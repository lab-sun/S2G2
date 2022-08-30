import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map


def feature_selection(input, dim, index):
    # feature selection
    # input: [N, ?, ?, ...]
    # dim: scalar > 0
    # index: [N, idx]
    # input [B, N, (H*W)], index [2, (H*W)]
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]   #view.shape ([B, 1, -1])
    expanse = list(input.size())
    expanse[0] = -1
    expanse[dim] = -1   #expand.shape  ([-1, 128, -1])
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)  #return shape ([B, C, H, W])


class CrossViewTransformer(nn.Module):
    def __init__(self, in_dim):
        super(CrossViewTransformer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.f_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=3, stride=1, padding=1,
                                bias=True)

        self.res_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, front_x, cross_x, front_x_hat):
        m_batchsize, C, width, height = front_x.size()
        proj_query = self.query_conv(cross_x).view(m_batchsize, -1, width * height)  # B x C x (N)
        proj_key = self.key_conv(front_x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C

        energy = torch.bmm(proj_key, proj_query)  # transpose check   energy.shape: ([B, (W*H), (W*H)])
        front_star, front_star_arg = torch.max(energy, dim=1)   # front_star.shape: ([B, (W*H)])  front_star_arg.shape: ([B, (W*H)])
        proj_value = self.value_conv(front_x_hat).view(m_batchsize, -1, width * height)  # B x C x N

        T = feature_selection(proj_value, 2, front_star_arg).view(front_star.size(0), -1, width, height)  #T.shape: ([B, C, H, W])

        S = front_star.view(front_star.size(0), 1, width, height)  #S.shape: ([B, 1, H, w])

        front_res = torch.cat((front_x, T), dim=1)
        front_res = self.f_conv(front_res)
        front_res = front_res * S
        output = front_x + front_res

        return output


if __name__ == '__main__':
    features = torch.rand(4,320,8,16)
    features2 = torch.rand(4,320,8,16)
    features3 = torch.rand(4,320,8,16)
    print("in CVT, let see features: {}, features2: {}, features3: {}".format(features.shape, features2.shape, features3.shape))

    attention3 = CrossViewTransformer(320)
    print(attention3(features, features2, features3).shape)
