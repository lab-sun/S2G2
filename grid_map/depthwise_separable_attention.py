import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
from torchsummary import summary

verbose = False

class SeparableAttention(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableAttention,self).__init__()

        self.deepth_att = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias),
            nn.BatchNorm2d(in_channels,eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
        self.pooling1 = MaxPool2d(kernel_size=[2,4], stride=[2,4], padding=0)
        self.pooling2 = MaxPool2d(kernel_size=[4,4], stride=[4,4], padding=0)
        self.point_att = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        inputs = x
        x = self.deepth_att(x)
        x = self.pooling1(x)
        x = self.deepth_att(x)
        x = self.pooling2(x)
        x = self.point_att(x)
        scale = torch.sigmoid(x)
        outputs = inputs * scale
        return outputs

if __name__ == "__main__":
    inputs = torch.randn(4,640,8,16).to("cuda")
    separable_att = SeparableAttention(640, 640).to("cuda")
    outputs = separable_att(inputs)
    print(outputs.shape)
    print(summary(separable_att,  input_size=(640,8,16)))