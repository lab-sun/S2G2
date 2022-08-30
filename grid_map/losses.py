# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1) 
    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')  #F.kl_div（）中第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵。如果现在想用Y指导X，第一个参数要传X，第二个要传Y，就是被指导的放在前面。


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

def vae_CE_loss(pred_map, map, mu, logvar):
    CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=
        torch.Tensor([0.6225708,  2.53963754, 15.46416047, 0.52885405]).to('cuda:0'), ignore_index=4, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return 0.9*CE + 0.1*KLD, CE, KLD

# soft dice loss
class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()

  def forward(self, input, target):
    bs = target.size(0)
    smooth = 1

    input_flat = input.view(bs, -1)
    target_flat = target.view(bs, -1)

    intersection = input_flat * target_flat

    loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / bs
    #loss = 1-loss

    return loss.sum()

class MulticlassDiceLoss(nn.Module):
  def __init__(self):
    super(MulticlassDiceLoss, self).__init__()

  def forward(self, input, target, weights=None):
    assert input.shape == target.shape

    num_class = input.shape[1]
    dice = DiceLoss()
    totalLoss = 0

    input = input.log_softmax(dim=1).exp()
    target = target.log_softmax(dim=1).exp()

    for i in range(num_class):
      diceLoss = dice(input[:,i], target[:,i])
      if weights is not None:
        diceLoss *= weights[i]
      totalLoss += diceLoss
    return totalLoss   #Returns the sum over all examples. Divide by the batch size afterwards if you want the mean.

