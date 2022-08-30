"""Utility functions and classes"""

import sys
import torch
import numpy as np
import time
import imageio
import os
import skimage.io
import matplotlib as mpl
from PIL import Image

import functools


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val.__class__ == torch.Tensor:
            val = val.item()
        if n.__class__ == torch.Tensor:
            n = n.item()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())



def vis_with_FOVmsk(curr_map):
    mask = imageio.imread('misc/mask_64.png')
    mask = mask[:, :, 0]
    valid_FOV_index = mask == 0

    color_list = np.array([[128, 64, 128],   #深紫 road
                        [244, 35, 232],      #洋红 sidewalk
                        [152, 251, 152],     # 绿  terrian
                        [255, 0, 0],         # 红  Non free-space
                        [0, 0, 0]], dtype=np.uint8)   # 黑  Out of FOV
    curr_map[valid_FOV_index] = 4

    # print(curr_map)
    curr_map = np.repeat(np.repeat(curr_map, 8, axis=0), 8, axis=1).reshape(-1)
    curr_map_c = np.zeros((64*64*8*8, 3), dtype=np.uint8)
    for i in range(64*64*8*8):
        # print(curr_map[i])
        curr_map_c[i, :] = color_list[curr_map[i]]

    curr_map_c = np.reshape(curr_map_c, (64*8, 64*8, 3))

    return curr_map_c


def mask_pred(img_tensor, state, args, save_dir):
    img_list = []
    num_img = img_tensor.shape[0]
    
    is_save = args.image_save_flag
    img_save_dir = save_dir
    # save_root = args.image_save_dir
    # date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # save_dir = save_root + "/" + date + "/" + state + "/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)    
    map_to_save = np.reshape(np.argmax(img_tensor.numpy().transpose((0,2,3,1)), axis=3), [num_img, 64, 64]).astype(np.uint8)
    for i in range(num_img):
        img = map_to_save[i]
        img_with_mask = vis_with_FOVmsk(img)
        now = time.strftime("%H-%M-%S", time.localtime())
        if is_save:
            #img_save_dir = img_save_dir + "/" + state + "/"
            skimage.io.imsave(img_save_dir + now + "_" + state + "_" + str(i) +"_map.png", img)
            skimage.io.imsave(img_save_dir + now + "_" + state + "_" + str(i) +"_map_mask.png", img_with_mask)
        img_with_mask = torch.from_numpy(img_with_mask.transpose((2,0,1)))
        img_list.append(img_with_mask)
    return img_list


def mask_gt_map(img_tensor):
    img_list = []
    num_img = img_tensor.shape[0]
    
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for i in range(num_img):
        img = img_tensor[i].squeeze()
        img_with_mask = vis_with_FOVmsk(img)
        now = time.strftime("%H-%M-%S", time.localtime())
        img_with_mask = torch.from_numpy(img_with_mask.transpose((2,0,1)))
        img_list.append(img_with_mask)
    return img_list


##################################################################################################################################################
# HRNet
# HRNet bu_helper.py

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
    relu_inplace = True



############################################################################################################
def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class