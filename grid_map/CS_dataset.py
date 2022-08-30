import pandas as pd
import os
import torch
import random
import math
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

verbose = False

class OccMapDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.examples = pd.read_csv(csv_file, header=None)
        self.transform = transform

        imgs = []
        for i in range(len(self.examples)):
            img_path = self.examples.iloc[i, 0]
            map_path = self.examples.iloc[i, 1]
            img = (img_path, map_path)
            imgs.append(img)
        self.imgs = imgs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        rgb = io.imread(self.examples.iloc[item, 0])
        map = io.imread(self.examples.iloc[item, 1])
        if verbose: print("we are in CS_dataset.py, let see rgb and map imread: ", rgb.shape, map.shape)

        rgb_name = self.examples.iloc[item, 0].split('/')[-1]
        map_name = self.examples.iloc[item, 1].split('/')[-1]
        if verbose: print('in CS_dataset, rgb_name', rgb_name)
        if verbose: print('in CS_dataset, map_name', map_name)
        
        rgb = np.asarray(rgb)
        resize = transforms.Compose([transforms.ToPILImage(),transforms.Resize((256, 512)), transforms.ToTensor()])
        rgb_original = resize(rgb)

        map = np.asarray(map)
        example = {'rgb': rgb,
                   'map': map, }


        
        for func in self.transform:
            if verbose: print("!!!we are in CS_dataset.py, let see func", func)
            if verbose: print(example['rgb'].shape)
            if verbose: print(example['map'].shape)
            rgb, map = func(example['rgb'], example['map'])
            example['rgb'] = rgb
            example['map'] = map
        sample = Rescale((256,512))(example)
        sample = ToTensor_Norm()(sample)
            
        if verbose: print("!!!we are in CS_dataset.py, let see sample['map']", sample['map'].shape)
        return sample, rgb_original

class ToTensor_Norm(object):
    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        trans = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])])
        rgb = trans(rgb)

        map = torch.from_numpy(map)
        return {'rgb': rgb,
                'map': map}

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        rgb = transform.resize(rgb, self.output_size, mode='constant', preserve_range=False, anti_aliasing=False)
        if verbose: print("we are in CS_dataset.py, Rescale, let see rgb.shape: ", rgb.shape)

        if verbose: print("we are in CS_dataset.py, Rescale, finished")
        return {'rgb': rgb,
                'map': map}

class Img_distro(object):
    def __init__(self, rot_deg, pix_offset):
        self.rot_deg = rot_deg
        self.pix_offset = pix_offset

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        tran_mat = transform.AffineTransform(translation=(0, self.pix_offset))
        shifted = transform.warp(rgb, tran_mat, preserve_range=True)

        rotated = transform.rotate(shifted, self.rot_deg)

        return {'rgb': rotated,
                'map': map}

class Normalize(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(rgb)
        return {'rgb': rgb,
                'map': map}

###############################################################################################################
# Random Noise
class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img, label):
        img = np.array(img)
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0,1,2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8').convert('RGB'))
        return img, label


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img, label):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img>255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img, label



class RandomFlip():
    def __init__(self, prob=0.5):
        #super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1].copy()
            label = label[:,::-1].copy()
        return image, label


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        #super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            image = (image * bright_factor).astype(image.dtype)

        return image, label




