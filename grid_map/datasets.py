import torchvision.transforms as transforms

from . import data
from .utils import export
from .CS_dataset import ToTensor_Norm, Rescale, Normalize, AddSaltPepperNoise, AddGaussianNoise, RandomFlip, RandomBrightness, ToTensor_Norm


@export
def cityscapes():
    channel_states = dict(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    
    train_transformation = [
        #Rescale((256, 512)),
        RandomFlip(0.5),
        RandomBrightness(),
        #ToTensor_Norm(),
    ]
    
    
    eval_transformation = [
        #Rescale((256, 512)),
        #ToTensor_Norm(),
    ]

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/Cityscapes/',
        'num_classes': 4
    }