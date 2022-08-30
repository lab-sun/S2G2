import os
import time, datetime
import logging

import numpy as np
import torch
from torch._C import AggregationType
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.utils
import yaml
from torchsummary import summary
from pytorch_toolbelt import losses as L

from sklearn.metrics import confusion_matrix

from grid_map import losses, datasets, CS_dataset, cli, data, architectures, ramps, cbam
from grid_map.MeanTeacher import IF_RESNET
from grid_map.utils import *
from grid_map.run_context import RunContext
from grid_map.EfficientNet import EfficientNet as EffNet
from grid_map.depthwise_separable_attention import SeparableAttention as Att
import grid_map


from tensorboardX import SummaryWriter


IF_TEA = True
IF_RESNET = False
NO_LABEL = 4
LOG = logging.getLogger('main')

args = None
global_step = 0
start_datetime = datetime.datetime.now().replace(microsecond=0)
img_save_dir = "None"

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args): 
    """
    create a combinated data_loader of labeled and unlabeled data for the training, evaluation and test
    :param train_transformation: the transformation methods for the training data
    :param eval_transformation: the transformation methods for the eval data
    :param datadir: dataroot for the train, eval and test data
    :param args: parameters
    :return: train, eval, test dataloader
    """
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    testdir = os.path.join(datadir, args.test_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])  

    #create dataset
    dataset = CS_dataset.OccMapDataset(traindir, train_transformation)  

    # reading the list of labeled data
    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(',') for line in f.read().splitlines())  
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)  

    # get batch sampler
    if args.exclude_unlabeled:  
        sampler = SubsetRandomSampler(labeled_idxs)  
        print("1. in create_data_loaders, sampler: ", sampler.indices)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)  
    elif args.labeled_batch_size:  
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    # get data_loaders
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)  
    
    eval_loader = torch.utils.data.DataLoader(
        CS_dataset.OccMapDataset(evaldir, eval_transformation),
        batch_size=2,   
        shuffle=True,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        CS_dataset.OccMapDataset(testdir, eval_transformation),
        batch_size=2,   
        shuffle=True,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader, test_loader

# create training model  
def get_models(args):
    models = {}
    model_params_to_train = {}
    base_param = []
    teacher_param =[]
    student_param = []
    doubleAtt_param = []
    fea_channels = 448
    if IF_RESNET:   #ResNet as backbone
        models['encoder'] = grid_map.Front_fea_Encoder(152, args.img_height, args.img_width, args.pretrained)
        models['CycledViewProjection'] = grid_map.CycledViewProjection(height_dim=32, width_dim=64)   #after resnet the size of the feature map is N*128*32*64
        models['CrossViewTransformer'] = grid_map.CrossViewTransformer(512)
        models['Student'] = grid_map.Stu_TeaNet(152, args.img_height, args.img_width, args.pretrained, num_class=args.num_class)
        if IF_TEA:
            models['Teacher'] = grid_map.Stu_TeaNet(152, args.img_height, args.img_width, args.pretrained, num_class=args.num_class)
           

    else:   #EfficientNet as backbone
        models['encoder'] = EffNet.from_pretrained('efficientnet-b4')
        models['CycledViewProjection'] = grid_map.CycledViewProjection(in_channels=fea_channels, out_channels=fea_channels)  #after EfficientNet the size of the feature map is N*640*8*16
        models['CrossViewTransformer'] = grid_map.CrossViewTransformer(fea_channels)
        models['Attention'] = Att(fea_channels,fea_channels)
        models['Student'] = grid_map.Stu_TeaNet(fea_channels, args.img_height, args.img_width, args.pretrained, num_class=args.num_class)
        if IF_TEA:
            models['Teacher'] = grid_map.Stu_TeaNet(fea_channels, args.img_height, args.img_width, args.pretrained, num_class=args.num_class)

    # detach the params in Teacher network
    for key in models.keys():
        models[key].to("cuda")  
        if "Teacher" in key:
            for param in models[key].parameters():
                param.detach_()
                param.requires_grad=False
            teacher_param += list(models[key].parameters())
        elif "Student" in key:
            student_param += list(models[key].parameters())
        elif "Attention" in key:
            doubleAtt_param +=list(models[key].parameters())
        else:
            base_param += list(models[key].parameters())
    base_param += student_param  # base_param includes student_param
    model_params_to_train = {"tea_params": teacher_param,
                            "stu_params" : student_param,
                            "base_params": base_param,
                            "doubleAtt_param": doubleAtt_param}

    # model params
    return models, model_params_to_train

def model_process(model, input_var):
    outputs = {}
    if IF_RESNET: 
        features = model['encoder'](input_var)  
    else:
        features = model['encoder'].extract_features(input_var)  
    
    # Cross-view Transformation Module
    x_features = features
    transform_feature, retransform_feature = model['CycledViewProjection'](features)
    features = model['CrossViewTransformer'](features, transform_feature, retransform_feature)
    features_cbam = model['Attention'](transform_feature)
    outputs['student'] = model['Student'](features)  
    if IF_TEA:
        outputs['teacher'] = model['Teacher'](features_cbam)   
        outputs['teacher'] = Variable(outputs['teacher'].detach().data, requires_grad=False)
    return outputs, x_features, retransform_feature, features, features_cbam 


def test(test_loader, model):
    global start_datetime


    for key in model.keys():
        model[key].to("cuda")
        model[key].eval()

    num_classes = 4
    conf_total_stu = np.zeros((num_classes, num_classes))
    conf_total_tea = np.zeros((num_classes, num_classes))
    for i, (sample, original_img) in enumerate(test_loader):
        start_t = time.time()
        input = sample['rgb'].type(torch.FloatTensor)
        target = sample['map']
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target.cuda())
            target_var = target_var.long()

        outputs,_,_,_,_ = model_process(model, input_var)

        # calculate the confusion matrix
        label = target_var.cpu().numpy().squeeze().flatten()
        prediction_stu = outputs['student'].argmax(1).cpu().numpy().squeeze().flatten()
        if IF_TEA: prediction_tea = outputs['teacher'].argmax(1).cpu().numpy().squeeze().flatten()
        conf_stu = confusion_matrix(y_true=label, y_pred=prediction_stu, labels=[0,1,2,3])
        conf_total_stu += conf_stu
        if IF_TEA: conf_tea = confusion_matrix(y_true=label, y_pred=prediction_tea, labels=[0,1,2,3])
        if IF_TEA: conf_total_tea += conf_tea


        print('Test: {model_name}, iter [{global_step}/{total_step}], {num_img:.2f}img/sec, time {time}'\
            .format(model_name=args.arch, global_step=(i+1), total_step=len(test_loader),
            num_img=len(original_img)/(time.time()-start_t),
            time=datetime.datetime.now().replace(microsecond=0)-start_datetime
            ))


    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total_stu)
    average_precision = precision_per_class.mean()
    average_recall = recall_per_class.mean()
    average_IoU = iou_per_class.mean()
                
    LOG.info(' * Results in TEST: average_precision {average_precision:.6f}  average_recall {average_recall:.6f}  average_IoU {average_IoU:.6f}\n'
          .format(average_precision=average_precision, average_recall=average_recall, average_IoU=average_IoU))
    



def main(context):
    global global_step
    global start_datetime

    print("###############loading dataset: ", args.dataset)
    dataset_config = datasets.__dict__[args.dataset]()  
    num_classes = dataset_config.pop('num_classes')
    _, _, test_loader = create_data_loaders(**dataset_config, args=args)

    # create the net model
    LOG.info("=> creating {pretrained} model '{arch}'".format(
        pretrained='pre-trained' if args.pretrained else '',
        arch=args.arch
    ))
    model, _ = get_models(args=args) 

    # resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        pretrained_dict = checkpoint['state_dict']
        for key in model.keys():
            model_dict = model[key].state_dict()
            pretrained_dict[key] = {k:v for k,v in pretrained_dict[key].items() if k in model_dict}
            model_dict.update(pretrained_dict[key])
            model[key].load_state_dict(model_dict)
        LOG.info("=> loaded checkpoint '{}' ".format(args.resume))
        

    # start to test
    LOG.info("Testing the model:")
    test(test_loader, model)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))   
