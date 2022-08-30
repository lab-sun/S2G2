import re
import argparse
import os
import shutil
import time, datetime
import math
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

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IF_TEA = True
IF_RESNET = False
NO_LABEL = 4
LOG = logging.getLogger('main')

args = None
best_iou_tea = 0
best_iou_stu = 0
best_epoch_tea = 0
best_epoch_stu = 0
is_best_tea, is_best_stu = False, False
global_step = 0
start_datetime = datetime.datetime.now().replace(microsecond=0)
img_save_dir = "None"
text_dir = './runs/record_txt/'   # the file to store the results
if not os.path.exists(text_dir):
        os.makedirs(text_dir)

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
    if IF_RESNET:   # ResNet as backbone
        models['encoder'] = grid_map.Front_fea_Encoder(152, args.img_height, args.img_width, args.pretrained)
        models['CycledViewProjection'] = grid_map.CycledViewProjection(height_dim=32, width_dim=64)   #after resnet the size of the feature map is N*128*32*64
        models['CrossViewTransformer'] = grid_map.CrossViewTransformer(512)
        models['Student'] = grid_map.Stu_TeaNet(152, args.img_height, args.img_width, args.pretrained, num_class=args.num_class)
        if IF_TEA:
            models['Teacher'] = grid_map.Stu_TeaNet(152, args.img_height, args.img_width, args.pretrained, num_class=args.num_class)
           

    else:   # EfficientNet as backbone
        #models['encoder'] = EffNet.from_pretrained('efficientnet-b7', if_front_encoder=True)
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



# calculate the current consistency weight according to the epoch
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# update the parameters fo the teacher network
def update_ema_variables(model, alpha, epoch, consistency_rampup):
    """
    Use the true average until the exponential average is more correct
    :param model: the network model
    :param alpha: EMA decay, to adjust the weights of the current stu params and the previous tea params during update.
    :param global_step: global step
    :param consistency_rampup: the steps in ramp-up phase  
    """

    alpha = min(1 - 1/(global_step+1), alpha)
    stu_params = model['Student'].parameters()
    tea_params = model['Teacher'].parameters()
    for tea_param, stu_param in zip(tea_params, stu_params):
        tea_param.data.mul_(alpha).add_(stu_param.data, alpha=1-alpha)  # alpha * tea_params + （1-alpha）* stu_params
    return alpha  

def save_checkpoint(state, is_best_tea, is_best_stu, dirpath, epoch):
    filename = 'checkpoint.pth'
    checkpoint_path = os.path.join(dirpath, filename)
    best_path_tea = os.path.join(dirpath, 'best_tea.pth')
    best_path_stu = os.path.join(dirpath, 'best_stu.pth')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best_tea:
        shutil.copyfile(checkpoint_path, best_path_tea)
        LOG.info("--- checkpoint copied to %s ---" % best_path_tea)
    if is_best_stu:
        shutil.copyfile(checkpoint_path, best_path_stu)
        LOG.info("--- checkpoint copied to %s ---" % best_path_stu)

def train(train_loader, model, optimizer, scheduler, epoch, trainwriter, lr_exp_update_flag):
    global global_step
    global img_save_dir

    # consistency_loss type
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == 'dice':
        consistency_criterion = losses.MulticlassDiceLoss()
    else:
        assert False, args.consistency_type

    # residual_loss 
    residual_logit_criterion = losses.symmetric_mse_loss
    # seg_loss: dice loss
    seg_criterion_dice = L.DiceLoss(mode='multiclass', ignore_index=NO_LABEL).cuda()   
    # seg_loss: CE loss
    seg_criterion = L.SoftCrossEntropyLoss(reduction='sum', smooth_factor = 0.1, ignore_index=NO_LABEL).cuda()
    # cross_view_loss
    view_criterion = nn.L1Loss()
    # BEV loss
    BEV_criterion = nn.L1Loss()

    # swich to train mode
    for key in model.keys():
        model[key].to("cuda")
        model[key].train()

    num_classes = 4
    conf_total = np.zeros((num_classes, num_classes))

    running_loss = 0.0

    for i, (sample, original_img) in enumerate(train_loader):
        input = sample['rgb'].type(torch.FloatTensor)
        target = sample['map']
        start_t = time.time()

        outputs = {}
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda().long()

        minibatch_size = args.labeled_batch_size * 64 * 64

        # the outputs from network
        outputs, x_features, retransform_feature, features, features_cbam = model_process(model, input_var)

        # CE loss
        seg_loss_stu = seg_criterion(outputs['student'], target_var) / minibatch_size
        if IF_TEA:
            seg_loss_tea = seg_criterion(outputs['teacher'], target_var) / minibatch_size
        # dice loss
        dice_loss_stu = seg_criterion_dice(outputs['student'], target_var) 
        if IF_TEA:
            dice_loss_tea = seg_criterion_dice(outputs['teacher'], target_var) 
        # consistency loss
        if IF_TEA:
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * consistency_criterion(outputs['student'], outputs['teacher']) / minibatch_size
        else:
            consistency_weight = 0
            consistency_loss = 0
        # restruction loss
        if args.cycle_view_weight >= 0:
            cycle_view_weight = args.cycle_view_weight
            cycleview_loss = cycle_view_weight * view_criterion(x_features, retransform_feature)
        else:
            cycleview_loss = 0
        # BEV loss
        if args.BEV_weight >= 0:
            BEV_weight = args.BEV_weight
            BEV_loss = BEV_weight * BEV_criterion(features, features_cbam)
        else:
            BEV_loss = 0
        
        # total loss
        loss = seg_loss_stu + dice_loss_stu + consistency_loss + cycleview_loss + BEV_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        running_loss += loss.item()

        # Calculate gradient and update student network weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # update teacher network weight
        if IF_TEA:
            alpha = update_ema_variables(model, args.ema_decay, global_step, args.consistency_rampup*len(train_loader))

        lr_this_epo = 0
        lr_this_epo_att = 0
        
        if len(optimizer.param_groups) == 1:
            for param_group in optimizer.param_groups:
                lr_this_epo = param_group['lr']
        elif len(optimizer.param_groups) == 2:
            lr_this_epo = optimizer.param_groups[0]['lr']
            lr_this_epo_att = optimizer.param_groups[1]['lr']
        
        print('Train: {model_name}, epo [{epoch}/{epoch_max}], iter [{global_step}/{total_step}], lr {lr:.8f}, {num_img:.2f}img/sec, loss {loss:.4f}, time {time}'\
            .format(model_name=args.arch,  epoch=epoch, epoch_max=args.epochs, global_step=(i+1), total_step=len(train_loader),
            lr=lr_this_epo, num_img=len(original_img)/(time.time()-start_t), loss=float(loss), 
            time=datetime.datetime.now().replace(microsecond=0)-start_datetime
            ))

        # calculate the confusion matrix
        label = target_var.cpu().numpy().squeeze().flatten()
        prediction = outputs['student'].argmax(1).cpu().numpy().squeeze().flatten()
        conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3])
        conf_total += conf

        # log down the results
        if i % args.print_freq == 0:
            LOG.info(
                'Train: [{epoch}][{global_step}/{total_step}]\t'
                'Time {time:.4f}\t'
                'LR {lr:.8f}\t'
                'Loss {loss:.8f}\t'
                'Seg_Loss {seg_loss:.8f}\t'
                'Dice_loss {dice_loss:.8f}\t'
                'Cons_loss {consistency_loss:.8f}\t'
                'View_loss {view_loss:.8f}\t'
                'BEV_loss {BEV_loss:.8f}\t'
                'Cons_weight {consistency_weight:.8f}\n'.format(
                    epoch=epoch, global_step=(i+1), total_step=len(train_loader), 
                    time=(time.time()-start_t), lr=lr_this_epo, loss=loss, seg_loss=seg_loss_stu, 
                    dice_loss=dice_loss_stu, consistency_loss=consistency_loss,  
                    view_loss=cycleview_loss, BEV_loss=BEV_loss, consistency_weight=consistency_weight
                )
            )
            with open(text_dir + 'results.txt', 'a') as f:
                f.write(
                    'Train epoch: [{epoch}][{global_step}/{total_step}]\t'
                    'Time {time}\t'
                    'LR {lr:.8f}\t'
                    'Loss {loss:.8f}'
                    'Seg_Loss {seg_loss:.8f}\t'
                    'Dice_loss {dice_loss:.8f}\t'
                    'Cons_loss {consistency_loss:.8f}\t'
                    'View_loss {view_loss:.8f}\t'
                    'BEV_loss {BEV_loss:.8f}\t'
                    'Cons_weight {consistency_weight:.8f}\n'.format(
                        epoch=epoch, global_step=(i+1), total_step=len(train_loader), 
                        time=(time.time()-start_t), lr=lr_this_epo, loss=loss, seg_loss=seg_loss_stu, 
                        dice_loss=dice_loss_stu, consistency_loss=consistency_loss,  
                        view_loss=cycleview_loss, BEV_loss=BEV_loss, consistency_weight=consistency_weight
                    )
                )
    
        # tensorboard shows the loss curves
        trainwriter.add_scalar('train/Seg_Loss_stu', seg_loss_stu, global_step)
        if IF_TEA: trainwriter.add_scalar('train/Seg_Loss_tea', seg_loss_tea, global_step)
        trainwriter.add_scalar('train/dice_loss_stu', dice_loss_stu, global_step)
        if IF_TEA: trainwriter.add_scalar('train/dice_loss_tea', dice_loss_tea, global_step)
        trainwriter.add_scalar('train/CrossView_Loss', cycleview_loss, global_step)
        trainwriter.add_scalar('train/Cons_Loss', consistency_loss, global_step)
        trainwriter.add_scalar('train/BEV_loss', BEV_loss, global_step)
        trainwriter.add_scalar('train/Loss', loss, global_step)
        trainwriter.add_scalar('train/lr', lr_this_epo, global_step)
        trainwriter.add_scalar('train/cons_weight', consistency_weight, global_step)
        if IF_TEA: trainwriter.add_scalar('train/alpha', alpha, global_step)   
        

        # tensorboard shows the images
        if i % (len(train_loader) // 1) == 0: 
                grid_image = torchvision.utils.make_grid(original_img[:-3], 3, normalize=False)  
                trainwriter.add_image('train/Image_train', grid_image, global_step)  
                show_gt = mask_gt_map(target_var[-3:].clone().cpu().data)
                grid_image = torchvision.utils.make_grid(show_gt, 3, normalize=False)  
                trainwriter.add_image('train/ground_truth', grid_image, global_step)
                show_img = mask_pred(outputs['student'][-3:].clone().cpu().data, state="Train_stu", args=args, save_dir=img_save_dir)
                grid_image = torchvision.utils.make_grid(show_img, 3, normalize=False)   
                trainwriter.add_image('train/student_model_output', grid_image, global_step)
                if IF_TEA: 
                    show_img = mask_pred(outputs['teacher'][-3:].clone().cpu().data, state="Train_tea", args=args, save_dir=img_save_dir)
                    grid_image = torchvision.utils.make_grid(show_img, 3, normalize=False)  
                    trainwriter.add_image('train/teacher_model_output', grid_image, global_step)
    
    # calculate the evaluation metrics
    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
    average_precision = precision_per_class.mean()
    average_recall = recall_per_class.mean()
    average_IoU = iou_per_class.mean()
    average_loss = running_loss / len(train_loader)
    LOG.info(' * average_precision {average_precision:.6f}\taverage_recall {average_recall:.6f}\taverage_IoU {average_IoU:.6f}\n'
        .format(average_precision=average_precision, average_recall=average_recall, average_IoU=average_IoU))
    with open(text_dir + 'results.txt', 'a') as f:
        f.write(' *Train ==> average_precision {average_precision:.6f}\taverage_recall {average_recall:.6f}\taverage_IoU {average_IoU:.6f}\n'
        .format(average_precision=average_precision, average_recall=average_recall, average_IoU=average_IoU))
    
    # tensorboard shows the metrics curves
    trainwriter.add_scalar('Epoch_Loss', average_loss, epoch+1)
    trainwriter.add_scalar('Average_Precision', average_precision, epoch+1)
    trainwriter.add_scalar('Average_Recall', average_recall, epoch+1)
    trainwriter.add_scalar('Average_IoU', average_IoU, epoch+1)
    
    
def validate(eval_loader, model, global_step, epoch, evalwriter, ema_evalwriter):
    global best_iou_tea
    global best_iou_stu
    global best_epoch_tea
    global best_epoch_stu
    global is_best_tea, is_best_stu
    global start_datetime
    
    # define the loss functions
    seg_criterion = L.SoftCrossEntropyLoss(reduction='sum', smooth_factor = 0.1, ignore_index=NO_LABEL).cuda()
    seg_criterion_dice = L.DiceLoss(mode='multiclass', ignore_index=NO_LABEL).cuda()

    for key in model.keys():
        model[key].to("cuda")
        model[key].eval()

    running_loss_stu = 0.0
    running_loss_tea = 0.0
    num_classes = 4
    conf_total_stu = np.zeros((num_classes, num_classes))
    conf_total_tea = np.zeros((num_classes, num_classes))
    for i, (sample, original_img) in enumerate(eval_loader):
        start_t = time.time()
        input = sample['rgb'].type(torch.FloatTensor)
        target = sample['map']
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target.cuda())
            target_var = target_var.long()

        minibatch_size = len(target_var) * 64 * 64

        outputs,_,_,_,_ = model_process(model, input_var)
        
        label = target_var.cpu().numpy().squeeze().flatten()
        prediction_stu = outputs['student'].argmax(1).cpu().numpy().squeeze().flatten()
        if IF_TEA: prediction_tea = outputs['teacher'].argmax(1).cpu().numpy().squeeze().flatten()
        conf_stu = confusion_matrix(y_true=label, y_pred=prediction_stu, labels=[0,1,2,3])
        conf_total_stu += conf_stu
        if IF_TEA: conf_tea = confusion_matrix(y_true=label, y_pred=prediction_tea, labels=[0,1,2,3])
        if IF_TEA: conf_total_tea += conf_tea
        
        seg_loss_stu = seg_criterion(outputs['student'], target_var) / minibatch_size
        if IF_TEA: seg_loss_tea = seg_criterion(outputs['teacher'], target_var) / minibatch_size
        dice_loss_stu = seg_criterion_dice(outputs['student'], target_var)
        if IF_TEA: dice_loss_tea = seg_criterion_dice(outputs['teacher'], target_var)

        loss_stu = seg_loss_stu + dice_loss_stu
        if IF_TEA: loss_tea = seg_loss_tea + dice_loss_tea

        running_loss_stu += loss_stu.item()
        if IF_TEA: running_loss_tea += loss_tea.item()
        if not IF_TEA:
            loss_tea = 0

        print('Val: {model_name}, epo [{epoch}/{epoch_max}], iter [{global_step}/{total_step}], {num_img:.2f}img/sec, STU_loss {stu_loss:.4f}, TEA_loss {tea_loss:.4f}, time {time}'\
            .format(model_name=args.arch,  epoch=epoch, epoch_max=args.epochs, global_step=(i+1), total_step=len(eval_loader),
            num_img=len(original_img)/(time.time()-start_t), stu_loss=float(loss_stu), tea_loss=float(loss_tea),
            time=datetime.datetime.now().replace(microsecond=0)-start_datetime
            ))

        if i % args.print_freq == 0:
            LOG.info(
                'Val: [{epoch}][{global_step}/{total_step}]\t'
                'Time {time}\t'
                'Seg_loss_stu {seg_loss_stu:.8f}\t'
                'Seg_loss_tea {seg_loss_tea:.8f}\n'
                .format(
                    epoch=epoch, global_step=(i+1), total_step=len(eval_loader),
                    time=(time.time()-start_t),
                    seg_loss_stu=loss_stu,
                    seg_loss_tea=loss_tea)
                )
            with open(text_dir + 'results.txt', 'a') as f:
                f.write(
                'Val: [{epoch}][{global_step}/{total_step}]\t'
                'Time {time}\t'
                'Seg_loss_stu {seg_loss_stu:.8f}\t'
                'Seg_loss_tea {seg_loss_tea:.8f}\n'
                .format(
                    epoch=epoch, global_step=(i+1), total_step=len(eval_loader),
                    time=(time.time()-start_t),
                    seg_loss_stu=loss_stu,
                    seg_loss_tea=loss_tea)
                )

        evalwriter.add_scalar('val/STU_Seg_Loss', loss_stu, global_step)
        ema_evalwriter.add_scalar('val/TEA_Seg_Loss', loss_tea, global_step)

        if i % (len(eval_loader) // 1) == 0:
            grid_image = torchvision.utils.make_grid(original_img[:3], 3, normalize=False)  
            evalwriter.add_image('VAL/origal_img', grid_image, global_step)  
            show_gt = mask_gt_map(target_var[:3].clone().cpu().data)
            grid_image = torchvision.utils.make_grid(show_gt, 3, normalize=False)  
            evalwriter.add_image('VAL/ground_truth', grid_image, global_step)
            show_img = mask_pred(outputs['student'][:3].clone().cpu().data, state="Val_stu", args=args, save_dir=img_save_dir)
            grid_image = torchvision.utils.make_grid(show_img, 3, normalize=False)   
            evalwriter.add_image('VAL/Student_output', grid_image, global_step)
            if IF_TEA:
                show_img = mask_pred(outputs['teacher'][:3].clone().cpu().data, state="Val_tea", args=args, save_dir=img_save_dir)
                grid_image = torchvision.utils.make_grid(show_img, 3, normalize=False)   
                evalwriter.add_image('VAL/Teacher_output', grid_image, global_step)


    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total_stu)
    average_precision_stu = precision_per_class.mean()
    average_recall_stu = recall_per_class.mean()
    average_IoU_stu = iou_per_class.mean()

    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total_tea)
    average_precision_tea = precision_per_class.mean()
    average_recall_tea = recall_per_class.mean()
    average_IoU_tea = iou_per_class.mean()
                
    
    val_loss_stu = running_loss_stu/len(eval_loader)
    val_loss_tea = running_loss_tea/len(eval_loader)

    
    LOG.info(' * Student in VAL: STU_average_precision {average_precision:.6f}\tSTU_average_recall {average_recall:.6f}\tSTU_average_IoU {average_IoU:.6f}\n'
          .format(average_precision=average_precision_stu, average_recall=average_recall_stu, average_IoU=average_IoU_stu))
    
    LOG.info(' * Teacher in VAL: TEA_average_precision {average_precision:.6f}\tTEA_average_recall {average_recall:.6f}\tTEA_average_IoU {average_IoU:.6f}\n'
          .format(average_precision=average_precision_tea, average_recall=average_recall_tea, average_IoU=average_IoU_tea))

    
    with open(text_dir + 'results.txt', 'a') as f:
        f.write(' * ==> Student in VAL: average_precision {average_precision:.6f}\taverage_recall {average_recall:.6f}\taverage_IoU {average_IoU:.6f}\n'
          .format(average_precision=average_precision_stu, average_recall=average_recall_stu, average_IoU=average_IoU_stu))
        f.write(' * ==> Teacher in VAL: average_precision {average_precision:.6f}\taverage_recall {average_recall:.6f}\taverage_IoU {average_IoU:.6f}\n'
          .format(average_precision=average_precision_tea, average_recall=average_recall_tea, average_IoU=average_IoU_tea))
    
    
    evalwriter.add_scalar('Epoch_Loss', val_loss_stu, epoch+1)
    evalwriter.add_scalar('Average_Precision', average_precision_stu, epoch+1)
    evalwriter.add_scalar('Average_Recall', average_recall_stu, epoch+1)
    evalwriter.add_scalar('Average_IoU', average_IoU_stu, epoch+1)
            
    ema_evalwriter.add_scalar('Epoch_Loss', val_loss_tea, epoch+1)
    ema_evalwriter.add_scalar('Average_Precision', average_precision_tea, epoch+1)
    ema_evalwriter.add_scalar('Average_Recall', average_recall_tea, epoch+1)
    ema_evalwriter.add_scalar('Average_IoU', average_IoU_tea, epoch+1)
    
    # student best iou
    is_best_stu = average_IoU_stu > best_iou_stu
    best_iou_stu = max(average_IoU_stu, best_iou_stu)
    if is_best_stu:
        best_epoch_stu = epoch + 1
    LOG.info("***beat VAL iou from <student> is %.6f at %s epoch ***" % (best_iou_stu, best_epoch_stu))
    # teacher best iou
    is_best_tea = average_IoU_tea > best_iou_tea
    best_iou_tea = max(average_IoU_tea, best_iou_tea)
    if is_best_tea:
        best_epoch_tea = epoch + 1
    LOG.info("***beat VAL iou from <teacher> is %.6f at %s epoch ***" % (best_iou_tea, best_epoch_tea))

    with open(text_dir + 'results.txt', 'a') as f:
        f.write("\n ***beat VAL iou from TEACHER is %.6f at %s epoch ***\n" % (best_iou_tea, best_epoch_tea))
        f.write("\n ***beat VAL iou from STUDENT is %.6f at %s epoch ***\n" % (best_iou_stu, best_epoch_stu))


def main(context):
    global global_step
    global best_iou_tea
    global best_iou_stu
    global best_epoch_tea
    global best_epoch_stu
    global is_best_tea, is_best_stu
    global img_save_dir
    global start_datetime


    # checkpoint and log
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")

    # tensorboard
    summary_work_dir = time.strftime("./runs/%Y-%m-%dT%H:%M", time.localtime())
    if not os.path.exists(summary_work_dir):
        os.makedirs(summary_work_dir)
    # Tensorboard initialization.
    trainwriter = SummaryWriter('{}/{}'.format(summary_work_dir, 'Train'))
    evalwriter = SummaryWriter('{}/{}'.format(summary_work_dir, 'STU_Eval'))
    ema_evalwriter = SummaryWriter('{}/{}'.format(summary_work_dir, 'TEA_Eval'))
    textwriter = SummaryWriter('{}/{}'.format(summary_work_dir, 'text'))

    # record output to a txt
    with open(text_dir + 'results.txt', "r+") as f:   
        f.truncate()

    # save_path of the output images
    if args.image_save_flag:
        img_save_root = args.image_save_dir
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        img_save_dir = img_save_root + "/" + date + "/"
        
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
    
    # load the dataset
    print("###############loading dataset: ", args.dataset)
    dataset_config = datasets.__dict__[args.dataset]()  
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, test_loader = create_data_loaders(**dataset_config, args=args)

    # create the net model
    LOG.info("=> creating {pretrained} model '{arch}'".format(
        pretrained='pre-trained' if args.pretrained else '',
        arch=args.arch
    ))
    model, params_to_train = get_models(args=args) 
    
    # Optimization
    model_params_to_train = [{"params": params_to_train['base_params'], "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay},
                             {"params": params_to_train['doubleAtt_param'], "lr": args.lr*1, "momentum": args.momentum, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(model_params_to_train)
    #optimizer = torch.optim.SGD(model_params_to_train)
    if args.lr_scheduler_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.65)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    else:
        scheduler = None

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_iou_tea = checkpoint['best_iou_tea']
        best_iou_stu = checkpoint['best_iou_stu']
        best_epoch_tea = checkpoint['best_epoch_tea']
        best_epoch_stu = checkpoint['best_epoch_stu']
        pretrained_dict = checkpoint['state_dict']
        for key in model.keys():
            model_dict = model[key].state_dict()
            pretrained_dict[key] = {k:v for k,v in pretrained_dict[key].items() if k in model_dict}
            model_dict.update(pretrained_dict[key])
            model[key].load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        LOG.info("=> best_iou_tea is {} at epoch {}".format(best_iou_tea, best_epoch_tea))
        LOG.info("=> best_iou_stu is {} at epoch {}".format(best_iou_stu, best_epoch_stu))
    
    lr_exp_update_flag = False

    # Training
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        with open(text_dir + 'results.txt', "r+") as f:   
            f.truncate()

        # # # train for each epoch
        train(train_loader, model, optimizer, scheduler, epoch, trainwriter, lr_exp_update_flag)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        # # # validation for each epoch
        start_time = time.time()
        LOG.info("Evaluating the model:")
        validate(eval_loader, model, global_step, epoch, evalwriter, ema_evalwriter)

        with open(text_dir + 'results.txt', "r") as file:   #将中间结构写入tensorboard
            textwriter.add_text('print', file.read().replace('\n', '  \n'), epoch+1)


        # update the learning rate
        scheduler.step()

        # save the model
        state_dicts = {}
        for model_name, model_item in model.items():
            state_dict = model_item.state_dict()
            state_dicts[model_name] = state_dict
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': state_dicts,   
                'best_iou_tea': best_iou_tea,
                'best_iou_stu': best_iou_stu,  
                'best_epoch_tea': best_epoch_tea,
                'best_epoch_stu': best_epoch_stu,
                'optimizer' : optimizer.state_dict(),
            }, is_best_tea, is_best_stu, checkpoint_path, epoch + 1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))   
