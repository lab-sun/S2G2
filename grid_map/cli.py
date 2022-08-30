import re
import argparse
import logging

from . import  architectures, datasets


LOG = logging.getLogger('main')



def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # dataset & nerwork structure
    parser.add_argument('--dataset', metavar='DATASET', default='cityscapes',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))

    
    # dataloader
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")

    # lr & optimizer
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--lr-scheduler-type', default='exp', type=str,
                        metavar='LR', choices=['cos', 'exp'], help='the learning rate type')
    parser.add_argument('--lr-steps', default=[50], type=float, nargs="+",  # attention
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--lr-type', '--learning-rate-type', default='cos', type=str,
                        metavar='LR', choices=['cos', 'exp'], help='the learning rate type')
    parser.add_argument('--exp-gamma', default=0.9, type=float, 
                        metavar='LR',help='lr decays by gamma every epoch')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    
    # network input
    parser.add_argument('--img-height', default=128, type=int,
                        help='the height of the input image')
    parser.add_argument('--img-width', default=512, type=int,
                        help='the width of the input image')
    parser.add_argument('--pretrained', default=True, type=str2bool,
                        help='use pre-trained model')
    parser.add_argument('--num-class', default=4, type=int,
                        help='the number of the semantic class')

    # path
    parser.add_argument('--labels', default='./data-local/Cityscapes/CS_train_64_labeled.csv', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--train-subdir', default='CS_train_extra191.csv', type=str,
                        help='the CSV file for training data')   #全是有标签的是CS_train_64.csv
    parser.add_argument('--eval-subdir', default='CS_val_64.csv', type=str,
                        help='the CSV file for eval data')
    parser.add_argument('--test-subdir', default='CS_test_64.csv', type=str,
                        help='the CSV file for eval data')
    

    # loss weights
    parser.add_argument('--cycle-view-weight', default=None, type=float, metavar='WEIGHT',
                        help='use cycle view loss with given weight (default: None)')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl', 'dice'], help='consistency loss type to use')
    parser.add_argument('--consistency', default=5, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=100, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--BEV-weight', default=None, type=float, metavar='WEIGHT',
                        help='use BEV loss with given weight (default: None)')
    
    # training related
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--test-epochs', default=1, type=int,
                        metavar='EPOCHS', help='test frequency in epochs, 0 to turn test off (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    # image saving
    parser.add_argument('--image-save-dir', default='./results/images', type=str,
                        help='output images saving dir')
    parser.add_argument('--image-save-flag', default=False, type=str2bool,
                        help='whether save the output images')
                        
    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
