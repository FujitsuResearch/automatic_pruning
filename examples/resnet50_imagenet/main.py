# main.py COPYRIGHT Fujitsu Limited 2022

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import copy
from collections import OrderedDict
from resnet50 import ResNet50
import sys
sys.path.append('../../')
from auto_prune_imagenet import auto_prune

#===================================================================================
parser = argparse.ArgumentParser(description='PyTorch ImageNet Automatic Pruning')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers')
# for re-training
parser.add_argument('--data', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
parser.add_argument('--nesterov', default=False)
parser.add_argument('--scheduler_timing', type=str, default='epoch',
                    help="set LR change timing by LR_scheduler. 'epoch': execute scheduler.step() for each epoch. 'iter' : Execute scheduler.step() for each iteration")
# for stepLR scheduler
parser.add_argument('--lr-milestone', type=list, default=[30, 60])
parser.add_argument('--lr-gamma', type=float, default=0.1)
# for auto pruning
parser.add_argument('--acc_control', type=float, default=0.05,
                    help='control parameter for pruned model accuracy')
parser.add_argument('--loss_margin', type=float, default=0.1,
                    help='control parameter for loss function margin to derive threshold')
parser.add_argument('--rates', nargs='*', type=float, default=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    help='candidates for pruning rates')
parser.add_argument('--max_search_times', type=int, default=1000,
                    help='maximum number of times for pruning rate search')
parser.add_argument('--epochs', type=int, default=90,
                    help='re-training epochs')
parser.add_argument('--pretrained_model_path', type=str, default='./resnet50-5c106cde.pth',
                    help='pre-trained model filepath')
parser.add_argument('--pruned_model_path', type=str, default='./pruned_imagenet_resnet50.pt',
                    help='pruned model filepath')
#===================================================================================

model_info = OrderedDict()
model_info['conv1'] = {'arg': 'ch_conv1'}
model_info['bn1']   = {'arg': 'ch_conv1'}


model_info['layer1.0.conv1'] = {'arg': 'ch_l10_1'}
model_info['layer1.0.bn1']   = {'arg': 'ch_l10_1'}
model_info['layer1.0.conv2'] = {'arg': 'ch_l10_2'}
model_info['layer1.0.bn2']   = {'arg': 'ch_l10_2'}
model_info['layer1.0.conv3'] = {'arg': 'ch_l10_3'}
model_info['layer1.0.bn3']   = {'arg': 'ch_l10_3'}
model_info['layer1.0.downsample.0'] = {'arg': 'ch_l10_ds', 'prev': ['bn1']}
model_info['layer1.0.downsample.1'] = {'arg': 'ch_l10_ds'}

model_info['layer1.1.conv1'] = {'arg': 'ch_l11_1', 'prev': ['layer1.0.downsample.1', 'layer1.0.bn3']}
model_info['layer1.1.bn1']   = {'arg': 'ch_l11_1'}
model_info['layer1.1.conv2'] = {'arg': 'ch_l11_2'}
model_info['layer1.1.bn2']   = {'arg': 'ch_l11_2'}
model_info['layer1.1.conv3'] = {'arg': 'ch_l11_3'}
model_info['layer1.1.bn3']   = {'arg': 'ch_l11_3'}

model_info['layer1.2.conv1'] = {'arg': 'ch_l12_1', 'prev': ['layer1.0.downsample.1', 'layer1.0.bn3', 'layer1.1.bn3']}
model_info['layer1.2.bn1']   = {'arg': 'ch_l12_1'}
model_info['layer1.2.conv2'] = {'arg': 'ch_l12_2'}
model_info['layer1.2.bn2']   = {'arg': 'ch_l12_2'}
model_info['layer1.2.conv3'] = {'arg': 'ch_l12_3'}
model_info['layer1.2.bn3']   = {'arg': 'ch_l12_3'}



model_info['layer2.0.conv1'] = {'arg': 'ch_l20_1', 'prev': ['layer1.0.downsample.1', 'layer1.0.bn3', 'layer1.1.bn3', 'layer1.2.bn3']}
model_info['layer2.0.bn1']   = {'arg': 'ch_l20_1'}
model_info['layer2.0.conv2'] = {'arg': 'ch_l20_2'}
model_info['layer2.0.bn2']   = {'arg': 'ch_l20_2'}
model_info['layer2.0.conv3'] = {'arg': 'ch_l20_3'}
model_info['layer2.0.bn3']   = {'arg': 'ch_l20_3'}
model_info['layer2.0.downsample.0'] = {'arg': 'ch_l20_ds', 'prev': ['layer1.0.downsample.1', 'layer1.0.bn3', 'layer1.1.bn3', 'layer1.2.bn3']}
model_info['layer2.0.downsample.1'] = {'arg': 'ch_l20_ds'}

model_info['layer2.1.conv1'] = {'arg': 'ch_l21_1', 'prev': ['layer2.0.downsample.1', 'layer2.0.bn3']}
model_info['layer2.1.bn1']   = {'arg': 'ch_l21_1'}
model_info['layer2.1.conv2'] = {'arg': 'ch_l21_2'}
model_info['layer2.1.bn2']   = {'arg': 'ch_l21_2'}
model_info['layer2.1.conv3'] = {'arg': 'ch_l21_3'}
model_info['layer2.1.bn3']   = {'arg': 'ch_l21_3'}

model_info['layer2.2.conv1'] = {'arg': 'ch_l22_1', 'prev': ['layer2.0.downsample.1', 'layer2.0.bn3', 'layer2.1.bn3']}
model_info['layer2.2.bn1']   = {'arg': 'ch_l22_1'}
model_info['layer2.2.conv2'] = {'arg': 'ch_l22_2'}
model_info['layer2.2.bn2']   = {'arg': 'ch_l22_2'}
model_info['layer2.2.conv3'] = {'arg': 'ch_l22_3'}
model_info['layer2.2.bn3']   = {'arg': 'ch_l22_3'}

model_info['layer2.3.conv1'] = {'arg': 'ch_l23_1', 'prev': ['layer2.0.downsample.1', 'layer2.0.bn3', 'layer2.1.bn3', 'layer2.2.bn3']}
model_info['layer2.3.bn1']   = {'arg': 'ch_l23_1'}
model_info['layer2.3.conv2'] = {'arg': 'ch_l23_2'}
model_info['layer2.3.bn2']   = {'arg': 'ch_l23_2'}
model_info['layer2.3.conv3'] = {'arg': 'ch_l23_3'}
model_info['layer2.3.bn3']   = {'arg': 'ch_l23_3'}



model_info['layer3.0.conv1'] = {'arg': 'ch_l30_1', 'prev': ['layer2.0.downsample.1', 'layer2.0.bn3', 'layer2.1.bn3', 'layer2.2.bn3', 'layer2.3.bn3']}
model_info['layer3.0.bn1']   = {'arg': 'ch_l30_1'}
model_info['layer3.0.conv2'] = {'arg': 'ch_l30_2'}
model_info['layer3.0.bn2']   = {'arg': 'ch_l30_2'}
model_info['layer3.0.conv3'] = {'arg': 'ch_l30_3'}
model_info['layer3.0.bn3']   = {'arg': 'ch_l30_3'}
model_info['layer3.0.downsample.0'] = {'arg': 'ch_l30_ds', 'prev': ['layer2.0.downsample.1', 'layer2.0.bn3', 'layer2.1.bn3', 'layer2.2.bn3', 'layer2.3.bn3']}
model_info['layer3.0.downsample.1'] = {'arg': 'ch_l30_ds'}

model_info['layer3.1.conv1'] = {'arg': 'ch_l31_1', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3']}
model_info['layer3.1.bn1']   = {'arg': 'ch_l31_1'}
model_info['layer3.1.conv2'] = {'arg': 'ch_l31_2'}
model_info['layer3.1.bn2']   = {'arg': 'ch_l31_2'}
model_info['layer3.1.conv3'] = {'arg': 'ch_l31_3'}
model_info['layer3.1.bn3']   = {'arg': 'ch_l31_3'}

model_info['layer3.2.conv1'] = {'arg': 'ch_l32_1', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3', 'layer3.1.bn3']}
model_info['layer3.2.bn1']   = {'arg': 'ch_l32_1'}
model_info['layer3.2.conv2'] = {'arg': 'ch_l32_2'}
model_info['layer3.2.bn2']   = {'arg': 'ch_l32_2'}
model_info['layer3.2.conv3'] = {'arg': 'ch_l32_3'}
model_info['layer3.2.bn3']   = {'arg': 'ch_l32_3'}

model_info['layer3.3.conv1'] = {'arg': 'ch_l33_1', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3', 'layer3.1.bn3', 'layer3.2.bn3']}
model_info['layer3.3.bn1']   = {'arg': 'ch_l33_1'}
model_info['layer3.3.conv2'] = {'arg': 'ch_l33_2'}
model_info['layer3.3.bn2']   = {'arg': 'ch_l33_2'}
model_info['layer3.3.conv3'] = {'arg': 'ch_l33_3'}
model_info['layer3.3.bn3']   = {'arg': 'ch_l33_3'}

model_info['layer3.4.conv1'] = {'arg': 'ch_l34_1', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3', 'layer3.1.bn3', 'layer3.2.bn3', 'layer3.3.bn3']}
model_info['layer3.4.bn1']   = {'arg': 'ch_l34_1'}
model_info['layer3.4.conv2'] = {'arg': 'ch_l34_2'}
model_info['layer3.4.bn2']   = {'arg': 'ch_l34_2'}
model_info['layer3.4.conv3'] = {'arg': 'ch_l34_3'}
model_info['layer3.4.bn3']   = {'arg': 'ch_l34_3'}

model_info['layer3.5.conv1'] = {'arg': 'ch_l35_1', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3', 'layer3.1.bn3', 'layer3.2.bn3', 'layer3.3.bn3', 'layer3.4.bn3']}
model_info['layer3.5.bn1']   = {'arg': 'ch_l35_1'}
model_info['layer3.5.conv2'] = {'arg': 'ch_l35_2'}
model_info['layer3.5.bn2']   = {'arg': 'ch_l35_2'}
model_info['layer3.5.conv3'] = {'arg': 'ch_l35_3'}
model_info['layer3.5.bn3']   = {'arg': 'ch_l35_3'}



model_info['layer4.0.conv1'] = {'arg': 'ch_l40_1', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3', 'layer3.1.bn3', 'layer3.2.bn3', 'layer3.3.bn3', 'layer3.4.bn3', 'layer3.5.bn3']}
model_info['layer4.0.bn1']   = {'arg': 'ch_l40_1'}
model_info['layer4.0.conv2'] = {'arg': 'ch_l40_2'}
model_info['layer4.0.bn2']   = {'arg': 'ch_l40_2'}
model_info['layer4.0.conv3'] = {'arg': 'ch_l40_3'}
model_info['layer4.0.bn3']   = {'arg': 'ch_l40_3'}
model_info['layer4.0.downsample.0'] = {'arg': 'ch_l40_ds', 'prev': ['layer3.0.downsample.1', 'layer3.0.bn3', 'layer3.1.bn3', 'layer3.2.bn3', 'layer3.3.bn3', 'layer3.4.bn3', 'layer3.5.bn3']}
model_info['layer4.0.downsample.1'] = {'arg': 'ch_l40_ds'}

model_info['layer4.1.conv1'] = {'arg': 'ch_l41_1', 'prev': ['layer4.0.downsample.1', 'layer4.0.bn3']}
model_info['layer4.1.bn1']   = {'arg': 'ch_l41_1'}
model_info['layer4.1.conv2'] = {'arg': 'ch_l41_2'}
model_info['layer4.1.bn2']   = {'arg': 'ch_l41_2'}
model_info['layer4.1.conv3'] = {'arg': 'ch_l41_3'}
model_info['layer4.1.bn3']   = {'arg': 'ch_l41_3'}

model_info['layer4.2.conv1'] = {'arg': 'ch_l42_1', 'prev': ['layer4.0.downsample.1', 'layer4.0.bn3', 'layer4.1.bn3']}
model_info['layer4.2.bn1']   = {'arg': 'ch_l42_1'}
model_info['layer4.2.conv2'] = {'arg': 'ch_l42_2'}
model_info['layer4.2.bn2']   = {'arg': 'ch_l42_2'}
model_info['layer4.2.conv3'] = {'arg': 'ch_l42_3'}
model_info['layer4.2.bn3']   = {'arg': 'ch_l42_3'}

model_info['fc'] = {'arg': None, 'prev': ['layer4.0.downsample.1', 'layer4.0.bn3', 'layer4.1.bn3', 'layer4.2.bn3']}
#===================================================================================

def main():
    args = parser.parse_args()
    print(f'args: {args}')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
  
    # load model
    model = ResNet50()
    print('===== model: before pruning ==========')
    print(model)
    
    # optionally pretrained_model_path from a checkpoint
    if args.pretrained_model_path:
        if os.path.isfile(args.pretrained_model_path):
            print("=> loading checkpoint '{}'".format(args.pretrained_model_path))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained_model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained_model_path, map_location=loc)
            model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_model_path))

    use_DataParallel = False

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        use_DataParallel = True
            
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # calculate accuracy with unpruned trained model    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    Ab = validate(val_loader, model, criterion, args)
    print('Accuracy before pruning:', Ab)

    # copy weight for pre-trained model
    if torch.cuda.device_count() > 1 and use_DataParallel:
        weights = copy.deepcopy(model.module.state_dict())
    else:
        weights = copy.deepcopy(model.state_dict())

    ##### tune pruning rate #####
    print('===== start pruning rate tuning =====')
    # set optimizer
    optim_params = dict(lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov)
    ### set LR scheduler
    # step LR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    scheduler_params = dict(milestones=args.lr_milestone, gamma=args.lr_gamma)

    weights, Afinal, n_args_channels = auto_prune(ResNet50, model_info, weights, Ab,
                                                  train_loader, val_loader, criterion,
                                                  optim_type='SGD',
                                                  optim_params=optim_params,
                                                  lr_scheduler=scheduler,
                                                  scheduler_params=scheduler_params,
                                                  update_lr=args.scheduler_timing,
                                                  use_gpu=torch.cuda.is_available(),
                                                  use_DataParallel=use_DataParallel,
                                                  acc_control=args.acc_control,
                                                  loss_margin=args.loss_margin,
                                                  rates=args.rates,
                                                  max_search_times=args.max_search_times,
                                                  epochs=args.epochs,
                                                  model_path=args.pretrained_model_path,
                                                  pruned_model_path=args.pruned_model_path,
                                                  args=args)

    print('===== model: after pruning ==========')
    print(model)
    print('===== Results =====')
    print('Model size before pruning (Byte):', os.path.getsize(args.pretrained_model_path))
    if os.path.exists(args.pruned_model_path):
        print('Model size after pruning  (Byte):',
              os.path.getsize(args.pruned_model_path))
        print('Compression rate                : {:.3f}'.format(
            1-os.path.getsize(args.pruned_model_path)/os.path.getsize(args.pretrained_model_path)))
    else:
        print('Pretrained model can not be pruned...')
    print('Acc. before pruning: {:.2f}'.format(Ab))
    print('Acc. after pruning : {:.2f}'.format(Afinal))
    print('Arguments name & number of channels for pruned model: ', n_args_channels)



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
