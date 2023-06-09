# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.cls_function import validate, validate_observe_train
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('--epsilon',
                        help='epsilon for perturbation range',
                        type=int,
                        default=8)
    parser.add_argument('--step_size',
                        help='step size alpha for bias iter',
                        type=float,
                        default=2)
    parser.add_argument('--num_steps',
                        help='num of steps for iterative perturbation',
                        type=int,
                        default=10)
    parser.add_argument('--attack',
                        help='specify which type of attack to use, default: pgd',
                        type=str,
                        default="pgd")
    parser.add_argument('--TrainGrad',
                        help='which type of grad to have trained: real, phantom unroll, phantom neumann',
                        required=True,
                        type=str)
    parser.add_argument('--AttackGrad',
                        help='which type of grad to attack: real, phantom unroll, phantom neumann',
                        required=True,
                        type=str)
    parser.add_argument('--isObserveTrain',
                        help='if True, use train loader for observation',
                        action='store_true')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    log_file_setting = "train-{}-attack-{}".format(args.TrainGrad, args.AttackGrad)
    log_file = "{}-{}.log".format(log_file_setting, args.attack)
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid', log_file)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir, '{}-checkpoint.pth.tar'.format(args.TrainGrad))

        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file)['state_dict'])

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    dataset_name = config.DATASET.DATASET
    topk = (1,5) if dataset_name == 'imagenet' else (1,)
    # Data loading code
    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
 #           normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        valid_dataset = datasets.ImageFolder(valdir, transform_valid)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference
        
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        transform_train = transforms.Compose([
            transforms.ToTensor(),
#            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
#            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True, transform=transform_valid)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate_observe_train(config, valid_loader if not args.isObserveTrain else train_loader, model, criterion, None, epoch=-1, output_dir=final_output_dir,
             tb_log_dir=tb_log_dir, writer_dict=None, topk=topk, spectral_radius_mode=config.DEQ.SPECTRAL_RADIUS_MODE,
             epsilon=args.epsilon/255, step_size=args.step_size/255, num_steps=args.num_steps, attack=args.attack, AttackGrad=args.AttackGrad, TrainGrad=args.TrainGrad, logfile=log_file_setting, isTrain=args.isObserveTrain)

if __name__ == '__main__':
    main()
