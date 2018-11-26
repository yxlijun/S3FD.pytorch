#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data import *
from s3fd import build_s3fd
from utils.augmentations import S3FDAugmentation
from layers.modules import MultiBoxLoss


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='WIDER', choices=['WIDER', 'AFW'],
                    type=str, help='WIDER or AFW')
parser.add_argument('--dataset_root', default=WIDER_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


def train():
    cfg = wider
    dataset = WIDERDetection(root=args.dataset_root,
                             transform=S3FDAugmentation(cfg['min_dim'], MEANS),
                             training=True)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    s3fd_net = build_s3fd('train', cfg['min_dim'], cfg['num_classes'])
    net = s3fd_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        s3fd_net.load_weights(args.resume)

    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Load base network....')
        s3fd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
        cudnn.benckmark = True

    if not args.resume:
        print('Initializing weights...')
        s3fd_net.extras.apply(weights_init)
        s3fd_net.loc.apply(weights_init)
        s3fd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], [0.1, 0.35, 0.5], 3, 0, True, 3, 0.5,
                             args.cuda)
    #criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 3, 0, True, 3, 0.5,
    #                         args.cuda)

    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)
    step_index = 0
    iteration = 0
    # loss counters

    net.train()
    for epoch in xrange(250):
        loc_loss = 0
        conf_loss = 0
        for batch_idx, (images, targets) in enumerate(data_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            if iteration % 10 == 0:
                locloss = loc_loss / (batch_idx + 1)
                confloss = conf_loss / (batch_idx + 1)
                tloss = locloss + confloss
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || loc_Loss: %.4f || conf_Loss:%.4f || Loss:%.4f' %
                      (locloss,confloss,tloss), end=' ')

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/s3fd_WIDER_' +
                           repr(iteration) + '.pth')
            iteration += 1


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
