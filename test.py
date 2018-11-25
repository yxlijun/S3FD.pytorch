#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from data import *
from s3fd import build_s3fd
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='S3FD Face Detection')
parser.add_argument('--trained_model', default='weights/s3fd_640_WIDER.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--wider_root', default=WIDER_ROOT,
                    help='Location of VOC root directory')
args = parser.parse_args()


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    pass


def test_wider():
    cfg = wider
    num_classes = cfg['num_classes']
    net = build_s3fd('test', cfg['min_dim'], num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finish loading model!')

    test_dataset = WIDERDetection(root=args.wider_root,
                                  transform=None,
                                  training=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    test_net(args.save_folder, net, args.cuda, test_dataset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)
    
