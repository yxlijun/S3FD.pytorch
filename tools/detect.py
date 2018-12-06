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

import cv2
import time
import numpy as np

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import S3FDBasicTransform


parser = argparse.ArgumentParser(description='s3df evaluatuon fddb')
parser.add_argument('--trained_model', type=str,
                    default='weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.1, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


net = build_s3fd('train', cfg.INPUT_SIZE, cfg.NUM_CLASSES)
net.load_state_dict(torch.load(args.trained_model))
net.eval()


def dyna_anchor(imh,imw):
	pass
