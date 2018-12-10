#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


from easydict import EasyDict

cfg = EasyDict()

# dataset config
cfg.HOME = '/home/data/lj/'
cfg.TRAIN_FILE = './data/train.txt'
cfg.VAL_FILE = './data/val.txt'

# evalution config
cfg.FDDB_DIR = '/home/data/lj/FDDB'
cfg.WIDER_DIR = '/home/data/lj/WIDER'
cfg.AFW_DIR = '/home/data/lj/AFW'
cfg.PASCAL_DIR = '/home/data/lj/PASCAL_FACE'


cfg.MEANS = (104, 117, 123)
# train config
cfg.LR_STEPS = (198, 250)
cfg.EPOCHES = 300


# anchor config
cfg.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
cfg.INPUT_SIZE = 640
cfg.STEPS = [4, 8, 16, 32, 64, 128]
cfg.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
cfg.CLIP = True
cfg.VARIANCE = [0.1, 0.2]

##
cfg.FILTER_THRESH = 6 / cfg.INPUT_SIZE

# loss config
cfg.NUM_CLASSES = 2
cfg.OVERLAP_THRESH = [0.1, 0.35, 0.5]
cfg.NEG_POS_RATIOS = 3
cfg.GAMMA = 4


# detection config
cfg.NMS_THRESH = 0.5
cfg.TOP_K = 1000
cfg.CONF_THRESH = 0.05
