#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function



HOME = '/home/data/lj/'


MEANS = (104, 117, 123)


wider = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [160, 80, 40, 20, 10, 5],
    'min_dim':  640,
    'steps': [4, 8, 16, 32, 64, 128],
    'min_sizes': [16, 32, 64, 128, 256, 512],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'WIDER',
}
