#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
import torch
import numpy as np
from data.config import cfg
import torch.utils.data as data

from utils.augmentations import S3FDValTransform
from layers.functions import PriorBox
from data.wider_face import WIDERDetection, detection_collate

from layers.bbox_utils import match, match_ssd, decode

import matplotlib.pyplot as plt


dataset = WIDERDetection(
    cfg.TRAIN_FILE, transform=S3FDValTransform(cfg.INPUT_SIZE), train=False)

data_loader = data.DataLoader(dataset, 64,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=detection_collate,
                              pin_memory=True)

anchor_boxes = PriorBox(cfg).forward()
num_priors = anchor_boxes.size(0)
variance = cfg.VARIANCE

savepath = 'tmp'
if not os.path.exists(savepath):
    os.makedirs(savepath)

filename = os.path.join(savepath, 'match_anchor.pkl')


def anchor_match_count():
    anchor_scale_map = {16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0}
    thresh = cfg.OVERLAP_THRESH
    sfd_scales = []
    for idx, (_, target) in enumerate(data_loader):
        num = len(target)

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for index in range(num):
            truths = target[index][:, :-1]
            labels = target[index][:, -1]

            match(thresh, truths, anchor_boxes,
                  variance, labels, loc_t, conf_t, index)

        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_t)
        defaults = anchor_boxes.view(-1, num_priors,
                                     4).expand_as(loc_t).contiguous().view(-1, 4).clone()
        loc_t = loc_t.view(-1, 4)
        decoded_boxes = decode(loc_t, defaults, variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        match_boxes = decoded_boxes[pos_idx].view(-1, 4)

        ori_boxes = match_boxes * cfg.INPUT_SIZE
        wh = ori_boxes[:, 2:] - ori_boxes[:, 0:2]

        scales = torch.sqrt(wh[:, 0] * wh[:, 1])
        scales = scales.numpy().astype(np.int)
        print(scales.shape)

        sfd_scales += [scales]

    sfd_scales = np.concatenate(sfd_scales, axis=0)
    sfd_result = all_np(sfd_scales)

    return sfd_result


def anchor_match_ssd_count():
    anchor_scale_map = {16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0}
    thresh = 0.5
    ssd_scales = []
    for idx, (_, target) in enumerate(data_loader):
        num = len(target)

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for index in range(num):
            truths = target[index][:, :-1]
            labels = target[index][:, -1]

            match_ssd(thresh, truths, anchor_boxes,
                      variance, labels, loc_t, conf_t, index)

        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_t)
        defaults = anchor_boxes.view(-1, num_priors,
                                     4).expand_as(loc_t).contiguous().view(-1, 4).clone()
        loc_t = loc_t.view(-1, 4)
        decoded_boxes = decode(loc_t, defaults, variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        match_boxes = decoded_boxes[pos_idx].view(-1, 4)

        ori_boxes = match_boxes * cfg.INPUT_SIZE
        wh = ori_boxes[:, 2:] - ori_boxes[:, 0:2]

        scales = torch.sqrt(wh[:, 0] * wh[:, 1])
        scales = scales.numpy().astype(np.int)
        print(scales.shape)

        ssd_scales += [scales]

    ssd_scales = np.concatenate(ssd_scales, axis=0)
    ssd_result = all_np(ssd_scales)
    return ssd_result


def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def save_pkl():
    sfd_count = anchor_match_count()
    ssd_count = anchor_match_ssd_count()

    result = {'sfd': sfd_count, 'ssd': ssd_count}
    file = open(filename, 'wb')
    pickle.dump(result, file)
    file.close()


def plot_anchor_match():
    if not os.path.exists(filename):
        save_pkl()

    with open(filename, 'rb') as f:
        result = pickle.load(f)

    sfd_res = result['sfd']
    ssd_res = result['ssd']

    sfd_count = []
    ssd_count = []

    frames = range(0, 660)

    sfd_feat_count = np.zeros(len(frames))
    ssd_feat_count = np.zeros(len(frames))
    feat_maps = np.array([16, 32, 64, 128, 256, 512])

    for i in frames:
        if i in sfd_res.keys():
            sfd_count += [sfd_res[i]]
        else:
            sfd_count += [0]
        if i in ssd_res.keys():
            ssd_count += [ssd_res[i]]
        else:
            ssd_count += [0]

    sfd_count = np.array(sfd_count)
    ssd_count = np.array(ssd_count)

    for feat in feat_maps:
        if feat in sfd_res.keys():
            sfd_feat_count[feat] = sfd_res[feat]
        if feat in ssd_res.keys():
            ssd_feat_count[feat] = ssd_res[feat]

    n = 280
    plt.plot(frames[:n], sfd_count[:n], 'r', label='sfd matching method')
    plt.plot(frames[:n], sfd_feat_count[:n], 'b',
             label='sfd in [16,32,64...] match')

    plt.plot(frames[:n], ssd_count[:n], 'g-', label='ssd matching method')
    plt.plot(frames[:n], ssd_feat_count[:n], 'c-',
             label='ssd in [16,32,64...] match')

    fig1 = plt.figure(1)
    axes = plt.subplot(111)
    axes.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes.grid(True)
    plt.legend(loc="upper right")

    plt.ylabel('Num of match anchor ratio')   # set ystick label
    plt.xlabel('Scale of face')  # set xstck label

    plt.show()


if __name__ == '__main__':
    plot_anchor_match()
