#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import torch
import numpy as np
import os.path as osp
import torch.utils.data as data


from data.config import cfg
from utils.augmentations import S3FDAugmentation, S3FDValTransform, jaccard_numpy
import random


class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, transform=None, train=True):
        super(WIDERDetection, self).__init__()
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            self.fnames.append(line[0])
            num_faces = int(line[1])
            box = []
            label = []
            for i in xrange(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                box.append([x, y, x + w, y + h])
                label.append(c)
            self.boxes.append(box)
            self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)

        return img, target

    def pull_item(self, index):
        img_path = self.fnames[index]
        img = cv2.imread(img_path)

        height, width, channel = img.shape

        boxes = np.array(self.boxes[index])
        label = np.array(self.labels[index])

        if self.transform is not None:
            if self.train:
                img, boxes, label = self.random_crop(img, boxes, label)
            else:
                img, boxes, label = self.filter_sample(img, boxes, label)
            img, boxes, label = self.transform(img, boxes, label)

            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, label[:, np.newaxis]))
        '''
        boxes*=640
        for bbox in boxes:
            left_up = (int(bbox[0]), int(bbox[1]))
            right_bottom = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
        cv2.imwrite('image.jpg', img)
        '''
        
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def random_getim(self):
        idx = random.randrange(0, self.num_samples)
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        box = np.array(self.boxes[idx])
        label = np.array(self.labels[idx])

        return img, box, label

    def filter_sample(self, image, boxes, labels):
        height, width, _ = image.shape
        while True:
            boxes_uniform = boxes / \
                np.array([width, height, width, height])

            boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
            mask = (boxwh[:, 0] > cfg.FILTER_THRESH) & (
                boxwh[:, 1] > cfg.FILTER_THRESH)
            if not mask.any():
                image, boxes, labels = self.random_getim()
                height, width, _ = image.shape
                continue

            boxes = boxes[mask, :]
            labels = labels[mask]

            return image, boxes, labels

    def random_crop(self, image, boxes, labels):
        imh, imw, _ = image.shape
        short_size = min(imh, imw)
        sample_options = [None, 0.3, 0.5, 0.7, 0.9]
        while True:
            mode = np.random.choice(sample_options)
            if mode is None:
                boxes_uniform = boxes / (np.array([imw, imh, imw, imh]))
                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > cfg.FILTER_THRESH) & (
                    boxwh[:, 1] > cfg.FILTER_THRESH)
                if not mask.any():
                    image, boxes, labels = self.random_getim()
                    imh, imw, _ = image.shape
                    short_size = min(imh, imw)
                    continue

                boxes = boxes[mask, :]
                labels = labels[mask]

                return image, boxes, labels

            for _ in range(10):
                w = random.randrange(int(0.3 * short_size), short_size)
                h = w

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)

                rect = np.array([x, y, x + w, y + h])

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                current_image = image[rect[1]:rect[3], rect[0]:rect[2], :]

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(
                    current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                boxes_uniform = current_boxes / np.array([imw, imh, imw, imh])

                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > cfg.FILTER_THRESH) & (
                    boxwh[:, 1] > cfg.FILTER_THRESH)
                if not mask.any():
                    image, boxes, labels = self.random_getim()
                    imh, imw, _ = image.shape
                    short_size = min(imh, imw)
                    continue

                current_boxes = current_boxes[mask, :].copy()
                # take only matching gt labels
                current_labels = current_labels[mask]
                return current_image, current_boxes, current_labels
    '''
    def random_crop(self, image, boxes, labels):
        sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(sample_options)
            if mode is None:
                boxes_uniform = boxes / \
                    np.array([width, height, width, height])

                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > cfg.FILTER_THRESH) & (
                    boxwh[:, 1] > cfg.FILTER_THRESH)
                if not mask.any():
                    image, boxes, labels = self.random_getim()
                    height, width, _ = image.shape
                    continue

                boxes = boxes[mask, :]
                labels = labels[mask]

                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)
                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                # cut the crop from the image
                current_image = current_image[
                    rect[1]:rect[3], rect[0]:rect[2], :]
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(
                    current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                boxes_uniform = current_boxes / \
                    np.array([width, height, width, height])

                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > cfg.FILTER_THRESH) & (
                    boxwh[:, 1] > cfg.FILTER_THRESH)
                if not mask.any():
                    image, boxes, labels = self.random_getim()
                    height, width, _ = image.shape
                    continue

                current_boxes = current_boxes[mask, :].copy()
                # take only matching gt labels
                current_labels = current_labels[mask]
                return current_image, current_boxes, current_labels

        '''
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    dataset = WIDERDetection(
        cfg.TRAIN_FILE, transform=S3FDAugmentation(), train=True)
    # for i in range(1000):
    dataset.pull_item(1006)
