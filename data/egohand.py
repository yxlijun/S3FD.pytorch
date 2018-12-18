#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess


class HandDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(HandDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in xrange(num_faces):
                xmin = float(line[2 + 4 * i])
                ymin = float(line[3 + 4 * i])
                xmax = float(line[4 + 4 * i])
                ymax = float(line[5 + 4 * i])
                box.append([xmin, ymin, xmax, ymax])
                label.append(1)
            self.fnames.append(line[0])
            self.boxes.append(box)
            self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return img, target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        '''
        #img = Image.fromarray(img)        
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in target:
            bbox = (bbox[:-1] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.show()
        '''
        return torch.from_numpy(img), target, im_height, im_width
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


    def pull_image(self,index):
        img_path = self.fnames[index]
        img = Image.open(img_path)
        if img.mode=='L':
            img.convert('RGB')
        img = np.array(img)
        return img


if __name__ == '__main__':
    from data.config import cfg
    dataset = HandDetection(cfg.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(2)
