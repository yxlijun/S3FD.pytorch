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

from .config import HOME 
from utils.augmentations import S3FDAugmentation


WIDER_ROOT = os.path.join(HOME, 'WIDER')


def parse_wider_file(root, file_path):
    with open(file_path, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3]),0]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in xrange(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self,
                 root,
                 transform=None,
                 training=True):
        super(WIDERDetection, self).__init__()
        self.root = root
        self.data_file = osp.join(root, 'wider_face_split', 'wider_face_train_bbx_gt.txt') \
            if training else osp.join(root, 'wider_face_split', 'wider_face_val_bbx_gt.txt')

        self.data_path = osp.join(root, 'WIDER_train', 'images') \
            if training else osp.join(self.root, 'WIDER_val', 'images')

        self.transform = transform

        self.img_paths, self.img_faces = parse_wider_file(
            self.data_path, self.data_file)

    def __getitem__(self, index):
        im,gt,h,w = self.pull_item(index)
        
        return im,gt 

    def __len__(self):
        return len(self.img_paths)

    def pull_item(self, index):
        imgpath = self.img_paths[index]
        img = cv2.imread(imgpath)
        height, width, channels = img.shape

        target = self.img_faces[index]
        target = self.norm_hw(target, height, width)

        '''
        for bbox in target:
        	left_up = (bbox[0],bbox[1])
        	right_bottom = (bbox[2],bbox[3])
        	cv2.rectangle(img,left_up,right_bottom,(0,0,255),2)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        '''

        if self.transform is not None:
            target = np.array(target)
            img, boxes,labels = self.transform(img, target[:,:4],target[:,4])

            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes,np.expand_dims(labels,axis=1)))
        '''
        target = boxes
        for bbox in target:
        	left_up = (int(bbox[0]),int(bbox[1]))
        	right_bottom = (int(bbox[2]),int(bbox[3]))
        	cv2.rectangle(img,left_up,right_bottom,(0,0,255),2)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        '''
        return torch.from_numpy(img).permute(2,0,1),target,height,width

    def norm_hw(self, target, height, width):
        for i, bbox in enumerate(target):
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            target[i][0] = bbox[0] / width
            target[i][1] = bbox[1] / height
            target[i][2] = bbox[2] / width
            target[i][3] = bbox[3] / height
        return target



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
	d = WIDERDetection(WIDER_ROOT,transform=S3FDAugmentation())
	d.pull_item(2)