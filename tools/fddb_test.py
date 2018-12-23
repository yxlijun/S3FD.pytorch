#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

parser = argparse.ArgumentParser(description='s3fd evaluatuon fddb')
parser.add_argument('--model', type=str,
                    default='weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.1, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


FDDB_IMG_DIR = os.path.join(cfg.FACE.FDDB_DIR, 'images')
FDDB_FOLD_DIR = os.path.join(cfg.FACE.FDDB_DIR, 'FDDB-folds')
FDDB_RESULT_DIR = os.path.join(cfg.FACE.FDDB_DIR, 's3fd')
FDDB_RESULT_IMG_DIR = os.path.join(FDDB_RESULT_DIR, 'images')

if not os.path.exists(FDDB_RESULT_IMG_DIR):
    os.makedirs(FDDB_RESULT_IMG_DIR)


def detect_face(net, img, thresh):
    height, width, _ = img.shape
    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    bboxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            box = []
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(np.int)
            j += 1
            box += [pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1], score]
            bboxes += [box]

    return bboxes


if __name__ == '__main__':
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    counter = 0

    for i in range(10):
        txt_in = os.path.join(FDDB_FOLD_DIR, 'FDDB-fold-%02d.txt' % (i + 1))
        txt_out = os.path.join(FDDB_RESULT_DIR, 'fold-%02d-out.txt' % (i + 1))
        answer_in = os.path.join(
            FDDB_FOLD_DIR, 'FDDB-fold-%02d-ellipseList.txt' % (i + 1))
        with open(txt_in, 'r') as fr:
            lines = fr.readlines()
        fout = open(txt_out, 'w')
        ain = open(answer_in, 'r')
        for line in lines:
            line = line.strip()
            img_file = os.path.join(FDDB_IMG_DIR, line + '.jpg')
            out_file = os.path.join(
                FDDB_RESULT_IMG_DIR, line.replace('/', '_') + '.jpg')
            counter += 1
            t1 = time.time()
            #img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = Image.open(img_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)
            bboxes = detect_face(net, img, args.thresh)
            t2 = time.time()
            print('Detect %04d th image costs %.4f' % (counter, t2 - t1))
            fout.write('%s\n' % line)
            fout.write('%d\n' % len(bboxes))
            for bbox in bboxes:
                x1, y1, w, h, score = bbox
                fout.write('%d %d %d %d %lf\n' % (x1, y1, w, h, score))
            ain.readline()
            n = int(ain.readline().strip())
            for i in range(n):
                line = ain.readline().strip()
                line_data = [float(_) for _ in line.split(' ')[:5]]
                major_axis_radius, minor_axis_radius, angle, center_x, center_y = line_data
                angle = angle / 3.1415926 * 180.
                center_x, center_y = int(center_x), int(center_y)
                major_axis_radius, minor_axis_radius = int(
                    major_axis_radius), int(minor_axis_radius)
                cv2.ellipse(img, (center_x, center_y), (major_axis_radius,
                                                        minor_axis_radius), angle, 0, 360, (255, 0, 0), 2)

            for bbox in bboxes:
                x1, y1, w, h, score = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(out_file, img)
        fout.close()
        ain.close()
