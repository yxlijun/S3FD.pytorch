#-*-coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import csv

from data.config import cfg

if not os.path.exists('./data'):
    os.makedirs('./data')

TRAIN_ROOT = os.path.join(cfg.HAND.DIR, 'images', 'train')
TEST_ROOT = os.path.join(cfg.HAND.DIR, 'images', 'test')


def generate_file(csv_file, target_file, root):
    filenames = []
    bboxes = []
    with open(csv_file, 'rb') as sd:
        lines = csv.DictReader(sd)
        for line in lines:
            filenames.append(os.path.join(root, line['filename']))
            bbox = [int(line['xmin']), int(line['ymin']),
                    int(line['xmax']), int(line['ymax'])]
            bboxes.append(bbox)

    filenames = np.array(filenames)
    bboxes = np.array(bboxes)
    uniq_filenames = np.unique(filenames)

    fout = open(target_file, 'w')

    for name in uniq_filenames:
        idx = np.where(filenames == name)[0]
        bbox = bboxes[idx]
        fout.write('{} '.format(name))
        fout.write(('{} ').format(len(bbox)))
        for loc in bbox:
            x1, y1, x2, y2 = loc
            fout.write('{} {} {} {} '.format(x1, y1, x2, y2))
        fout.write('\n')
    fout.close()


if __name__ == '__main__':
    train_csv_file = os.path.join(TRAIN_ROOT, 'train_labels.csv')
    test_csv_file = os.path.join(TEST_ROOT, 'test_labels.csv')
    generate_file(train_csv_file, cfg.HAND.TRAIN_FILE, TRAIN_ROOT)
    generate_file(test_csv_file, cfg.HAND.VAL_FILE, TEST_ROOT)
