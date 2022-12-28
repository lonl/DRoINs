import os
from imdb import imdb
import ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from ..fast_rcnn.config import cfg

class car(imdb):
    def __init__(self, image_set, data_path = None):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._data_path = data_path
        self._classes = ('__background__', 'car',
                         'trucks', 'tractors', 'campingcars', 'motobike', 'bus',
                         'vans', 'others', 'pickups', 'boats', 'plane')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index('train.txt')

        self._roidb_handler = self.gt_roidb
        self.picture_name = []

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self,index):
        image_path = os.path.join(self._data_path,index)
        return image_path

    def _load_image_set_index(self, imagelist):
        image_set_file = os.path.join(self._data_path,imagelist)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_annotation(self,index):

        picture = []

        filename = os.path.join(self._data_path,index,'diff.txt')
        f = open(filename)
        lines = f.readlines()
        num_objs = len(lines) - 3

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # gt_classes = np.zeros((num_objs), dtype=np.int32)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # seg_areas = np.zeros((num_objs), dtype=np.float32)

        # gt2_id = lines.index('gt2\n') - 1
        # diff_id = lines.index('diff\n') - 2
        # for item in ['gt1\n','gt2\n','diff\n']:
        #   lines.remove(item)

        ix = 0
        ids = []
        for line in lines:
            a = line.strip().split()
            if not len(a) == 5:
                ids.append(lines.index(a[0]+'\n'))
                picture.append(a[0])
            else:
                gt, xcenter, ycenter, width, height = a
                gt, xcenter, ycenter, width, height = int(gt), float(xcenter), float(ycenter), float(width),float(height)

                x1 = float((xcenter-width/2)*1024) - 1
                if x1 < 0:
                    x1 = 0
                y1 = float((ycenter-height/2)*1024) - 1
                if y1 < 0:
                    y1 = 0
                x2 = float((xcenter+width/2)*1024) - 1
                if x2 >= 1024:
                    x2 = 1023
                y2 = float((ycenter+height/2)*1024) - 1
                if y2 >= 1024:
                    y2 = 1023

                boxes[ix,:] = [x1, y1, x2, y2]
                ix += 1

        gt2_id = ids[1] - 1
        diff_id = ids[2] - 2
        boxes1 = boxes[:gt2_id,:]
        boxes2 = boxes[gt2_id:diff_id,:]
        diff = boxes[diff_id:,:]

        self.picture_name.append(picture)

        return {'boxes1':boxes1,
                'boxes2':boxes2,
                'dif_boxes':diff,
                'flipped': False,
                }

    def gt_roidb(self):

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


