# --------------------------------------------------------
# TFFRCNN - Resnet50
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by miraclebiu
# --------------------------------------------------------
import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class Resnet50_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []

        self.data1 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data1')
        self.data2 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data2')

        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data1':self.data1, 'data2':self.data2, 'im_info':self.im_info })
        self.trainable = trainable
        self.setup()

    def setup(self):
        with tf.variable_scope("siamese") as scope:
            feature1, roi1, confidence1= self.object_detection('data1', 'gt_boxes1')
            scope.reuse_variables()
            feature2, roi2, confidence2 = self.object_detection('data2', 'gt_boxes2')

        (self.feed(roi1, confidence1, roi2, confidence2)
         .crp(name='crp'))

        roi = self.get_output('crp')
        inputs = self.crop(feature1, roi)
        targets = self.crop(feature2, roi)

        inputs_norm = tf.tanh(inputs)
        targets_norm = tf.tanh(targets)

        with tf.variable_scope("siamese") as scope:
            true_pair = tf.concat([inputs_norm, targets_norm], axis=3)
            score1 = self.Discriminator(true_pair)
            self.score1 = tf.sigmoid(score1)

    def object_detection(self,x, box):
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed(x)
         .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
         .batch_normalization(relu=True, name='bn_conv1', is_training=False)
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(name='bn2a_branch1', is_training=False, relu=False))

        (self.feed('pool1')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(relu=True, name='bn2a_branch2a', is_training=False)
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(relu=True, name='bn2a_branch2b', is_training=False)
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
         .batch_normalization(name='bn2a_branch2c', is_training=False, relu=False))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(relu=True, name='bn2b_branch2a', is_training=False)
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(relu=True, name='bn2b_branch2b', is_training=False)
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(name='bn2b_branch2c', is_training=False, relu=False))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
         .batch_normalization(relu=True, name='bn2c_branch2a', is_training=False)
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
         .batch_normalization(relu=True, name='bn2c_branch2b', is_training=False)
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
         .batch_normalization(name='bn2c_branch2c', is_training=False, relu=False))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add(name='res2c')
         .relu(name='res2c_relu')
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1', padding='VALID')
         .batch_normalization(name='bn3a_branch1', is_training=False, relu=False))

        (self.feed('res2c_relu')
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a', padding='VALID')
         .batch_normalization(relu=True, name='bn3a_branch2a', is_training=False)
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(relu=True, name='bn3a_branch2b', is_training=False)
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
         .batch_normalization(name='bn3a_branch2c', is_training=False, relu=False))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
         .batch_normalization(relu=True, name='bn3b_branch2a', is_training=False)
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
         .batch_normalization(relu=True, name='bn3b_branch2b', is_training=False)
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
         .batch_normalization(name='bn3b_branch2c', is_training=False, relu=False))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
         .add(name='res3b')
         .relu(name='res3b_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
         .batch_normalization(relu=True, name='bn3c_branch2a', is_training=False)
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
         .batch_normalization(relu=True, name='bn3c_branch2b', is_training=False)
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
         .batch_normalization(name='bn3c_branch2c', is_training=False, relu=False))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
         .add(name='res3c')
         .relu(name='res3c_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
         .batch_normalization(relu=True, name='bn3d_branch2a', is_training=False)
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
         .batch_normalization(relu=True, name='bn3d_branch2b', is_training=False)
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
         .batch_normalization(name='bn3d_branch2c', is_training=False, relu=False))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
         .add(name='res3d')
         .relu(name='res3d_relu')
         .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1', padding='VALID')
         .batch_normalization(name='bn4a_branch1', is_training=False, relu=False))

        (self.feed('res3d_relu')
         .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a', padding='VALID')
         .batch_normalization(relu=True, name='bn4a_branch2a', is_training=False)
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(relu=True, name='bn4a_branch2b', is_training=False)
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
         .batch_normalization(name='bn4a_branch2c', is_training=False, relu=False))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(relu=True, name='bn4b_branch2a', is_training=False)
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(relu=True, name='bn4b_branch2b', is_training=False)
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
         .batch_normalization(name='bn4b_branch2c', is_training=False, relu=False))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
         .batch_normalization(relu=True, name='bn4c_branch2a', is_training=False)
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
         .batch_normalization(relu=True, name='bn4c_branch2b', is_training=False)
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
         .batch_normalization(name='bn4c_branch2c', is_training=False, relu=False))

        (self.feed('res4b_relu',
                   'bn4c_branch2c')
         .add(name='res4c')
         .relu(name='res4c_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
         .batch_normalization(relu=True, name='bn4d_branch2a', is_training=False)
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
         .batch_normalization(relu=True, name='bn4d_branch2b', is_training=False)
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
         .batch_normalization(name='bn4d_branch2c', is_training=False, relu=False))

        (self.feed('res4c_relu',
                   'bn4d_branch2c')
         .add(name='res4d')
         .relu(name='res4d_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
         .batch_normalization(relu=True, name='bn4e_branch2a', is_training=False)
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
         .batch_normalization(relu=True, name='bn4e_branch2b', is_training=False)
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
         .batch_normalization(name='bn4e_branch2c', is_training=False, relu=False))

        (self.feed('res4d_relu',
                   'bn4e_branch2c')
         .add(name='res4e')
         .relu(name='res4e_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
         .batch_normalization(relu=True, name='bn4f_branch2a', is_training=False)
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
         .batch_normalization(relu=True, name='bn4f_branch2b', is_training=False)
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
         .batch_normalization(name='bn4f_branch2c', is_training=False, relu=False))

        (self.feed('res4e_relu',
                   'bn4f_branch2c')
         .add(name='res4f')
         .relu(name='res4f_relu'))

        # ========= RPN ============
        (self.feed('res4f_relu')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))
        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        # ========= RoI Proposal ============
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rpn_rois'))

        # (self.feed('rpn_rois', 'gt_boxes', 'gt_ishard', 'dontcare_areas')
        #  .proposal_target_layer(n_classes, name='roi-data'))
        roi = self.get_output('rpn_rois')

        return self.get_output('res4f_relu'), roi[0], roi[1]

    def crop(self, feature, roi):
        (self.feed(feature, roi)
         .roi_pool(8, 8, 1.0 / 16, name='res5a_branch2a_roipooling'))
        return self.get_output('res5a_branch2a_roipooling')


    def Discriminator(self, pair):
        (self.feed(pair)
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a', padding='VALID')
         .batch_normalization(relu=True, name='bn5a_branch2a', is_training=False)
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
         .batch_normalization(relu=True, name='bn5a_branch2b', is_training=False)
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
         .batch_normalization(name='bn5a_branch2c', is_training=False, relu=False))

        (self.feed(pair)
         .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1', padding='VALID')
         .batch_normalization(name='bn5a_branch1', is_training=False, relu=False))

        (self.feed('bn5a_branch2c', 'bn5a_branch1')
         .add(name='res5a')
         .relu(name='res5a_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
         .batch_normalization(relu=True, name='bn5b_branch2a', is_training=False)
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
         .batch_normalization(relu=True, name='bn5b_branch2b', is_training=False)
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
         .batch_normalization(name='bn5b_branch2c', is_training=False, relu=False))
        # pdb.set_trace()
        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add(name='res5b')
         .relu(name='res5b_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
         .batch_normalization(relu=True, name='bn5c_branch2a', is_training=False)
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
         .batch_normalization(relu=True, name='bn5c_branch2b', is_training=False)
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
         .batch_normalization(name='bn5c_branch2c', is_training=False, relu=False))
        # pdb.set_trace()
        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add(name='res5c')
         .relu(name='res5c_relu')
         .fc(2, relu=False, name='cls_score'))
        return self.get_output('cls_score')