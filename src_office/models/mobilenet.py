# -*- coding: utf-8 -*-
"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf

from nets.mobilenet import mobilenet_v2 as mobilenet_v2_builder

slim = tf.contrib.slim

def mobilenet_v2(inputs,
              num_classes,
              depth_multiplier=1.0,
              finegrain_classification_mode=True,
              padding='SAME',
              flag_global_pool=True):
    """mobilenet_v2 model"""
    logits, endpoints = mobilenet_v2_builder.mobilenet(inputs,
                                               num_classes=num_classes,
                                               depth_multiplier=depth_multiplier,
                                               finegrain_classification_mode=finegrain_classification_mode,
                                               padding=padding,
                                               flag_global_pool=flag_global_pool,
                                               )
    endpoints['output'] = logits
    print(logits.shape)
    return logits, endpoints

def training_scope(weight_decay,
                   is_training,
                   stddev,
                   dropout_keep_prob,
                   bn_decay):
    return mobilenet_v2_builder.training_scope(weight_decay=weight_decay,
                                       is_training=is_training,
                                       stddev=stddev,
                                       dropout_keep_prob=dropout_keep_prob,
                                       bn_decay=bn_decay)
