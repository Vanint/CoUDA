#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os, sys
import os.path as osp

import time
import numpy as np
import cv2
import utils
import scipy
from functools import partial
from models.mobilenet import mobilenet_v2, training_scope
from collections import OrderedDict
from load_yaml_env import load_env
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import argparse   
import random
import network

#import tensorflow_probability as tfp
slim = tf.contrib.slim
CLASS_NUMBER = 31

OPTIMIZER_DICT={
        'sgd':tf.train.GradientDescentOptimizer,
        'adagrad':tf.train.AdagradOptimizer,
        'adadelta':tf.train.AdadeltaOptimizer,
        'ftrl':tf.train.FtrlOptimizer,
        'adam':tf.train.AdamOptimizer,
        'rmsprop':tf.train.RMSPropOptimizer
        }
PREFETCH_SIZE = 144 


def read_lines(fname):
    data = open(fname).readlines()
    fnames = []
    labels = []
    for line in data:
        fnames.append(line.split()[0])
        labels.append(int(line.split()[1]))
    return fnames, labels


def train_image_process(fname):
    image_string = tf.read_file(fname)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [256, 256])
    image = tf.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    return image

def test_prep(fname, label):
    image_string = tf.read_file(fname)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [256, 256])
    image = tf.image.central_crop(image, 0.875)
    return image, label, image, label     

def train_prep1(fname, label):
    
    source_data = train_image_process(fname["source"])
    source_label = label["source_label"]
    target_data = train_image_process(fname["target"]) 
    target_label = label["target_label"] 
    return source_data, source_label, target_data, target_label  

def build_psp(net):
    pyrimid = []
    tensors = {}
    img_dim = net.shape[1:4]
    with tf.variable_scope('psp_net'):
        # pooling
        pyrimid.append(slim.avg_pool2d(net, [16,16],
                                       stride=[16,16]))
        pyrimid.append(slim.avg_pool2d(net, [8,8],
                                       stride=[8,8]))
        pyrimid.append(slim.avg_pool2d(net, [4,4],
                                       stride=[4,4]))
        pyrimid.append(slim.avg_pool2d(net, [2,2],
                                       stride=[2,2]))
        # convolution
        for i in range(4):
            pyrimid[i] = slim.conv2d(pyrimid[i], img_dim[2], 1, scope='conv_p%d'%i)
            pyrimid[i] = slim.batch_norm(pyrimid[i])
            pyrimid[i] = tf.nn.relu(pyrimid[i])
            tensors['psp_net/conv_p%d'%i] = pyrimid[i]
        # upsample
        for i in range(4):
            pyrimid[i] = tf.image.resize_images(pyrimid[i], img_dim[:2])
        # merge
        pyrimid.append(net)
        net = tf.concat(pyrimid, 3)
        net = slim.conv2d(net, img_dim[2], 3, scope='conv_pmerge')
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        tensors['psp_net/conv_pmerge'] = pyrimid[i]
    return net, tensors


def get_mobilenet_v2(images, num_classes, is_training, reuse=None, flag_psp=False):
    """ return mobilenet v2 model
    Arguments:
        images: input image tensor
        num_classes: int, number of output classes
        is_training: bool tensor, placeholder for training mask
        reuse: bool, whether reuse the weight
    Return:
        (output, end_points)
        output: output tensor of network
        end_points: a dictionary of network endpoint tensors
    """
    weight_decay=0.00004
    stddev=0.09
    dropout_keep_prob=0.8
    bn_decay=0.997
    depth_multiplier=1.4
    finegrain_classification_mode=True
    padding='SAME'
    flag_global_pool=True
    with slim.arg_scope(training_scope(weight_decay=weight_decay,
                                       is_training=is_training,
                                       stddev=stddev,
                                       dropout_keep_prob=dropout_keep_prob,
                                       bn_decay=bn_decay)):
        if flag_psp:
            logits, end_points = mobilenet_v2(images, 512,
                                              depth_multiplier, finegrain_classification_mode,
                                              padding, flag_global_pool)
            logits = slim.batch_norm(logits)
            logits = tf.nn.relu(logits)
            logits, tmp_tensors = build_psp(logits)
            # class convolution
            logits = slim.conv2d(logits, num_classes, 1, scope='conv_seg',
                              biases_initializer=tf.zeros_initializer(),
                              padding=padding)
            end_points['output'] = logits
        else:
            logits, end_points = mobilenet_v2(images, num_classes,
                                          depth_multiplier, finegrain_classification_mode,
                                          padding, flag_global_pool)
        if num_classes is not None:
            if num_classes > 1:
                net = slim.softmax(logits, scope='predictions')
                end_points['predictions'] = net
            else:
                net = tf.sigmoid(logits, name='predictions')
                end_points['predictions'] = net
        return logits, end_points


def build_model(img_size, num_classes, tensor_is_training=None, tensor_img=None, tensor_label=None, scope=None):
    ''' build a complete model of mobile net training
        In practise, you can only use part of the code to build
        the part you need.
    Arguments:
        img_size: array like, (H, W, C)
        num_classes: int, number of classes of classifier
        is_training: bool tensor,
    Return: tensors, variables, nodes
        tensors: dictionary, all useful tesors of outputs
        variables: dictionary, all variables in the network
        nodes: dictionary, all input nodes 
    '''
    # place holder
    if scope:
        with tf.variable_scope(scope):
            if tensor_img is not None:
                imgs_node = tensor_img
            else:
                imgs_node = tf.placeholder(tf.float32, [None,]+list(img_size), name="image_input")
            if tensor_label is not None:
                labels_node = tensor_label
            else:
                labels_node = tf.placeholder(tf.float32, [None, num_classes])
            if tensor_is_training is not None:
                is_training_node = tensor_is_training
            else:
                is_training_node = tf.placeholder(tf.bool, [])
        
            nodes = {}
            nodes['image'] = imgs_node
            nodes['target'] = labels_node
            nodes['is_training'] = is_training_node
        
            # build mobilenet v2 
            network, tensors = get_mobilenet_v2(imgs_node,
                                              num_classes=num_classes,
                                              is_training=is_training_node)
            variables = dict([k,v] for k,v in enumerate(slim.get_variables_to_restore()))
    else:
         if tensor_img is not None:
            imgs_node = tensor_img
         else:
            imgs_node = tf.placeholder(tf.float32, [None,]+list(img_size), name="image_input")
         if tensor_label is not None:
            labels_node = tensor_label
         else:
            labels_node = tf.placeholder(tf.float32, [None, num_classes])
         if tensor_is_training is not None:
            is_training_node = tensor_is_training
         else:
            is_training_node = tf.placeholder(tf.bool, [])
    
         nodes = {}
         nodes['image'] = imgs_node
         nodes['target'] = labels_node
         nodes['is_training'] = is_training_node
    
         # build mobilenet v2 
         network, tensors = get_mobilenet_v2(imgs_node,
                                          num_classes=num_classes,
                                          is_training=is_training_node)
         variables = dict([k,v] for k,v in enumerate(slim.get_variables_to_restore())) 
         
    return tensors, variables, nodes

def load_weights(variables, weight_file, sess, flag_verb=0,scope=None):
    ''' restore variables
    ''' 
    reader = tf.train.NewCheckpointReader(weight_file)
    var_map = reader.get_variable_to_shape_map()
    graph = tf.get_default_graph()
    variables_to_restore = []
    variables_found = []
    variables_not_found = []
    for vname in variables:
        var = variables[vname]
        var_name = var.name.split(':')[0]
        if var_name in var_map and \
                var_map[var_name] == graph.get_tensor_by_name(var.name).get_shape() and \
                var_name not in variables_found:
            variables_to_restore.append(var)
            variables_found.append(var_name)
        else:
            if var_name not in variables_found:
                variables_not_found.append(var_name)
    if len(variables_to_restore) > 0:
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, weight_file)
    if flag_verb:
        print("Variables restored: ",variables_found)
        print("Variables not restored: ",variables_not_found)




def build_classification_loss(
                     logits,
                     labels, 
                     #nodes,
                     tensor_weight = None):
    ''' 
    Arguments:
        num_classes: int, number of classes of classifier
        tensors: list, tensors generated by build_model
        variables: list, varialbes generated by build_model
        nodes: list, nodes generated by build_mode
    Return: losses, nodes
        losses: list
        nodes: dictionary, append extra nodes to input nodes
        tensors: dictionary, append extra tensors to input tensors
    ''' 
    num_classes = CLASS_NUMBER
    ## training losses
    losses = [0] * 1 
    if num_classes == 1:
        losses[0] = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                           labels, logits))
    else:
        losses[0] = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                           labels, logits)) 
    return losses[0] 

def build_entropy_loss(logits):
    y_pred=tf.nn.softmax(logits)
    size = tf.cast(tf.shape(logits)[0],tf.float32) 
    loss = tf.cond(tf.equal(size,0), lambda : 0.0, lambda : tf.reduce_sum(-y_pred*tf.log(y_pred), axis=1))
    return tf.reduce_mean(loss)

def focal_loss(labels, logits, gamma =2): 
    y_pred=tf.nn.softmax(logits) 
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)  
    loss = tf.reduce_sum(L,axis=1)
    return loss

def build_focal_loss(logits,
                     labels,
                     tensor_weight = None):
    num_classes = CLASS_NUMBER
    gamma = np.float(os.environ.get('gamma'))
    size = tf.cast(tf.shape(logits)[0],tf.float32) 
    if num_classes == 1:
        losses = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                           labels, logits))
    else:
        losses = tf.cond(tf.equal(size,0), lambda : tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, logits)), 
                                              lambda : tf.reduce_mean(focal_loss(labels, logits, gamma)))
    
    return losses


def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
  """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.

  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def build_mmd_loss(source_tensor, target_tensor, scope=None):
  """Adds a similarity loss term, the MMD between two representations.

  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas)) 
  loss_value = maximum_mean_discrepancy(
      source_tensor, target_tensor, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) 
 

  return loss_value



def flip_gradient(x, nodes, l=10):  
    nodes['z_step'] = tf.placeholder(tf.int32, [])
    l_step =  2/(1+tf.exp(- tf.cast(nodes['z_step'],tf.float32) * l)) - 1
    grad_name = "FlipGradient%s" % tf.cast(nodes['z_step'], tf.string)
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * l_step]
    
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(x)
         
    return y, nodes
 


def build_dann_loss(source_tensor, target_tensor, nodes, scope=None):
  """Adds the domain adversarial (DANN) loss.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.variable_scope(scope):
    source_batch_size = tf.shape(source_tensor)[0]
    target_batch_size = tf.shape(target_tensor)[0]
    samples = tf.concat(axis=0, values=[source_tensor, target_tensor])
    samples = slim.flatten(samples)

    domain_selection_mask = tf.concat(
        axis=0, values=[tf.zeros((source_batch_size, 1)), tf.ones((target_batch_size, 1))])
    gamma = np.float(os.environ.get('gamma'))
    # Perform the gradient reversal and be careful with the shape. 
    grl, nodes = flip_gradient(samples, nodes)  
    grl = tf.reshape(tf.convert_to_tensor(grl), (-1, samples.get_shape().as_list()[1]))

    grl = slim.fully_connected(grl, 1024, scope='fc1')  
    grl = tf.nn.leaky_relu(grl)
    grl = slim.fully_connected(grl, 1024, scope='fc2')  
    grl = tf.nn.leaky_relu(grl)
    logits = slim.fully_connected(grl, 1, activation_fn=None, scope='fc3')

  
    def cross_entropy(logits,domain_selection_mask):
        domain_predictions = tf.sigmoid(logits)
        domain_loss = tf.losses.log_loss(domain_selection_mask, domain_predictions)
        return domain_loss
    
    def ls_distance(logits, source_batch_size, target_batch_size):
        source_adversary_logits, target_adversary_logits = tf.split(logits, [source_batch_size, target_batch_size], 0)
        domain_loss = tf.reduce_mean((source_adversary_logits - 1)**2) +  tf.reduce_mean((target_adversary_logits)**2)
        return domain_loss
    
    def focal_loss_sigmoid(logits,domain_selection_mask, gamma):
        y_pred = tf.sigmoid(logits)
        domain_loss=tf.reduce_mean(-domain_selection_mask* ((1-y_pred)*gamma)*tf.log(y_pred)- (1-domain_selection_mask)* (y_pred**gamma)*tf.log(1-y_pred))
        return domain_loss
    domain_loss = tf.cond(tf.equal(source_batch_size,0)|tf.equal(target_batch_size,0), 
                                                 lambda : cross_entropy(logits, domain_selection_mask),  
                                                 lambda : ls_distance(logits, source_batch_size, target_batch_size)) 
  return domain_loss, nodes


def build_weighted_dann_loss(source_tensor, target_tensor, nodes, weight, variables, scope=None):
  """Adds the domain adversarial (DANN) loss.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.variable_scope(scope):
    source_batch_size = tf.shape(source_tensor)[0]
    target_batch_size = tf.shape(target_tensor)[0]
    samples = tf.concat(axis=0, values=[source_tensor, target_tensor])
    samples = slim.flatten(samples)

    domain_selection_mask = tf.concat(
        axis=0, values=[tf.zeros((source_batch_size, 1)), tf.ones((target_batch_size, 1))])
    gamma = np.float(os.environ.get('gamma'))
    # Perform the gradient reversal and be careful with the shape. 
    grl, nodes = flip_gradient(samples, nodes)  
    grl = tf.reshape(tf.convert_to_tensor(grl), (-1, samples.get_shape().as_list()[1]))

    grl = slim.fully_connected(grl, 1024, scope='fc1') # 512
    grl = tf.nn.leaky_relu(grl)
    grl = slim.fully_connected(grl, 1024, scope='fc2') # 512
    grl = tf.nn.leaky_relu(grl)
    logits = slim.fully_connected(grl, 1, activation_fn=None, scope='fc3') 

    def cross_entropy(logits,domain_selection_mask,weight):
        domain_predictions = tf.sigmoid(logits)
        domain_loss = tf.losses.log_loss(domain_selection_mask, domain_predictions, weight)
        return domain_loss
    
    def ls_distance(logits, source_batch_size, target_batch_size, weight):
        source_adversary_logits, target_adversary_logits = tf.split(logits, [source_batch_size, target_batch_size], 0)
        source_weight, target_weight = tf.split(weight, [source_batch_size, target_batch_size], 0)
        domain_loss = tf.reduce_mean(tf.multiply(source_weight, (source_adversary_logits - 1)**2)) +  tf.reduce_mean(tf.multiply(target_weight, (target_adversary_logits)**2))
        return domain_loss
    
    def focal_loss_sigmoid(logits,domain_selection_mask, gamma):
        y_pred = tf.sigmoid(logits)
        domain_loss=tf.reduce_mean(-domain_selection_mask* ((1-y_pred)*gamma)*tf.log(y_pred)- (1-domain_selection_mask)* (y_pred**gamma)*tf.log(1-y_pred))
        return domain_loss
    domain_loss = tf.cond(tf.equal(source_batch_size,0)|tf.equal(target_batch_size,0), 
                                                 lambda : cross_entropy(logits, domain_selection_mask, weight),  
                                                 lambda : ls_distance(logits, source_batch_size, target_batch_size, weight)) 
  return domain_loss, nodes


def build_reugularization(fine_tune_filename):
    print('Applying L2-SP regularization for all parameters (weights, biases, gammas, betas)...')
    reader = tf.train.NewCheckpointReader(fine_tune_filename)
    l2_losses_existing_layers = [] 
    for v in tf.trainable_variables():
        name = v.name.split(':')[0]
        pre_trained_weights = reader.get_tensor(name)
        l2_losses_existing_layers.append(tf.nn.l2_loss(v - pre_trained_weights))
    return tf.add_n(l2_losses_existing_layers) 

def inner_focal_loss(source_logits,source_label,gamma):
    y_pred=tf.nn.softmax(source_logits)  
    L=-source_label*((1-y_pred)**gamma)*tf.log(y_pred)   
    return L
  
def co_denoise(source_logits_1, source_logits_2, source_label,num_classes):
    gamma = np.float(os.environ.get('gamma')) 
    source_prediction_1 = tf.nn.softmax(source_logits_1)
    source_prediction_2 = tf.nn.softmax(source_logits_2)
    
    # noise transfomation matrisx
    with tf.variable_scope('denoise1'): 
        source_trans1_class1 = slim.fully_connected(source_logits_1, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.9,0.1,0.1]), scope='fc1-class1')  
        source_trans1_class2 = slim.fully_connected(source_logits_1, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.1,0.9,0.1]), scope='fc1-class2')
        source_trans1_class3 = slim.fully_connected(source_logits_1, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.1,0.1,0.9]), scope='fc1-class3')  
        
    with tf.variable_scope('denoise2'):     
        source_trans2_class1 = slim.fully_connected(source_logits_2, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.9,0.1,0.1]), scope='fc2-class1')  
        source_trans2_class2 = slim.fully_connected(source_logits_2, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.1,0.9,0.1]), scope='fc2-class2')
        source_trans2_class3 = slim.fully_connected(source_logits_2, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.1,0.1,0.9]), scope='fc2-class3') 
    
    
    
    source_new_prediction_1 = tf.expand_dims(source_prediction_1[:,0],1)*source_trans2_class1 + tf.expand_dims(source_prediction_1[:,1],1)*source_trans2_class2 + tf.expand_dims(source_prediction_1[:,2],1)*source_trans2_class3
    source_new_prediction_2 = tf.expand_dims(source_prediction_2[:,0],1)*source_trans1_class1 + tf.expand_dims(source_prediction_2[:,1],1)*source_trans1_class2 + tf.expand_dims(source_prediction_2[:,2],1)*source_trans1_class3
         
    L_1 = tf.reduce_mean(inner_focal_loss(source_new_prediction_1,source_label,gamma))
    L_2 = tf.reduce_mean(inner_focal_loss(source_new_prediction_2,source_label,gamma))
 
    return L_1, L_2
 

def prediction_refine(original_logits, num_classes): 
    # for 3-class classification
    source_prediction = tf.nn.softmax(original_logits) 
    
    # noise transfomation matrix
    with tf.variable_scope('refine',reuse=tf.AUTO_REUSE): 
        source_trans_class1 = slim.fully_connected(original_logits, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.9,0.1,0.1]), scope='fc1-class1')  
        source_trans_class2 = slim.fully_connected(original_logits, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.1,0.9,0.1]), scope='fc1-class2')
        source_trans_class3 = slim.fully_connected(original_logits, num_classes, activation_fn=tf.nn.softmax, 
                                              weights_initializer= tf.zeros_initializer(), biases_initializer=tf.constant_initializer([0.1,0.1,0.9]), scope='fc1-class3')  
        
        source_new_prediction = tf.expand_dims(source_prediction[:,0],1)*source_trans_class1 + tf.expand_dims(source_prediction[:,1],1)*source_trans_class2 + tf.expand_dims(source_prediction[:,2],1)*source_trans_class3

    return source_new_prediction


def prediction_refine_k(original_logits, num_classes,Q): 
    source_prediction = tf.nn.softmax(original_logits)
    Q_softmax = tf.nn.softmax(Q) 
    source_new_prediction =  tf.matmul(source_prediction, Q_softmax)

    return source_new_prediction, Q_softmax


 
def build_classifier_loss(logits_1, logits_2):
    # KL distance
    #classifier_loss = tfp.distributions.kl_divergence(logits_1,logits_2)
    
    #JS distance
    medium_dsitribution = (logits_1+ logits_2)/2   
    classifier_loss = tf.reduce_sum(logits_1*tf.log(logits_1/medium_dsitribution),axis=1) + tf.reduce_sum(logits_2*tf.log(logits_2/medium_dsitribution),axis=1)
    
    return tf.reduce_mean(classifier_loss)
        

def _parse_image_name(fname_image):
    try:
        fname_image = fname_image.decode()
    except AttributeError:
        pass
    fname_image = os.path.basename(fname_image)
    fname_image = str(fname_image)
    sub = fname_image.split('_')[0]
    return fname_image, sub

def _update_sample_weight(fname_sample_weight,
                          pred_dict,
                          sample_weight_config_up,
                          sample_weight_config_low):
    fp = open(fname_sample_weight,'w')
    for sub in pred_dict:
        sub_label = next(iter(pred_dict[sub].values()))[0]
        pred_prob = [pred_dict[sub][x][1] for x in pred_dict[sub]]
        sorted_prob = sorted(pred_prob)
        sub_left = int(len(sorted_prob)*sample_weight_config_low[sub_label])
        sub_left = sorted_prob[sub_left]
        sub_right = int(len(sorted_prob)*(1-sample_weight_config_up[sub_label]))
        sub_right = sorted_prob[sub_right]
        for img in pred_dict[sub]:
            if sub_left >= sub_right or pred_dict[sub][img][1] >= sub_right:
                fp.write('%s,1\n'%img)
            elif pred_dict[sub][img][1] <= sub_left:
                fp.write('%s,0\n'%img)
            else:
                val = (pred_dict[sub][img][1]-sub_left)/(sub_right-sub_left)
                fp.write('%s,%f\n'%(img,val))
    fp.close()

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
        #print("discriminator:", var)
    return var_dict

def train_mobilenet(
          output_folder = './',
          num_epochs = 100,
          init_lr = 0.0001,
          batch_size = 32,
          init_mode = None,
          init_weight = None,
          verbose = 2, #0/1/2, 0: silent, 1: every epoch, 2: every batch
          s_dset_path='./domain_adaptation_images/office/amazon_list.txt',
          t_dset_path='./domain_adaptation_images/office/amazon_list.txt',
          test_dset_path='./domain_adaptation_images/office/amazon_list.txt',
          ):
    ''' training net ''' 
    ## output file name
    otype = 'mnet_v2'
    if init_mode:
        otype += '_%s'%init_mode
    num_labels = CLASS_NUMBER 

    ######################################################    prepare dataset    ###########################################################
    s_fnames, s_labels = read_lines(s_dset_path)
    t_fnames, t_labels = read_lines(t_dset_path)
    test_fnames, test_labels = read_lines(test_dset_path)
    
    if len(s_fnames) > len(t_fnames):
        t_n_fnames = t_fnames * (len(s_fnames) // len(t_fnames))
        t_n_labels = t_labels * (len(s_fnames) // len(t_fnames))
        s_sample = True
    else:
        facotr = len(t_fnames) // len(s_fnames) 
        s_fnames *= facotr
        s_labels *= facotr
        s_sample = False
    repeat_num = 20
    t_input_fnames = []
    s_input_fnames = []
    s_input_labels = [] 
    t_input_labels = []
    for j in range(repeat_num):
        if s_sample:
            sample_index = random.sample(range(len(s_fnames)), len(t_n_fnames))
            s_fnames_sample = [s_fnames[i] for i in sample_index]
            s_labels_sample = [s_labels[i] for i in sample_index]
            s_input_fnames += s_fnames_sample
            t_input_fnames += t_n_fnames
            s_input_labels += s_labels_sample 
            t_input_labels += t_n_labels
        else:
            sample_index = random.sample(range(len(t_fnames)), len(s_fnames))
            t_fnames_sample = [t_fnames[i] for i in sample_index]
            t_labels_sample = [t_labels[i] for i in sample_index]
            s_input_fnames += s_fnames
            t_input_fnames += t_fnames_sample
            s_input_labels += s_labels     
            t_input_labels += t_labels_sample

 

    features = {"source":s_input_fnames, "target":t_input_fnames}
    labels = {"source_label":s_input_labels, "target_label":t_input_labels}
    
    train_element = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(buffer_size=1000).map(train_prep1, num_parallel_calls=4).batch(batch_size).prefetch(144)
    test_batch_size = 1
    test_element=tf.data.Dataset.from_tensor_slices((test_fnames, test_labels)).map(test_prep, num_parallel_calls=4).batch(test_batch_size) 
    
    iterator = tf.data.Iterator.from_structure(train_element.output_types, train_element.output_shapes)
    next_element = iterator.get_next() 
    source_data = next_element[0]
    source_label = tf.one_hot(next_element[1], CLASS_NUMBER)  
    target_data = next_element[2]
    target_label =  tf.one_hot(next_element[3], CLASS_NUMBER)
    all_data = tf.concat([source_data,target_data],0)
    all_label = tf.concat([source_label,target_label],0)
    
    train_init_op = iterator.make_initializer(train_element)
    valid_init_op = iterator.make_initializer(test_element)    

    train_size = len(s_input_fnames)+len(t_input_fnames)
    eval_frequency =20 
    valid_size =  len(test_fnames)
 
    ######################################################   initialize model    ###########################################################
    tensors_1, variables_1, nodes_1 = build_model((224,224,3),
                                            num_labels,
                                            tensor_img=all_data,
                                            tensor_label=all_label,
                                            scope="network1")
    
    tensors_2, variables_2, nodes_2 = build_model((224,224,3),
                                            num_labels,
                                            tensor_img=all_data,
                                            tensor_label=all_label,
                                            scope="network2")
 
    source_mask = tf.concat([tf.ones(tf.shape(source_data)[0]),tf.zeros(tf.shape(target_data)[0])], 0)
    target_mask = tf.concat([tf.zeros(tf.shape(target_data)[0]),tf.ones(tf.shape(source_data)[0])], 0)
    num_classes = CLASS_NUMBER
    
    ###########################################################   prediction refine   ############################################################
    original_logits_1 = tf.reshape(tensors_1['output'],(-1,num_classes))
    original_logits_2 = tf.reshape(tensors_2['output'],(-1,num_classes))    
    Q = tf.Variable(tf.log(tf.ones([num_classes,num_classes])*(0.2/(num_classes-1)) + tf.eye(num_classes)*(0.8-(0.2/(num_classes-1)))),name="refine_matrix")
    all_logits_1,Q_softmax1 = prediction_refine_k(original_logits_1, num_classes,Q)   
    all_logits_2,Q_softmax2 = prediction_refine_k(original_logits_2, num_classes,Q)   
 
    ###########################################################   network 1   ############################################################
    single_logits_1 = tf.nn.softmax(tensors_1['predictions'])   
    logits_1 = tf.nn.softmax(all_logits_1)
    source_logits_1 = tf.boolean_mask(all_logits_1, source_mask)  
    source_label_1 = tf.boolean_mask(nodes_1['target'], source_mask)  
    target_logits_1 = tf.boolean_mask(all_logits_1, target_mask)  
    target_label_1 = tf.boolean_mask(nodes_1['target'], target_mask)    
    source_batch_size_1 = tf.cast(tf.shape(source_logits_1)[0],tf.float32) 
    target_batch_size_1 = tf.cast(tf.shape(target_logits_1)[0],tf.float32) 
    source_acc_1 = (100/source_batch_size_1) * tf.reduce_sum(tf.cast(tf.equal(tf.argmax(source_logits_1,1),tf.argmax(source_label_1,1)),tf.float32))
    target_acc_1 = (100/target_batch_size_1) * tf.reduce_sum(tf.cast(tf.equal(tf.argmax(target_logits_1,1),tf.argmax(target_label_1,1)),tf.float32))
    
    
    ###########################################################   network 2   ############################################################
    single_logits_2 = tf.nn.softmax(tensors_2['predictions'])  
    logits_2 = tf.nn.softmax(all_logits_2)
    source_logits_2 = tf.boolean_mask(all_logits_2, source_mask)  
    source_label_2 = tf.boolean_mask(nodes_2['target'], source_mask)  
    target_logits_2 = tf.boolean_mask(all_logits_2, target_mask)  
    target_label_2 = tf.boolean_mask(nodes_2['target'], target_mask)    
    source_batch_size_2 = tf.cast(tf.shape(source_logits_2)[0],tf.float32) 
    target_batch_size_2 = tf.cast(tf.shape(target_logits_2)[0],tf.float32) 
    source_acc_2 = (100/source_batch_size_2) * tf.reduce_sum(tf.cast(tf.equal(tf.argmax(source_logits_2,1),tf.argmax(source_label_2,1)),tf.float32))
    target_acc_2 = (100/target_batch_size_2) * tf.reduce_sum(tf.cast(tf.equal(tf.argmax(target_logits_2,1),tf.argmax(target_label_2,1)),tf.float32))


    ###########################################################   final prediction   ############################################################
    final_logits = (single_logits_1 + single_logits_2)/2 

    #####################################################   Adaptation  instance Weight   ######################################################
    
    def matmul(A,B):
        return tf.squeeze(tf.matmul(A,B),2)
    
    logit_1_temp = tf.transpose(tf.expand_dims(single_logits_1,2), [0,2,1])  
    logit_2_temp = tf.transpose(tf.expand_dims(single_logits_2,2), [0,2,1])  
    numerator = matmul(logit_1_temp, tf.expand_dims(single_logits_2,2)) 
    denominator = tf.sqrt(matmul(logit_1_temp,tf.expand_dims(single_logits_1,2))) * tf.sqrt(matmul(logit_2_temp,tf.expand_dims(single_logits_2,2)))
    weight = tf.divide(numerator,denominator)
    weight_abs = tf.abs(weight)
    source_weight = tf.boolean_mask(weight_abs , source_mask) 
    target_weight = tf.boolean_mask(weight_abs , target_mask) 
    domain_weight = 1 - tf.concat(axis=0, values=[source_weight, target_weight])
    
     
    
    #######################################################   loss of network 1   ############################################################
    # build classification loss  
    classification_loss_1 = build_focal_loss(source_logits_1, source_label_1)     

    # Entropy for minimizing unstable distributions: target or all 
    # entropy_loss_1 = build_entropy_loss(all_logits_1)
        
    # domain adaptation loss i
    GAP_1 = tensors_1['global_pool']
    GAP_1 = tf.squeeze(GAP_1, [1, 2])
    source_tensors_1 = tf.boolean_mask(GAP_1 , source_mask)  
    target_tensors_1 = tf.boolean_mask(GAP_1 , target_mask)        
     
    # MMD loss 
    #domain_loss = build_mmd_loss(source_tensors, target_tensors)      
    
    # DANN loss    
    #domain_loss_1, nodes_1 = build_dann_loss(source_tensors_1, target_tensors_1, nodes_1, scope="dann_network1")
    domain_loss_1, nodes_1 = build_weighted_dann_loss(source_tensors_1, target_tensors_1, nodes_1, 1 + domain_weight, variables_1, scope="network1-dann")
    
 
     
    ######################################################   loss of network 2   ############################################################    
    # build classification loss  
    classification_loss_2 = build_focal_loss(source_logits_2, source_label_2)  

    # Entropy for minimizing unstable distributions: target or all 
    #entropy_loss_2 = build_entropy_loss(all_logits_2)
 
    # domain adaptation loss 
    GAP_2 = tensors_2['global_pool']
    GAP_2 = tf.squeeze(GAP_2, [1, 2])
    source_tensors_2 = tf.boolean_mask(GAP_2 , source_mask)  
    target_tensors_2 = tf.boolean_mask(GAP_2 , target_mask)       
    
    # MMD loss 
    #domain_loss = build_mmd_loss(source_tensors, target_tensors)       
    
    # DANN loss    
    #omain_loss_2, nodes_2 = build_dann_loss(source_tensors_2, target_tensors_2, nodes_2, scope="dann_network2")
    domain_loss_2, nodes_2 = build_weighted_dann_loss(source_tensors_2, target_tensors_2, nodes_2, 1 + domain_weight, variables_2, scope="network2-dann")
    
    
    
    
    ###########################################################   kick off training   ###########################################################   
    step = 0
    all_variables = slim.get_variables_to_restore() 
    variables_network1 = [v for v in all_variables if v.name.split('/')[0] == 'network1'] 
    variables_network2 = [v for v in all_variables if v.name.split('/')[0] == 'network2'] 
    regularization_loss_1 = 0.000005 * tf.add_n( 
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in variables_network1
       if 'batch_normalization' not in v.name])
    regularization_loss_2 = 0.000005 * tf.add_n( 
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in variables_network2
       if 'batch_normalization' not in v.name])  
   
    alpha = np.float(os.environ.get('alpha'))
    eta = np.float(os.environ.get('eta'))
    
    loss_1 = classification_loss_1 + alpha * domain_loss_1 +regularization_loss_1 
    loss_2 = classification_loss_2 + alpha * domain_loss_2 +regularization_loss_2 
    loss = loss_1+ loss_2 
    
    # classifeir maximization
    classifier_loss = - eta * build_classifier_loss(single_logits_1, single_logits_2)
    
    learning_rate = tf.train.exponential_decay(init_lr,
                                               step,
                                               100000,
                                               0.99,
                                               staircase=False)
    optimizer = OPTIMIZER_DICT[os.environ.get('OPTIMIZER').lower()](learning_rate)

    variables_of_classifier = [v for v in all_variables if 'Logits' in v.name]   
    train_op1 = slim.learning.create_train_op(loss_1, optimizer)
    train_op2 = slim.learning.create_train_op(loss_2, optimizer)         
    train_op3 = optimizer.minimize(classifier_loss, var_list=variables_of_classifier)
 
    valid_best_1 = 0
    train_best_1 = 1e10
    valid_best_2 = 0
    train_best_2 = 1e10    
    valid_best_co = 0
    train_best_co = 1e10   
    F1_best_co = 0
    
    # summary
    tf.summary.scalar('network1/acc/source_train_acc', source_acc_1)
    tf.summary.scalar('network1/acc/target_train_acc', target_acc_1)
    tf.summary.scalar('network1/loss/classification loss', classification_loss_1)
    #tf.summary.scalar('network1/loss/entropy loss', entropy_loss_1) 
    tf.summary.scalar('network1/loss/domain loss', domain_loss_1)
    tf.summary.scalar('network1/loss/total loss', loss_1)  
    tf.summary.scalar('network2/acc/source_train_acc', source_acc_2)
    tf.summary.scalar('network2/acc/target_train_acc', target_acc_2)
    tf.summary.scalar('network2/loss/classification loss', classification_loss_2)
    #tf.summary.scalar('network2/loss/entropy loss', entropy_loss_2) 
    tf.summary.scalar('network2/loss/domain loss', domain_loss_2)
    tf.summary.scalar('network2/loss/total loss', loss_2)  
    tf.summary.scalar('loss/classifier loss', classifier_loss) 
    tf.summary.scalar('loss/total loss', loss+classifier_loss) 
    merged_summary = tf.summary.merge_all() 
    
    saver = tf.train.Saver() 
    variables_network1_ckpt = [v for v in all_variables if v.name.split('/')[0] == 'network1' and 'Logits' not in v.name] 
    variables_network2_ckpt = [v for v in all_variables if v.name.split('/')[0] == 'network2' and 'Logits' not in v.name]          
    
    saver1 = tf.train.Saver(variables_network1_ckpt) 
    saver2 = tf.train.Saver(variables_network2_ckpt)
    # saver3 = tf.train.Saver(variables_refine)        
    
    
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True   
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(output_folder, sess.graph)
        ## initialize value
        tf.global_variables_initializer().run()
        ## load value
        if init_weight and init_mode: 
            saver1.restore(sess, init_weight+"-network1")
            saver2.restore(sess, init_weight+"-network2")
            ##saver3.restore(sess, init_weight+"-co")
            
        ## start training
        sess.run(train_init_op)
        start_time = time.time()
        states_1 = [0]*20
        states_2 = [0]*20
        states_co = [0]*20
        counts_correct_1 = [0]*num_labels
        counts_correct_2 = [0]*num_labels
        counts_correct_co = [0]*num_labels
        counts_all_1 = [0]*num_labels
        counts_all_2 = [0]*num_labels
        counts_all_co = [0]*num_labels
        
        print('step\tepoch\ttime\tloss_N1\tacc_N1',end=''), 
        print('\tloss_N2\tacc_N2',end=''),     
        print('\tloss_co\tacc_co',end=''),              
        print('') 
        while 1:   
            # Run the optimizer to update weights.
            outputs = sess.run([train_op1, train_op2,
                                loss_1, loss_2,
                                logits_1,logits_2]+[all_label] +[merged_summary, final_logits, loss, train_op3,Q_softmax1],
                                feed_dict={nodes_1['is_training']:True, nodes_2['is_training']:True, nodes_1['z_step']: step, nodes_2['z_step']: step}) 
            if np.isnan(outputs[2]) or np.isnan(outputs[3]):
                continue 
             
            step+=1 
            if step%10==0:
                train_writer.add_summary(outputs[7], step)
            class_labels = np.array(outputs[6])
            # count
            states_1[0] += 1
            states_2[0] += 1
            states_co[0] += 1
            # training loss
            states_1[1] += outputs[2]
            states_2[1] += outputs[3]
            states_co[1] += outputs[9]
            # accu
            states_1[2] += _accu_rate(outputs[4], class_labels)
            states_2[2] += _accu_rate(outputs[5], class_labels)
            states_co[2] += _accu_rate(outputs[8], class_labels)
            # count each class
            for i in range(batch_size):
                if outputs[4].shape[1] > 1:
                    pred_label = np.argmax(outputs[4][i])
                    truth_label = np.argmax(class_labels[i])
                else:
                    pred_label = int(outputs[4][i]>0)
                    truth_label = int(class_labels[i]>0.5)
                if pred_label == truth_label:
                    counts_correct_1[pred_label]+=1
                    counts_all_1[pred_label]+=1
                else:
                    counts_all_1[pred_label]+=1
                    counts_all_1[truth_label]+=1            
            for i in range(batch_size):
                if outputs[5].shape[1] > 1:
                    pred_label = np.argmax(outputs[5][i])
                    truth_label = np.argmax(class_labels[i])
                else:
                    pred_label = int(outputs[5][i]>0)
                    truth_label = int(class_labels[i]>0.5)
                if pred_label == truth_label:
                    counts_correct_2[pred_label]+=1
                    counts_all_2[pred_label]+=1
                else:
                    counts_all_2[pred_label]+=1
                    counts_all_2[truth_label]+=1       
            for i in range(batch_size):
                if outputs[8].shape[1] > 1:
                    pred_label = np.argmax(outputs[8][i])
                    truth_label = np.argmax(class_labels[i])
                else:
                    pred_label = int(outputs[8][i]>0)
                    truth_label = int(class_labels[i]>0.5)
                if pred_label == truth_label:
                    counts_correct_co[pred_label]+=1
                    counts_all_co[pred_label]+=1
                else:
                    counts_all_co[pred_label]+=1
                    counts_all_co[truth_label]+=1           
                    
            elapsed_time = time.time() - start_time
            if verbose==2 or (verbose==1 and step % eval_frequency == 0):
                print('\n%d\t%.2f\t%.1fs' %\
                        (step, float(step)*batch_size/train_size, elapsed_time),
                      end=''),
                for i in range(1,3):
                    print('\t%.2f'%(states_1[i]/states_1[0]),end=''),
                for i in range(1,3):
                    print('\t%.2f'%(states_2[i]/states_2[0]),end=''),                                
                for i in range(1,3):
                    print('\t%.2f'%(states_co[i]/states_co[0]),end=''),                                              
                sys.stdout.flush()
                
            # predict on validation data once reach the evaluation frequency
            if step % eval_frequency == 0:
                if states_1[1]/states_1[0] < train_best_1:
                    train_best_1 = states_1[1]/states_1[0]
                    print('\ntbest-network1:',train_best_1,end=''),
                if states_2[1]/states_2[0] < train_best_2:
                    train_best_2 = states_2[1]/states_2[0]
                    print('\ntbest-network2:',train_best_2,end=''), 
                if states_co[1]/states_co[0] < train_best_co:
                    train_best_co = states_co[1]/states_co[0]
                    print('\ntbest-network-co:',train_best_co,end=''),                       
                sess.run(valid_init_op)
                # fetch some extra nodes' data
                start_time = time.time()
                states_1 = [0]*20
                states_2 = [0]*20
                states_co = [0]*20
                counts_correct_1 = [0]*num_labels
                counts_correct_2 = [0]*num_labels
                counts_correct_co = [0]*num_labels
                counts_all_1 = [0]*num_labels
                counts_all_2 = [0]*num_labels
                counts_all_co = [0]*num_labels
                count_co = np.zeros([num_labels,num_labels])
                print("")
                # start evaluation on validation data
                vstep = 0
                while 1:
                    vstep+=1
                    # Run the prediction  
                    outputs = sess.run([classification_loss_1, classification_loss_2, logits_1,logits_2]+[all_label]+[final_logits, loss], 
                                        feed_dict={nodes_1['is_training']:False, nodes_2['is_training']:False}) 
                     
                    class_labels = np.array(outputs[4][0:test_batch_size]) 
                    outputs[2] = outputs[2][0:test_batch_size]
                    outputs[3] = outputs[3][0:test_batch_size]                   
                    outputs[5] = outputs[5][0:test_batch_size]
                    # count
                    states_1[0] += 1
                    states_2[0] += 1
                    states_co[0] += 1
                    # training loss
                    states_1[1] += outputs[0]
                    states_2[1] += outputs[1]
                    states_co[1] += outputs[6]
                    # accu  
                    states_1[2] += _accu_rate(outputs[2], class_labels)
                    states_2[2] += _accu_rate(outputs[3], class_labels)
                    states_co[2] += _accu_rate(outputs[5], class_labels)


                    # count each class
                    for i in range(test_batch_size):
                        if outputs[2].shape[1] > 1:
                            pred_label = np.argmax(outputs[2][i])
                            truth_label = np.argmax(class_labels[i])
                        else:
                            pred_label = int(outputs[2][i]>0)
                            truth_label = int(class_labels[i]>0.5)
                        if pred_label == truth_label:
                            counts_correct_1[pred_label]+=1
                            counts_all_1[pred_label]+=1
                        else:
                            counts_all_1[pred_label]+=1
                            counts_all_1[truth_label]+=1            
                    for i in range(test_batch_size):
                        if outputs[3].shape[1] > 1:
                            pred_label = np.argmax(outputs[3][i])
                            truth_label = np.argmax(class_labels[i])
                        else:
                            pred_label = int(outputs[3][i]>0)
                            truth_label = int(class_labels[i]>0.5)
                        if pred_label == truth_label:
                            counts_correct_2[pred_label]+=1
                            counts_all_2[pred_label]+=1
                        else:
                            counts_all_2[pred_label]+=1
                            counts_all_2[truth_label]+=1                               
                            
                    for i in range(test_batch_size):
                        if outputs[5].shape[1] > 1:
                            pred_label = np.argmax(outputs[5][i])
                            truth_label = np.argmax(class_labels[i])
                        else:
                            pred_label = int(outputs[5][i]>0)
                            truth_label = int(class_labels[i]>0.5)
                        if pred_label == truth_label:
                            counts_correct_co[pred_label]+=1
                            counts_all_co[pred_label]+=1
                        else:
                            counts_all_co[pred_label]+=1
                            counts_all_co[truth_label]+=1                                
                        count_co[pred_label,truth_label] +=1          
                        
                    if (verbose == 2 and vstep >= valid_size):
                        elapsed_time = time.time() - start_time
                        print('\nvalid\t%.2f\t%.1fs' %\
                                (float(step)*batch_size/train_size, elapsed_time),
                              end=''),
         
                        for i in range(1,3):
                            print('\t%.2f'%(states_1[i]/states_1[0]),end=''),
                        for i in range(1,3):
                            print('\t%.2f'%(states_2[i]/states_2[0]),end=''),          
                        for i in range(1,3):
                            print('\t%.2f'%(states_co[i]/states_co[0]),end=''),                                              
                        sys.stdout.flush()


                    # print some extra information once reach the evaluation end
                    if vstep >= valid_size: 
                        if states_1[2]/states_1[0] > valid_best_1:
                            saver.save(sess, '%s/%s-vbest-network1'%(output_folder, otype))
                            valid_best_1 = states_1[2]/states_1[0]
                            print('\nvbest-network1:',valid_best_1,end=''),
                        if states_2[2]/states_2[0] > valid_best_2:
                            saver.save(sess, '%s/%s-vbest-network2'%(output_folder, otype))
                            valid_best_2 = states_2[2]/states_2[0]
                            print('\nvbest-network2:',valid_best_2,end=''),  
                        if states_co[2]/states_co[0] > valid_best_co:
                            saver.save(sess, '%s/%s-vbest-acc'%(output_folder, otype))
                            valid_best_co = states_co[2]/states_co[0]
                            print('\nvbest-acc:',valid_best_co,end=''),  
                        
                        pred_all = np.sum(count_co,axis=1)
                        truth_all = np.sum(count_co,axis=0)
                        positive_pre = np.diag(count_co)
                        for i in range(len(truth_all)):
                            if truth_all[i]==0:
                                truth_all[i]=1                    
    
                        recall = positive_pre/truth_all
                        precision = positive_pre/pred_all
                        for i in range(len(recall)):
                            if np.isnan(recall[i]):
                                recall[i]=0
                        for j in range(len(precision)):
                            if np.isnan(precision[j]):
                                precision[j]=0                
                            if recall[j]+precision[j]==0: 
                                precision[j]=1
                        F1 = (2*recall*precision)/(recall+precision)
                        F1 = (np.sum(F1)/31)*100		
                        print("")	
                        print("F1 measure",F1)
 
                        if F1 > F1_best_co:
                            saver.save(sess, '%s/%s-vbest-co'%(output_folder, otype))
                            F1_best_co =F1
                            print('\nvbest-co F1:',F1_best_co,end=''),  
                                                
                        # clean up
                        start_time = time.time()
                        states_1 = [0]*20
                        states_2 = [0]*20
                        states_co = [0]*20
                        counts_correct_1 = [0]*num_labels
                        counts_correct_2 = [0]*num_labels
                        counts_correct_co = [0]*num_labels
                        counts_all_1 = [0]*num_labels
                        counts_all_2 = [0]*num_labels
                        counts_all_co = [0]*num_labels
                        sess.run(train_init_op)
                        print("")
                        break
            if step > int(num_epochs*train_size)//batch_size:
                break

def _accu_rate(predictions, labels):
    """Return the error rate based on dense predictions and dense labels.""" 
    if labels.shape[-1] > 1:
        return 100.0 * \
           np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / \
           predictions.shape[0]
    else:
        return 100.0 * \
           np.sum((predictions>0.5) == labels) / \
           predictions.shape[0]


def _seg_dice(truth, pred, weight):
    ''' compute 1-dice loss of segmentation
    Arguments:
        truth: tensor, (N,H,W,C), segmentation truth
        pred: tensor, (N,H,W,C), segmentation prediction
        weight: tensor, (N,H,W,1), pixel level seg
    Return:
        dice loss
    '''
    wsum= np.sum(weight)
    if wsum > 0:
        dice = np.sum(truth*pred*weight)*2/((truth+pred)*weight)
    else:
        dice = -1
    return dice

if __name__ == '__main__':
    if len(sys.argv)>1:
        load_env(sys.argv[1])
    init_weight = os.environ.get('INIT_MODEL_FILE')
    init_mode = os.environ.get('INIT_MODEL_NAME')
    if init_weight.lower() == 'none' or \
               init_mode is None or \
               init_mode.lower() == 'none':
        init_mode = None
        init_weight = None
    train_mobilenet(
            output_folder=os.environ.get('OUTPUT_FOLDER'),
            num_epochs=int(os.environ.get('EPOCH')),
            init_lr=float(os.environ.get('INIT_LR')),
            batch_size=int(os.environ.get('BATCH_SIZE')),
            init_mode=init_mode,
            init_weight=init_weight,
            verbose=int(os.environ.get('VERBOSE')), 
            s_dset_path=str(os.environ.get('S_DSET_PATH')),
            t_dset_path=str(os.environ.get('T_DSET_PATH')),
            test_dset_path=str(os.environ.get('TEST_DSET_PATH')), 
            )

