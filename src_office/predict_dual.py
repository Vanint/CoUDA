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
    return image, label 

def train_prep(fname, label):
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


def prediction_refine(original_logits, num_classes): 
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
    #source_prediction_2 = tf.nn.softmax(original_logits_2)
    Q_softmax = tf.nn.softmax(Q) #
     
    source_new_prediction =  tf.matmul(source_prediction, Q_softmax)

    return source_new_prediction

        

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
          test_dset_path='./domain_adaptation_images/office/amazon_list.txt',
          ):
    ''' training net
    ''' 
    ## output file name
    otype = 'mnet_v2'
    if init_mode:
        otype += '_%s'%init_mode

    ## prepare for dynamic sample weight 
    num_labels = CLASS_NUMBER
    ######################################################    prepare dataset    ###########################################################
     
    test_fnames, test_labels = read_lines(test_dset_path)
    test_element=tf.data.Dataset.from_tensor_slices((test_fnames, test_labels)).map(test_prep, num_parallel_calls=4).batch(1) 
    iterator = tf.data.Iterator.from_structure(test_element.output_types, test_element.output_shapes)
    next_element = iterator.get_next() 

    target_data = next_element[0]
    target_label =  tf.one_hot(next_element[1], CLASS_NUMBER)
    source_data = target_data
    source_label = target_label
    all_data = tf.concat([source_data,target_data],0)
    all_label = tf.concat([source_label,target_label],0)
    valid_init_op = iterator.make_initializer(test_element)    

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

 
    num_classes = CLASS_NUMBER
    
    ###########################################################   prediction refine   ################################################### 
    original_logits_1 = tf.reshape(tensors_1['output'],(-1,num_classes))
    original_logits_2 = tf.reshape(tensors_2['output'],(-1,num_classes))   
    Q = tf.Variable(tf.log(tf.ones([num_classes,num_classes])*(0.1/(num_classes-1)) + tf.eye(num_classes)*(0.9-(0.1/(num_classes-1)))),name="refine_matrix")
    all_logits_1 = prediction_refine_k(original_logits_1, num_classes,Q)   
    all_logits_2 = prediction_refine_k(original_logits_2, num_classes,Q)   
 
    logits_1 = tf.nn.softmax(all_logits_1)  # network 1 
    logits_2 = tf.nn.softmax(all_logits_2)  # network 2  
    final_logits = (logits_1 + logits_2)/2  # final prediction
    
    
    # saver
    saver = tf.train.Saver()
    valid_size =   len(test_fnames)
    valid_best = 0    
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True   
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        ## load value
        if init_weight and init_mode:
            saver.restore(sess, init_weight+"-co")
        ## start evaluation
        
        print('step\tepoch\ttime\tacc_N1',end=''),
        print('\tacc_N2',end=''),    
        print('\tacc_co',end=''),             
        print('') 
                   
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
        print("")
        # start evaluation on validation data
        vstep = 0
        count = np.zeros([num_labels,num_labels])
        while 1:
            vstep+=1
            # Run the prediction   
            outputs = sess.run([logits_1,logits_2,final_logits,all_label],feed_dict={nodes_1['is_training']:False, nodes_2['is_training']:False})   
             
            class_labels = np.array(outputs[3][0:batch_size]) 
            outputs[0] = outputs[0][0:batch_size]
            outputs[1] = outputs[1][0:batch_size]                   
            outputs[2] = outputs[2][0:batch_size]
            # count
            states_1[0] += 1
            states_2[0] += 1
            states_co[0] += 1 
            # accu  
            states_1[1] += _accu_rate(outputs[0], class_labels)
            states_2[1] += _accu_rate(outputs[1], class_labels)
            states_co[1] += _accu_rate(outputs[2], class_labels)
 
            # count each class
            for i in range(batch_size):
                if outputs[0].shape[1] > 1:
                    pred_label = np.argmax(outputs[0][i])
                    truth_label = np.argmax(class_labels[i])
                else:
                    pred_label = int(outputs[0][i]>0)
                    truth_label = int(class_labels[i]>0.5)
                if pred_label == truth_label:
                    counts_correct_1[pred_label]+=1
                    counts_all_1[pred_label]+=1
                else:
                    counts_all_1[pred_label]+=1
                    counts_all_1[truth_label]+=1    
                    
            for i in range(batch_size):
                if outputs[1].shape[1] > 1:
                    pred_label = np.argmax(outputs[1][i])
                    truth_label = np.argmax(class_labels[i])
                else:
                    pred_label = int(outputs[1][i]>0)
                    truth_label = int(class_labels[i]>0.5)
                if pred_label == truth_label:
                    counts_correct_2[pred_label]+=1
                    counts_all_2[pred_label]+=1
                else:
                    counts_all_2[pred_label]+=1
                    counts_all_2[truth_label]+=1                               
                    
            for i in range(batch_size):
                if outputs[2].shape[1] > 1:
                    pred_label = np.argmax(outputs[2][i])
                    truth_label = np.argmax(class_labels[i])
                else:
                    pred_label = int(outputs[2][i]>0)
                    truth_label = int(class_labels[i]>0.5)
                if pred_label == truth_label:
                    counts_correct_co[pred_label]+=1
                    counts_all_co[pred_label]+=1
                else:
                    counts_all_co[pred_label]+=1
                    counts_all_co[truth_label]+=1         
                count[pred_label,truth_label] +=1                         
                     
            if (verbose == 2 and vstep == len(test_fnames)):
                elapsed_time = time.time() - start_time
                print('\nvalid\t%.2f\t%.1fs' %\
                        (float(vstep), elapsed_time),
                      end=''),
 
                for i in range(1,2):
                    print('\t%.2f'%(states_1[i]/states_1[0]),end=''),

                for i in range(1,2):
                    print('\t%.2f'%(states_2[i]/states_2[0]),end=''),                                       
                for i in range(1,2):
                    print('\t%.2f'%(states_co[i]/states_co[0]),end=''),                                                                  
                sys.stdout.flush() 

            if vstep >= valid_size:
                # evaluae result and save  
                if states_co[1]/states_co[0] > valid_best: 
                    valid_best = states_co[1]/states_co[0]
                    print('\nvbest:',valid_best,end=''),
                    
                pred_all = np.sum(count,axis=1)
                truth_all = np.sum(count,axis=0)
                positive_pre = np.diag(count)
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
                print("precision",np.sum(precision)/31)
                print("recall",np.sum(recall)/31)
                print("F1 measure",F1)                     
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
            test_dset_path=str(os.environ.get('TEST_DSET_PATH')), 
            )


