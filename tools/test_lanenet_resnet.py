#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import numpy as np
import os.path as ops
import argparse
import math
import tensorflow as tf
import glog as log
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from lanenet_model import lanenet_merge_model_try as lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test


CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=4)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()



def test_lanenet(image_path, weights_path, use_gpu, image_list, batch_size, save_dir):

    """
    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    tp = 0
    fp = 0
    anno_lane_num = 0
    detect_lane_num = 0
    # acc = 0
    # total = 0
    acc_mean = []
    acc_bg_mean = []

    test_dataset = lanenet_data_processor_test.DataSet(image_path, batch_size)
    input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
    input_instance_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_instance_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)
    imgs_instance = tf.map_fn(test_dataset.process_label_instance, input_instance_tensor, dtype=tf.int32)
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')
    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=weights_path)
        for i in range(math.ceil(len(image_list) / batch_size)):    
            print('progress:', i, i / math.ceil(len(image_list) / batch_size))
            paths, labels, labels_instance_paths = test_dataset.next_batch()
            instance_seg_image, existence_output, instance_seg_label = sess.run([binary_seg_ret, instance_seg_ret, imgs_instance],
                                                            feed_dict={input_tensor: paths, input_instance_tensor: labels_instance_paths})
            # instance_seg_label = sess.run(imgs_instance, feed_dict={input_tensor: labels_instance_paths})  # instance segment labels
            instance_seg_out = tf.argmax(instance_seg_image, axis=-1)  # instance segment outputs    
            
            anno_lane_num += sum(sum(np.array(labels).astype(int)))
            # accuracy, accuracy_back, IoU_lst = accuracy_helper(instance_seg_label, instance_seg_out)
            # IoU_lst = sess.run(IoU_lst)
            for cnt, image_name in enumerate(paths):  
                # instance_seg_label = test_dataset.process_label_instance(labels_instance_paths)
                accuracy, accuracy_back, IoU_lst = accuracy_helper(instance_seg_label[cnt], instance_seg_out[cnt])
                # IoU_lst = sess.run(IoU_lst)
                # instance_seg_label = test_dataset.process_label_instance(labels_instance_paths[cnt])              
                # print('metrics:', sess.run([accuracy, accuracy_back, IoU_lst]))  
                check_nan = tf.cond(tf.is_nan(accuracy), lambda: 1, lambda: 0)            
                if check_nan.eval() == 0:
                    acc_mean.append(accuracy)
                acc_bg_mean.append(accuracy_back)
                parent_path = os.path.dirname(image_name)
                directory = os.path.join(save_dir, 'vgg_SCNN_DULR', parent_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_exist = open(os.path.join(directory, os.path.basename(image_name)[:-3] + 'exist.txt'), 'w')
                image_height = (instance_seg_image[cnt, :, :, 0] * 255).astype(int).shape[0]
                image_width = (instance_seg_image[cnt, :, :, 0] * 255).astype(int).shape[1]
                image_tosave = np.zeros(shape = [image_height, image_width])
                for cnt_img in range(4):
                    cv2.imwrite(os.path.join(directory, os.path.basename(image_name)[:-4] + '_' + str(cnt_img + 1) + '_avg.png'),
                            (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int))
                    
                    if existence_output[cnt, cnt_img] > 0.5:
                        image_tosave += (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int)
                        file_exist.write('1 ')
                        detect_lane_num += 1
                        iou_compare = tf.cond(tf.greater(IoU_lst[cnt_img], 0.15), lambda: 1, lambda: 0)
                        if int(labels[cnt][cnt_img]) == 1 and iou_compare.eval() == 1:
                            tp += 1
                        #     acc += 1
                        else:
                            fp += 1
                    else:
                        file_exist.write('0 ')
                    #     if int(labels[cnt][cnt_img]) == 1:
                    #         fn += 1
                    #     else:
                    #         acc += 1
                    # total += 1
                cv2.imwrite(os.path.join(directory, os.path.basename(image_name)[:-4] + '_avg.png'), image_tosave)
                file_exist.close()
            # if i % 50  == 0 and i >0:
            #     precision = tp/detect_lane_num
            #     recall = tp/anno_lane_num
            #     f1 = 2*precision*recall/(precision+recall)
            #     print("!!!!!", tp, detect_lane_num, anno_lane_num, precision, recall)
            #     print('f1: ', f1)
            #     print('accuracy: ', sess.run([tf.reduce_mean(acc_mean), tf.reduce_mean(acc_bg_mean)]), 
            #     ' accuracy_back: ')
            # if i % 20 == 0 and i > 0:
            #     sess.close()
            #     sess = tf.Session(config=sess_config)
            #     sess.run(tf.global_variables_initializer())
            #     saver.restore(sess=sess, save_path=weights_path)
        sess.close()
    # accuracy = acc/total
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # f1 = 2*precision*recall/(precision+recall)
    print(fp)
    print('final accuracy: ', ' final accuracy_back: ', 
    sess.run([tf.reduce_mean(acc_mean), tf.reduce_mean(acc_bg_mean)]))
    print('final f1: ', f1)
    return

def accuracy_helper(instance_seg_label, instance_seg_out):
# calculate instance accuracy
    true0 = tf.cast(tf.equal(instance_seg_label, 0),tf.int32)
    out0 = tf.cast(tf.equal(instance_seg_out, 0), tf.int32)
    true1 = tf.cast(tf.equal(instance_seg_label, 1),tf.int32)
    out1 = tf.cast(tf.equal(instance_seg_out, 1), tf.int32)
    true2 = tf.cast(tf.equal(instance_seg_label, 2),tf.int32)
    out2 = tf.cast(tf.equal(instance_seg_out, 2), tf.int32)
    true3 = tf.cast(tf.equal(instance_seg_label, 3),tf.int32)
    out3 = tf.cast(tf.equal(instance_seg_out, 3), tf.int32)
    true4 = tf.cast(tf.equal(instance_seg_label, 4),tf.int32)
    out4 = tf.cast(tf.equal(instance_seg_out, 4), tf.int32)

    pred_0 = tf.count_nonzero(tf.multiply(true0, out0), dtype=tf.int32)
    pred_1 = tf.count_nonzero(tf.multiply(true1, out1), dtype=tf.int32)
    pred_2 = tf.count_nonzero(tf.multiply(true2, out2), dtype=tf.int32)
    pred_3 = tf.count_nonzero(tf.multiply(true3, out3), dtype=tf.int32)
    pred_4 = tf.count_nonzero(tf.multiply(true4, out4), dtype=tf.int32)
    
    gt_all = tf.count_nonzero(tf.cast(tf.greater(instance_seg_label, 0), tf.int32), dtype=tf.int32)
    gt_back = tf.count_nonzero(tf.cast(tf.equal(instance_seg_label, 0), tf.int32), dtype=tf.int32)

    pred_all = tf.add(tf.add(tf.add(pred_1, pred_2), pred_3), pred_4)

    accuracy = tf.divide(tf.cast(pred_all, tf.float32), tf.cast(gt_all, tf.float32))
    accuracy_back = tf.divide(tf.cast(pred_0, tf.float32), tf.cast(gt_back, tf.float32))    

    # Compute mIoU of Lanes
    overlap_1 = pred_1
    true1_count = tf.count_nonzero(true1, dtype=tf.int32)
    out1_count = tf.count_nonzero(out1, dtype=tf.int32)
    union_1 = tf.add(true1_count, out1_count)
    union_1 = tf.subtract(union_1, overlap_1)
    IoU_1 = tf.divide(tf.cast(overlap_1, tf.float32), tf.cast(union_1, tf.float32))

    overlap_2 = pred_2
    true2_count = tf.count_nonzero(true2, dtype=tf.int32)
    out2_count = tf.count_nonzero(out2, dtype=tf.int32)
    union_2 = tf.add(true2_count, out2_count)
    union_2 = tf.subtract(union_2, overlap_2)
    IoU_2 = tf.divide(tf.cast(overlap_2, tf.float32), tf.cast(union_2, tf.float32))

    overlap_3 = pred_3
    true3_count = tf.count_nonzero(true3, dtype=tf.int32)
    out3_count = tf.count_nonzero(out3, dtype=tf.int32)
    union_3 = tf.add(true3_count, out3_count)
    union_3 = tf.subtract(union_3, overlap_3)
    IoU_3 = tf.divide(tf.cast(overlap_3, tf.float32), tf.cast(union_3, tf.float32))

    overlap_4 = pred_4
    true4_count = tf.count_nonzero(true4, dtype=tf.int32)
    out4_count = tf.count_nonzero(out4, dtype=tf.int32)
    union_4 = tf.add(true4_count, out4_count)
    union_4 = tf.subtract(union_4, overlap_4)
    IoU_4 = tf.divide(tf.cast(overlap_4, tf.float32), tf.cast(union_4, tf.float32))

    tf.get_variable_scope().reuse_variables()
    # out_all = tf.add(tf.add(tf.add(out1_count, out2_count), out3_count), out4_count)
    # true_all = tf.add(tf.add(tf.add(true1_count, true2_count), true3_count), true4_count)
    # IoU = tf.reduce_mean(tf.stack([IoU_1, IoU_2, IoU_3, IoU_4]))
    # precision = pred_all/out_all
    # recall = pred_all/true_all
    # f1 = 2*precision*recall/(precision+recall)
    return accuracy, accuracy_back, [IoU_1, IoU_2, IoU_3, IoU_4]


def forward(test_dataset, net, phase, scope):
    img_batch, label_instance_batch, label_existence_batch = batch_queue.dequeue()
    inference = net.inference(img_batch, phase, 'lanenet_loss')
    out_logits = tf.add_n(tf.get_collection('instance_seg_logits', scope))
    # calculate the accuracy
    out_logits = tf.nn.softmax(logits=out_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1)

    pred_0 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 0), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 0), tf.int32)),
                              dtype=tf.int32)
    pred_1 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 1), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 1), tf.int32)),
                              dtype=tf.int32)
    pred_2 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 2), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 2), tf.int32)),
                              dtype=tf.int32)
    pred_3 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 3), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 3), tf.int32)),
                              dtype=tf.int32)
    pred_4 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(label_instance_batch, 4), tf.int32),
                                          tf.cast(tf.equal(out_logits_out, 4), tf.int32)),
                              dtype=tf.int32)
    gt_all = tf.count_nonzero(tf.cast(tf.greater(label_instance_batch, 0), tf.int32), dtype=tf.int32)
    gt_back = tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 0), tf.int32), dtype=tf.int32)

    pred_all = tf.add(tf.add(tf.add(pred_1, pred_2), pred_3), pred_4)

    accuracy = tf.divide(tf.cast(pred_all, tf.float32), tf.cast(gt_all, tf.float32))
    accuracy_back = tf.divide(tf.cast(pred_0, tf.float32), tf.cast(gt_back, tf.float32))

    # Compute mIoU of Lanes
    overlap_1 = pred_1
    union_1 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 1),
                                              tf.int32), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 1),
                                              tf.int32), dtype=tf.int32))
    union_1 = tf.subtract(union_1, overlap_1)
    IoU_1 = tf.divide(tf.cast(overlap_1, tf.float32), tf.cast(union_1, tf.float32))

    overlap_2 = pred_2
    union_2 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 2),
                                              tf.int32), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 2),
                                              tf.int32), dtype=tf.int32))
    union_2 = tf.subtract(union_2, overlap_2)
    IoU_2 = tf.divide(tf.cast(overlap_2, tf.float32), tf.cast(union_2, tf.float32))

    overlap_3 = pred_3
    union_3 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 3),
                                              tf.int32), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 3),
                                              tf.int32), dtype=tf.int32))
    union_3 = tf.subtract(union_3, overlap_3)
    IoU_3 = tf.divide(tf.cast(overlap_3, tf.float32), tf.cast(union_3, tf.float32))

    overlap_4 = pred_4
    union_4 = tf.add(tf.count_nonzero(tf.cast(tf.equal(label_instance_batch, 4),
                                              tf.int64), dtype=tf.int32),
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 4),
                                              tf.int64), dtype=tf.int32))
    union_4 = tf.subtract(union_4, overlap_4)
    IoU_4 = tf.divide(tf.cast(overlap_4, tf.float32), tf.cast(union_4, tf.float32))

    # IoU = tf.reduce_mean(tf.stack([IoU_1, IoU_2, IoU_3, IoU_4]))

    tf.get_variable_scope().reuse_variables()

    return accuracy, accuracy_back, [IoU_1, IoU_2, IoU_3, IoU_4] # out_logits_out


def test_net(dataset_dir, weights_path=None, net_flag='vgg'):
    test_dataset_file = ops.join(dataset_dir, 'test0_normal_gt.txt')

    assert ops.exists(test_dataset_file)

    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    test_dataset = lanenet_data_processor.DataSet(test_dataset_file)

    net = lanenet_merge_model.LaneNet()

    test_img, test_label_instance, test_label_existence = test_dataset.next_batch(CFG.TEST.BATCH_SIZE)
    test_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [test_img, test_label_instance, test_label_existence], capacity=2 * CFG.TRAIN.GPU_NUM,
        num_threads=CFG.TRAIN.CPU_NUM)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(CFG.TRAIN.GPU_NUM):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('test_%d' % i) as scope:
                    accuracy, accuracy_back, IoU_lst = forward(test_batch_queue, net, phase, scope)

    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)

    with tf.Session(config=sess_config) as sess:
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path=weights_path)
            
        tf.train.start_queue_runners(sess=sess)
        
        epoch_accuracy_dict = {}
        for epoch in range(1):

            val_accuracy_mean = []
            val_accuracy_back_mean = []

            for epoch_val in range(int(len(test_dataset) / CFG.TEST.TEST_BATCH_SIZE / CFG.TRAIN.GPU_NUM)):
                accuracy, accuracy_back, IoU_lst = \
                    sess.run(
                        [accuracy, accuracy_back, IoU_lst],
                        feed_dict={phase: 'test'})


                val_accuracy_mean.append(val_accuracy)
                val_accuracy_back_mean.append(val_accuracy_back)

    
        

if __name__ == '__main__':
    # init args
    args = init_args()
    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)
    save_dir = os.path.join(args.image_path, 'predicts')
    if args.save_dir is not None:
        save_dir = args.save_dir

    img_name = []
    with open(str(args.image_path), 'r') as g:
        for line in g.readlines():
            img_name.append(line.strip())

    test_lanenet(args.image_path, args.weights_path, args.use_gpu, img_name, args.batch_size, save_dir)