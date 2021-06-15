#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午2:04
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : dilation_encoder.py
# @IDE: PyCharm Community Edition

from collections import OrderedDict
import math

import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from config import global_config

CFG = global_config.cfg

# replace VGG16Encoder with ResNetEncoder
class ResNetEncoder(cnn_basenet.CNNBaseModel):
    """
    ResNet encoder 
    """

    def __init__(self, phase, res_n):
        """

        :param phase:
        """
        super(ResNetEncoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()
        self.res_n = res_n

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _conv_dilated_stage(self, input_tensor, k_size, out_dims, name,
                            dilation=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.dilation_conv(input_tensor=input_tensor, out_dims=out_dims,
                                      k_size=k_size, rate=dilation,
                                      use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            fc = self.fullyconnect(inputdata=input_tensor, out_dim=out_dims, use_bias=use_bias,
                                   name='fc')

            bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _resblock_first(self, x, out_channel, _is_training, strides, name='unit'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):

            # Shortcut
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self.conv2d(inputdata=x, out_channel=out_channel, kernel_size=1, stride=strides, name='shortcut')
            
            # Residual
            x = self.conv2d(inputdata=x, out_channel=out_channel, kernel_size=3, stride=strides, name='conv_1')
            x = self.layerbn(inputdata=x, is_training=self._is_training, name='bn_1')
            # x = self.res_bn(x=x, is_training=self._is_training, name='bn_1')
            x = self.relu(inputdata=x, name='relu_1')
            x = self.conv2d(inputdata=x, out_channel=out_channel, kernel_size=3, stride=1, name='conv_2')
            x = self.layerbn(inputdata=x, is_training=self._is_training, name='bn_2')
            # x = self.res_bn(x=x, is_training=self._is_training, name='bn_2')
            x = x + shortcut
            x = self.relu(inputdata=x, name='relu_2')

        return x


    def _resblock(self, x, _is_training, name='unit'):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
        # with tf.variable_scope(name, reuse=True):
            # Shortcut
            shortcut = x
            # Residual
            x = self.conv2d(inputdata=x, out_channel=num_channel, kernel_size=3, stride=1, name='conv_1')
            x = self.layerbn(inputdata=x, is_training=self._is_training, name='bn_1')
            # x = self.res_bn(x=x, is_training=self._is_training, name='bn_1')
            x = self.relu(inputdata=x, name='relu_1')
            x = self.conv2d(inputdata=x, out_channel=num_channel, kernel_size=3, stride=1, name='conv_2')
            x = self.layerbn(inputdata=x, is_training=self._is_training, name='bn_2')
            # x = self.res_bn(x=x, is_training=self._is_training, name='bn_2')
            x = x + shortcut
            x = self.relu(inputdata=x, name='relu_2')

        return x


    def encode(self, input_tensor, name):
        """
        ResNet encoder
        :param input_tensor:
        :param name:
        :return: ResNet features
        """
        ret = OrderedDict()
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope(name):
            with tf.variable_scope('conv1'):
                x = self.conv2d(inputdata=input_tensor, out_channel=filters[0], kernel_size=kernels[0], stride=strides[0], name='conv')
                x = self.layerbn(inputdata=x, is_training=self._is_training, name='bn')
                # x = self.res_bn(x=x, is_training=self._is_training, name='bn')
                x = self.relu(inputdata=x, name='relu')
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

            # conv2_x
            x = self._resblock(x, self._is_training, name='conv2_1')
            x = self._resblock(x, self._is_training, name='conv2_2')

            # conv3_x
            x = self._resblock_first(x, filters[2], self._is_training, strides[2], name='conv3_1')
            x = self._resblock(x, self._is_training, name='conv3_2')

            # conv4_x
            x = self._resblock_first(x, filters[3], self._is_training, strides[3], name='conv4_1')
            x = self._resblock(x, self._is_training, name='conv4_2')

            # conv5_x
            x = self._resblock_first(x, filters[4], self._is_training, strides[4], name='conv5_1')
            x = self._resblock(x, self._is_training, name='conv5_2')

            ### add dilated convolution ###

            # conv stage 5_1
            conv_6_1 = self._conv_dilated_stage(input_tensor=x, k_size=3,
                                                out_dims=512, dilation=2, name='conv6_1')

            # conv stage 5_2
            conv_6_2 = self._conv_dilated_stage(input_tensor=conv_6_1, k_size=3,
                                                out_dims=512, dilation=2, name='conv6_2')

            # conv stage 5_3
            conv_6_3 = self._conv_dilated_stage(input_tensor=conv_6_2, k_size=3,
                                                out_dims=512, dilation=2, name='conv6_3')

            # added part of SCNN #

            # conv stage 5_4
            conv_6_4 = self._conv_dilated_stage(input_tensor=conv_6_3, k_size=3,
                                                out_dims=1024, dilation=4, name='conv6_4')

            # conv stage 5_5
            conv_6_5 = self._conv_stage(input_tensor=conv_6_4, k_size=1,
                                        out_dims=128, name='conv6_5')  # 8 x 36 x 100 x 128

            # add message passing #

            # top to down #

            """
            feature_list_old = []
            feature_list_new = []
            for cnt in range(conv_5_5.get_shape().as_list()[1]):
                feature_list_old.append(tf.expand_dims(conv_5_5[:, cnt, :, :], axis=1))
            feature_list_new.append(tf.expand_dims(conv_5_5[:, 0, :, :], axis=1))

            w1 = tf.get_variable('W1', [1, 9, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                                 # initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_1"):
                conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[1])
                feature_list_new.append(conv_6_1)

            for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_6_1", reuse=True):
                    conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[cnt])
                    feature_list_new.append(conv_6_1)

            # down to top #
            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(CFG.TRAIN.IMG_HEIGHT / 8) - 1
            feature_list_new.append(feature_list_old[length])

            w2 = tf.get_variable('W2', [1, 9, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                                 # initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_2"):
                conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[length - 1])
                feature_list_new.append(conv_6_2)

            for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_6_2", reuse=True):
                    conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[length - cnt])
                    feature_list_new.append(conv_6_2)

            feature_list_new.reverse()

            processed_feature = tf.stack(feature_list_new, axis=1)
            processed_feature = tf.squeeze(processed_feature, axis=2)

            # left to right #

            feature_list_old = []
            feature_list_new = []
            for cnt in range(processed_feature.get_shape().as_list()[2]):
                feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
            feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

            w3 = tf.get_variable('W3', [9, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                                 # initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_3"):
                conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[1])
                feature_list_new.append(conv_6_3)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_6_3", reuse=True):
                    conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[cnt])
                    feature_list_new.append(conv_6_3)

            # right to left #

            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(CFG.TRAIN.IMG_WIDTH / 8) - 1
            feature_list_new.append(feature_list_old[length])

            w4 = tf.get_variable('W4', [9, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
                                 # initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_4"):
                conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[length - 1])
                feature_list_new.append(conv_6_4)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_6_4", reuse=True):
                    conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[length - cnt])
                    feature_list_new.append(conv_6_4)

            feature_list_new.reverse()
            processed_feature = tf.stack(feature_list_new, axis=2)
            processed_feature = tf.squeeze(processed_feature, axis=3)

            """

            processed_feature = conv_6_5

            #######################

            dropout_output = self.dropout(processed_feature, 0.9, is_training=self._is_training,
                                          name='dropout')  # 0.9 denotes the probability of being kept

            conv_output = self.conv2d(inputdata=dropout_output, out_channel=5,
                                      kernel_size=1, use_bias=True, name='conv_6')

            ret['prob_output'] = tf.image.resize_images(conv_output, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
            
            ### add lane existence prediction branch ###

            # spatial softmax #
            features = conv_output  # N x H x W x C
            softmax = tf.nn.softmax(features)

            avg_pool = self.avgpooling(softmax, kernel_size=2, stride=2)
            _, H, W, C = avg_pool.get_shape().as_list()
            reshape_output = tf.reshape(avg_pool, [-1, H * W * C])
            fc_output = self.fullyconnect(reshape_output, 128)
            relu_output = self.relu(inputdata=fc_output, name='relu6')
            fc_output = self.fullyconnect(relu_output, 4)
            existence_output = fc_output

            ret['existence_output'] = existence_output

        return ret


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                       name='input')
    encoder = ResNetEncoder(phase=tf.constant('train', dtype=tf.string), res_n=18)
    ret = encoder.encode(a, name='encode')
    print(ret)
