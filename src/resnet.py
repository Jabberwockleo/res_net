#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: resnet.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

import math
import tensorflow as tf

def apply_batchnorm_and_relu(layer, is_training):
    """Apply batchnorm for resnet v2(K. He 2016) followed by ReLu.

    Args:
        layer: input layer
        is_training: flag for moving average acculumator
    Returns:
        output layer
    """
    layer = tf.layers.batch_normalization(
        inputs=layer,
        axis=3,
        training=is_training, # moving average mode
        momentum=0.997, # moving average param
        epsilon=1e-5, # moving average param
        center=True, # add offset of beta
        scale=True, # multiply by gamma
        fused=True # recommended by tensorflow sample project
    )
    layer = tf.nn.relu(layer)
    return layer

def apply_conv2d_with_padding(layer, filter_num, kernel_size, stride):
    """Apply 2D convolution with paddings.

    Args:
        layer: input layer
        filter_num: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        stride: An integer, specifying the strides of the convolution along the height and width. A single integer to specify the same value for all spatial dimensions.
    Returns:
        output layer
    """
    # paddings
    if stride > 1:
        # constant stride of 0 if stride > 1
        # SAME padding if stride == 1
        pad_total = kernel_size - 1
        pad_head = math.floor(pad_total / 2)
        pad_tail = pad_total - pad_head
        layer = tf.pad(layer, [[0, 0], [pad_head, pad_tail], [pad_head, pad_tail], [0, 0]])

    # conv2d
    layer = tf.layers.conv2d(
        inputs=layer,
        filters=filter_num,
        kernel_size=kernel_size,
        strides=stride,
        padding=("SAME" if stride == 1 else "VALID"),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer()
    )
    return layer

def add_standard_building_block(layer, shortcut_fn, filter_num, stride, is_training):
    """Standard building block for 34-layer ResNet

    Args:
        layer: input layer
        shortcut_fn: tranform function applied for shortcut
        filter_num: the number of filters in the convolution
        stride: stride of the convolution along the height and width
        is_training: flag for moving average acculumator
    Returns:
        output layer
    """
    shortcut_connection = layer
    layer = apply_batchnorm_and_relu(layer, is_training)
    if shortcut_fn is not None:
        shortcut_connection = shortcut_fn(layer)
    layer = apply_conv2d_with_padding(layer, filter_num, 3, stride)
    layer = apply_batchnorm_and_relu(layer, is_training)
    layer = apply_conv2d_with_padding(layer, filter_num, 3, 1)
    layer = layer + shortcut_connection
    return layer

def add_bottleneck_building_block(layer, shortcut_fn, filter_num, stride, is_training):
    """Bottleneck building block for deeper ResNet (50/101/152/..)

    Args:
        layer: input layer
        shortcut_fn: tranform function applied for shortcut
        filter_num: the number of filters in the convolution
        stride: stride of the convolution along the height and width
        is_training: flag for moving average acculumator
    Returns:
        output layer
    """
    shortcut_connection = layer
    layer = apply_batchnorm_and_relu(layer, is_training)
    if shortcut_fn is not None:
        shortcut_connection = shortcut_fn(layer)
    layer = apply_conv2d_with_padding(layer, filter_num, 1, 1)
    layer = apply_batchnorm_and_relu(layer, is_training)
    layer = apply_conv2d_with_padding(layer, filter_num, 3, stride)
    layer = apply_batchnorm_and_relu(layer, is_training)
    layer = apply_conv2d_with_padding(layer, 4*filter_num, 1, 1)
    layer = layer + shortcut_connection
    return layer


if __name__ == "__main__":
    pass
