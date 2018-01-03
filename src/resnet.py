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
import config

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
        scale=True # multiply by gamma
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
        pad_head = int(math.floor(pad_total / 2))
        pad_tail = pad_total - pad_head
        layer = tf.pad(layer, [[0, 0], [pad_head, pad_tail], [pad_head, pad_tail], [0, 0]])

    # conv2d
    layer = tf.layers.conv2d(
        inputs=layer,
        filters=filter_num,
        kernel_size=kernel_size,
        strides=stride,
        padding=("SAME" if stride == 1 else "VALID"),
        use_bias=False
    )
    return layer

def add_standard_building_block(layer, shortcut_fn, filter_num, stride, is_training):
    """Standard building block for shallow ResNet (e.g. 34)

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
    """Bottleneck building block for deeper ResNet (e.g. 50/101/152/..)

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
    # bottleneck building block has 4*original_filter_number
    layer = apply_conv2d_with_padding(layer, 4*filter_num, 1, 1)
    layer = layer + shortcut_connection
    return layer

def add_layer_compound(layer, filter_num, building_block_fn, block_num, stride, is_training):
    """Construct compound layer of building blocks

    Compound layers differs in number of channels (filer_num) and layer width/height
    Args:
        layer: input layer
        filter_num: number of output channels
        building_block_fn: function used to apply building block
        block_num: number of building blocks in this layer compound
        stride: stride of the convolution along the height and width
        is_training: flag for moving average acculumator
    Returns:
        output layer
    """
    def shortcut_projection_fn(layer):
        """Projection tranform from input to output size
        Args:
            layer: input layer
        Returns:
            output layer
        """
        if building_block_fn is add_bottleneck_building_block:
            # bottleneck building block has 4*original_filter_number
            layer = apply_conv2d_with_padding(layer, 4*filter_num, 1, stride)
            return layer
        else:
            layer = apply_conv2d_with_padding(layer, filter_num, 1, stride)
            return layer

    # first building block do the width/height/depth projection
    layer = building_block_fn(layer, shortcut_projection_fn, filter_num, stride, is_training)

    # other building blocks apply stride = 1 and identity shortcut
    for _ in xrange(1, block_num):
        layer = building_block_fn(layer, None, filter_num, 1, is_training)

    return layer

def build_graph(x, is_training):
    """Build compute graph

    ResNet model for cifar-10 dataset
    Args:
        x: input image data
        is_training: True for training, False for prediction
    Returns:
        output layer (logits of classes)
    """
    # input transform
    layer = apply_conv2d_with_padding(
        layer=x,
        filter_num=16,
        kernel_size=3,
        stride=1
    )
    # size: 32x32, depth: 16

    # 1st compound
    layer = add_layer_compound(
        layer=layer,
        filter_num=16,
        building_block_fn=add_standard_building_block,
        block_num=5,
        stride=1,
        is_training=is_training
    )
    # size: 32x32, depth: 16

    # 2nd compound
    layer = add_layer_compound(
        layer=layer,
        filter_num=32,
        building_block_fn=add_standard_building_block,
        block_num=5,
        stride=2,
        is_training=is_training
    )
    # size: 16x16, depth: 32

    # 3rd compound
    layer = add_layer_compound(
        layer=layer,
        filter_num=64,
        building_block_fn=add_standard_building_block,
        block_num=5,
        stride=2,
        is_training=is_training
    )
    # size: 8x8, depth: 64

    # projection output
    layer = apply_batchnorm_and_relu(layer, is_training)
    layer = tf.layers.average_pooling2d(
        inputs=layer,
        pool_size=8,
        strides=1,
        padding="VALID",
        data_format="channels_last"
    )
    layer = tf.reshape(layer, [-1, 64])
    layer = tf.layers.dense(
        inputs=layer,
        units=config.CLASS_NUM
    )
    return layer

if __name__ == "__main__":
    with tf.Session() as sess:
        X = tf.placeholder(
            tf.float32,
            [None, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH]
        )
        logits = build_graph(X, True)
