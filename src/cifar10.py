#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: cifar10.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

import os
import tensorflow as tf
import numpy as np
import config
import resnet

def model_fn(features, labels, mode, params):
    """Estimator API model function to build an custom tf.estimator.Estimator

    1. Configure the model via TensorFlow operations
    2. Define the loss function for training/evaluation
    3. Define the training operation/optimizer
    4. Generate predictions
    5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    detailed: https://www.tensorflow.org/extend/estimators

    Returns:
        tf.estimator.EstimatorSpec instance
    """
    # summaries
    tf.summary.image('images', features, max_outputs=10)

    # Input
    X = tf.reshape(features, [-1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH])

    # graph
    logits = resnet.build_graph(X, mode == tf.estimator.ModeKeys.TRAIN)

    # predictions
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # Mode: predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Mode: train or evaluate
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    cross_entropy = tf.identity(loss, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO
        pass

def input_fn(is_training):
    """Input function streaming train/predict data

    Args:
        is_training: True for get training data, False for prediction

    Returns:
        images, labels (generator object)
    """
    def data_file_paths(is_training):
        """Get file names

        Args:
            is_training: True for get training data, False for prediction
        Returns:
            array of filenames
        """
        if is_training:
            return [os.path.join(config.DATA_DIR, 'cifar-10-batches-bin/data_batch_%d.bin' % (i + 1)) \
                    for i in xrange(5)]
        else:
            return [os.path.join(config.DATA_DIR, 'cifar-10-batches-bin/test_batch.bin')]

    data_row_length = config.IMG_HEIGHT * config.IMG_WIDTH * config.IMG_DEPTH + 1 # 1 is the label byte
    # dataset instance
    dataset = tf.data.FixedLengthRecordDataset(
        data_file_paths(is_training),
        data_row_length
    )

    if is_training:
        # shuffle dataset, prepare number of total training images
        dataset = dataset.shuffle(buffer_size=config.TRAIN_IMG_NUM)

    def parse_row(row_data):
        """Parse data row to label, image

        Args:
            row_data: string representation of one row of data

        Returns:
            image, label tensor
        """
        pixel_num = config.IMG_DEPTH * config.IMG_HEIGHT * config.IMG_WIDTH
        row_vec = tf.decode_raw(row_data, tf.uint8)
        label = tf.one_hot(tf.cast(row_vec[0], tf.int32), config.CLASS_NUM)
        img_chw = tf.reshape(row_vec[1:pixel_num + 1], [config.IMG_DEPTH, config.IMG_HEIGHT, config.IMG_WIDTH])
        # convert to shape [h, w, c]
        image = tf.cast(tf.transpose(img_chw, [1, 2, 0]), tf.float32)
        return image, label

    # parse to tensor, label pairs
    dataset = dataset.map(parse_row)

    def preprocess_image(image, is_training):
        """Preprocess image

        Randomly crop or affine transform if training
        """
        if is_training:
            image = tf.image.resize_image_with_crop_or_pad(
                image,
                config.IMG_HEIGHT + 8,
                config.IMG_WIDTH + 8
            )
            image = tf.random_crop(
                image,
                [config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_DEPTH]
            )
            image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        return image

    # preprocess image
    dataset = dataset.map(
        lambda image, label: (preprocess_image(image, is_training), label)
    )

    # batch config
    dataset.prefetch(config.BATCH_SIZE * 2)
    dataset.repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    # data generator
    images, labels = iterator.get_next()
    return images, labels


if __name__ == "__main__":
    with tf.Session() as sess:
        X, Y = input_fn(True)
        for i in xrange(2):
            print "== iter [%d]" % i
            print "Y:", tf.shape(Y)
            print "Y:", sess.run(Y)
            #print "X:", tf.shape(X)
            #print "X:", sess.run(X)
    pass
