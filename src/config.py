#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: config.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

# environment config
DATA_DIR = "../data/"
MODEL_DIR = "../model/"
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

# hyper parameters
LEARNING_RATE = 0.01
EPOCH_NUM = 1000
BATCH_SIZE = 100
TRAIN_IMG_NUM = 50000
VALIDATION_IMG_NUM = 10000
WEIGHT_DECAY = 2e-4
MOMENTUM = 0.9

# dataset parameters
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_DEPTH = 3
CLASS_NUM = 10
FILE_NUM = 5
