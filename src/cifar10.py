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
import tarfile
import config

if __name__ == "__main__":
    file_path = config.DATA_DIR + config.DATA_URL.split("/")[-1]
    tarfile.open(file_path, 'r:gz').extractall(config.DATA_DIR)
    pass
