#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:04:00 2019

@author: aditya
"""

#Parameters
learning_rate = 0.01
num_steps = 100
batch_size = 128
display_step = 1

#Network Parameters
n_hidden1 = 300
n_hidden2 = 300
n_hidden3 = 300
num_input = 784
num_classes = 10


#Training Parameters
checkpoint_every = 100
checkpoint_dir = './runs/'