#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:08:06 2019

@author: aditya
"""

import tensorflow as tf
import hy_param

X = tf.placeholder("float",[None,hy_param.num_input],name="input_x")
Y = tf.placeholder("float",[None,hy_param.num_classes],name="input_y")


weights = {
    'h1' : tf.Variable(tf.random_normal([hy_param.num_input,hy_param.n_hidden1])),
    'h2' : tf.Variable(tf.random_normal([hy_param.n_hidden1,hy_param.n_hidden2])),
    'out': tf.Variable(tf.random_normal([hy_param.n_hidden2,hy_param.num_classes]))
           }
           
biases = {
    'b1': tf.Variable(tf.random_normal([hy_param.n_hidden1])),
    'b2': tf.Variable(tf.random_normal([hy_param.n_hidden2])),
    'out': tf.Variable(tf.random_normal([hy_param.num_classes]))
          }
          

#defining the operation of the hidden layers
layer_1 = tf.add(tf.matmul(X,weights['h1']),biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])

# Output fully connected layer with a neuron for each class
logits = tf.matmul(layer_2,weights['out']) + biases['out']
                   
# Performing softmax operation
prediction = tf.nn.softmax(logits,name='prediction')

#Define Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=hy_param.learning_rate)
train_op = optimizer.minimize(loss_op)


#Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)
,name='accuracy')