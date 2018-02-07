# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:42:23 2018

@author: Andrei Ionut Damian
"""

import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(288, 288, 3))
vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', 
                                          include_top=False, input_tensor=inputs)
x = tf.keras.layers.Conv2D(filters=1000, 
           kernel_size=(1, 1))(vgg16.output)
x = tf.keras.layers.Conv2DTranspose(filters=1000, 
                    kernel_size=(64, 64),
                    strides=(32, 32),
                    padding='same',)(x)
