#!/bin/python
import os
import sys
sys.path.append('/home/kal/TF_models/bin/')
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import matplotlib; matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import pylab
import seaborn
import tf_memory_limit
import ctcfgen

from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler
from keras.utils import plot_model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation
from keras.optimizers import SGD, Adam
import keras.backend as K
from scipy.ndimage.interpolation import shift

import numpy as np
import math
import time

import tensorflow as tf


# get the generator
gen = ctcfgen.CTCFGeneratorhdf5('/home/kal/TF_models/seq_only/seq_classifier/ctcfgen_data.hdf5')
g = gen.pairgen()
g_val = gen.pairgen(mode='val')
num_batches = gen.get_num_training_examples() // 32

# make a covlution neural net with one layer and a tanh (0 centered)! 
#later, try fixing the weights on said layer to predifined position weight matrix weights

input = Input(batch_shape=(32, 256, 4))
add_RC_to_batch = Lambda(lambda x: K.concatenate([x, x[:, ::-1, ::-1]], axis=0), output_shape=lambda s: (2 * s[0], s[1], s[2]))
pwm_conv = Conv1D(1, 32, padding='valid', input_shape=(256,4))
max_by_direction = Lambda(lambda x: K.maximum(K.max(x[:x.shape[0]//2, :, :], axis=1), K.max(x[x.shape[0]//2:, ::-1, :], axis=1)), name='stackmax', output_shape=lambda s: (s[0] // 2, 1))
linearize = Activation('tanh')
predictions = linearize(max_by_direction(pwm_conv(add_RC_to_batch(input))))
model = Model(inputs=[input], outputs=[predictions])

# compile the network
optimizer = Adam(beta_1=0.95, lr=0.0005, epsilon=.1)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# train the network
History = model.fit_generator(g, steps_per_epoch=num_batches, epochs=10, validation_data=g_val, validation_steps=100)

model.save('/home/kal/TF_models/seq_only/pwm_model/final_output.hdf5')
