#!/bin/python
import os
import sys
sys.path.insert(0,"/home/kal/CTCF/modules/")
sys.path.insert(0,"/home/kal/CTCF/mass_CTCF/modules/")
sys.path.insert(0,"/home/kal/CTCF/ATAC_CTCF/modules/")
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import matplotlib; matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import pylab
import seaborn
import tf_memory_limit

from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler
from keras.utils import plot_model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation
from keras.optimizers import SGD, Adam
import keras.backend as K
from scipy.ndimage.interpolation import shift

import numpy as np
import math
import time
import pickle

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('out_dir', '/home/kal/CTCF/ATAC_CTCF/output', 'Location to write output files.')
tf.app.flags.DEFINE_string('gen_path', None, 'Location of generator (not yet supported)')
tf.app.flags.DEFINE_string('loss', 'mean_squared_error', 'Loss function to use')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs of training data to run.')
tf.app.flags.DEFINE_bool('change_e', True, 'Change epsilon for adam optimizer')
tf.app.flags.DEFINE_bool('finer_epochs', True, 'Use  1/50th epoch size')
tf.app.flags.DEFINE_bool('log', True, 'Use log of counts')

import ctcfdhsgen
import convnet
def main(argv=None):
    """ Train a network to the given specifications."""

    #Define some params
    input_window = FLAGS.input_window
    batch_size = FLAGS.batch_size
    # Get the time-tag for the model.
    timestr = time.strftime("%Y%m%d_%H%M%S")
    better_out_dir = os.path.join(FLAGS.out_dir, timestr)
    os.makedirs(better_out_dir)
 
    # Input *is* one hot encoded - and type np.uint8.
    input = Input(batch_shape=(batch_size, input_window, 5))
    # add the reverse complement of the sequence to the batch -- this way it only has to learn one orentation of the sequence. 
    add_RC_to_batch = Lambda(lambda x: K.concatenate([x, K.concatenate([x[:, :, :4][:,::-1, ::-1],x[:,::-1,4:5]], axis=2)], axis=0), output_shape=lambda s: (2 * s[0], s[1], s[2]))
    # Output a (batchsize, sequencelength, 1) vector.
    correct_conv_list = [[int(x) for x in cell.split('.')] for cell in FLAGS.conv_list.split('_')]
    per_base_score = convnet.BasicConv(input_window, correct_conv_list, num_outputs=1, drop_rate=FLAGS.drop_rate)
    # Take wide-window convolution scan with fixed wieghts to smooth out peaks.
    wide_scan = Conv1D(1, 50, use_bias=False, kernel_initializer='ones', trainable=False, name='wide_scan', padding='valid')
    # Get the forward/reverse sequence maximum. 
    max_by_direction = Lambda(lambda x: K.maximum(K.max(x[:batch_size, :, :], axis=1), K.max(x[batch_size:, ::-1, :], axis=1)), name='stackmax', output_shape=lambda s: (s[0] // 2, 1))
    # Add a custom bias layer
    final_bias = convnet.Bias(1, name='bias')

    # build the model
    predictions = final_bias(max_by_direction(wide_scan(per_base_score(add_RC_to_batch(input)))))
    model = Model(inputs=[input], outputs=[predictions])

    # save a graph of the model configuration
    plot_model(model, to_file=os.path.join(better_out_dir, timestr + '_' + FLAGS.conv_list + '_model.png'), show_shapes=True)

    # create optimizers for the three learning phases with learning rate 1/10th of previous at each step
    optimizer_1 = Adam(beta_1=0.95, lr=0.0005, epsilon=.1)
    if FLAGS.change_e:
        optimizer_2 = Adam(beta_1=0.95, lr=0.00005, epsilon=.001)
        optimizer_3 = Adam(beta_1=0.95, lr=0.000005, epsilon=.00001)
    else:
        optimizer_2 = Adam(beta_1=0.95, lr=0.00005)
        optimizer_3 = Adam(beta_1=0.95, lr=0.000005)

    # Create a data generator object.
    if FLAGS.gen_path != None:
        print('Loading generator')
        if '.hdf5' in FLAGS.gen_path:
            gen = ctcfdhsgen.CTCFGeneratorhdf5(FLAGS.gen_path)
        elif '.pk1' in FLAGS.gen_path:
            with open(FLAGS.gen_path, 'rb') as input:
                gen = pickle.load(input)
    else:
        print('Creating Generator')
        try:
            gen = ctcfdhsgen.CTCFDHSGen()
        except OSError:
            print('Please provide a file path or data path.')
            raise OSError('No file path to data provided.')

    # create generator with labels
    g = gen.batch_gen(log=FLAGS.log)
    g_val = gen.batch_gen(mode='val', log=FLAGS.log)

    # Create a callback to save the model weights.
    checkpath = os.path.join(better_out_dir, '_'.join([timestr, 'weights_1_{epoch:02d}_{val_loss:.2f}.hdf5']))
    checkpointer = ModelCheckpoint(checkpath, verbose=1, monitor='val_loss', mode='min') 
    callbacks_list = [checkpointer]

    # Train the model
    print('Training Model')
    # Figure out the number of epochs to run
    num_batches = gen.get_num_training_examples() // FLAGS.batch_size

    # complie the first iteration of the training params and model
    model.compile(loss=FLAGS.loss, optimizer=optimizer_1)

    # Train the model and get the loss/acc stats in a History object
    # note that a full validation is not run after every epoch -- just 25 batches (for computational speed)
    if FLAGS.finer_epochs:
        History_1 = model.fit_generator(g, num_batches // 10, epochs=FLAGS.num_epochs*10//3, validation_data=g_val, validation_steps=25, callbacks=callbacks_list)
        # Create a callback to save the model weights.
        checkpath = os.path.join(better_out_dir, '_'.join([timestr, 'weights_2_{epoch:02d}_{val_loss:.2f}.hdf5']))
        checkpointer = ModelCheckpoint(checkpath, verbose=1, monitor='val_loss', mode='min')
        callbacks_list = [checkpointer]
        model.compile(loss=FLAGS.loss, optimizer=optimizer_2)
        History_2 = model.fit_generator(g, num_batches // 10, epochs=FLAGS.num_epochs*10//3, validation_data=g_val, validation_steps=25, callbacks=callbacks_list)
        # Create a callback to save the model weights.
        checkpath = os.path.join(better_out_dir, '_'.join([timestr, 'weights_3_{epoch:02d}_{val_loss:.2f}.hdf5']))
        checkpointer = ModelCheckpoint(checkpath, verbose=1, monitor='val_loss', mode='min')
        callbacks_list = [checkpointer]
        model.compile(loss=FLAGS.loss, optimizer=optimizer_3)
        History_3 = model.fit_generator(g, num_batches // 10, epochs=FLAGS.num_epochs*10//3, validation_data=g_val, validation_steps=25, callbacks=callbacks_list)       
    else:
        History = model.fit_generator(g, num_batches, epochs=FLAGS.num_epochs, validation_data=g_val, validation_steps=25, callbacks=callbacks_list)

    # Write out the loss and accuracy histories
    with open(os.path.join(better_out_dir, timestr +'_history1.pk1'), 'wb') as output:
        pickle.dump(History_1.history, output, -1)
        print('Created history pickle')
    if History_2 != None:
        with open(os.path.join(better_out_dir, timestr +'_history2.pk1'), 'wb') as output:
            pickle.dump(History_2.history, output, -1)
            print('Created history pickle')
        with open(os.path.join(better_out_dir, timestr +'_history3.pk1'), 'wb') as output:
            pickle.dump(History_3.history, output, -1)
            print('Created history pickle')

if __name__ == '__main__':
    tf.app.run()
import gc; gc.collect()
