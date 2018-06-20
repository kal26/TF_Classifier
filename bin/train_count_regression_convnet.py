#!/bin/python
import os
import sys
sys.path.append('/home/kal/TF_models/bin/')
sys.path.append('/home/kal/TF_models/seq_only/bin/')
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import tf_memory_limit
#import general use packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
#import keras related packages
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler
from keras.utils import plot_model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf
#import custom packages
import helper
import viz_sequence
import train_TFmodel
import sequence
import seq_only_gen


def make_model(out_path, conv_string, gen, fine_grain=True, num_epochs=10, verb=0):
    """Construct and train a convolutional transcription factor prediciton network."""
    #Define some params
    input_window = 256
    batch_size = 32
    drop_rate = 0.1
    conv_list = [[int(x) for x in cell.split('.')] for cell in conv_string.split('_')]
    # Get the time-tag for the model.
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # make a file system
    weights_path = os.path.join(out_path, 'intermediate_weights')
    os.makedirs(weights_path)
    history_path = os.path.join(out_path, 'history')
    os.makedirs(history_path)

    # get a data generator
    
    
    # define the model construction
    # A one how input with reverse complement -- > series of convolutions --> smoothing --> maximum over directions --> bias --> activation
    # Input *is* one hot encoded - and type np.uint8.
    input = Input(batch_shape=(batch_size, input_window, 4))
    # add reverse complement so the model only has to learn one direciton
    add_RC_to_batch = Lambda(lambda x: K.concatenate([x, x[:, ::-1, ::-1]], axis=0), output_shape=lambda s: (2 * s[0], s[1], s[2]))
    # output an acitvaiton at each base form convolutions of convolutions
    per_base_score = train_TFmodel.BasicConv(input_window, conv_list, final_activation=None, num_outputs=1, drop_rate=drop_rate)
    # Take wide-window convolution scan with fixed wieghts to smooth out peaks.
    wide_scan = Conv1D(1, 50, use_bias=False, kernel_initializer='ones', trainable=False, name='wide_scan', padding='valid')
    # Get the forward/reverse sequence maximum. 
    max_by_direction = Lambda(lambda x: K.maximum(K.max(x[:x.shape[0]//2, :, :], axis=1), K.max(x[x.shape[0]//2:, ::-1, :], axis=1)), name='stackmax', output_shape=lambda s: (s[0] // 2, 1))
    # Add a custom bias layer
    final_bias = train_TFmodel.Bias(1, name='bias')

    # build the model
    predictions = final_bias(max_by_direction(wide_scan(per_base_score(add_RC_to_batch(input)))))
    model = Model(inputs=[input], outputs=[predictions])
    # save a graph of the model configuration
    plot_model(model, to_file=os.path.join(out_path, timestr + '_' + conv_string + '_model.png'), show_shapes=True)

    # create optimizers for the three learning phases with learning rate 1/10th of previous at each step
    #optimizer_1 = Adam(beta_1=0.95, lr=0.0005)
    optimizer_1 = Adam(beta_1=0.95, lr=0.0005, epsilon=.1)
    optimizer_2 = Adam(beta_1=0.95, lr=0.00005, epsilon=.001)
    
    #train the model
    num_batches = TFgen.get_num_training_examples() // batch_size
    if fine_grain:
        # go for 250 'epochs' in each of the three training stages
        # each 'epoch' is actually only 1/50 of the data -- so 5 real epochs
        # compile and run the first iteration of the training params and model
        model.compile(loss=loss_func, optimizer=optimizer_1, metrics=['mse'])
        History_1 = model.fit_generator(traingen, num_batches // 50, epochs=num_epochs*50 // 3, validation_data=valgen, validation_steps=20, callbacks=callbacks_list, verbose=verb)
        # compile and train the second iteration
        checkpath = os.path.join(weights_path, '_'.join([timestr, 'weights_2_{epoch:02d}_{val_loss:.2f}.hdf5']))
        checkpointer = ModelCheckpoint(checkpath, verbose=verb, monitor='val_loss', mode='max')
        callbacks_list = [checkpointer]
        model.compile(loss=loss_func, optimizer=optimizer_2, metrics=['mse'])
        History_2 = model.fit_generator(traingen, num_batches // 50, epochs=num_epochs*50 // 3, validation_data=valgen, validation_steps=20, callbacks=callbacks_list, verbose=verb)
        # compile and train the third iteration
        checkpath = os.path.join(weights_path, '_'.join([timestr, 'weights_3_{epoch:02d}_{val_loss:.2f}.hdf5']))
        checkpointer = ModelCheckpoint(checkpath, verbose=verb, monitor='val_loss', mode='min')
        callbacks_list = [checkpointer]
        model.compile(loss=loss_func, optimizer=optimizer_3, metrics=['mse'])
        History_3 = model.fit_generator(traingen, num_batches // 50, epochs=num_epochs*50 // 3, validation_data=valgen, validation_steps=20, callbacks=callbacks_list, verbose=verb)
    else:
        # just do three big epochs
        model.compile(loss=loss_func, optimizer=optimizer_1, metrics=['mse'])
        History_1 = model.fit_generator(traingen, num_batches, epochs=num_epochs // 3, validation_data=valgen, validation_steps=20, callbacks=callbacks_list, verbose=verb)
        # compile and train the second iteration
        checkpath = os.path.join(weights_path, '_'.join([timestr, 'weights_2_{epoch:02d}_{val_loss:.2f}.hdf5']))
        checkpointer = ModelCheckpoint(checkpath, verbose=verb, monitor='val_loss', mode='max')
        callbacks_list = [checkpointer]
        model.compile(loss=loss_func, optimizer=optimizer_2, metrics=['mse'])
        History_2 = model.fit_generator(traingen, num_batches, epochs=num_epochs // 3, validation_data=valgen, validation_steps=20, callbacks=callbacks_list, verbose=verb)
        # compile and train the third iteration
        checkpath = os.path.join(weights_path, '_'.join([timestr, 'weights_3_{epoch:02d}_{val_loss:.2f}.hdf5']))
        checkpointer = ModelCheckpoint(checkpath, verbose=verb, monitor='val_loss', mode='min')
        callbacks_list = [checkpointer]
        model.compile(loss=loss_func, optimizer=optimizer_3, metrics=['mse'])
        History_3 = model.fit_generator(traingen, num_batches, epochs=num_epochs // 3, validation_data=valgen, validation_steps=20, callbacks=callbacks_list, verbose=verb)
    # Write out the loss and accuracy histories
    with open(os.path.join(history_path, timestr +'_history1.pk1'), 'wb') as output:
        pickle.dump(History_1.history, output, -1)
    with open(os.path.join(history_path, timestr +'_history2.pk1'), 'wb') as output:
       pickle.dump(History_2.history, output, -1)
    with open(os.path.join(history_path, timestr +'_history3.pk1'), 'wb') as output:
        pickle.dump(History_3.history, output, -1)
    print('Created history pickle')

    # save the final model
    model.save(os.path.join(out_path, 'final_model.hdf5'))

def main(argv=None):
    # Get the time-tag for the model.
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # make a file system
    out_path = os.path.join(FLAGS.out_dir, timestr + '_convnet')
    os.makedirs(out_path)
    make_model(out_path, FLAGS.conv_list, FLAGS.gen_path)


if __name__ == '__main__':
    tf.app.run()
import gc; gc.collect()


#!/bin/python
import os
import sys
sys.path.append('/home/kal/TF_models/bin/')
#!/bin/python
import os
import sys
#!/bin/python
import os
import sys
sys.path.append('/home/kal/TF_models/bin/')
