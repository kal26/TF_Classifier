import tf_memory_limit

from keras.models import Model
from keras.layers import Conv1D, Dropout, Activation, add, Lambda, concatenate, multiply, SpatialDropout1D, Layer, Input
from keras.layers.advanced_activations import LeakyReLU
from keras import activations
from keras import initializers
from keras.callbacks import Callback
from keras.engine import InputSpec
import numpy as np
import time
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('drop_rate', 0.1, 'Percentage of features to drop between layers.')
tf.app.flags.DEFINE_string('conv_list', '32.3_32.32_16.3_8.3', 'List of convolution params - (num_hidden, kernel_size)')
tf.app.flags.DEFINE_bool('last_drop', False, 'Drop features on the last layer')
tf.app.flags.DEFINE_bool('middle_drops', True, 'Drop features on the intermediate layers')

class BasicBlock(object):
    """"Basic convolution block with dropout and weight decay."""
    def __init__(self, filters, kernel_width=3, decay=0.9999, drop_rate=0.1):
        """Initialize a convolution block.
        
        Arguments:
            filters -- Number of neurons in the convolution.
        Keywords:
            kernal_width -- Width of convolutional window.
            decay -- Decay of weights after each training update.
            drop_rate -- Fraction of neurons to drop.
        """    
        super(BasicBlock, self).__init__()
        self.conv = Conv1D(filters, kernel_width, padding='valid', activation=None)#Add leaky relu?
        # weight decay
        self.conv.add_update([(w, K.in_train_phase(decay * w, w)) for w in self.conv.trainable_weights])

        if drop_rate > 0 and FLAGS.middle_drops:
            self.dropout = SpatialDropout1D(drop_rate)
        else:
            self.dropout = None

    def __call__(self, x):
        """Run the convolution"""
        out = self.conv(x) #Add leaky relu?
        out = LeakyReLU()(out)
        if self.dropout != None:
            out = self.dropout(out)
        return out

class BasicConv(object):
    """Object for a series of convolutions."""
    def __init__(self, num_inputs, conv_list, num_outputs=1, drop_rate=0.0, final_activation=None):
        """Initialize a BasicConv object.
     
            Args:
                num_inputs -- Number of excpected inputs. 
                conv_list -- List of tuples of (num_hidden, kernel_size), one for each convlution in the net.
            Keywords:
                num_outputs -- Number of final outputs required.
                drop_rate -- Percentage of features to drop on each level.
        """        
        super(BasicConv, self).__init__()

        self.drop_rate = drop_rate 

        blockgen = BasicBlock
        print('Convolutions used: ' + str(conv_list) + ' [neurons, filter]')
        self.conv_in = Conv1D(conv_list[0][0], conv_list[0][1], padding='valid', activation=None, input_shape=(num_inputs, 5))

        self.hidden_layers = []
        for params in conv_list[1:]:
            self.hidden_layers.append(blockgen(params[0], kernel_width=params[1], drop_rate=drop_rate))

        self.conv_out = Conv1D(num_outputs, 1, padding='valid', activation=final_activation)

    def __call__(self, x):
        x = self.conv_in(x)
        x = LeakyReLU()(x)
        self.hidden_outputs = []
        self.hidden_outputs.append(x)
        if self.drop_rate > 0 and FLAGS.middle_drops:
            x = SpatialDropout1D(self.drop_rate)(x)
        for h in self.hidden_layers:
            x = h(x)
            self.hidden_outputs.append(x)
        x = self.conv_out(x)
        x = LeakyReLU()(x)
        if self.drop_rate > 0 and FLAGS.last_drop:
            x = SpatialDropout1D(self.drop_rate)(x)
        return x

class Bias(Layer):
    
    def __init__(self, units, activation=None, bias_initializer='zeros', **kwargs):
         super(Bias, self).__init__(**kwargs)
         self.units = units
         self.activation = activations.get(activation)
         self.bias_initializer = initializers.get(bias_initializer)
         self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.b = self.add_weight(shape=(self.units,), initializer=self.bias_initializer, name='bias')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.bias_add(inputs, self.b)
        if self.activation is not None:
             output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
 
    def get_config(self):
        config = {'units' : self.units, 'activation' : activations.serialize(self.activation), 'bias_initializer': initializers.serialize(self.bias_initializer)}
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NBatchLogger(Callback):
    def __init__(self, batch_count, validation_data):
        self.loss=[]
        self.accuracy=[]
        self.val_loss=[]
        self.val_accuracy=[]
        self.batch_count = batch_count
        self.validation_data = validation_data

    def on_batch_end(self, batch, logs={}):
        if logs.get('batch') % self.batch_count == 0:
            self.loss.append(logs.get('loss'))
            val_loss = self.model.evaluate(self, self.validation_data[0], self.validation_data[1], batch_size=FLAGS.batch_size)
            y_pred = self.model.predict(self.model.training_data[0])
            accuracy = accuracy_score(self.model.training_data[1], y_pred)
            self.accuracy.append(accuracy)
            val_y_pred = self.model.predict(self.validation_data[0])
            val_accuracy = accuracy_score(self.validation_data[1], val_y_pred)
            self.val_accuracy.append(val_accuracy)


if __name__ == '__main__':
    tf.app.run()
    network = ConvNet(5, FLAGS.conv_list, drop_rate=FLAGS.drop_rate)
    input = Input(shape=(2*FLAGS.input_window, 5))
    output = network(input)
    model = Model(input=input, output=output)
    plot_model(model, to_file='model.png')
