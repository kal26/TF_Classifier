import tf_memory_limit
#import keras related packages
from keras.models import Model
from keras.layers import Conv1D, Dropout, Activation, add, Lambda, concatenate, multiply, SpatialDropout1D, Layer, Input
from keras.engine import InputSpec
from keras import activations
from keras import initializers
import keras.backend as K
import tensorflow as tf
#import general use packages
from itertools import zip_longest, product, chain, repeat
import numpy as np
#import custom packages
import sequence

#some batch forming methods
def blank_batch(seq, batch_size=32):
     """Make a batch blank but for the given sequence in position 0."""
     seq = sequence.encode_to_onehot(seq)
     batch = np.zeros((batch_size, seq.shape[0], seq.shape[1]), dtype=np.uint8)
     batch[0] = seq
     return batch

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def filled_batch(iterable, batch_size=32, fillvalue=np.asarray([False]*256*4).reshape(256,4)):
    """Make batches of the given size until running out of elements, then buffer."""
    groups = grouper(iterable, batch_size, fillvalue=fillvalue)
    while True:
        yield np.asarray(next(groups))

def random_seq():
    """return a random Sequence."""
    random = np.random.choice(np.fromstring('acgt', np.uint8), size=256)
    return sequence.Sequence(random)

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [sequence length, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

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


class BasicBlock(object):
    """"Basic convolution block with dropout and weight decay."""
    def __init__(self, filters, kernel_width=3, decay=0.9999, drop_rate=0.1, middle_drops=True):
        """Initialize a convolution block.
        
        Arguments:
            filters -- Number of neurons in the convolution.
        Keywords:
            kernal_width -- Width of convolutional window.
            decay -- Decay of weights after each training update.
            drop_rate -- Fraction of neurons to drop.
            middle_drops -- Drop after intermediate layers.
        """
        super(BasicBlock, self).__init__()
        self.conv = Conv1D(filters, kernel_width, padding='valid', activation='relu')#Add leaky relu?
        # weight decay
        self.conv.add_update([(w, K.in_train_phase(decay * w, w)) for w in self.conv.trainable_weights])

        if drop_rate > 0 and middle_drops:
            self.dropout = SpatialDropout1D(drop_rate)
        else:
            self.dropout = None

    def __call__(self, x):
        """Run the convolution"""
        out = self.conv(x) #Add leaky relu?
        if self.dropout != None:
            out = self.dropout(out)
        return out

class BasicConv(object):
    """Object for a series of convolutions."""
    def __init__(self, num_inputs, conv_list, num_outputs=1, drop_rate=0.0, final_activation='relu', middle_drops=True, last_drop=False):
        """Initialize a BasicConv object.
     
            Args:
                num_inputs -- Number of excpected inputs. 
                conv_list -- List of tuples of (num_hidden, kernel_size), one for each convlution in the net.
            Keywords:
                num_outputs -- Number of final outputs required.
                drop_rate -- Percentage of features to drop on each level.
                middle_drops -- Drop after intermediate layers.
                last_drop -- Drop after final layer.
        """
        super(BasicConv, self).__init__()

        self.drop_rate = drop_rate
        self.middle_drops = middle_drops
        self.last_drop = last_drop

        blockgen = BasicBlock
        print('Convolutions used: ' + str(conv_list) + ' [neurons, filter]')
        self.conv_in = Conv1D(conv_list[0][0], conv_list[0][1], padding='valid', activation='relu', input_shape=(num_inputs, 4))

        self.hidden_layers = []
        for params in conv_list[1:]:
            self.hidden_layers.append(blockgen(params[0], kernel_width=params[1], drop_rate=drop_rate))

        self.conv_out = Conv1D(num_outputs, 1, padding='valid', activation=final_activation, name='final_conv')

    def __call__(self, x):
        x = self.conv_in(x)
        self.hidden_outputs = []
        self.hidden_outputs.append(x)
        if self.drop_rate > 0 and self.middle_drops:
            x = SpatialDropout1D(self.drop_rate)(x)
        for h in self.hidden_layers:
            x = h(x)
            self.hidden_outputs.append(x)
        x = self.conv_out(x)
        if self.drop_rate > 0 and self.last_drop:
            x = SpatialDropout1D(self.drop_rate)(x)
        return x

