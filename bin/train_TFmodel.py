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

def filled_batch(iterable, batch_size=32, fillvalue=np.zeros((256, 4))):
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
    logits: [batch_size, n_class] unnormalized log-probs
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

