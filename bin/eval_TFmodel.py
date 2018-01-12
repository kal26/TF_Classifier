import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import tf_memory_limit
import numpy as np
import train_TFmodel
import sequence
from keras import backend as K

#helper functions

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm."""
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def rejection(seq_a, seq_b):
    """utiliy function to compute rejection of a from b"""
    out = list()
    for a, b in zip(seq_a, seq_b):
        if np.linalg.norm(b) == 0:
            out.append(a)
        else:
            out.append(a - ((np.dot(a, b) / (np.linalg.norm(b)**2)) * b))
    return np.asarray(out)

def softmax(y):
    """Take softmax of the given sequence or batch at the lowest array level."""
    if y.ndim == 1:
        return _softmax(y)
    else:
        return np.asarray([softmax(small) for small in y])
 

def _softmax(x):
    """Take softmax of an array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class TFmodel(object):
    """Transcription factor classification keras model."""

    def __init__(self, model_path, output_path=None):
        """Create a new model object.

        Arguments:
            model_path -- path to a trained model with weights and arcitecture.
        Keywords:
            output_path -- directory to write out requested files to. 
        """
        self.model = load_model(model_path, custom_objects={'Bias':Bias})
        self.out_path = output_path
        self.layer_dict = dict([(layer.name, layer) for layer in model.layers]) 
        self.get_act = K.function([model.input, K.learning_phase()], [layer_dict['bias'].output])

    def __str__(self):
        """Printable version of the model."""
    
    def __repr__(self):
        """Representaiton of the model."""

    def get_activation(self, generator, distribution_repeats=32):
        """Return predctions for the given sequence or generator.
          
        Arguments:
            generator -- A sequence or generator of correct length sequences.
        Keywords:
            distribution_repeats -- Number of sequences to sample from a distribution.
        Output:
            outact -- List or value for activation.
        """
        activation = list()
        #assume iterable
        try:
            test = next(generator)
            if isinstance(test, sequence.SeqDist):
                dist = True
                #distribution
                #converts distributions to discrete sequences and averages
                def gen():
                    for i in range(distribution_repeats):
                        yield test.discrete_seq()
                    for elem in generator:
                        for i in range(distribution_repeats):
                            yield elem.discrete_seq()
                g = gen()
                batch_gen = train_TFmodel.filled_batch(g)
            else:
                dist = False
                def stackgen():
                    yield test
                    for elem in generator:
                        yield elem
                g = stackgen()
                batch_gen = train_TFmodel.filled_batch(g)
            # get the numbers
            for batch in batch_gen:
                activation.append(self.get_act([batch, 0]))
            activation = np.asarray(activation).flatten()
            if dist:
                #average every so often
                ids = np.arange(len(activation))//distribution_repeats
                outact = np.bincount(ids, activations)/np.bincount(ids) 
                return outact
            return activaton
        except TypeError:
            #acutally not iterable
            return self.get_act([train_TFmodel.blank_batch(generator), 0])[0][0] 
            
    def dream(seq, num_iterations=20):
        """Dream a sequence for the given number of steps."""
