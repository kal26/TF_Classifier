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
            count = 1
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
                        count = count + 1
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
            return activaton[:count]
        except TypeError:
            #acutally not iterable
            return self.get_act([train_TFmodel.blank_batch(generator), 0])[0][0] 
            
    def dream(seq, iterate_op=None, num_iterations=20):
        """Dream a sequence for the given number of steps.
         
        Arguments:
            seq -- SeqDist object to iterate over.
        Keywords:
            iterate_op -- operation to get the update step, default is maximize output. 
        Returns:
            dream_seq -- result of the iterations. 
        """
        # get an iterate operation
        if iterate_op = None:
            iterate_op = self.build_iterate()
        # dreaming won't work off of true zero probabilities - if these exist we must add a pseudocount
        if np.count_nonzero(seq.dist) != np.size:
            dream_seq = SeqDist(softmax(seq.dist + .000001))
        else:
            dream_seq = SeqDist(seq.dist)
        # find a good step size 
        update_grads = iterate_op([next(trainTF_model.filled_batch(dream_seq.discrete_gen()))])
        step = 8/np.amax(update_grads)
        # apply the updates
        for i in range(num_iterations):
            batch_gen = trainTF_model.filled_batch(dream_seq.discrete_gen())
            update_grads = iterate_op([next(batch_gen)])
            # we apply the update in log space so a zero update won't change anything
            update = np.average(update_grads, axis=0)*dream_seq.dist*step
            dream_seq = SeqDist(softmax(np.log(dream_seq.dist + update))) 
        return dream_seq

    def build_iterate(self, layer_name='final_output', filter_index=0):
        """ Build a interation operation for use with dreaming method.
     
        Keywords:
            

    def localize(row, genome):
        """ Find the section of a bed file row giving maximum acitvation.

        Arguments:
            row -- bed file row.
            genome -- genome associated with the bed file. 
        Output:
            max_tile -- Sequence object for the maximum predicting 256 bp region.
            max_pred -- prediction value for the row. 
        """
        # break the sequence into overlapping tiles
        tile_seqs = list()
        num_tiles = int((row['end']-row['start']) / input_window) + ((row['end']-row['start']) % input_window > 0)
        for idx in range(num_tiles):
            if row['start'] + idx*input_window - input_window//2 > 0:
                seq = genome[row['chr']][row['start'] + idx*input_window - input_window//2:row['start'] + (idx+1)*input_window - input_window//2].lower()
                tile_seqs.append(seqeunce.encode_to_onehot(seq))
            else:
                buffered_seq = np.zeros((256,4))
                buffered_seq[:row['start'] + (idx+1)*input_window - input_window//2] = genome[row['chr']][0:row['start'] + (idx+1)*input_window - input_window//2]
                tile_seqs.append(seqeunce.encode_to_onehot(buffered_seq))
            seq = genome[row['chr']][row['start'] + idx*input_window:row['start'] + (idx+1)*input_window].lower()
            tile_seqs.append(seqeunce.encode_to_onehot(seq))
        #configure the tiled sequences
        tile_seqs= np.asarray(tile_seqs)
        tile_iter = iter(tile_seqs)
        # get a batch generator
        batches = train_TFmodel.filled_batch(tile_iter)
        # figure out where the max prediction is coming from
        preds = list()
        for batch in batches:
            preds.append(self.predict_on_batch(batch))
        preds = np.asarray(preds).reshape((-1))[:tile_seqs.shape[0]]
        # get a tile centered there
        max_pred = np.max(preds)
        max_tile = tile_seq[np.argmax(preds)]
        return Sequence(max_tile), max_pred
        



