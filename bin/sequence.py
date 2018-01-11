import sys
import numpy as np
from itertools import zip_longest, product, chain, repeat

one_hot_encoder = np.fromstring('acgt', np.uint8)


# some sequence encoding methods
def encode_to_string(seq):
    "return a string from string, uint8, or onehot"
    if isinstance(seq, str):
        return seq
    else if isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return seq.tostring().decode('UTF-8')
        else:
        #onehot array
            return np.asarray([one_hot_encoder[i] for i in np.nonzero(seq)[1]]).tostring().decode('UTF-8')
    else:
        raise TypeError('Sequence is not an accepted type')

def encode_to_uint8(seq):
    "return a uint8 from string, uint8, or onehot"
    if isinstance(seq, str):
        return np.fromstring(seq, dtype=np.uint8)
    else if isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return seq
        else:
        #onehot array
            return np.asarray([one_hot_encoder[i] for i in np.nonzero(seq)[1]])
    else:
        raise TypeError('Sequence is not an accepted type')

def encode_to_onehot(seq):
    "return a onehot from string, uint8, or onehot"
    if isinstance(seq, str):
        return np.asarray([np.equal(char, one_hot_encoder) for char in np.fromstring(seq, dtype=np.uint8)])
    else if isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return np.asarray([np.equal(char, one_hot_encoder) for char in seq])
        else:
        #onehot array
            return seq
    else:
        raise TypeError('Sequence is not an accepted type')

def rc(seq):
    """Takes a seq to its reverse complement of same type."""
    onehot = encode_to_onehot(seq)
    rc = onehot[:, ::-1, ::-1]
    if isinstance(seq, str):
        return encode_to_string(rc)
    else if isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return encode_to_uint8(rc)
        else:
        #onehot array
            return rc
    else:
        raise TypeError('Sequence is not an accepted type')

#some batch forming methods
def blank_batch(seq, batch_size=32):
     """Make a batch blank but for the given sequence in position 0."""
     seq = encode_to_onehot(seq)
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

class Sequence(object):
    """ Encoding and variations on a sequence."""
    
    def __init__(self, nucleotides):
        """ Create a sequence object.
        
        Arguments:
            nucleotides -- Sequence in string, np.uint8, or one-hot form.
        """
        self.onehot = encode_to_onehot(nucleotides)   
    def __string__(self):
        "ACTG representation of the sequence."
        return encode_to_string(self.onehot)
    def __repr__(self):
        " information about the sequence."
        return 'Sequence() length ' + str(self.onehot.shape[0])

    def sequential_mutant_gen(self):
    """generate sequences with a blank mutation."""
        for idx in range(self.onehot.shape[0]):
            new_seq = np.copy(self.onehot)
            if idx == 0:
                new_seq[idx:idx+1] = np.fromstring('x', np.uint8)
            elif idx == len(seq)-1:
                new_seq[idx-1:idx] = np.fromstring('x', np.uint8)
            else:
                new_seq[idx-1:idx+1] = np.fromstring('x', np.uint8)
            yield new_seq

    def ngram_mutant_gen(self, n=5, padding='valid'):
        """ Generate ngram mutants trying every possible amino acid combination of a length n in a sequence.

        Keywords:
            n -- width of the motif to mutate.
            padding -- valid or same, similar to keras funcitonality.
        """
        done = False
        if padding != 'valid':
            print('Alternative padding not yet supported')
        while not done:
            for idx in range(len(self.onehot)):
                if n//2 <= idx <= len(self.onehot) - n//2 - 1:
                    first = idx-n//2
                    last = idx+n//2+1
                    #standard case
                    ngrams = product(one_hots, repeat=n)
                    for gram in ngrams:
                        new_seq = np.copy(self.onehot)
                        new_seq[first:last] = np.asarray(gram)
                        yield new_seq
            done = True

    def double_mutant_gen(self, n=1):
        """Generate every possible double mutant."""
        for mut1_seq in self.ngram_mutant_gen(n=n):
            for mut2_seq in Sequence(mut1_seq).ngram_mutant_gen(n=n):
                yield mut2_seq

    def insertion_mutant_gen(self, n=1):
    """Generate every n length insertion."""
        done = False
        while not done:
            for idx in range(len(sel.onehot)):
                ngrams = product(one_hots, repeat=n)
                for gram in ngrams:
                    new_seq = np.insert(self.onehot, idx, gram, axis=0)
                    yield new_seq[:256]
            done = True

    def deletion_mutant_gen(self, n=1):
    """Generate every deletion mutant."""
        done = False
        while not done:
            ngrams = product(one_hots, repeat=n)
            gram = next(ngrams)
            for start_idx in range(len(self.onehot)-n):
                del_idx = range(start_idx, start_idx+n)
                new_seq = np.delete(self.onehot, del_idx, axis=0)
                new_seq = np.append(new_seq, gram, axis=0)
                yield new_seq
            done = True

    def importance(self, model, viz=False, start=None, end=None, plot=False):
        """ generate the gradient based importance of a sequence according to a given model.
        
        Arguments:
             model -- the keras model to run the seqeunce through.
             viz -- sequence logo of importance?
             start -- plot only past this nucleotide.
             end -- plot only to this nucleotide.
             plot -- generate a gain-loss plot?
        
        Returns:
             diffs -- difference at each position to score.
             average_diffs -- base by base importance value. 
             masked_diffs -- importance for bases in origonal sequence.
        """
         score = model.get_act(blank_batch(self.onehot), 0])[0][0][0]
         mutant_preds = self.act_mutagenisis(model)
         diffs = mutant_preds - score

        # we want the difference for each nucleotide at a position minus the average difference at that position
        average_diffs = list()
        for base_seq, base_preds in zip(self.onehot, mutant_preds):
            this_base = list()
            for idx in range(4):
                this_base.append(base_preds[idx] - np.average(base_preds))
            average_diffs.append(list(this_base))

        average_diffs = np.asarray(average_diffs)

        # masked by the actual base
        masked_diffs = (self.onehot * average_diffs)

        if plot:
            # plot the gain-loss curve 
            plt.figure(figsize=(30,2.5))
            plt.plot(np.amax(diffs, axis=1)[start:end])
            plt.plot(np.amin(diffs, axis=1)[start:end])
            plt.title('Prediciton Difference for a Mutagenisis Scan')
            plt.ylabel('importance (difference)')
            plt.xlabel('nucleotide')
            plt.show()

        if viz:
            print('Prediciton Difference')
            viz_sequence.plot_weights(average_diffs[start:end])
            print('Masked average prediciton difference')
            viz_sequence.plot_weights(masked_diffs[start:end])
            print('Information Content of Softmax average prediction difference')
            viz_sequence.plot_icweights(softmax(average_diffs[start:end])[0])
            print('Information Content of Softmax prediction difference')
            viz_sequence.plot_icweights(softmax(diffs[start:end])

    return all_diffs


