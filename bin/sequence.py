import sys
import numpy as np
from itertools import zip_longest, product, chain, repeat
import viz_sequence
import train_TFmodel

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

class Sequence(object):
    """ Encoding and variations on a sequence."""
    
    def __init__(self, nucleotides):
        """ Create a sequence object.
        
        Arguments:
            nucleotides -- Sequence in string, np.uint8, or one-hot form.
        """
        self.onehot = encode_to_onehot(nucleotides)   
    def __string__(self):
        """ACTG representation of the sequence."""
        return encode_to_string(self.onehot)

    def __repr__(self):
        """Information about the sequence."""
        return 'Sequence() length ' + str(self.onehot.shape[0])
 
    def logo(self, start=None, end=None):
        """Plot a sequence logo from start to end."""
        viz_sequence.plot_weights(self.onehot[start:end])

    def sequential_mutant_gen(self):
        """Generate sequences with a blank mutation."""
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

    def act_mutagenisis(self, TFmodel):
        """Prediction value for a single base mutation in each position.
         
        Arguments:
            TFmodel -- the keras model used to make predictions.
        Returns:
            mutant_preds -- predictions for each base in each position.
        """
        #get a mutant batch generator
        mutant_gen = self.ngram_mutant_generator()
        #approximate base importance as a large step 'gradient'
        mutant_preds=list()
        for batch in train_TFmodel.filled_batch(mutant_gen):
            mutant_preds.append(TFmodel.get_act([batch, 0]))
        #get the correct shape
        mutant_preds = np.asarray(mutant_preds).reshape((-1, 4))
        return mutant_preds

    def importance(self, TFmodel, viz=False, start=None, end=None, plot=False):
        """Generate the gradient based importance of a sequence according to a given model.
        
        Arguments:
             TFmodel -- the keras model to run the seqeunce through.
             viz -- sequence logo of importance?
             start -- plot only past this nucleotide.
             end -- plot only to this nucleotide.
             plot -- generate a gain-loss plot?
        Returns:
             diffs -- difference at each position to score.
             average_diffs -- base by base importance value. 
             masked_diffs -- importance for bases in origonal sequence.
        """
         score = TFmodel.get_act(train_TFmodel.blank_batch(self.onehot), 0])[0][0][0]
         mutant_preds = self.act_mutagenisis(TFmodel)
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
        return diffs, average_diffs, masked diffs

class SeqDist(Sequence):
    """A sequence, but as a probability distribution."""

    def __init__(self, distribution):
        """Create a new sequence distribution object."""
        if isinstance(seq, np.ndarray) and not (seq.dtype == np.uint8):
            # right type!
            self.distribution = distribution 
            self.onehot = np.amax(distribution, axis=1)
        else:
            raise TypeError('Sequence is not an accepted type')
 
    def __repr__(self):
        """Information about the sequence."""
        return 'DistributionSequence() length ' + str(self.onehot.shape[0])

    def logo(self, start=None, end=None):
         """Plot a sequence logo from start to end."""
        viz_sequence.plot_weights(self.distribution[start:end])        

    
# process the memes
def process_meme(meme_path, transform=True):
    """Extract a meme distribution and process.
   
    Arguments:
        meme_path -- file path to a .meme file.
    Keywords:
        transform -- apply normalization and a log transform?
    Returns:
        meme_list -- List of DistSeq() meme and reverse complements.
    """
    with open(meme_path, 'r') as infile:
        meme_length = -1
        memes = list()
        for line in infile.readlines():
            if 'letter-probability matrix' in line:
                meme_length = int(line.split()[5])
                this_meme_lines = list()
            elif meme_length > 0:
                this_meme_lines.append([float(item.strip()) for item in line.split()])
                meme_length = meme_length - 1
            elif meme_length == 0:
                this_meme = np.asarray(this_meme_lines)
                memes.append(this_meme)
                meme_length = -1
        if meme_length == 0:
            this_meme = np.asarray(this_meme_lines)
            memes.append(this_meme)
            meme_length = -1
    # add rcs of memes
    rcs = list()
    for meme in memes:
        rcs.append(meme[::-1, ::-1])
    memes = memes + rcs
    #transofrm the memes
    if transform:
        psuedocount=0.05
        transformed_memes = list()
        for meme in memes:
            meme = meme + psuedocount
            norms = np.repeat(np.linalg.norm(meme, axis=1), 4).reshape((-1, 4))
            meme = np.log(meme/norms)
            min = np.amin(meme)
            meme = meme - min
            transformed_memes.append(meme)
    else:
        transformed_memes = memes
    #make distribution objects
    meme_list = [SeqDist(distribution) for distribution in transformed_memes] 
    return mem_list

CTCF_memes = process_meme('/home/kal/TF_models/data/memes/CTCF.meme')
mystery_memes = process_meme('/home/kal/TF_models/data/memes/mystery_motif.meme')
