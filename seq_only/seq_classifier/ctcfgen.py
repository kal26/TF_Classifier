import sys
sys.path.insert(0,"/home/kal/CTCF/modules/")
sys.path.insert(0,"/home/kal/CTCF/mass_CTCF/modules/")
import ucscgenome
import random
from scipy.signal import correlate2d
import time
import os
from itertools import zip_longest, product, chain, repeat
import keras.backend as K
import pandas
import numpy as np
from intervaltree import IntervalTree
from tqdm import tqdm
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import viz_sequence
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '/home/kal/CTCF/wgEncodeRegTfbsClusteredV3.bed.gz', 'Path to the file of data')
tf.app.flags.DEFINE_string('gen_path', None, 'Path to the CTCFgen object')
tf.app.flags.DEFINE_integer('input_window', 256, 'Size of the prediciton window to pull from the geomes.')
tf.app.flags.DEFINE_integer('num_mutations', 50, 'Number of mutations to make via error-prone pcr like evaluation.')
tf.app.flags.DEFINE_integer('mutmask_size', 5, 'Size of the window to be zeroed out via an alanine scan like evaluation.')
tf.app.flags.DEFINE_integer('shift_padding', 80, 'How much padding to leave on the edges for shifting ChIP-seq peak data')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of examples in each batch.')

one_hot_encoder = np.fromstring('acgt', np.uint8)
def encode(seq):
    """Takes a np.uint8 array to one-hot."""
    if seq.dtype == np.uint8:
        return np.asarray([np.equal(char, one_hot_encoder) for char in seq])

def encode_string(seq):
    return encode(np.fromstring(seq, dtype=np.uint8))

def decode(seq):
    """Takes a one-hot to np.uint8."""
    
    return np.asarray([one_hot_encoder[i] for i in np.nonzero(seq)[1]])

def get_string(seq):
    """Takes a one-hot to string."""
    return decode(seq).tostring().decode('UTF-8')


def rc(seq):
    """Takes a one-hot seq to its reverse complement."""
    return seq[:, ::-1, ::-1]

def sequential_mutant_gen(seq):
    for idx in range(len(seq)):
        new_seq = np.copy(seq)
        if idx == 0:
            new_seq[idx:idx+1] = np.fromstring('x', np.uint8)
        elif idx == len(seq)-1:
            new_seq[idx-1:idx] = np.fromstring('x', np.uint8) 
        else:
            new_seq[idx-1:idx+1] = np.fromstring('x', np.uint8)
        yield new_seq

def double_mutant_gen(seq, n=1):
    """Generate every possible double mutant."""
    for mut1_seq in ngram_mutant_gen(seq, n=n):
        for mut2_seq in ngram_mutant_gen(mut1_seq, n=n):
            yield mut2_seq


def ngram_mutant_gen(seq, n=5, padding='valid'):
    """ Generate kgram mutants trying every possible amino acid combination of a length k-window in sequence.
    Arguments:
        seq -- One hot encoded seq to mutate.
    Keywords:
        n -- width of the motif to mutate.
        padding -- valid or same, similar to keras funcitonality.
    """
    done = False
    one_hots = encode(np.fromstring('acgt', np.uint8))
    if padding != 'valid':
        print('Alternative padding not yet supported')
    while not done:
        for idx in range(len(seq)):
            if n//2 <= idx <= len(seq) - n//2 - 1:
                first = idx-n//2
                last = idx+n//2+1
                #standard case
                ngrams = product(one_hots, repeat=n)
                for gram in ngrams:
                    new_seq = np.copy(seq)
                    new_seq[first:last] = np.asarray(gram)
                    yield new_seq
        done = True

def indel_mutant_gen(seq, n=1):
    done = False
    one_hots = encode(np.fromstring('acgt', np.uint8))
    while not done:
        for idx in range(len(seq)):
            ngrams = product(one_hots, repeat=n)
            for gram in ngrams:
                new_seq = np.insert(seq, idx, gram, axis=0)
                yield new_seq[:256]
        ngrams = product(one_hots, repeat=n)
        gram = next(ngrams)
        for start_idx in range(len(seq)-n):
            del_idx = range(start_idx, start_idx+n)
            new_seq = np.delete(seq, del_idx, axis=0)
            new_seq = np.append(new_seq, gram, axis=0)
            yield new_seq
        done = True

def blank_batch(seq, batch_size):
     batch = np.zeros((batch_size, seq.shape[0], seq.shape[1]), dtype=np.uint8)
     batch[0] = seq
     return batch

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def filled_batch(iterable, batch_size, fillvalue=np.zeros((FLAGS.input_window, 4))):
    groups = grouper(iterable, batch_size, fillvalue=fillvalue)
    while True:
        yield np.asarray(next(groups))

class CTCFGenerator(object):
    def __init__(self):
        # read CTCF peaks
        self.peaks = pandas.read_table(FLAGS.data_path, header=None)
        self.peaks.columns = 'chr start end name score expCount expNums expScores'.split()
        self.peaks = self.peaks[self.peaks.name.isin(['CTCF'])]
        print(self.peaks.head())
        widths = self.peaks.end - self.peaks.start
        # only one of the peaks is actually not 150 wide
        print('Getting genome.')
        self.genome = ucscgenome.Genome('/home/kal/.ucscgenome/hg19.2bit')
        # positives - center of each peak with 256(defalut) bp window
        # negatives - positives shifted left/right by 1024 (masked to not overlap any positives)
        # negatives - shuffled positives sequences

        # for testing - hold out chr8

        prediction_window = FLAGS.input_window
        self.half_window = prediction_window // 2
        self.negative_shift = prediction_window * 4

        self.num_training_examples = sum(self.peaks.chr != 'chr8')
        print('Number of training examples: ' + str(self.num_training_examples))

        # build intervaltrees for peaks to make sure our negatives (shifted positives)
        # are true negatives
        print('Building itrtree')
        self.peak_intervals = {chr: IntervalTree() for chr in self.peaks.chr.unique()}
        for chr in self.peaks.chr.unique():
            self.peak_intervals[chr][len(self.genome[chr]):len(self.genome[chr])+1] = 1
        for idx, row in tqdm(self.peaks.iterrows()):
            self.peak_intervals[row.chr][(row.start - self.half_window):(row.end + self.half_window)] = 1

    def pos_training_gen(self, mode='train', once=False):
        done = False
        if mode == 'test':
            indices = np.nonzero(self.peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 0] 
        elif mode =='val':
            indices = np.nonzero(self.peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 1]
        elif mode == 'all':
            indices = range(len(self.peaks))
        else:
            indices = np.nonzero(self.peaks.chr != 'chr8')[0]
        while not done:
            np.random.shuffle(indices)
            for idx in indices:
                row = self.peaks.iloc[idx]
                center = (row.start + row.end) // 2
                yield self.genome[row.chr][(center - self.half_window):(center + self.half_window)].lower()
            done = once

    def neg_training_gen_shifted(self, mode='train', once=False):
        done = False
        if mode == 'test':
            indices = np.nonzero(self.peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 0]
        elif mode =='val':
            indices = np.nonzero(self.peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 1]
        elif mode == 'all':
            indices = range(len(self.peaks))
        else:
            indices = np.nonzero(self.peaks.chr != 'chr8')[0]
        while not done:
            np.random.shuffle(indices)
            for idx in indices:
                row = self.peaks.iloc[idx]
                for center in (row.start - self.negative_shift, row.end + self.negative_shift):
                    if len(self.peak_intervals[row.chr][(center - self.half_window):(center + self.half_window)]) == 0:
                        yield self.genome[row.chr][(center - self.half_window):(center + self.half_window)].lower()
            done = once

    def neg_training_gen_scrambled(self, mode='train', once=False):
        done = False
        while not done:
            posgen = self.pos_training_gen(mode=mode, once=True)
            for p in posgen:
                yield ''.join(random.sample(p,len(p)))
            done = once

    def neg_training_gen(self, mode='train', once=False):
       for n1, n2 in zip_longest(self.neg_training_gen_shifted(mode=mode, once=once), self.neg_training_gen_scrambled(mode=mode, once=once)):
           if n1 != None:
               yield n1
           if n2 != None:
               yield n2


    def make_hdf5(self):
        # Get the time-tag for the data.
        timestr = time.strftime("%Y%m%d_%H%M%S")
        dt = "S" + str(FLAGS.input_window)
        print('Writing hdf5 File')
        hf5 = h5py.File(os.path.join('/home/kal/CTCF/output/', timestr + '_data.hdf5'), 'w')
        hf5.create_dataset('train_pos', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(self.pos_training_gen(mode='train', once=True), dt, count=-1)], chunks=True)
        print('Finished positive training')
        hf5.create_dataset('test_pos', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(self.pos_training_gen(mode='test', once=True), dt, count=-1)], chunks=True)
        print('Finished positive testing')
        hf5.create_dataset('val_pos', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(self.pos_training_gen(mode='val', once=True), dt, count=-1)], chunks=True)
        print('Finished positive validation')
        hf5.create_dataset('train_neg', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(self.neg_training_gen(mode='train', once=True), dt, count=-1)], chunks=True)
        print('Finished negative training')
        hf5.create_dataset('test_neg', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(self.neg_training_gen(mode='test', once=True), dt, count=-1)], chunks=True)
        print('Finished negative testing')
        hf5.create_dataset('val_neg', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(self.neg_training_gen(mode='val', once=True), dt, count=-1)], chunks=True)
        print('Finished negative validation')
        hf5.close()
        print('Wrote to file')

    def get_num_training_examples(self):
        return self.num_training_examples

    def mutant_gen(self, mode='train'):
        while True:
            posgen = self.pos_training_gen(mode=mode)
            for p in posgen:
                rand_indecies = [np.randint(len(p)) for i in FLAGS.num_mutations]
                for index in rand_indecies:
                    p[index] = np.random.choice(['a', 'c', 'g', 't'])
                yield p 

    def masked_gen(self, mode='train'):
        while True:
            posgen = self.pos_training_gen(mode=mode)
            for p in posgen:
                start_index = np.randint(len(p)-FLAGS.mutmask_size)
                p[start_index:start_index + FLAGS.mutmask_size] = 'x'
                yield p 

    def pairgen(self, mode='train', batch_size=FLAGS.batch_size):
        p =  self.pos_training_gen(mode=mode)
        n = self.neg_training_gen(mode=mode)
        labels = np.zeros(batch_size)
        labels[:(batch_size // 2)] = 1

        while True:
            pos_seqs = [encode(next(p)) for i in range(FLAGS.batch_size // 2)]
            neg_seqs = [encode(next(n)) for i in range(FLAGS.batch_size // 2)]
            yield np.asarray(pos_seqs + neg_seqs), labels

class CTCFGeneratorhdf5(CTCFGenerator):
    """Generator from a hd5f file."""
    def __init__(self, filepath):
        self.hf5 = h5py.File(filepath, 'r')
        self.train_pos = self.hf5['train_pos']
        self.test_pos = self.hf5['test_pos']
        self.val_pos = self.hf5['val_pos']
        self.train_neg = self.hf5['test_neg']
        self.test_neg = self.hf5['test_neg']
        self.val_neg = self.hf5['val_neg']
        self.num_training_examples= self.train_pos.shape[0]

    def pos_training_gen(self, mode='train', once=False):
        done = False
        if mode == 'test':
            indices = np.asarray(range(self.test_pos.shape[0]))
        elif mode =='val':
            indices = np.asarray(range(self.val_pos.shape[0]))
        else:
            indices = np.asarray(range(self.train_pos.shape[0]))
        while not done:
            np.random.shuffle(indices)
            for idx in indices:
                if mode == 'test':
                    yield self.test_pos[idx]
                elif mode =='val':
                    yield self.val_pos[idx]
                else:
                    yield self.train_pos[idx]
            done = once

    def pos_training_gen_shifted(self, mode='train', once=False):
        done = False
        if mode == 'test':
            indices = np.asarray(range(self.test_pos.shape[0]))
        elif mode =='val':
            indices = np.asarray(range(self.val_pos.shape[0]))
        else:
            indices = np.asarray(range(self.train_pos.shape[0]))
        while not done:
            mixes = np.asarray([[random.randint(FLAGS.shift_padding, FLAGS.input_window-FLAGS.shift_padding), index] for index in np.random.shuffle(indices)]*10)
            for idx, shf in mixes:
                if mode == 'test':
                    yield np.roll(self.test_pos[idx], shf, axis=0)
                elif mode =='val':
                    yield np.roll(self.val_pos[idx], shf, axis=0)
                else:
                    yield np.roll(self.train_pos[idx], shf, axis=0)
            done = once

    def neg_training_gen(self, mode='train', once=False):
        done = False
        if mode == 'test':
            indices = np.asarray(range(self.test_neg.shape[0]))
        elif mode =='val':
            indices = np.asarray(range(self.val_neg.shape[0]))
        else:
            indices = np.asarray(range(self.train_neg.shape[0]))
        while not done:
            np.random.shuffle(indices)
            for idx in indices:
                if mode == 'test':
                    yield self.test_neg[idx]
                elif mode =='val':
                    yield self.val_neg[idx]
                else:
                    yield self.train_neg[idx]
            done = once
    
    def neg_training_gen_shifted(self, mode='train', once=False):
        print('ERROR: method not well defined for hdf5 generators')
        g = self.neg_training_gen(mode=mode, once=once)
        while True: 
             yield next(g)

    def neg_training_gen_scrambled(self, mode='train', once=False):
        print('ERROR: method not well defined for hdf5 generators')
        g = self.neg_training_gen(mode=mode, once=once)
        while True:
             yield next(g)


def act_mutagenisis(seq, model, batch_size=32):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    get_act = K.function([model.input, K.learning_phase()], [layer_dict['bias'].output])

    # do a mutant scan
    mutant_window=1

    # get a mutant batch generator
    mutant_gen = ngram_mutant_gen(seq, n=mutant_window)
    g = filled_batch(mutant_gen, batch_size)
    
    # base importances as large-step gradients
    # score with base there - average of scores without base there
    mutant_preds = list()
    for batch in g:
        mutant_preds.append(get_act([batch, 0])[0])

    return np.asarray(mutant_preds).reshape((-1, 4))  

def get_importance(seq, model, start=None, end=None, plot=False, viz=1):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    get_act = K.function([model.input, K.learning_phase()], [layer_dict['bias'].output])

    pred = model.predict(np.asarray([seq]*32))[0][0]
    score = get_act([[seq]*32, 0])[0][0][0]
    print('Prediction ' + str(pred))

    mutant_preds = act_mutagenisis(seq, model)
      
    diffs = mutant_preds - score
   
    all_diffs = list()
    for base_seq, base_preds in zip(seq, mutant_preds):
        this_base = list()
        for idx in range(4):
            this_base.append(base_preds[idx] - np.average(base_preds))
        all_diffs.append(list(this_base))

    all_diffs=np.asarray(all_diffs)

    score_diff = list()
    for base_seq, base_preds in zip(seq, mutant_preds):
        idx = np.where(base_seq)
        score_diff.append(base_preds[idx] - np.average(base_preds))
      
    score_diff = (seq * score_diff)
    
    if plot:
        # plot the gain-loss curve 
        plt.figure(figsize=(30,2.5))

        plt.plot(np.amax(diffs, axis=1)[start:end])
        plt.plot(np.amin(diffs, axis=1)[start:end])
        plt.title('Prediciton Difference for a Mutagenisis Scan')
        plt.ylabel('importance (difference)')
        plt.xlabel('nucleotide')
        plt.show()
    
    if viz > 1:
        print('Prediciton Difference')
        viz_sequence.plot_weights(all_diffs[start:end])

    if viz > 0:
        print('Masked average prediciton difference')
        viz_sequence.plot_weights(score_diff[start:end])

    if viz > 2:
        print('Softmax average prediction difference')
        viz_sequence.plot_icweights(softmax([all_diffs[start:end]])[0])
    
    return all_diffs

# process the memes
def process_meme(meme_path):
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

    
    psuedocount=0.05
    # get the transformed memes
    transformed_memes = list()
    for meme in memes:
        meme = meme + psuedocount 
        norms = np.repeat(np.linalg.norm(meme, axis=1), 4).reshape((-1, 4))
        meme = np.log(meme/norms)
        min = np.amin(meme)
        meme = meme - min
        transformed_memes.append(meme)
    
    return transformed_memes

CTCF_memes = process_meme('/home/kal/data/CTCF.meme')
mystery_memes = process_meme('/home/kal/CTCF/mystery_seq/data/mystery_motif.meme')


def get_pwm(input_seqs, meme=[[-10]], position=None, meme_library=CTCF_memes, get_score=True, get_loc=False, get_everything=False):  
    # get position and meme if not specified
    if meme_library == 'mystery_memes':
        meme_library = mystery_memes
    if (meme[0][0] == -10 or position == None):
        get_loc = True
        for seq in input_seqs:
            best_score = -np.inf
            for test_meme, i in zip(meme_library, range(len(meme_library))):
                correlations = correlate2d(seq, test_meme, mode='valid')
                if np.max(correlations) > best_score:
                    best_score = np.max(correlations)
                    best_location = np.argmax(correlations)
                    best_filter = test_meme
                    meme_index = i
        position = best_location
        meme = best_filter
    else:
        get_loc = False
        meme_index = None
    
    # get the pwms
    output_scores = list()
    seq = input_seqs[0]
    pwm = np.zeros(input_seqs[0].shape)
    pwm[position:position+meme.shape[0]] = meme*seq[position:position+meme.shape[0]]
    for seq in input_seqs:
        output_scores.append(np.copy(pwm))  

    if get_everything:
        corr = correlate2d(seq[position:position+meme.shape[0]], meme, mode='valid')
        score = np.max(corr)
        return score, pwm, meme, meme_index, position
    
    if get_score:
        corr = correlate2d(seq[position:position+meme.shape[0]], meme, mode='valid')
        return pwm, np.max(corr)
            
    if get_loc:
        return np.asarray(output_scores), meme, position
    else:
        return np.asarray(output_scores)



def main(argv=None):
    gen = CTCFGenerator()
    gen.make_hdf5()



if __name__ == '__main__':
    tf.app.run()
import gc; gc.collect()
