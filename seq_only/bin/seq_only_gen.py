import os
import sys
sys.path.append('/home/kal/TF_models/bin/')
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import tf_memory_limit
#import general use packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from intervaltree import IntervalTree
from itertools import zip_longest
import h5py
import random
import ucscgenome
from tqdm import tqdm
#import keras related packages
from keras import backend as K
from keras.models import load_model, Model, Input
import tensorflow as tf
#import custom packages
import helper
import viz_sequence
import train_TFmodel
import sequence

def create_from_bed(bed_path, out_path, columns=None):
    """Create an hdf5 file from a bed file."""
    # read CTCF peaks
    full = pd.read_table(bed_path, header=None)
    if columns == None:
        columns = 'chr start end name score expCount expNums expScores'
    full.columns = columns.split()
    peaks = full[full.name.isin(['CTCF'])]
    genome = ucscgenome.Genome('/home/kal/.ucscgenome/hg19.2bit')
    # positives - center of each peak with 256(defalut) bp window
    # negatives - positives shifted left/right by 1024 (masked to not overlap any positives)
    # negatives - shuffled positives sequences
    # for testing - hold out chr8
    prediction_window = 256
    half_window = prediction_window // 2
    negative_shift = prediction_window * 4
    num_training_examples = sum(peaks.chr != 'chr8')
    print('Number of training examples: ' + str(num_training_examples))
    # build intervaltrees for peaks to make sure our negatives (shifted positives)
    # are true negatives
    print('Building itrtree')
    peak_intervals = {chr: IntervalTree() for chr in peaks.chr.unique()}
    for chr in peaks.chr.unique():
        peak_intervals[chr][len(genome[chr]):len(genome[chr])+1] = 1
    for idx, row in tqdm(peaks.iterrows()):
        peak_intervals[row.chr][(row.start - half_window):(row.end + half_window)] = 1

    def pos_gen(mode='train'):
        if mode == 'test':
            indices = np.nonzero(peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 0]
        elif mode =='val':
            indices = np.nonzero(peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 1]
        else:
            indices = np.nonzero(peaks.chr != 'chr8')[0]
        for idx in indices:
            row = peaks.iloc[idx]
            center = (row.start + row.end) // 2
            yield genome[row.chr][(center - half_window):(center + half_window)].lower()
 
    def pos_gen_strength(mode='train'):
        if mode == 'test':
            indices = np.nonzero(peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 0]
        elif mode =='val':
            indices = np.nonzero(peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 1]
        else:
            indices = np.nonzero(peaks.chr != 'chr8')[0]
        for idx in indices:
            row = peaks.iloc[idx]
            yield row.score

    def neg_gen_shifted(mode='train'):
        if mode == 'test':
            indices = np.nonzero(peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 0]
        elif mode =='val':
            indices = np.nonzero(peaks.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 1]
        else:
            indices = np.nonzero(peaks.chr != 'chr8')[0]
        for idx in indices:
            row = peaks.iloc[idx]
            for center in (row.start - negative_shift, row.end + negative_shift):
                if len(peak_intervals[row.chr][(center - half_window):(center + half_window)]) == 0:
                    yield genome[row.chr][(center - half_window):(center + half_window)].lower()

    def neg_gen_scrambled(mode='train'):
        posgen = pos_gen(mode=mode)
        for p in posgen:
            yield ''.join(random.sample(p,len(p)))

    def neg_gen(mode='train'):
        for n1, n2 in zip_longest(neg_gen_shifted(mode=mode), neg_gen_scrambled(mode=mode)):
            if n1 != None:
                yield n1
            if n2 != None:
                yield n2

    # Write out a file for the data.
    dt = "S" + str(256)
    print('Writing hdf5 File')
    hf5 = h5py.File(out_path, 'w')
    hf5.create_dataset('train_pos_str', data=np.fromiter(pos_gen_strength(mode='train'), np.uint32, count=-1), chunks=True)
    hf5.create_dataset('train_pos', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(pos_gen(mode='train'), dt, count=-1)], chunks=True)
    print('Finished positive training')
    hf5.create_dataset('test_pos_str', data=np.fromiter(pos_gen_strength(mode='test'), np.uint32, count=-1), chunks=True)
    hf5.create_dataset('test_pos', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(pos_gen(mode='test'), dt, count=-1)], chunks=True)
    print('Finished positive testing')
    hf5.create_dataset('val_pos_str', data=np.fromiter(pos_gen_strength(mode='val'), np.uint32, count=-1), chunks=True)
    hf5.create_dataset('val_pos', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(pos_gen(mode='val'), dt, count=-1)], chunks=True)
    print('Finished positive validation')
    hf5.create_dataset('train_neg', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(neg_gen(mode='train'), dt, count=-1)], chunks=True)
    print('Finished negative training')
    hf5.create_dataset('test_neg', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(neg_gen(mode='test'), dt, count=-1)], chunks=True)
    print('Finished negative testing')
    hf5.create_dataset('val_neg', data=[np.fromstring(seq, np.uint8) for seq in np.fromiter(neg_gen(mode='val'), dt, count=-1)], chunks=True)
    print('Finished negative validation')
    hf5.close()
    print('Wrote to file')
 
class CTCFGenerator(object):
    def __init__(self, file_path):
        """Create a generator from an hdf5 file."""
        self.hf5 = h5py.File(file_path, 'r')
        self.train_pos = self.hf5['train_pos']
        self.train_pos_str = self.hf5['train_pos_str']
        self.test_pos = self.hf5['test_pos']
        self.test_pos_str = self.hf5['test_pos_str']
        self.val_pos = self.hf5['val_pos']
        self.val_pos_str = self.hf5['val_pos_str']
        self.train_neg = self.hf5['test_neg']
        self.test_neg = self.hf5['test_neg']
        self.val_neg = self.hf5['val_neg']
        self.num_training_examples = self.train_pos.shape[0]
        
    def pos_gen(self, mode='train', once=False):
        """Generate a positive seqeunce sample."""
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
                    yield self.test_pos[idx], self.test_pos_str[idx]
                elif mode =='val':
                    yield self.val_pos[idx], self.val_pos_str[idx]
                else:
                    yield self.train_pos[idx], self.train_pos_str[idx]
            done = once

    def neg_gen(self, mode='train', once=False):
        """Generate a negative sequence sample."""
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

    def pair_gen(self, mode='train', once=False, batch_size=32, strengths=False):
        """Generate batched of paired samples."""
        p = self.pos_gen(mode=mode, once=once)
        n = self.neg_gen(mode=mode, once=once)
        if not strengths:
            labels = np.zeros(batch_size)
            labels[:(batch_size // 2)] = 1
        while True:
            pos_seqs = list()
            neg_seqs = list()
            if strengths:
                scores = list()
            for i in range(batch_size // 2):
                pos_seq, score = next(p)
                neg_seq = next(n)
                pos_seqs.append(sequence.encode_to_onehot(pos_seq)) 
                neg_seqs.append(sequence.encode_to_onehot(neg_seq))
                if strengths:
                    scores.append(score)
            if strengths:
                labels = scores + np.zeros(batch_size // 2)
            yield np.asarray(pos_seqs + neg_seqs), labels

    def get_num_training_examples(self):
        return self.num_training_examples
