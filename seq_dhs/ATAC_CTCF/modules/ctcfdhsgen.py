#!/bin/python
import sys
sys.path.insert(0,"/home/kal/CTCF/modules/")
sys.path.insert(0,"/home/kal/CTCF/mass_CTCF/modules/")
import tf_memory_limit
import ucscgenome
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
import h5py
import pysam
from itertools import zip_longest
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of samples per batch.')
tf.app.flags.DEFINE_integer('input_window', 256, 'Size of the genome to grab.')

one_hot_encoder = np.fromstring('acgt', np.uint8)
def encode(seq):
    """Takes a np.uint8 array to one-hot."""
    if seq.dtype == np.uint8:
        return np.asarray([np.equal(char, one_hot_encoder) for char in seq])

def encode_string(seq):
    return encode(np.fromstring(seq, dtype=np.uint8))

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def filled_batch(iterable, batch_size, input_window, fillvalue=None):
    if fillvalue == None:
        fillvalue = np.zeros((input_window, 4))
    groups = grouper(iterable, batch_size, fillvalue=fillvalue)
    while True:
        yield np.asarray(next(groups))

class CTCFDHSGen(object):
    def __init__(self, regions='/home/kal/data/chipseq_ctcf_peaks.bed', 
cellline_CTCFs=None, cellline_DHSs=None, keys=None, input_window=FLAGS.input_window, batch_size=FLAGS.batch_size, column_names='chr start end . . . . .'):
        """ Create a generator for CTCFDHS data.

        Arguments:
            regions -- bed file with training points to use.
            cellline_CTCF -- list of CTCF peak files, one for each cell line to use.
            cellline_DHS -- list of files with DHS reads, one for each cell line to use. matching order to cellline_CTCF.
        """
        # read in the regions
        self.bed = pd.read_table(regions, header=None)
        self.bed.columns = column_names.split() 

        # set up a reference genome.
        self.genome = ucscgenome.Genome('/home/kal/.ucscgenome/hg19.2bit')

        # define some values
        self.batch_size = batch_size
        self.input_window = input_window
        self.half_window = input_window // 2

        # set up cell line stuff.
        if cellline_CTCFs == None:
            cellline_CTCFs = ['/home/kal/CTCF/ATAC_CTCF/data/K562/wgEncodeBroadHistoneK562CtcfStdAlnRep1.bam', 
            '/home/kal/CTCF/ATAC_CTCF/data/HUVEC/wgEncodeBroadHistoneHuvecCtcfStdAlnRep1.bam', 
            '/home/kal/CTCF/ATAC_CTCF/data/HCT116/CTCF_untreated.reheader.bam']
        if cellline_DHSs == None:
            cellline_DHSs = ['/home/kal/CTCF/ATAC_CTCF/data/K562/CombinedScreens.unique_alignment.bam', 
            '/home/kal/CTCF/ATAC_CTCF/data/HUVEC/wgEncodeUwDnaseHuvecAlnRep2.bam', 
            '/home/kal/CTCF/ATAC_CTCF/data/HCT116/wgEncodeUwDnaseHct116AlnRep1.bam']
        self.CTCF_dict = dict()
        self.DHS_dict = dict()
        self.cov_dict = dict()
        if keys == None:
            keys = range(len(cellline_CTCFs))
        self.keys = keys
        for key, CTCF, DHS in zip(self.keys, cellline_CTCFs, cellline_DHSs):
            self.CTCF_dict[key] = pysam.AlignmentFile(CTCF,'rb')
            self.DHS_dict[key] = pysam.AlignmentFile(DHS,'rb')
            total_bases = sum([len(self.genome[chr]) for chr in self.DHS_dict[key].references])
            covered_bases = self.DHS_dict[key].count()
            self.cov_dict[key] = covered_bases/total_bases


    def get_label(self, chr, start, end, key):
        """ Find if the read count of the region chr:start-end"""
        num_reads = len([r for r in self.CTCF_dict[key].fetch('chr1', start, end)])
        return num_reads

    def get_coverage(self, chr, start, end, key, CTCF=False):
        """ Get the per-base coverage of the region."""
        dhs = [0]*(end-start)
        if CTCF:
            my_dict = self.CTCF_dict
        else:
            my_dict = self.DHS_dict 
        # some of the bam files don't have all of the chromosomes, which throws an error
        try:
            for c in my_dict[key].pileup(chr, start, end):
                if c.pos < end:
                    dhs[c.pos-start] = c.n 
        except ValueError:
            print('Silencing ValueError for indexing to ' + str(chr) + ' for ' + str(key))
        # normalize for the cell type
        dhs = [value/self.cov_dict[key] for value in dhs]
        return dhs

    def datagen(self, mode='train', once=False, shuffle=True, get_idx=False, verb=0):
        done = False
        if mode == 'test':
            indices = np.nonzero(self.bed.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 0]
        elif mode =='val':
            indices = np.nonzero(self.bed.chr == 'chr8')[0]
            indices = [x for x in indices if x%2 == 1]
        elif mode == 'all':
            indices = range(len(self.peaks))
        elif mode =='train':
            indices = np.nonzero(self.bed.chr != 'chr8')[0]
        cell = np.repeat(self.keys, len(indices))
        indices = np.tile(indices, len(self.keys))
        combined = np.append(np.expand_dims(indices, axis=1), np.expand_dims(cell, axis=1), axis=1)        
        while not done:
            if shuffle:
                np.random.shuffle(combined)
            for combo in combined:
                # get out the datapoint we need
                idx = int(combo[0])
                cell_key = combo[1]
                row = self.bed.iloc[idx]
                if verb:
                    print('Combo ' + str(combo))
                # figure out our region
                center = (row.start + row.end) // 2
                window_start = center - self.half_window
                window_end = center + self.half_window
                if verb:
                    print('Start ' + str(window_start))
                    print('End ' + str(window_end))
                    print('Chr ' + str(row.chr))
                # get the sequence
                seq = self.genome[row.chr][window_start:window_end].lower()
                if verb:
                    print('Sequence ' + str(seq))
                encoded_seq = encode_string(seq)
                if verb:
                    print('Encoded ' + str(encoded_seq))
                # get the coverage
                dhs = self.get_coverage(row.chr, window_start, window_end, cell_key)
                if verb:
                    print('DHS ' + str(dhs))
                # append the coverage to the sequence
                both = np.append(encoded_seq, np.expand_dims(dhs, axis=1), axis=1)
                # get the log number of reads
                num_reads = self.get_label(row.chr, window_start, window_end, cell_key)
                if get_idx:
                    yield both, num_reads, str(idx) + str(cell_key)
                else:
                    yield both, num_reads
            done=once

    def batch_gen(self, mode='train', once=False):
        data = self.datagen(mode=mode, once=once)
        while True:
            counts = list()
            seqs = list()
            while len(seqs) < 32:
                combo = next(data)
                seqs.append(combo[0])
                counts.append(combo[1])
            yield np.asarray(seqs), np.asarray(counts)

    def get_num_examples(self):
        return sum(self.bed.chr != 'chr8')

    def make_hdf5(self, path, verb=0):
        indices = list(self.bed.index.values)
        cell = np.repeat(self.keys, len(indices))
        indices = np.tile(indices, len(self.keys))
        combined = np.append(np.expand_dims(indices, axis=1), np.expand_dims(cell, axis=1), axis=1)
        with h5py.File(path, 'w') as hf5:
            val = hf5.create_group('val')
            test = hf5.create_group('test')
            train = hf5.create_group('train')
            for combo in tqdm(combined):
                # get out the datapoint we need
                idx = int(combo[0])
                cell_key = combo[1]
                row = self.bed.iloc[idx]
                # figure out our region
                name = str(idx) + str(cell_key) + '_' + str(row.chr) + '_' + str(row.start) + '_' + str(row.end)
                center = (row.start + row.end) // 2
                window_start = center - self.half_window
                window_end = center + self.half_window
                # pull the seq and per-base coverage
                seq = self.genome[row.chr][window_start:window_end].lower()
                encoded_seq = encode_string(seq)
                dhs = self.get_coverage(row.chr, window_start, window_end, cell_key)
                ctcf = self.get_coverage(row.chr, window_start, window_end, cell_key, CTCF=True)
                # put all the data together
                both = np.append(encoded_seq, np.expand_dims(dhs, axis=1), axis=1)
                three = np.append(both, np.expand_dims(ctcf, axis=1), axis=1)
                # make the dataset
                if row.chr == 'chr8' and idx%2 == 0:
                    test.create_dataset(name, data=three)
                elif row.chr == 'chr8' and idx%2 == 1:
                    val.create_dataset(name, data=three)
                else:
                    train.create_dataset(name, data=three)

class CTCFGeneratorhdf5(object):
    """Generator from a hd5f file."""
    def __init__(self, filepath):
        self.hf5 = h5py.File(filepath, 'r')

    def get_num_examples(self, mode='train'):
        return len(np.asarray([str(key) for key in self.hf5[mode].keys()]))

    def datagen(self, mode='train', log=False, once=False):
        done=False
        grp = self.hf5[mode]
        keys = np.asarray([str(key) for key in grp.keys()])
        while not done:
            np.random.shuffle(keys)
            for key in keys:
                if log:
                    yield grp[key][:,:5], np.log(sum(grp[key][:,5])+.01)
                else:
                    yield grp[key][:,:5], sum(grp[key][:,5])
            done = once

    def batch_gen(self, mode='train', log=False, once=False):
        data = self.datagen(mode=mode, once=once, log=log)
        while True:
            counts = list()
            seqs = list()
            while len(seqs) < 32:
                combo = next(data)
                seqs.append(combo[0])
                counts.append(combo[1])
            yield np.asarray(seqs), np.asarray(counts)

if __name__ == '__main__':
    tf.app.run()
