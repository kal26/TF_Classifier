#!/bin/python

import re
import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import sys
sys.path.insert(0,"/home/kal/CTCF/modules/")
sys.path.insert(0,"/home/kal/CTCF/mass_CTCF/modules/")
import tf_memory_limit
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import ctcf_strength_gen
from convnet import Bias
from keras.models import load_model, Model
from keras.layers import Input, Activation
import viz_sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import logit, expit
from sklearn.metrics import precision_recall_curve
from sklearn.manifold import TSNE
from scipy.signal import correlate2d
import ucscgenome
import pickle

meme_path = '/home/kal/data/CTCF.meme'



class ctcfmodel(object):
    """ An instance of a trained ctcf model that can then be used for evalutation."""

    def __init__(self, outpath, gen_path='/home/kal/data/ctcf_strengthgen.hdf5', genome_path='/home/kal/.ucscgenome/hg19.2bit', bed_path=None, batch_size=32):
        """ Initialize a new ctcf model obejct.

        Arguments:
           outpath -- path to the folder where the outputs were generated.

        Keywords:
            genpath -- path to a data generator.
            genome_path -- path to a genome.
            bed_path -- path to an already generated and annotated atac peak bed file.
            batch_size -- size of batches accepted by model.
        """ 
        self.batch_size = batch_size
        # get an output direcotry
        self.out_dir = outpath 

        # get the genome
        self.genome = ucscgenome.Genome(genome_path)

        # load the historys
        num_pk1 = len([f for f in os.listdir(outpath) if f.endswith('.pk1') and os.path.isfile(os.path.join(outpath, f))])
        folder_name = os.path.basename(os.path.normpath(outpath))
        history_path = os.path.join(outpath, folder_name + '_history')
        if num_pk1 == 1:
            with open(history_path + '1.pk1', 'rb') as input:
                self.h = pickle.load(input)
                self.finer_epochs = False
        elif num_pk1 == 3:
            with open(history_path + '1.pk1', 'rb') as input:
                self.h1 = pickle.load(input)
            with open(history_path + '2.pk1', 'rb') as input:
                self.h2 = pickle.load(input)
            with open(history_path + '3.pk1', 'rb') as input:
                self.h3 = pickle.load(input) 
            self.finer_epochs = True

        print('Loaded training history.')

        # load the potential model paths
        model_paths = list()
        for file in os.listdir(outpath):
            if 'weights_3_24' in file and file.endswith(".hdf5"):
                model_paths.append(os.path.join(outpath, file))
        # Find the model with the highest val_acc
        def extract_number(f):
            s = f.split('_')[-1].rsplit('.', maxsplit=1)
            return (float(s[0]) if s else -1, f)
        model_path = min(model_paths, key=extract_number) 
        print('model path:' + str(model_path))

        # load the model
        self.model = load_model(model_path, custom_objects={'Bias':Bias})
        # and the layer names
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers])

        print('Loaded the model.')

        # get a data generator
        self.gen = ctcf_strength_gen.CTCFGeneratorhdf5(gen_path)
        print('Loaded the data generator.')

        if bed_path == None:
            self.generate_bed()
            print('Generated a bed file.')
        else:
            self.peaks = pd.read_table(bed_path, header=None)
            self.peaks.columns = 'chr start end ctcf pwm ml'.split()
            print('Loaded a bed file.')

        # get the constrained peaks
        f = open(os.path.join(self.out_dir, 'constrained_atac.bed'), 'w')
        subprocess.run(['bedtools', 'coverage',  '-counts', '-a', os.path.join(self.out_dir, 'end_atac.bed'), '-b', '/home/kal/data/K526_atac_sorted.bed'], stdout=f)
        self.constrained_peaks = pd.read_table(os.path.join(self.out_dir, 'constrained_atac.bed'), header=None)
        self.constrained_peaks.columns = 'chr start end oldctcf pwm ml counts'.split()

        self.constrained_peaks['ctcf'] = self.constrained_peaks['counts'] > 0

        print('Constrained the peaks')

        # get a subset of localized sequences
        self.sample_peaks = self.peaks.sample(100*self.batch_size)
        signal_seqs = list()
        for index, row in self.sample_peaks.iterrows():
            this_seq, max_pred = localize(row, self.model, self.genome)    
            signal_seqs.append(this_seq)
        self.sample_peaks['signal_seq'] = signal_seqs
        print('Localized a subset of sequences.')


    def get_history_graph(self, key, show=False):
        """ Create a graph of the given key over time and save the graph.
    
        Arguments:
            key -- history key to graph.
       
        Keywords:
            show -- show the plot?
        """
        # Summarize history for the key
        if self.finer_epochs:
            plt.plot(group_stats('val_' + key, self.h1, self.h2, self.h3))
            plt.plot(group_stats(key, self.h1, self.h2, self.h3))
        else:
            plt.plot(self.h[key])
            plt.plot(self.h['val_' + key])
        plt.title('model '+ key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='upper left')
        plt.savefig(os.path.join(self.out_dir, 'model_' + key + '.png'), bbox_inches='tight')
        if show:
            plt.show()
    def get_hexbin(self, const=False, show=False):
        """ Make hexbin plots for positvie and negative samples."""
        if const:
            pos_peaks = self.constrained_peaks.loc[self.constrained_peaks['ctcf'] == 1]
            neg_peaks = self.constrained_peaks.loc[self.constrained_peaks['ctcf'] == 0]
        else:
            pos_peaks = self.peaks.loc[self.peaks['ctcf'] == 1]
            neg_peaks = self.peaks.loc[self.peaks['ctcf'] == 0]
        # make a plot of the two precitions vs each other
        plt.hexbin(pos_peaks['pwm'].tolist(), pos_peaks['ml'].tolist(), gridsize=50, bins='log', cmap='plasma')
        plt.title('ML Predictions vs PWM score for atac peaks with ctcf')
        plt.savefig(os.path.join(self.out_dir, 'positive_hexbin.png'), bbox_inches='tight')
        if show:
            plt.show()

        plt.hexbin(neg_peaks['pwm'].tolist(), neg_peaks['ml'].tolist(), gridsize=50, bins='log', cmap='plasma')
        plt.title('ML Predictions vs PWM score for atac peaks without ctcf')
        plt.savefig(os.path.join(self.out_dir, 'negative_hexbin.png'), bbox_inches='tight')
        if show:
            plt.show()
   

    def get_pr(self, show=False, mode='atac', gain=False, const=False):
        if mode == 'atac':
            return self.get_atac_pr(show=show, const=const, gain=gain)
        elif mode == 'test':
            return self.get_test_pr(show=show, gain=gain)

    def get_test_pr(self, show=False, gai[n=False):
        batch_size = self.batch_size
        g = self.gen.batch_gen(mode='test')
        preds = list()
        scores = list()
        pwms = list()
        for (batch, labels), index in zip(g, range(self.gen.test_pos.shape[0] // batch_size)):
             # Get predicitons.
             preds.append(self.model.predict(batch).flatten())
             scores.append(labels)
             pwms.append(get_pwm_score(batch))

        preds = np.asarray(preds).flatten()
        scores = np.asarray(scores).flatten() 
        pwms = np.asarray(pwms).flatten()
        labels = [score!=0 for score in scores]

        # plot a pr curve
        precision, recall, thresholds = precision_recall_curve(labels, preds, pos_label=1)
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(os.path.join(self.out_dir, 'test_prc.png'), bbox_inches='tight') 
        if show:
            plt.show()
        if not gain:
            return precision, recall
        else:
            # get pr gain
            prop_pos = sum(labels)/len(labels)
            print('Proportion positive:' + str(prop_pos))
            precision_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in precision]
            recall_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in recall]

            plt.plot(recall_gain, precision_gain, label='Deep Learning')
            plt.legend()
            plt.title('Precision-Recall Gain Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.savefig(os.path.join(self.out_dir, 'test_prgain_curve.png'), bbox_inches='tight')
            if show:
                 plt.show()
            return precision_gain, recall_gain

    def get_atac_pr(self, const=False, show=False, gain=False):
        """ Get a precision recall and precision recall gain curve. 

        Keywords:
            show -- show the graph?
 
        Returns:
            deep_precision_gain -- precision for ml model normalized for proportion of positive samples.
            deep_recall_gain -- recall for ml model normalized for proportion of positive samples.
            pwm_precision_gain -- precision for pwm model normalized for proportion of positive samples.
            pwm_recall_gain -- recall for pwm model normalized for proportion of positive samples.

        """
        if const:
            deep_precision, deep_recall, thresholds = precision_recall_curve(self.constrained_peaks['ctcf'].tolist(),
                                                                 self.constrained_peaks['ml'].tolist(), pos_label=1)

            pwm_precision, pwm_recall, thresholds = precision_recall_curve(self.constrained_peaks['ctcf'].tolist(),
                                                                 self.constrained_peaks['pwm'].tolist(), pos_label=1)
        else:
            deep_precision, deep_recall, thresholds = precision_recall_curve(self.peaks['ctcf'].tolist(), 
                                                                 self.peaks['ml'].tolist(), pos_label=1)

            pwm_precision, pwm_recall, thresholds = precision_recall_curve(self.peaks['ctcf'].tolist(), 
                                                                 self.peaks['pwm'].tolist(), pos_label=1)
    
        plt.plot(deep_recall, deep_precision, label='Deep Learning')
        plt.plot(pwm_recall, pwm_precision, label = 'Position-Weight Matrix')
        plt.legend()
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(os.path.join(self.out_dir, 'pr_curve.png'), bbox_inches='tight')
        if show:
             plt.show()

        # get pr gain
        if const:
            prop_pos = len(self.constrained_peaks.loc[self.constrained_peaks['ctcf'] == 1])/len(self.constrained_peaks)
        else:
            prop_pos = len(self.peaks.loc[self.peaks['ctcf'] == 1])/len(self.peaks)
        print('Proportion positive:' + str(prop_pos))
        deep_precision_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in deep_precision]
        deep_recall_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in deep_recall]

        pwm_precision_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in pwm_precision]
        pwm_recall_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in pwm_recall]

        plt.plot(deep_recall_gain, deep_precision_gain, label='Deep Learning')
        plt.plot(pwm_recall_gain, pwm_precision_gain, label = 'Position-Weight Matrix')
        plt.legend()
        plt.title('Precision-Recall Gain Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(os.path.join(self.out_dir, 'prgain_curve.png'), bbox_inches='tight')
        if show:
             plt.show()
        if not gain:
            return deep_precision, deep_recall, pwm_precision, pwm_recall
        else:
            # get pr gain
            if const:
                prop_pos = len(self.constrained_peaks.loc[self.constrained_peaks['ctcf'] == 1])/len(self.constrained_peaks)
            else:
                prop_pos = len(self.peaks.loc[self.peaks['ctcf'] == 1])/len(self.peaks)
            print('Proportion positive:' + str(prop_pos))
            deep_precision_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in deep_precision]
            deep_recall_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in deep_recall]

            pwm_precision_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in pwm_precision]
            pwm_recall_gain = [(x-prop_pos)/((1-prop_pos)*x) for x in pwm_recall]

            plt.plot(deep_recall_gain, deep_precision_gain, label='Deep Learning')
            plt.plot(pwm_recall_gain, pwm_precision_gain, label = 'Position-Weight Matrix')
            plt.legend()
            plt.title('Precision-Recall Gain Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.savefig(os.path.join(self.out_dir, 'prgain_curve.png'), bbox_inches='tight')
            if show:
                 plt.show()
            return deep_precision_gain, deep_recall_gain, pwm_precision_gain, pwm_recall_gain
 

    def get_tsne(self, layer_name, show=False):
        """ Get a tsne visualization of the neuron activations.

        Arguments:
            layer_name -- layer to pull activations for (key in layer dict).

        Keywords:
            show -- show the final graph?
        """
        # build a function to get nueron activations.
        seqs = self.model.input
        get_activations = K.function([seqs, K.learning_phase()], [self.layer_dict[layer_name].output, self.model.output])

        # put the sequences into batches
        g = ctcf_strength_gen.filled_batch(iter(self.sample_peaks['signal_seq']), batch_size=self.batch_size)
        # get the layer activation for each sequence.
        base_activations = list()
        for input_batch in g:
            activations, predictions = get_activations([input_batch, 0])
            base_activations.append(np.append(activations[:32], activations[32:], axis=1))
        
        # reshape and take the maximum of each neuron to collapse the feature space to something more reasonable.
        base_activations = np.asarray(base_activations)
        base_activations = base_activations.reshape((-1, base_activations.shape[2], base_activations.shape[3]))[:self.sample_peaks.shape[0]]
        max_activations = np.amax(base_activations, axis=1)

        # build a t-sne model
        tsnemodel = TSNE(n_components=2, random_state=0)
        divisions = tsnemodel.fit_transform(max_activations)
        heatmap = plt.scatter(divisions[:,0], divisions[:,1], c=self.sample_peaks['ml'], s=10+self.sample_peaks['ctcf']*10, cmap='plasma')
        cbar = plt.colorbar(heatmap)
        plt.title('T-SNE for ' + layer_name + ' Kernel Maximum Activations')
        plt.savefig(os.path.join(self.out_dir, layer_name + '_tsne.png'), bbox_inches='tight')
        if show:
             plt.show()      
        return divisions

    def varience_plot(self, layer_name, show=False):
        """ Plot covarience of neuron activations with themselves and scores.
   
        Arguments:
            layer_name -- Layer to get neuron activations from (in layer_dict)

        Keywords:
            show -- show the plot?
        """
        # build a function to get activations
        seqs = self.model.input
        get_activations = K.function([seqs, K.learning_phase()], [self.layer_dict[layer_name].output, self.model.output])
       
        # put the sequences into batches.
        g = ctcf_strength_gen.filled_batch(iter(self.sample_peaks['signal_seq']), batch_size=self.batch_size)

        # get activations.
        activations = list()
        for i in range(100):
            input_seqs = next(g)
            a, p = get_activations([input_seqs, 0])
            both = a[:self.batch_size] + a[self.batch_size:]
            activations.append([value for value in np.max(both, axis=1)])
      
        activations = [act for l in activations for act in l]
        activations = np.asarray(activations)

        # plot activation covarience
        plt.title('Correlations for ' + str(layer_name))

        ml = np.repeat(np.expand_dims(self.sample_peaks['ml'], axis=1), 2, axis=1)
        pwm = np.repeat(np.expand_dims(self.sample_peaks['pwm'], axis=1), 2, axis=1)
        diff = np.repeat(np.expand_dims(self.sample_peaks['pwm'] - self.sample_peaks['ml'], axis=1), 2, axis=1)
        variables = np.append(activations, ml, axis=1)
        variables = np.append(variables, pwm, axis=1)
        variables = np.append(variables, diff, axis=1)
        label = ['ml', 'pwm', 'diffs']

        cov = np.corrcoef(variables, rowvar=False)
        plt.yticks([activations.shape[1] + 1, activations.shape[1] + 3, activations.shape[1] + 5], ('ml', 'pwm', 'diffs'))
        plt.xticks([])

        plt.imshow(cov, cmap='plasma')
        plt.savefig(os.path.join(self.out_dir, layer_name + '_corrcoef.png'), bbox_inches='tight')
        if show:
             plt.show()     



    def generate_pd(self):
        """ generate and annotate a dataframe.
        """
        # iterate throught the sequences
        for seq, ctcf in gen.datagen(mode='test'):
           peaks[seq]
           peaks = gen.get_table(mode='all')
        peaks.columns = 'chr start end'.split() 

        # get predictions for the bed file
        preds, pwms, ctcf = predict_peaks(self.model, self.genome, peaks)
        print('Number predictions: ' +str(np.asarray(preds).flatten().shape))

        # get only the predictions we need:
        num_samples = file_len(split_atac_path)
        flat_preds = np.asarray(preds).flatten()[:num_samples]
        peaks['ml_preds'] = flat_preds

        # get the pwm predictions
        pwms = get_pwmscore(peaks['signal_seqs']

        # write out the annotated bed file
        peaks.to_csv(os.path.join(self.out_dir, 'split_atac_ml.bed'), sep='\t', index=False, header=False)
        print('Generated annotated bed file.')

        # sort the split peaks output.
        f = open(os.path.join(self.out_dir, 'sorted_split_atac_ml.bed'), 'w')
        subprocess.run(['sort', '-k1,1',  '-k2,2n', '-i', os.path.join(self.out_dir, 'split_atac_ml.bed')], stdout=f)

        # map the ml predictions onto the full atac peaks
        f = open(os.path.join(self.out_dir, 'full_atac.bed'), 'w')
        subprocess.run(['bedtools', 'map', '-c', '4', '-null', '0.0', '-o', 'max', '-a', full_atac_path, '-b', os.path.join(self.out_dir, 'sorted_split_atac_ml.bed')], stdout=f)

        # generate a final bed file
        peaks = pd.read_table(os.path.join(self.out_dir, 'full_atac.bed'), header=None)
        peaks.columns = 'chr start end numreads numbases length precentcoverage pwm_score ml_preds'.split()
        print('Generated a full peak mapped annotated bed file.')

        # make a ctcf label column:
        labels = []
        for index, row in peaks.iterrows():
            labels.append(row['numreads']>0)
        peaks['ctcf_label'] = labels

        # write out a file with only what we need
        finaldf = peaks.filter(['chr', 'start', 'end', 'ctcf_label', 'pwm_score'])
        finaldf['ml'] = peaks['ml_preds']
        finaldf.to_csv(os.path.join(self.out_dir, 'end_atac.bed'), sep='\t', index=False, header=False)

        # add the bed path and column names to the object.
        self.peaks = pd.read_table(os.path.join(self.out_dir, 'end_atac.bed'), header=None)
        self.peaks.columns = 'chr start end ctcf pwm ml'.split()

def group_stats(key, h1, h2, h3):
    # Summarize history for accuracy
    out1 = np.copy(h1[key])
    out2 = np.copy(h2[key])
    out3 = np.copy(h3[key])
    return np.concatenate([out1, out2, out3])

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def predict_bed(model, genome, bed_path, column_names='chr start end', input_window=256, batch_size=32):
    """Predict from a bed file.

    Arguments:
        model -- kerase model to use to create predictions.
        genome -- genome to pull sequences from.
        bed_path -- path to a bed file that needs predictions.

    Keywords:
        column_names -- to generate a dataframe from the bed file.
        input_window -- length of sequences to pass to the model.
        batch_size -- size of batches to be passed to the model.
    """
    peaks = pd.read_table(bed_path, header=None)
    peaks.columns = column_names.split()
    return predict_peaks(model, genome, peaks, input_window=input_window, batch_size=batch_size)

def predict_peaks(model, genome, peaks, input_window=256, batch_size=32):
    """Predict based off of a model and bed generated dataframe.

    Arguments:
        model -- kerase model to use to create predictions.
        genome -- genome to pull sequences from.
        peaks -- optionally, pass the dataframe instead of the bed file.

    Keywords:
        snv -- predict single nucleotide varients.
        input_window -- length of sequences to pass to the model.
        batch_size -- size of batches to be passed to the model.

    Returns:
        preds -- predictions, one for each row in the bed file buffered to fit batch size. 
        pwms -- pwm scores
        ctcf -- 
    """
    half_window = input_window // 2

    def seq_gen():
        done = False
        first = True
        batches = 0
        sequences = 0
        iterations = 0
        while not done:
                for index, row in peaks.iterrows():
                    if first:
                        pad_seq = np.zeros((1,256,4))
                        if row.start < 0:
                            row.start = 0
                        seq = ctcf_strength_gen.encode(np.fromstring(genome[row.chr][row.start:row.end].lower(), 
                                                           dtype=np.uint8))
                        if seq.shape != (256,4):
                                print(row.start)
                                print(row.end)
                        pad_seq[0, :seq.shape[0], :seq.shape[1]] = seq
                        batch = pad_seq
                        first = False
                    else:
                        if np.asarray(batch).shape == (32, 256, 4):
                            batches +=1
                            yield np.asarray(batch)
                            pad_seq = np.zeros((1,256,4))
                            if row.start < 0:
                                 row.start = 0
                            seq = ctcf_strength_gen.encode(np.fromstring(genome[row.chr][row.start:row.end].lower(), 
                                                           dtype=np.uint8))
                            if seq.shape != (256,4):
                                print(row.start)
                                print(row.end)
                            pad_seq[0, :seq.shape[0], :seq.shape[1]] = seq
                            batch = pad_seq
                        elif len(batch) == 32:
                            print('What in the what?!?')
                            print(batch)
                            print(batch.shape)
                        else:                        
                            pad_seq = np.zeros((1,256,4))
                            if row.start < 0:
                                row.start = 0
                            seq = ctcf_strength_gen.encode(np.fromstring(genome[row.chr][row.start:row.end].lower(), 
                                                           dtype=np.uint8))
                            if seq.shape != (256,4):
                                print(row.start)
                                print(row.end)
                            pad_seq[0, :seq.shape[0], :seq.shape[1]] = seq
                            batch = np.append(batch, pad_seq, axis=0)
                    sequences += 1
                    
                print('Batches pulled: ' + str(batches))
                final = np.zeros((batch_size, input_window, 4))
                final[:batch.shape[0],:batch.shape[1]] = batch
                print('Sequences pulled: ' + str(sequences))
                print('Did final')
                yield final
                done = True
            
    g = seq_gen()
    preds = list()
    print('Expected batch count: ' + str((peaks.shape[0] // batch_size) + (peaks.shape[0] % batch_size > 0)))
    
    for i in range((peaks.shape[0] // batch_size) + (peaks.shape[0] % batch_size > 0)):
        if i % 500 == 0:
            print('Batch number: ' + str(i))
        batch = next(g)
        if batch.shape == (32, 256, 4):
            preds.append(model.predict_on_batch(batch))
        else:
            print(batch.shape)

    return preds

def from_long(seq, model, input_window=256, batch_size=32, verb=False):
    """ Returns an input window size piece of the sequence with the maximum prediction.
    
    Arugments:
         seq -- one-hot encoded sequence.
         model -- model to localize the sequence with.

    Keywords:
         input_window -- length of the sequence to return.
         batch_size -- batch size accepted by the model.
         verb -- verbosity?

    Returns:
         max_tile -- peice of sequence with maximum response.
         max_pred -- prediction from max_tile.
         index -- index within the sequence where the tile starts.
    """
    # check for a short sequence.
    if len(seq) <= 256:
        print('The sequence you passed to long_seq is not long.') 
        max_tile = [0,0,0,0]*input_window
        idx = (input_window-len(seq))//2
        max_tile[idx:idx + len(seq)] = seq.copy()
        pred = model.predict(max_tile)
        return max_tile, pred, idx
        
          
    # get the indexes of the tiles.
    idxs = list(range(0, len(seq)-input_window, input_window)) + list(range(input_window//2, len(seq)-input_window, input_window))
    idxs.append(len(seq) - input_window)   
    idxs.append(max(len(seq) -(input_window//2 +input_window), 0))
                                            
    # make tiles.
    first_split = np.split(seq, list(range(input_window, len(seq), input_window)))[:-1]
    second_split = np.split(seq, list(range(input_window//2, len(seq), input_window)))[1:-1]
    tile_seqs = first_split + second_split 
    tile_seqs = np.asarray(tile_seqs)

    if verb:
        print(tile_seqs.shape)
        print(np.asarray([seq[-input_window:].copy()]).shape)
    tile_seqs = np.append(tile_seqs, np.asarray([seq[-input_window:].copy()]), axis=0)
    tile_seqs = np.append(tile_seqs, np.asarray([seq[-(input_window//2 +input_window):-input_window//2].copy()]), axis=0)    

    # make an iterable tile generator.
    tile_seqs = np.asarray(tile_seqs)
    tile_iter = iter(tile_seqs)
 
    # get a batch generator
    batches = ctcf_strength_gen.filled_batch(tile_iter, batch_size=batch_size)

    # figure out where the max prediction is coming from
    preds = list()
    batch_list = list()
    for batch in batches:
        preds.append(model.predict_on_batch(batch))
        batch_list.append(batch)
    preds = np.asarray(preds).reshape((-1))[:tile_seqs.shape[0]]
    batch_list = np.asarray(batch_list).reshape((-1, input_window, 4))[:tile_seqs.shape[0]]

    # get the tile that produced that predicion.
    max_idx = np.argmax(preds)
    max_pred = np.max(preds)
    max_tile = batch_list[max_idx]
    return max_tile, max_pred, idxs[max_idx] 


def localize(row, model, genome, input_window=256, batch_size=32, get_idx=False, verb=False):
    """Find the input_window bp window responsible for a ml prediction in a bed file row.

    Arguments:
        row -- pd dataframe row with start and end parameters. 
        model -- keras model to make predictions on.
        genome -- genome to pull sequences from.

    Keywords:
        input_window -- size of window to localize on.
        batch_size -- batch_size accepted by the model.
        verb -- print output?

    Returns:
        max_tile -- input_window sized sequence that gave maximum prediciton from the model.
        max_pred -- prediction value for max_tile.
    """
    # break the sequence into overlapping tiles
    tile_seqs = list()
    num_tiles = int((row['end']-row['start']) / input_window) + ((row['end']-row['start']) % input_window > 0)
    if verb:
        print(num_tiles)
    for idx in range(num_tiles):
        if row['start'] + idx*input_window - input_window//2 > 0:
            seq = genome[row['chr']][row['start'] + idx*input_window - input_window//2:row['start'] + (idx+1)*input_window - input_window//2].lower()
            tile_seqs.append(ctcf_strength_gen.encode(np.fromstring(seq, dtype=np.uint8)))
        else:
            buffered_seq = np.zeros((256,4))
            buffered_seq[:row['start'] + (idx+1)*input_window - input_window//2] = genome[row['chr']][0:row['start'] + (idx+1)*input_window - input_window//2]
            tile_seqs.append(ctcf_strength_gen.encode(np.fromstring(buffered_seq).lower(), dtype=np.uint8))
        seq = genome[row['chr']][row['start'] + idx*input_window:row['start'] + (idx+1)*input_window].lower()
        tile_seqs.append(ctcf_strength_gen.encode(np.fromstring(seq, dtype=np.uint8)))
        
    tile_seqs= np.asarray(tile_seqs)
    tile_iter = iter(tile_seqs)
    
    # get a batch generator
    batches = ctcf_strength_gen.filled_batch(tile_iter, batch_size=batch_size)
    
    # figure out where the max prediction is coming from
    preds = list()
    batch_list = list()
    for batch in batches:
        preds.append(model.predict_on_batch(batch))
        batch_list.append(batch)
    preds = np.asarray(preds).reshape((-1))[:tile_seqs.shape[0]]
    batch_list = np.asarray(batch_list).reshape((-1, input_window, 4))[:tile_seqs.shape[0]]

    # get a tile centered there
    max_idx = np.argmax(preds)
    max_pred = np.max(preds)
    max_tile = batch_list[max_idx]
    if verb:
        print(max_idx)
        print(max_pred)
        print(preds)
    
    if get_idx:
        return max_tile, max_pred, max_idx*input_window + row['start']
    return max_tile, max_pred

# process the memes
with open(meme_path, 'r') as infile:
    meme_length = -1
    CTCF_memes = list()
    for line in infile.readlines():
        if 'letter-probability matrix' in line:
            meme_length = int(line.split()[5])
            this_meme_lines = list()
        elif meme_length > 0:
            this_meme_lines.append([float(item.strip()) for item in line.split()])
            meme_length = meme_length - 1
        elif meme_length == 0:
            this_meme = np.asarray(this_meme_lines)
            CTCF_memes.append(this_meme)
            meme_length = -1
    if meme_length == 0:
        this_meme = np.asarray(this_meme_lines)
        CTCF_memes.append(this_meme)
        meme_length = -1
        
# add rcs of memes
rcs = list()
for meme in CTCF_memes:
    rcs.append(meme[::-1, ::-1])
CTCF_memes = CTCF_memes + rcs

    
psuedocount=0.1
# get the transformed memes
transformed_memes = list()
for meme in CTCF_memes:
    # add psuedo count
    #print('MEME:')
    #viz_sequence.plot_weights(meme)
    meme = meme + psuedocount 
    #viz_sequence.plot_weights(meme)
    # normalize
    #print('normalized:')
    norms = np.repeat(np.linalg.norm(meme, axis=1), 4).reshape((-1, 4))
    meme = meme/norms
    #viz_sequence.plot_weights(meme)
    # log transform
    #print('Log transformed')
    meme = np.log(meme)
    #viz_sequence.plot_weights(meme)
    # shift up
    #print('shift')
    min = np.amin(meme)
    meme = meme - min
    #viz_sequence.plot_weights(meme)
    transformed_memes.append(meme)

def get_pwm_score(input_seqs, meme_library=transformed_memes):  
    scores = list()
    for seq in input_seqs:
        best_score = -np.inf
        for test_meme in meme_library:
            correlations = correlate2d(seq, test_meme, mode='valid')
            if np.max(correlations) > best_score:
                best_score = np.max(correlations)
                best_location = np.argmax(correlations)
                best_filter = test_meme
        scores.append(best_score)
    return np.asarray(scores)
