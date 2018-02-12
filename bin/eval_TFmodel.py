import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import tf_memory_limit
#import general use packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ucscgenome
from tqdm import tqdm
#import keras related packages
from keras import backend as K
from keras.models import load_model, Model, Input
from keras.layers import Input, Lambda
import tensorflow as tf
#import custom packages
import helper
import viz_sequence
import train_TFmodel
import sequence

class TFmodel(object):
    """Transcription factor classification keras model."""

    def __init__(self, full_path, output_path=None, model_path=None):
        """Create a new model object.

        Arguments:
            model_path -- path to a trained mode's directory with
                          final_model.hdf5
                          32.3_32.3_16.3_8.3_model.png
                          history/
                          intermediate_weights/
                          atac_analysis/
                          evaluation/
        Keywords:
            output_path -- directory to write out requested files to (defalut is to evaluation or atac analysis). 
            model_path -- actual path to the model, default is 'final_model.hdf5'.
        """
        self.full_path = full_path
        if model_path == None:
            self.model_path = os.path.join(self.full_path, 'final_model.hdf5')
        else:
            self.model_path = model_path
        self.model = load_model(self.model_path, custom_objects={'Bias':train_TFmodel.Bias})
        if output_path != None:
            self.out_path = output_path
        else:
            self.out_path = os.path.join(full_path, 'evaluation')
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers]) 
        try:
            self.get_act = K.function([self.model.input, K.learning_phase()], [self.layer_dict['bias'].output])
        except KeyError:
            print('Loading model without Bias layer')
            self.get_act = K.function([self.model.input, K.learning_phase()], [self.model.output])

    def __str__(self):
        """Printable version of the model."""
        return 'TFmodel() at ' + self.model_path
    
    def __repr__(self):
        """Representaiton of the model."""
        return 'TFmodel() at ' + self.model_path

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
            return activation
        except TypeError:
            #acutally not iterable
            if isinstance(generator, sequence.SeqDist):
                dist = True
                #distribution
                #converts distributions to discrete sequences and averages
                def gen():
                    for i in range(distribution_repeats):
                        yield generator.discrete_seq()
                g = gen()
                batch_gen = train_TFmodel.filled_batch(g)
                for batch in batch_gen:
                    activation.append(self.get_act([batch, 0]))
                activation = np.asarray(activation).flatten()
                return np.sum(activation)/activation.shape[0]
            else:
                return self.get_act([train_TFmodel.blank_batch(generator.seq), 0])[0][0][0] 
    def gumbel_dream(self, seq, dream_type, temp=10, layer_name='final_output', filter_index=0, meme_library=None, num_iterations=20, step=None, viz=False):
        """ Dream a sequence for the given number of steps employing the gumbel-softmax reparamterization trick.

        Arguments:
            seq -- SeqDist object to iterate over.
            dream_type -- type of dreaming to do. 
                standard: update is average gradient * step
                adverse: update is standard - .05
                blocked: dream only outside the pwm region (should I allow the max pwm to move around? doesn't currently.)
        Keywords:
            temp -- for gumbel softmax.
            layer_name -- name of the layer to optimize.
            filter_index -- which of the neurons at this filter to optimize.
            meme_library -- memes to use if applicable (default is CTCF)
            num_iterations -- how many iterations to increment over.
            step -- default is 1/10th the initial maximum gradient
            viz -- sequence logo of importance?
        Returns:
            dream_seq -- result of the iterations.
        """
        # dreaming won't work off of true zero probabilities - if these exist we must add a pseudocount
        if np.count_nonzero(seq.seq) != np.size(seq.seq):
            print('Discrete Sequence passed - converting to a distibution via pseudocount')
            dream_seq = sequence.SeqDist(helper.softmax(3*seq.seq + 1))
        else:
            dream_seq = sequence.SeqDist(seq.seq)

        # get a gradient grabbing op
        #input underlying distribution as (batch_size, 256, 4) duplications of the sequence
        dist = Input(shape=((256,4)), name='distribution')
        logits_dist = tf.reshape(dist, [-1,4])
        # sample and reshape back (shape=(batch_size, 256, 4))
        # set hard=True for ST Gumbel-Softmax
        sampled_seq = tf.reshape(train_TFmodel.gumbel_softmax(logits_dist, temp, hard=True),[-1, 256, 4])
        sampled_seq = self.model.input
        if layer_name == 'final_output':
            loss = self.model.output
        else:
            max_by_direction = Lambda(lambda x: K.maximum(K.max(x[:x.shape[0]//2, :, :], axis=1), K.max(x[x.shape[0]//2:, ::-1, :], axis=1)), name='stackmax', output_shape=lambda s: (s[0] // 2, 1))
            layer_output = max_by_direction(self.layer_dict[layer_name].output)
            loss = layer_output[:, filter_index] #each batch and nuceotide at this neuron.
        # compute the gradient of the input seq wrt this loss and average to get the update (sampeling already weights for probability)
        update = K.mean(K.gradients(loss, sampled_seq)[0], axis=0)
        #get a function
        update_op = K.function([sampled_seq, K.learning_phase()], [update])

        #find a step size
        if step == None:
            step = 1/(np.amax(update_op([[dream_seq.seq]*32, 0])[0]))
            print('Step ' + str(step))
        # print the initial sequence
        if viz:
            print('Initial Sequence')
            seq.logo()
            print('Model Prediction: ' + str(self.model.predict(train_TFmodel.blank_batch(dream_seq.discrete_seq()))[0][0]))
            self.get_importance(dream_seq, viz=True)
            print('PWM score: ' + str(dream_seq.find_pwm(viz=True)[2]))

        #iterate and dream
        for i in range(num_iterations):
            update = update_op([[dream_seq.seq]*32, 0])[0]
            if dream_type == 'standard':
                dream_seq.seq = helper.softmax(np.log(dream_seq.seq) + update*step)
            elif dream_type == 'adverse':
                dream_seq.seq = helper.softmax(np.log(dream_seq.seq) + update*step -1) 
            elif dream_type == 'blocked':
                meme, position, _ = dream_seq.find_pwm(meme_library=meme_library)
                update[position:position+meme.seq.shape[0]] = 0
                dream_seq.seq = helper.softmax(np.log(dream_seq.seq) + update*step)
            if i%(num_iterations//4) == 0 and viz:
                print('Sequence after ' + str(i) + ' iterations')
                viz_sequence.plot_icweights(dream_seq.seq)

        #print the final sequence
        if viz:
            print('Final sequence')
            dream_seq.logo()
            print('Model Prediction: ' + str(self.model.predict(train_TFmodel.blank_batch(dream_seq.discrete_seq()))[0][0]))
            self.get_importance(dream_seq, viz=True)
            print('PWM score: ' + str(dream_seq.find_pwm(viz=True)[2]))
        return dream_seq     

    def dream(self, seq, dream_type='standard', iterate_op=None, layer_name='final_output', filter_index=0, meme_library=None, num_iterations=20, step=None, viz=False):
        """Dream a sequence for the given number of steps.
         
        Arguments:
            seq -- SeqDist object to iterate over.
        Keywords:
            dream_type -- type of dreaming to do
                standard: update is average gradient @ base * p(base) * step
                adversarial: update is standard - 1/10 * step
                blocked: dream only outside the pwm region (should I allow the max pwm to move around? doesn't currently.)
                constrained: dream orthogal to the pwm score (DOESN'T WORK)
                strict: gradients only apply to a base if that base was in the discrete sequence chosen. 
            iterate_op -- operation to get the update step, default is maximize output. 
            layer_name -- name of the layer to optimize.
            filter_index -- which of the neurons at this filter to optimize.
            meme_library -- memes to use if applicable (default is CTCF)
            num_iterations -- how many iterations to increment over.
            step -- default is 1/10th the initial maximum gradient
            viz -- sequence logo of importance?
        Returns:
            dream_seq -- result of the iterations. 
        """
        # get an iterate operation
        if iterate_op == None:
            iterate_op = self.build_iterate(layer_name=layer_name, filter_index=filter_index)
        # dreaming won't work off of true zero probabilities - if these exist we must add a pseudocount
        if np.count_nonzero(seq.seq) != np.size(seq.seq):
            print('Discrete Sequence passed - converting to a distibution via pseudocount')
            dream_seq = sequence.SeqDist(helper.softmax(3*seq.seq + 1))
        else:
            dream_seq = sequence.SeqDist(seq.seq)
        # find the meme position 
        meme, position, _ = seq.find_pwm(meme_library=meme_library)
        pwm_activation = seq.run_pwm(meme=meme, position=position)
        #print the initial sequence
        if viz:
            print('Inital sequence')
            viz_sequence.plot_icweights(dream_seq.seq)
            self.get_importance(dream_seq, viz=viz)
        # find a good step size 
        batch_gen = train_TFmodel.filled_batch(dream_seq.discrete_gen())
        batch = next(batch_gen)
        update_grads = iterate_op([batch, 0])[0]
        if step == None:
            step = 10/np.amax(update_grads)
            print('step: ' + str(step))
        # apply the updates
        for i in range(num_iterations):
            batch = next(batch_gen)
            update_grads = iterate_op([batch, 0])[0]
            # figure out the type of update to do
            if dream_type == 'adversarial':
                update = np.average(update_grads, axis=0)*dream_seq.seq*step -.1*step
            elif dream_type == 'blocked':
                update = np.average(update_grads, axis=0)*dream_seq.seq*step
                update[position:position+meme.seq.shape[0]] = 0
            elif dream_type == 'constrained':
                pwm_activation = seq.run_pwm(meme=meme, position=position)
                update = helper.rejection(np.average(update_grads, axis=0)*dream_seq.seq, pwm_activation)*step
            elif dream_type == 'strict':
                update = np.average(strict_grads, axis=0, weights=batch)*dream_seq.seq*step

            elif dream_type == 'standard':
                update = np.average(update_grads, axis=0)*dream_seq.seq*step
            else:
                print('Unrecognized dream type passed. Setting to standard.')
                update = np.average(update_grads, axis=0)*dream_seq.seq*step
            # we apply the update in log space so a zero update won't change anything
            dream_seq = np.log(dream_seq.seq) + update
            dream_seq = sequence.SeqDist(helper.softmax(dream_seq)) 
            #print intermediate sequences
            if i%(num_iterations//4) == 0 and viz:
                print('Sequence after ' + str(i) + ' iterations')
                viz_sequence.plot_icweights(dream_seq.seq)
        #print the final sequence
        if viz:
            print('Final sequence')
            viz_sequence.plot_icweights(dream_seq.seq)
            self.get_importance(dream_seq, viz=viz)
        return dream_seq

    def build_iterate(self, layer_name='final_output', filter_index=0):
        """ Build a interation operation for use with dreaming method.
     
        Keywords:
           layer_name -- layer dictionary enry to get the output from.
           filter_index -- inex of the filter to pull from the layer. 
        Output:
            iterate_op -- iteration operation returning gradients.
        """
        # set a placeholder input
        encoded_seq = self.model.input
        # build a function that sumes the activation of the nth filter of the layer considered
        if layer_name == 'final_output':
            activations = self.model.output
            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(K.mean(activations), encoded_seq)[0]
        else:
            layer_output = self.layer_dict[layer_name].output
            activations = layer_output[:, :, filter_index] #each batch and nuceotide at this neuron.
            # forward and reverse sequences
            combined_activation = K.mean(np.maximum(activations[:32], activations[32:]))
            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(combined_activation, encoded_seq)[0]
            # normalization trick: we normalize the gradient - not sure if I should use this
            # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        # this function returns the loss and grads given the input picture
        iterate_op = K.function([encoded_seq, K.learning_phase()], [grads])
        return iterate_op 

    def localize(self, row, genome):
        """ Find the section of a bed file row giving maximum acitvation.

        Arguments:
            row -- bed file row.
            genome -- genome associated with the bed file. 
        Output:
            max_tile -- Sequence object for the maximum predicting 256 bp region.
            max_pred -- prediction value for the row. 
        """
        # break the sequence into overlapping tiles
        input_window=256
        tile_seqs = list()
        num_tiles = int((row['end']-row['start']) / input_window) + ((row['end']-row['start']) % input_window > 0)
        for idx in range(num_tiles):
            try:
                if row['start'] + idx*input_window - input_window//2 > 0:
                    seq = genome[row['chr']][row['start'] + idx*input_window - input_window//2:row['start'] + (idx+1)*input_window - input_window//2].lower()
                    tile_seqs.append(sequence.encode_to_onehot(seq))
                else:
                    buffered_seq = np.zeros((256,4))
                    buffered_seq[:row['start'] + (idx+1)*input_window - input_window//2] = genome[row['chr']][0:row['start'] + (idx+1)*input_window - input_window//2]
                    tile_seqs.append(sequence.encode_to_onehot(buffered_seq))
                seq = genome[row['chr']][row['start'] + idx*input_window:row['start'] + (idx+1)*input_window].lower()
                tile_seqs.append(sequence.encode_to_onehot(seq))
            except ValueError:
                print('Weird value error row here:')
                print(row)
        #configure the tiled sequences
        tile_seqs= np.asarray(tile_seqs)
        tile_iter = iter(tile_seqs)
        # get a batch generator
        batches = train_TFmodel.filled_batch(tile_iter)
        # figure out where the max prediction is coming from
        preds = list()
        for batch in batches:
            try:
                 preds.append(self.model.predict_on_batch(batch))
            except ValueError:
                 print('Weird batch at ' + str(row))
                 print(batch.shape)
                 print(batch)
        preds = np.asarray(preds).reshape((-1))[:tile_seqs.shape[0]]
        # get a tile centered there
        try:
            max_pred = np.max(preds)
            max_tile = tile_seqs[np.argmax(preds)]
        except ValueError:
            print('No maximum pred?')
            print(row)
            print(preds)
        return sequence.Sequence(max_tile), max_pred
        
    def get_importance(self, seq, viz=False, start=None, end=None, plot=False):
        """Generate the gradient based importance of a sequence according to a given model.
        
        Arguments:
             seq -- the Sequence to run through the keras model.
             viz -- sequence logo of importance?
             start -- plot only past this nucleotide.
             end -- plot only to this nucleotide.
             plot -- generate a gain-loss plot?
        Outputs:
             diffs -- difference at each position to score.
             average_diffs -- base by base importance value. 
             masked_diffs -- importance for bases in origonal sequence.
        """
        score = self.get_activation(seq)
        mutant_preds = self.get_activation(seq.ngram_mutant_gen())
        #get the right shape
        mutant_preds = mutant_preds.reshape((-1, 4))[:len(seq.seq)]
        diffs = mutant_preds - score
        # we want the difference for each nucleotide at a position minus the average difference at that position
        average_diffs = list()
        for base_seq, base_preds in zip(seq.seq, mutant_preds):
            this_base = list()
            for idx in range(4):
                this_base.append(base_preds[idx] - np.average(base_preds))
            average_diffs.append(list(this_base))
        average_diffs = np.asarray(average_diffs)
        # masked by the actual base
        masked_diffs = (seq.seq * average_diffs)
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
            temp = .05
            #print('Prediciton Difference')
            #viz_sequence.plot_weights(average_diffs[start:end])
            print('Masked average prediciton difference')
            viz_sequence.plot_weights(masked_diffs[start:end])
            #print('Softmax prediction difference')
            #viz_sequence.plot_weights(helper.softmax(diffs[start:end]))
            print('Information Content of Softmax prediction difference')
            viz_sequence.plot_icweights(helper.softmax(diffs[start:end]/(temp*self.get_activation(seq))))
        return diffs, average_diffs, masked_diffs


    def predict_bed(self, peaks, genome=None):
        """Predict from a bed file.
    
        Arguments:
            peaks -- from the bed file.
        Keywords:
             genome -- default is hg19.
        Outputs:
            preds -- predictions for each row. 
        """
        # get the genome and bed file regions
        if genome == None:
             genome = ucscgenome.Genome('/home/kal/.ucscgenome/hg19.2bit')
        # predict over the rows
        preds = list()
        for index, row in tqdm(peaks.iterrows()):
            tile, pred = self.localize(row, genome)
            preds.append(pred)
        return np.asarray(preds).flatten()

    def predict_snv(self, peaks, genome=None):
        """Predict from a bed file with chr, position, refAllele, altAllele.

        Arguments:
            peaks -- the bed file in pd table form.
        Keywords:
            genome -- default is hg19.
        Outputs:
            refpreds -- predictions for each row with reference allele. 
            altpreds -- predictions for each row with alternate allele. 
        """
        # get the genome and bed file regions
        if genome == None:
             genome = ucscgenome.Genome('/home/kal/.ucscgenome/hg19.2bit')
        # predict over the rows
        refpreds = list()
        batchgen = train_TFmodel.filled_batch(snv_gen(peaks, genome, alt=False))
        for batch in batchgen:
            refpreds.append(self.model.predict_on_batch(batch))
        refpreds = np.asarray(refpreds).flatten()[:len(peaks)]
    
        altpreds = list()
        batchgen = train_TFmodel.filled_batch(snv_gen(peaks, genome, alt=True))
        for batch in batchgen:
            altpreds.append(self.model.predict_on_batch(batch))
        altpreds = np.asarray(altpreds).flatten()[:len(peaks)]
    
        return refpreds, altpreds

def snv_gen(peaks, genome, alt=False):
    """Generate sequnces from snv data.
    
    Arguments:
        peaks -- from a bed file.
        genome -- to pull bed from.
    Keywords:
        alt -- give alternate allele version.
    Returns:
        seq -- sequence with the alternate or refernce allele, centered around the position. """
    for index, row in peaks.iterrows():
        if row.position > 128:
            seq = sequence.encode_to_onehot(genome[row.chr][row.position-128:row.position+128])
            if alt:
                seq[128] = sequence.encode_to_onehot(row.altAllele.lower())
            else:
                seq[128] = sequence.encode_to_onehot(row.refAllele.lower())
        else:
            # sequence too close to begining
            seq = sequence.encode_to_onehot(genome[row.chr][0:256])
            if alt:
                seq[row.position] = sequence.encode_to_onehot(row.altAllele.lower())
            else:
                seq[row.position] = sequence.encode_to_onehot(row.refAllele.lower())
        if seq.shape != (256, 4):
            # seq too close to end?
            print('Sequence at ' +str(row.chr) + ' ' + str(row.position) + ' is too short!')
            offset = 0
            while seq.shape != (256, 4) and offset < 128:
                offset += 1
                seq = sequence.encode_to_onehot(genome[row.chr][row.position-128-offset:row.position+128-offset])
            if alt:
                seq[128-offset] = sequence.encode_to_onehot(row.altAllele.lower())
            else:
                seq[128-offset] = sequence.encode_to_onehot(row.refAllele.lower())
        if seq.shape == (256, 4):
            yield seq
        else:
            print('Sequence at ' +str(row.chr) + ' ' + str(row.position) + ' couldn\'t be fixed')

def group_stats(key, h1, h2, h3):
    # Summarize history for accuracy
    out1 = np.copy(h1[key])
    out2 = np.copy(h2[key])
    out3 = np.copy(h3[key])
    return np.concatenate([out1, out2, out3])
