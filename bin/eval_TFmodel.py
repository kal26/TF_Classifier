import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import tf_memory_limit
#import general use packages
import numpy as np
import matplotlib.pyplot as plt
#import keras related packages
from keras import backend as K
from keras.models import load_model, Model, Input
#import custom packages
import helper
import viz_sequence
import train_TFmodel
import sequence

class TFmodel(object):
    """Transcription factor classification keras model."""

    def __init__(self, model_path, output_path=None):
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
        """
        self.model_path = model_path
        self.model = load_model(os.path.join(model_path, 'final_model.hdf5'), custom_objects={'Bias':train_TFmodel.Bias})
        if output_path != None:
            self.out_path = output_path
        else:
            self.out_path = os.path.join(model_path, 'evaluation')
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers]) 
        self.get_act = K.function([self.model.input, K.learning_phase()], [self.layer_dict['bias'].output])

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
    def gumbel_dream(self, seq, temp=10, layer_name='final_output', filter_index=0, meme_library=None, num_iterations=20, step=None, viz=False):
        """ Dream a sequence for the given number of steps employing the gumbel-softmax reparamterization trick.

      Arguments:
            seq -- SeqDist object to iterate over.
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
        # print the initial sequence
        if viz:
            print('Initial Sequence')
            viz_sequence.plot_icweights(seq.seq)

        # get an gradient grabbing operation
        distribution = Input(tensor=n)
        sampled_seq = train_TFmodel.gumbel_softmax(distribution, temp, hard=True)
        if layer_name == 'final_output':
            activations = self.model.output
        else:
            layer_output = self.layer_dict[layer_name].output
        activations = layer_output[:, :, filter_index] #each batch and nuceotide at this neuron.
        # forward and reverse sequences
        combined_activation = K.mean(np.maximum(activations[:32], activations[32:]))
        # compute the gradient of the input seq wrt this loss
        grads = K.gradients(combined_activation, sampled_seq)[0]
        # average to get the update (sampeling already weights for probability)
        update = K.average(grads, axis=0)
        # this function returns the loss and grads given the input picture
        iterate_op = K.function([distribution, K.learning_phase()], [update])

        #iterate and dream
        dream_seq = sequence.SeqDist(seq.seq)
        for i in range(num_iterations):
            update = iterate_op([dreami_seq.seq, 0])[0]
            dream_seq.seq = dream_seq.seq + update
            if i%(num_iterations//4) == 0 and viz:
                print('Sequence after ' + str(i) + ' iterations')
                viz_sequence.plot_icweights(dream_seq.seq)
               #print the final sequence
        if viz:
            print('Final sequence')
            viz_sequence.plot_icweights(dream_seq.seq)
            self.get_importance(dream_seq, viz=viz)
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
            print('Prediciton Difference')
            viz_sequence.plot_weights(average_diffs[start:end])
            print('Masked average prediciton difference')
            viz_sequence.plot_weights(masked_diffs[start:end])
            print('Softmax prediction difference')
            viz_sequence.plot_weights(helper.softmax(diffs[start:end]))
            print('Information Content of Softmax prediction difference')
            viz_sequence.plot_icweights(helper.softmax(diffs[start:end]))
        return diffs, average_diffs, masked_diffs


def predict_bed(data_path, model, genome, column_names=None):
    """Predict from a bed file.
    
    Arguments:
        data_path -- to the bed file.
    Keywords:
        genome -- default is hg19.
        column_names -- default is chr start end
    Outputs:
        preds -- predictions for each row. 
    """
    peaks = pandas.read_table(data_path, header=None)
    if column_names == None:
        column_names = 'chr start end'
    peaks.columns = column_names.split()

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
                        seq = ctcfgen.encode(np.fromstring(genome[row.chr][row.start:row.end].lower(), 
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
                            seq = ctcfgen.encode(np.fromstring(genome[row.chr][row.start:row.end].lower(), 
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
                            seq = ctcfgen.encode(np.fromstring(genome[row.chr][row.start:row.end].lower(), 
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
    
    preds=[]
    
    print('Expected batch count: ' + str((peaks.shape[0] // batch_size) + (peaks.shape[0] % batch_size > 0)))
    
    for i in range((peaks.shape[0] // batch_size) + (peaks.shape[0] % batch_size > 0)):
        batch = next(g)
        if batch.shape == (32, 256, 4):
            preds.append(model.predict_on_batch(batch))
        else:
            print(batch.shape)

    return preds
