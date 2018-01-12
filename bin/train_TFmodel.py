from itertools import zip_longest, product, chain, repeat
import numpy as np


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


