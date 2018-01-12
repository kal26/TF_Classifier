import numpy as np

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
