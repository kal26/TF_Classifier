import numpy as np

#helper functions

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm."""
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def rejection(seq_a, seq_b):
    """utiliy function to compute rejection of a from b"""
    out = list()
    for a, b in zip(seq_a, seq_b):
        if np.linalg.norm(b) == 0:
            out.append(a)
        else:
            out.append(a - ((np.dot(a, b) / (np.linalg.norm(b)**2)) * b))
    return np.asarray(out)

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


