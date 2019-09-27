import contextlib
from warnings import warn
import numpy as np
import torch
import torch.nn as nn


def nop(x):
    return x


@contextlib.contextmanager
def nop_ctx():
    yield None


try:
    import numba
    maybe_jit = numba.jit
except Exception as e:
    warn("numba not found or failed to import, some operations may run slowly;\n"
         "Error message: {}".format(e))
    maybe_jit = nop


def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1,
                        allow_incomplete=True, callback=nop):
    """
    Iterates minibatches from an array of tensors/numpy arrays
    :param tensors: one or several arrays each 3
        Each tensor can be np.ndarray, torch.Tensor or any object that supports array indexing: tensor[[1,2,3]]

    :param batch_size: maximum number of rows in each output
    :param allow_incomplete: affects last batch if number of rows is not divisible by batch size,
        if True, the last batch will have less elements than others
        if False, the last batch will be dropped
    :param shuffle: if True, shuffles the order of data before every epoch (not inplace)
    :param epochs: number of full cycles over data before the iterator terminates
    :param callback: wrapper for range of batches (e.g. tqdm.tqdm_notebook)
    """
    indices = np.arange(len(tensors[0]))
    upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """
    Computes output by applying batch-parallel function to large task tensor in chunks
    :param function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...]
    :param args: one or many tensors, each [num_instances, ...]
    :param batch_size: maximum chunk size processed in one go
    :param out: memory buffer for out, defaults to torch.zeros of appropriate size and type
    :returns: function(task), computed in a memory-efficient way
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


@maybe_jit
def sliced_argmax(inp, slices, out=None):
    if out is None:
        out = np.full(len(slices) - 1, -1, dtype=np.int64)
    for i in range(len(slices) - 1):
        if slices[i] == slices[i + 1]: continue
        out[i] = np.argmax(inp[slices[i]: slices[i + 1]])
    return out


class Lambda(nn.Module):
    def __init__(self, func):
        """ A convenience module that applies a given function during forward pass """
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


