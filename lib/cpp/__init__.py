"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import random

import setuptools.sandbox
from multiprocessing import cpu_count
from lib import cpp
import numpy as np
import torch

package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
if not os.path.exists(osp.join(package_abspath, "_bindings.so")):
    # try build _bfs.so
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(
            osp.join(package_abspath, "setup.py"), ["clean", "build"]
        )
        os.system(
            "cp {}/build/lib*/*.so {}/_bindings.so".format(package_abspath, package_abspath)
        )
        assert os.path.exists(osp.join(package_abspath, "_bindings.so"))
    finally:
        os.chdir(workdir)

from . import _bindings
_bindings.set_seed(random.randint(0, 2 ** 16))


def set_seed(random_state):
    """ Sets random state for c++ dijkstra, """
    _bindings.set_seed(random_state)


def batch_dijkstra(
        slices,
        sliced_edges,
        sliced_adjacency_logits,
        sliced_weight_logits,
        initial_vertices,
        target_vertices,
        *,
        k_nearest,
        max_length,
        max_length_nearest=None,
        max_steps=None,
        deterministic=False,
        presample_edges=False,
        soft=False,
        n_jobs=None,
        validate=True,
):
    """
    Batch-parallel dijkstra algorithm in sliced format. This is a low-level function used by GraphEmbedding
    This algorithm accepts graph as sliced arrays of edges, weights and edge probabilities, read more in Note section
    :param slices: int32 vector, offsets for task at each row (see note)
    :param sliced_edges: int32 vector, indices of vertices directly available from vertex v_i (for each v_i)
    :param sliced_adjacency_logits: float32 vector, for each edge, a logit such that edge exists with probability
        p = sigmoid(logit) = 1 / (1 + e^(-logit))
    :param sliced_weight_logits: float32 vector, for each edge, a logit such that edge weight is defined as
        w(edge) = log(1 + exp(logit))
    :param initial_vertices: int32 vector of initial vertex indices (computes path from those)
    :param target_vertices: int32 , either vector  or matrix
        if vector[batch_size], batch of target vertex indices (computes path to those) * Param target_vertices
        if matrix[batch_suize, num_targets], each row corresponds to (multiple) target vertex ids for i-th input
    :param max_length: maximum length of paths to target
    :param k_nearest: number of paths to nearest neighbors, see returns
    :param max_length_nearest: maximum length of paths to nearest neighbors
    :param n_jobs: number of parallel jobs for computing dijkstra, if None use cpu count
    :param soft: if True, absent edges are actually still available if no other path exists
    :param deterministic: if True, edge probabilities over 0.5 are always present and below 0.5 are always ignored
    :param presample_edges: if True, samples edge probabilities in advance.
        Edges sampled as "present" will have logit of float min, others will be float max
    :param max_steps: if not None, terminates search after this many steps
    :param validate: check that all dtypes are correct. If false, runs the function regardless

    Note: sliced array is an 1d array that stores 2d task using the following scheme:
      [0, a_00, a_01, ...,  a_0m, a_10, a_11, ..., a_1n, ..., a_l0, a_l1, ..., a_lk]
          \----first vertex----/  \---second vertex---/       \----last vertex----/
    Slices for this array contain l + 1 elements: [1,   1 + m,   1 + m + n,   ...,   total length]

    :return: paths_to_target, paths_to_nearest
        :paths_to_target: int32 matrix [batch_size, max_length] containing edges padded with zeros
            edges are represented by indices in :sliced_edges:
        :paths_to_nearest: int32 tensor3d [batch_size, k_nearest, max_length_nearest]
            path_to_nearest are NOT SORTED by distance
    """
    n_jobs = n_jobs or cpu_count()
    if n_jobs < 0:
        n_jobs = cpu_count() - n_jobs + 1

    if max_steps is None:
        max_steps = -1

    batch_size = len(initial_vertices)
    max_length_nearest = max_length_nearest or max_length

    if validate:
        for arr in (slices, sliced_edges, sliced_adjacency_logits, sliced_weight_logits,
                    initial_vertices):
            assert isinstance(arr, np.ndarray), "expected np array but got {}".format(type(arr))
            assert arr.flags.c_contiguous, "please make sure array is contiguous (see np.ascontiguousarray)"
            assert arr.ndim == 1, "all arrays must be 1-dimensional"

        assert isinstance(target_vertices, np.ndarray)
        assert arr.flags.c_contiguous, "target paths must be contiguous (see np.ascontiguousarray)"
        assert np.ndim(target_vertices) in (1, 2), "target paths must be of either shape [batch_size] or" \
                                                   "[batch_size, num_targets] (batch_size is len(initial_vertices)"
        assert slices[0] == 1 and slices[-1] == len(sliced_edges)
        assert len(sliced_edges) == len(sliced_adjacency_logits) == len(sliced_weight_logits)
        assert len(initial_vertices) == len(target_vertices) == batch_size
        assert max(np.max(initial_vertices), np.max(target_vertices)) < len(slices) - 1, "vertex id exceeds n_vertices"
        assert slices.dtype == sliced_edges.dtype == np.int32
        assert sliced_adjacency_logits.dtype == sliced_weight_logits.dtype == np.float32
        assert initial_vertices.dtype == target_vertices.dtype == np.int32
        assert max_steps == -1 or max_steps >= k_nearest, "it is impossible to find all neighbors in this many steps"
        assert max_length > 0 and max_length_nearest > 0 and k_nearest >= 0
        assert isinstance(deterministic, bool)

    should_squeeze_target_paths = np.ndim(target_vertices) == 1
    if should_squeeze_target_paths:
        target_vertices = target_vertices[..., np.newaxis]
    target_paths = np.zeros([batch_size, target_vertices.shape[-1], max_length], 'int32')
    nearest_paths = np.zeros([batch_size, k_nearest, max_length_nearest], 'int32')

    if presample_edges:
        edge_logits = sliced_adjacency_logits
        min_value, max_value = np.finfo(edge_logits.dtype).min, np.finfo(edge_logits.dtype).max
        edge_exists = (torch.rand(len(edge_logits)) < torch.sigmoid(torch.as_tensor(edge_logits))).numpy()
        sliced_adjacency_logits = np.where(edge_exists, max_value, min_value)

    _bindings.batch_dijkstra(
        slices, sliced_edges,
        sliced_adjacency_logits,
        sliced_weight_logits,
        initial_vertices, target_vertices,
        target_paths, nearest_paths,
        deterministic, soft, max_steps, n_jobs
    )

    if should_squeeze_target_paths:
        target_paths = target_paths.reshape([batch_size, max_length])

    return target_paths, nearest_paths
