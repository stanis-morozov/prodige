"""
Baseline GloVe as in https://github.com/maciejkula/glove-python/
"""

import numba
import glove

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.sparse_adam import SparseAdam

from lib import GraphEmbedding


class GloveTrainer(nn.Module):
    def __init__(self, corpora: glove.Corpus, distance_embedding: GraphEmbedding, *,  alpha=0.75, max_count=100,
                 max_targets=10_000, bias_initial_std=0.01, **optimizer_kwargs):
        """
        Helper class to train glove model on a pre-computed cooccurence matrix
        :type corpora: glove.Corpus
        :type distance_embedding: GraphEmbedding
        :param alpha: power for glove weights
        :param max_count: frequency for glove weights after which weight is set to 1.0
        :param max_targets: default max number of targets per row in batch
        :param bias_initial_std: standard deviation for normal init of biases
        :param optimizer_kwargs: see SparseAdam parameters
        """
        super().__init__()
        self.alpha, self.max_count, self.max_targets = alpha, max_count, max_targets
        num_embeddings = len(corpora.dictionary)

        self.corpora = corpora
        self.cooc_matrix = (corpora.matrix + corpora.matrix.T).tocsr()
        self.cooc_matrix.sort_indices()

        weight_matrix = self.cooc_matrix.copy()
        weight_matrix.data = self.cooc_to_weight(weight_matrix.data)
        weight_matrix.sort_indices()

        self.row_total_weights = np.asarray(weight_matrix.sum(axis=-1)).reshape(-1)
        self.row_num_nonzeroes = np.diff(weight_matrix.indptr)
        self.row_probs = self.row_total_weights / np.sum(self.row_total_weights)

        self.token_to_ix = corpora.dictionary
        self.ix_to_token = {i: token for token, i in self.token_to_ix.items()}

        self.distance_embedding = distance_embedding
        self.biases = nn.Embedding(num_embeddings, 1, sparse=distance_embedding.sparse)
        nn.init.normal_(self.biases.weight, std=bias_initial_std)

        self.optimizer = SparseAdam(self.parameters(), **optimizer_kwargs)
        self.step = 0

    def cooc_to_weight(self, cooc):
        """ Convert raw co-occurence to sample weights """
        return np.minimum(1.0, (cooc / self.max_count)) ** self.alpha

    @staticmethod
    @numba.jit(nopython=True)
    def _sample_nonzeroes(matrix_values, matrix_indptr, matrix_indices, max_elems_per_row=10_000):
        """
        Sample up to :max_elems_per_row: nonzero indices for each row
        in sparse csr matrix defined by indptr and indices
        """
        num_rows = len(matrix_indptr) - 1
        output_indices = np.full((num_rows, max_elems_per_row), -1, dtype=np.int32)
        output_values = np.full((num_rows, max_elems_per_row), 0, dtype=np.float32)

        for row_i in range(num_rows):
            indices = matrix_indices[matrix_indptr[row_i]: matrix_indptr[row_i + 1]]
            values = matrix_values[matrix_indptr[row_i]: matrix_indptr[row_i + 1]]

            if len(indices) > max_elems_per_row:
                selector = np.random.choice(len(indices), replace=False, size=max_elems_per_row)
                indices = indices[selector]
                values = values[selector]

            output_indices[row_i, :len(indices)] = indices
            output_values[row_i, :len(values)] = values
        return output_indices, output_values

    def form_batch(self, batch_ii=None, batch_size=None, replace=False, max_elems_per_row=None):
        """ Sample training batch for given rows """
        assert (batch_ii is None) != (batch_size is None), "please provide either batch_ii or batch_size but not both"
        max_elems_per_row = max_elems_per_row or self.max_targets
        if batch_ii is None:
            batch_ii = np.random.choice(len(self.row_probs), size=batch_size, replace=replace, p=self.row_probs)

        batch_ii_repeated = np.repeat(batch_ii[:, None], repeats=max_elems_per_row, axis=1)
        cooc_rows = self.cooc_matrix[batch_ii]
        batch_jj, batch_cooc = self._sample_nonzeroes(
            cooc_rows.data, cooc_rows.indptr, cooc_rows.indices,
            max_elems_per_row=max_elems_per_row or self.max_targets)
        batch_jj_mask = batch_jj != -1
        batch_jj_masked = np.where(batch_jj_mask, batch_jj, batch_ii_repeated)

        glove_weights = self.cooc_to_weight(batch_cooc)
        sample_weights = glove_weights / (glove_weights * batch_jj_mask).sum(-1, keepdims=True)

        sample_weights = np.where(batch_jj_mask, sample_weights, 0)
        targets = np.log(np.where(batch_jj_mask, batch_cooc, 1))

        batch = dict(
            row_indices=batch_ii, col_matrix=batch_jj_masked, mask=batch_jj_mask, cooc=batch_cooc,
            glove_weights=glove_weights, sample_weights=sample_weights, targets=targets
        )
        return {key: torch.as_tensor(value) for key, value in batch.items()}

    def prune_edges(self, threshold=0.5, reset_optimizer=True, **optimizer_kwargs):
        """ Prune edges that are below :threshold: probability, create new optimizer for remaining edges """
        self.distance_embedding = self.distance_embedding.pruned(threshold=threshold)
        if reset_optimizer:
            self.optimizer = SparseAdam(self.parameters(), **optimizer_kwargs)
        return self
