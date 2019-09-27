from warnings import warn

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from collections import namedtuple
from itertools import chain
import scipy.stats

from .cpp import batch_dijkstra
from .utils import check_numpy, sliced_argmax
from collections import defaultdict
import scipy.sparse, scipy.sparse.csgraph

beta_quantile = lambda x, alpha=0.01, beta=0.5: scipy.stats.beta.ppf(x, alpha, beta)
inverse_softplus = lambda x: np.where(x > 10.0, x, np.log(np.expm1(np.clip(x, 1e-6, 10.0))))
inverse_sigmoid = lambda prob: -np.log(1. / np.clip(prob, 1e-6, 1.0 - 1e-6) - 1)


class GraphEmbedding(nn.Module):
    Edges = namedtuple("Edges", ["adjacent", "p_adjacent", "weights"])
    INF = torch.tensor(1e38, dtype=torch.float32)
    NEG_INF = -INF

    def __init__(self, edges_from, edges_to, *, initial_weights=1.0, initial_probs=0.9,
                 sparse=True, directed=True, default_distance=0.0, default_distance_trainable=True, **defaults):
        """
        Trainable graph embedding, a default class with many options

        :param edges_from: an array of sources of all edges, int[num_edges]
        :param edges_to: an array of destinations of all edges, int[num_edges]
        :param initial_weights: a scalar or array of distances of all edges, float[] or [num_edges]
        :param initial_probs: a scalar or array of probabilities of all edges, float[] or [num_edges]
        :param sparse: if True, gradients w.r.t. layer parameters are gonna be sparse (same as in nn.Embedding)
        :param directed: if False, forall i, j, edge(i, j) and edge(j, i) have shared weight and probability
        :param default_distance: a distance to return if agent was unable to find path to any of target vertices
        :param default_distance_trainable: if True, trains default distance alongside all other distances
        :param defaults: default parameters for forward pass, see lib.cpp.batch_dijkstra
        """
        super().__init__()
        if np.ndim(initial_weights) == 0:
            initial_weights = np.full(edges_from.shape, float(initial_weights), dtype=np.float32)
        if np.ndim(initial_probs) == 0:
            initial_probs = np.full(edges_from.shape, float(initial_probs), dtype=np.float32)
        assert len(edges_from) == len(edges_to) == len(initial_weights) == len(initial_probs)
        assert not (edges_from == edges_to).any(), "graph should not contain loops"
        assert np.min(initial_weights) >= 0.0
        edges_from, edges_to, initial_weights, initial_probs = map(
            check_numpy, [edges_from, edges_to, initial_weights, initial_probs])

        num_vertices = max(edges_from.max(), edges_to.max()) + 1
        initial_adjacency_logits = inverse_sigmoid(initial_probs)
        initial_weight_logits = inverse_softplus(initial_weights)

        adjacency = [list() for _ in range(num_vertices)]
        edge_to_weight_logits, edge_to_adjacency_logits = {}, {}
        for from_i, to_i, adj_logit, weight_logit in zip(
                edges_from, edges_to, initial_adjacency_logits, initial_weight_logits):
            edge = (int(from_i), int(to_i))
            if not directed:
                from_i, to_i = edge = tuple(sorted(edge))
            adjacency[from_i].append(to_i)
            edge_to_adjacency_logits[edge] = adj_logit
            edge_to_weight_logits[edge] = weight_logit

        adjacency = list(map(sorted, adjacency))
        edge_sources = list(chain(*([i] * len(adjacency[i]) for i in range(len(adjacency)))))
        edge_targets = list(chain(*adjacency))
        lengths = list(map(len, adjacency))

        # compute slices and edge_indices, add special "fake" edge at index 0
        self.edge_sources = np.array([0] + edge_sources, dtype=np.int32)
        self.edge_targets = np.array([0] + edge_targets, dtype=np.int32)
        self.slices = np.cumsum([1] + lengths).astype("int32")
        self.edge_adjacency_logits = nn.Parameter(torch.randn(len(self.edge_targets), 1), requires_grad=True)
        self.edge_weight_logits = nn.Parameter(torch.randn(len(self.edge_targets), 1), requires_grad=True)
        self.default_distance = nn.Parameter(torch.tensor(float(default_distance)).view(1, 1),
                                             requires_grad=default_distance_trainable)
        self.num_vertices, self.num_edges = num_vertices, len(self.edge_targets)

        # initialize weight and adjacency logits
        with torch.no_grad():
            flat_i = 1  # start from 1 to skip first "fake" edge
            for from_i, to_ix in enumerate(adjacency):
                for to_i in to_ix:
                    edge = (int(from_i), int(to_i))
                    self.edge_adjacency_logits.data[flat_i, 0] = float(edge_to_adjacency_logits[edge])
                    self.edge_weight_logits.data[flat_i, 0] = float(edge_to_weight_logits[edge])
                    flat_i += 1

        self.defaults = defaults
        self.directed = directed
        self.sparse = sparse

        if not directed:
            # prepare a special arrays that convert undirected edges to directed by duplicating them
            directed_edges_by_source = defaultdict(set)
            directed_edge_to_ix = {}

            for i, from_i, to_i in zip(np.arange(self.num_edges), self.edge_sources, self.edge_targets):
                if i == 0: continue
                directed_edges_by_source[from_i].add(to_i)
                directed_edges_by_source[to_i].add(from_i)
                directed_edge_to_ix[from_i, to_i] = i
                directed_edge_to_ix[to_i, from_i] = i

            directed_edges = [0]
            directed_slices = [1]
            directed_to_undirected_reorder = [0]
            for from_i in range(self.num_vertices):
                directed_edges_from_i = sorted(directed_edges_by_source[from_i])
                directed_slices.append(directed_slices[-1] + len(directed_edges_from_i))
                directed_edges.extend(directed_edges_from_i)
                directed_to_undirected_reorder.extend(
                    [directed_edge_to_ix[from_i, to_i] for to_i in directed_edges_from_i])

            self.directed_edge_indices = np.array(directed_edges, dtype='int32')
            self.directed_slices = np.array(directed_slices, dtype='int32')
            self.reorder_undirected_to_directed = torch.as_tensor(directed_to_undirected_reorder, dtype=torch.int64)

    def _get_logits(self, sliced_logits, sliced_indices):
        """ A private helper function that returns logits of corresponding to indices with (maybe) sparse grad """
        if not self.directed:
            sliced_indices = self.reorder_undirected_to_directed[sliced_indices]
        return F.embedding(sliced_indices, sliced_logits, sparse=self.sparse).view(*sliced_indices.shape)

    def _get_default_distance(self):
        """ A special magic that returns default distance in a way that gradients wrt that distance will be sparse"""
        return F.embedding(torch.zeros(1, dtype=torch.int64), self.default_distance, sparse=self.sparse).view([])

    def forward(self, from_ix, to_ix, **parameters):
        """
        Computes path from from_ix to to_ix, possibly returns a few other values
        :param from_ix: a vector of initial vertex indices, int32[batch_size]
        :param to_ix: a vector or matrix of target vertex indices, int32[batch_size] or [batch_size, num_targets]
        :param max_length: maximum length of paths (edges beyond this length are discarded, starting from from_ix)
        :param k_nearest: if given, returns distances and paths to this many nearest neighbors of each vertex
        :param deterministic: if True, always keeps edges with p > 0.5 and drops edges with p < 0.5, else samples;
            if not specified(default), uses deterministic=True in train mode, False in eval mode
        :param parameters: see lib.cpp.batch_dijkstra
        :return: a dict of many things {
            target_paths: from_ix->to_ix edge sequences from end to start, int32[*to_ix.shape, max_length]
            target_distances: sum of edge weights along path to each target (if exists), float32[*to_ix.shape]
            logp_target_paths: sum of edge log-probs along path to each target (if exists), float32[*to_ix.shape]
            found_target: whether pathfinding was successful, uint8[*to_ix.shape]
                if found_target[i] is 0, target_paths[i] and logp_target_paths[i] will be zeros,
                target_distances equals default_distance

            (v-- if k_nearest != 0, otherwise None)
            nearest_vertices: vertex ids of k_nearest neighbors of each vertex in from_ix, excluding oneself
            nearest_paths: edge sequences between from_ix and k_nearest vertices, pads with zeros if can't find enough
                shape int32[batch_size, k_nearest, max_length_nearest or max_length]
            nearest_distances: sum of edge weights on path to each neighbor, zero pad, float32[batch_size, k_nearest]
        }
        """
        # set defaults
        parameters = dict(self.defaults, **parameters)
        assert parameters.get('max_length') is not None, "Please specify max_length in either init or forward"
        parameters['deterministic'] = parameters.get('deterministic', not self.training)
        parameters['k_nearest'] = parameters.get('k_nearest', 0)

        assert from_ix.device == to_ix.device == torch.device('cpu'), "gpu not supported (yet)"
        from_ix = from_ix.to(dtype=torch.int32)
        to_ix = to_ix.to(dtype=torch.int32)

        # make sure padding edge has weight 0 and probability 1
        with torch.no_grad():
            self.edge_adjacency_logits.data[:1].fill_(self.INF)
            self.edge_weight_logits.data[:1].fill_(self.NEG_INF)

        if self.directed:
            slices, edge_indices = self.slices, self.edge_targets
            edge_adjacency_logits = self.edge_adjacency_logits
            edge_weight_logits = self.edge_weight_logits
        else:
            slices, edge_indices = self.directed_slices, self.directed_edge_indices
            edge_adjacency_logits = F.embedding(
                torch.as_tensor(self.reorder_undirected_to_directed),
                self.edge_adjacency_logits, sparse=self.sparse
            )  # [num_edges, 1]
            edge_weight_logits = F.embedding(
                torch.as_tensor(self.reorder_undirected_to_directed),
                self.edge_weight_logits, sparse=self.sparse
            )  # [num_edges, 1]

        target_paths, nearest_paths = batch_dijkstra(
            slices, edge_indices,
            edge_adjacency_logits.data.numpy().flatten(),
            edge_weight_logits.data.numpy().flatten(),
            from_ix.data.numpy(), to_ix.data.numpy(),
            **parameters
        )

        target_paths = torch.as_tensor(target_paths, dtype=torch.int64)  # [batch_size, max_length]
        target_distances = F.softplus(self._get_logits(self.edge_weight_logits, target_paths)).sum(dim=(-1))
        logp_target_paths = -F.softplus(-self._get_logits(self.edge_adjacency_logits, target_paths)).sum(dim=(-1))
        # ^--[batch_size, num_targets]

        # handle paths that are not found
        not_found_target = target_paths[..., 0] == 0
        if torch.any(not_found_target):
            is_not_loop = (from_ix[:, None] != to_ix.reshape(to_ix.shape[0], -1)).reshape(to_ix.shape)
            not_found_target = not_found_target & is_not_loop

        target_distances = torch.where(not_found_target, self._get_default_distance(), target_distances)

        if parameters['k_nearest'] != 0:
            nearest_paths = torch.as_tensor(np.copy(nearest_paths), dtype=torch.int64)
            nearest_distances = F.softplus(self._get_logits(self.edge_weight_logits, nearest_paths)).sum(dim=(-1))
            nearest_vertices = edge_indices[nearest_paths[..., 0]]
            # ^--[batch_size, k_nearest]
        else:
            nearest_paths = nearest_distances = nearest_vertices = None

        return dict(
            target_paths=target_paths,
            target_distances=target_distances,
            logp_target_paths=logp_target_paths,
            found_target=~not_found_target,

            nearest_paths=nearest_paths,
            nearest_distances=nearest_distances,
            nearest_vertices=nearest_vertices,
        )

    def get_edges(self, vertex):
        """ Returns information on edges corresponding to a single vertex(int scalar) """
        if self.directed:
            slices, edge_indices = self.slices, self.edge_targets
        else:
            slices, edge_indices = self.directed_slices, self.directed_edge_indices

        begin_i, end_i = slices[vertex], slices[vertex + 1]
        edge_span = torch.arange(begin_i, end_i, dtype=torch.int64,
                                 device=self.edge_adjacency_logits.device)

        return self.Edges(
            torch.as_tensor(edge_indices[begin_i: end_i]),
            torch.sigmoid(self._get_logits(self.edge_adjacency_logits, edge_span)),
            F.softplus(self._get_logits(self.edge_weight_logits, edge_span)),
        )

    def compute_l0_prior_penalty(self, lambd=1.0, free_edge=False, start_i=1, end_i=None, batch_size=None):
        """
        Computes negative L0_prior = - mean(1 - P(edge))
        :param lambd: scalar regularization coefficient, see https://arxiv.org/abs/1712.01312 for details
        :param free_edge: if True, the most likely edge of every vertex is not penalized
        :param batch_size: if specified, averages over this many random edges (default: all edges)
        """
        device = self.edge_adjacency_logits.device
        total_edges = (self.num_edges if self.directed else len(self.directed_edge_indices))
        end_i = end_i or total_edges
        regularized_indices = np.arange(start_i, end_i)

        if free_edge:
            if self.directed:
                logits = self.edge_adjacency_logits.data.numpy()
                slices = self.slices
            else:
                logits = self.edge_adjacency_logits.data.numpy()[self.reorder_undirected_to_directed]
                slices = self.directed_slices

            argmax_indices = slices[:-1] + sliced_argmax(logits.flatten(), slices)
            is_argmax = np.in1d(regularized_indices, argmax_indices)
            regularized_indices = regularized_indices[~is_argmax]

        if batch_size is not None:
            batch = torch.randint(0, len(regularized_indices), (batch_size,), device=device)
            regularized_indices = torch.as_tensor(regularized_indices)[batch]

        p_keep_edge = torch.sigmoid(self._get_logits(self.edge_adjacency_logits, regularized_indices))
        return lambd * p_keep_edge.mean()

    def compute_hierarchical_prior_penalty(self, *, nonzero_rate, start_i=1, end_i=None, smoothness=0.0, batch_size=None):
        """
        Computes quantile-wise penalty approximading EMD between the empirical distribution of edges and
        - in case smoothness == 0, bernoulli(nonzero_rate, 1 - nonzero_rate)
        - in case smoothness >0, beta(smoothness * nonzero_rate, smoothness * (1 - nonzero_rate))
        """
        assert 0 <= nonzero_rate <= 1 and smoothness >= 0
        device = self.edge_adjacency_logits.device
        end_i = end_i or self.num_edges

        if batch_size is None:
            logits_sorted = torch.sort(self.edge_adjacency_logits[start_i:end_i, 0])[0]
            batch_size = len(logits_sorted)
        else:
            ix = torch.randint(start_i, end_i, (batch_size,), device=device)
            logits_sorted = torch.sort(F.embedding(ix, self.edge_adjacency_logits, sparse=self.sparse).view(-1))[0]

        if smoothness != 0:
            alpha, beta = smoothness * nonzero_rate, smoothness * (1 - nonzero_rate)
            qq = np.linspace(0, 1, num=batch_size)
            reference = torch.as_tensor(beta_quantile(qq, alpha=alpha, beta=beta),
                                        device=device, dtype=torch.float32)
        else:
            reference = torch.zeros(batch_size, device=device, dtype=torch.float32)
            reference[-max(1, int(round(nonzero_rate * batch_size))):] = 1.0

        return F.mse_loss(reference, torch.sigmoid(logits_sorted))

    def compute_pairwise_distances(self, indices=None, edge_threshold=0.5, default=None):
        """
        Computes distances between all pairs of points,
        :param edge_threshold: minimal probability of edge for it to be kept
        :param indices: if given, comptues distances from only those indices (but TO all indices)
        :param default: use this distance if no path exists between two vertices (default=inf)
        :returns: distances matrix [len(indices) or num_vertices, num_vertices]
        """
        if indices is None:
            indices = np.arange(self.num_vertices)

        edges = defaultdict(lambda: float('inf')) # {(from, to) -> weight}
        for ix in range(self.num_vertices):
            adj, p_adj, weights = self.get_edges(ix)
            adj = adj[p_adj >= edge_threshold].detach().numpy()
            weights = weights[p_adj >= edge_threshold].detach().numpy()
            for (target, weight) in zip(adj, weights):
                edges[int(ix), int(target)] = min(edges[int(ix), int(target)], weight)

        from_to, weights = zip(*edges.items())
        froms, tos = zip(*from_to)
        sparse_edges = scipy.sparse.coo_matrix((weights, (froms, tos)), shape=[self.num_vertices, self.num_vertices])
        distances = scipy.sparse.csgraph.dijkstra(sparse_edges, directed=self.directed, indices=indices)
        if default is None:
            default = float(self._get_default_distance().item())
        distances[np.isinf(distances)] = default
        return distances

    def pruned(self, threshold=0.5):
        """
        Prunes graph edges by their adjacency probabilities
        :param threshold: prunes all edges that have probability less than threshold
        :return: GraphEmbedding
        """
        probs = torch.sigmoid(self.edge_adjacency_logits[1:, 0]).data.numpy()
        saved_mask = probs >= threshold

        edges_from = self.edge_sources[1:]
        edges_to = self.edge_targets[1:]
        weights = F.softplus(self.edge_weight_logits[1:, 0]).data.numpy()

        return GraphEmbedding(edges_from[saved_mask], edges_to[saved_mask], initial_weights=weights[saved_mask],
                              initial_probs=probs[saved_mask], sparse=self.sparse, directed=self.directed,
                              default_distance=self._get_default_distance().item(),
                              default_distance_trainable=self.default_distance.requires_grad,
                              **self.defaults)

    def report_model_size(self, threshold=0.5, bits_per_float=32, bits_per_int=32):
        """
        Reports model size
        :param threshold: prunes all edges that have probability less than threshold
        :return: a dict with 'size_bits', 'num_parameters' and a few other statistics
        """
        num_edges = check_numpy(
            torch.sigmoid(self.edge_adjacency_logits.flatten()[1:]) >= threshold
        ).astype('int64').sum()
        num_slices = len(self.slices)
        num_vertices = num_slices + 1
        trainable_default = int(self.default_distance.requires_grad)
        num_parameters = num_vertices + 2 * num_edges + trainable_default
        size_bits = (num_edges + num_vertices) * bits_per_int + (num_edges + trainable_default) * bits_per_float
        return locals()

    def extra_repr(self):
        edges_kept = np.sum(check_numpy(self.edge_adjacency_logits >= 0).astype('int64'))
        return "{} vertices, {} edges total, {} edges kept, {:.5} sparsity rate, default distance = {}".format(
            self.num_vertices, self.num_edges, edges_kept, 1. - edges_kept / self.num_edges,
            self.default_distance.item(),
        )


class InstanceEmbedding(nn.Module):
    DIRECTIONS = dict(REAL_TO_VIRTUAL=0, VIRTUAL_TO_REAL=1)

    def __init__(self, graph_embedding: GraphEmbedding, *, virtual_vertices, direction=None):
        """
        This is a graph-based embedding that maps individual indices into a vector of distances to k special nodes
        :param graph_embedding: GraphEmbedding containing with (n + v) vertices,
            where n = number of input indices (max index + 1) and v = :virtual_vertices:
        :param virtual_vertices: the number of (pre existing) special vertices used as targets when computing distance
        :param direction: either "REAL_TO_VIRTUAL" or "VIRTUAL_TO_REAL" or 0
            "REAL_TO_VIRTUAL" - runs dijkstra from input vertices to virual ones
            "VIRTUAL_TO_REAL" - runs dijkstra from virtual vertices to given input vertices
            None - defaults to whichever direction is more computationally efficient
                (REAL_TO_VIRTUAL if batch size > virtual_vertices else VIRTUAL_TO_REAL)
        """
        super().__init__()
        self.emb = graph_embedding
        self.virtual_vertices = virtual_vertices
        self.num_embeddings = graph_embedding.num_vertices - virtual_vertices
        self.direction = self.DIRECTIONS.get(direction, direction)

    def forward(self, input_ix, direction=None, **params):
        """
        :param input_ix: input indices to embed, int32[...]
        :param params: see lib.cpp.batch_dijkstra
        :param direction: override for direction provided in __init__
        :return: float32[..., virtual_vertices], distances from every input index to each of virtual vertices
        """
        virtual_ix = torch.arange(
            self.emb.num_vertices - self.virtual_vertices,
            self.emb.num_vertices,
            dtype=torch.int64,
        )

        # get all unique vertices
        unique_ix, reorder = torch.unique(input_ix, return_inverse=True)
        num_unique, num_hidden = unique_ix.shape[0], self.virtual_vertices

        # infer search direction (input -> virtual or virtual -> input)
        direction = direction or self.direction
        if direction is None:
            direction = self.DIRECTIONS['REAL_TO_VIRTUAL'] if num_unique <= num_hidden else self.DIRECTIONS[
                'VIRTUAL_TO_REAL']
            if self.emb.directed:
                direction_name = [name for name, i in self.DIRECTIONS.items() if i == direction][0]
                warn("InstanceEmbedding will default to direction = {} (embedding = {})".format(direction_name, self))
                self.direction = direction

        if direction == self.DIRECTIONS['REAL_TO_VIRTUAL']:
            from_ix = unique_ix
            to_ix = virtual_ix.repeat(len(from_ix), 1)
            pred = self.emb(from_ix, to_ix, **params)
            target_distances_flat = pred["target_distances"]
            # ^-- [num_unique, num_hidden]
            target_distances = target_distances_flat[reorder]

        elif direction == self.DIRECTIONS['VIRTUAL_TO_REAL']:
            from_ix = virtual_ix
            to_ix = unique_ix.repeat(len(from_ix), 1)  # [num_hidden, num_unique]
            pred = self.emb(from_ix, to_ix, **params)
            target_distances_flat = pred["target_distances"]
            # ^-- [num_hidden, num_unique]
            target_distances = target_distances_flat.t()[reorder]

        else:
            raise IndexError("Unknown direction {}".format(direction))

        self._cached_pred = pred
        return target_distances
