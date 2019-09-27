import numpy as np
from tqdm import tqdm
import glove
from multiprocessing import cpu_count

from ..compression import make_graph_from_vectors
from ... import iterate_minibatches, nop, GraphEmbedding
from glove import Corpus


def make_graph_from_corpora(
        corpora: Corpus, knn_edges, random_edges=0, directed=False, deduplicate=True, squared=False,
        lambd=1.0, batch_size=512, knn_indexes=1, knn_clusters=1, verbose=False, **kwargs):
    """
    Initializes graph by connecting words with highest cosine distances between contexts
    :param corpora: Corpora from glove_python
    :param knn_edges: number of nearest edges per word in corpora
    :param random_edges: number of random edges per word in corpora
    :param lambd: smoothing parameter for PMI scores
    :param deduplicate: if True (recommended), removes duplicate edges from graph
    :param directed: if enabled, treats (i, j) and (j, i) as the same edge
    :param verbose: if enabled, prints progress into stdout
    :param kwargs: other keyword args sent to :GraphEmbedding.__init__:
    :rtype: GraphEmbedding
    """
    assert isinstance(corpora, glove.corpus.Corpus)
    num_words = corpora.matrix.shape[0]

    # obtain symmetric matrices
    cooc_matrix = (corpora.matrix + corpora.matrix.T)
    freq_matrix = cooc_matrix.multiply(1. / (cooc_matrix.sum(axis=1) + lambd))
    freq_matrix = freq_matrix.tocsr()
    freq_matrix.sort_indices()

    edges_from, edges_to, distances = [], [], []

    if knn_edges:
        print("Searching for nearest neighbors")
        import pysparnn.cluster_index as ci
        index = ci.MultiClusterIndex(freq_matrix, np.arange(num_words), num_indexes=knn_indexes)

        neighbor_ix = np.zeros([num_words, knn_edges], 'int64')
        neighbor_distances = np.zeros([num_words, knn_edges], 'float64')
        for batch_ix, batch_freqs in iterate_minibatches(
                np.arange(num_words), freq_matrix, batch_size=batch_size,
                allow_incomplete=True, shuffle=False, callback=tqdm if verbose else nop):
            batch_knn = index.search(batch_freqs, k=knn_edges + 1, k_clusters=knn_clusters, return_distance=True)
            batch_distances, batch_neighbors = zip(*map(lambda row: zip(*row), batch_knn))
            neighbor_ix[batch_ix, :] = np.array(batch_neighbors, dtype='int64')[:, 1:]
            neighbor_distances[batch_ix, :] = np.array(batch_distances, dtype='float64')[:, 1:]

        if verbose: print("Adding knn edges")
        for from_i, (to_ix, batch_distances) in enumerate(zip(neighbor_ix, neighbor_distances)):
            for to_i, distance in zip(to_ix, batch_distances):
                if from_i == to_i: continue
                edges_from.append(from_i)
                edges_to.append(to_i)
                distances.append(distance)

    if random_edges:
        if verbose: print("Adding random edges")
        random_from = np.random.randint(0, num_words, num_words * random_edges)
        random_to = np.random.randint(0, num_words, num_words * random_edges)
        random_dots = np.asarray(freq_matrix[random_from].multiply(freq_matrix[random_to]).sum(-1)).reshape(-1)
        word_freq_norms = np.asarray(np.sqrt(freq_matrix.multiply(freq_matrix).sum(-1))).reshape(-1)
        random_distances = 1. - random_dots / (word_freq_norms[random_from] * word_freq_norms[random_to])

        for vertex_i, neighbor_i, distance in zip(random_from, random_to, random_distances):
            if vertex_i != neighbor_i:
                edges_from.append(vertex_i)
                edges_to.append(neighbor_i)
                distances.append(distance)

    if deduplicate:
        if verbose: print("Deduplicating edges")
        unique_edges_dict = {}  # {(from_i, to_i) : distance(i, j)}
        for from_i, to_i, distance in zip(edges_from, edges_to, distances):
            edge_iijj = int(from_i), int(to_i)
            if not directed:
                edge_iijj = tuple(sorted(edge_iijj))
            unique_edges_dict[edge_iijj] = distance

        edges_iijj, distances = zip(*unique_edges_dict.items())
        edges_from, edges_to = zip(*edges_iijj)

    edges_from, edges_to, distances = map(np.asanyarray, [edges_from, edges_to, distances])
    if verbose:
        print("Total edges: {}, mean edges per vertex: {}, mean distance: {}".format(
            len(edges_from), len(edges_from) / float(num_words), np.mean(distances)
        ))
    if not squared:
        distances = distances ** 0.5
    return GraphEmbedding(edges_from, edges_to, initial_weights=distances, directed=directed, **kwargs)


def make_graph_from_glove(corpora: Corpus, n_components=30, random_state=42, epochs=100, n_jobs=-1,
                          verbose=False, normalize=True, **kwargs):
    """
    Initializes graph by training GloVe and connecting nearest words
    :param corpora: Corpus from glove_python package
    :param n_components: GloVe vector dimensionality
    :param random_state: GloVe random state
    :param epochs: GloVe epochs
    :param n_jobs: number of threads used for both initialization and distance computation
    :param verbose: if True, prints progress in stdout
    :param normalize: whether to normalize word vectors before computing distances
    :param kwargs: see make_graph_from_vertices
    :return:
    """
    if n_jobs < 0:
        n_jobs = cpu_count() + 1 - n_jobs

    base_model = glove.Glove(n_components, random_state=random_state)
    if verbose:
        print("Training base GloVe model...")
    base_model.fit(corpora.matrix, epochs=epochs, no_threads=n_jobs, verbose=verbose)
    base_model.add_dictionary(corpora.dictionary)

    X = base_model.word_vectors
    if normalize:
        X = X / np.square(X).sum(-1, keepdims=True) ** 0.5

    emb = make_graph_from_vectors(
        X, n_jobs=n_jobs, verbose=verbose, **kwargs
    )

    return emb
