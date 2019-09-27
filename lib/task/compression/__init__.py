from warnings import warn

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ... import check_numpy, GraphEmbedding


def make_graph_from_vectors(X, *, knn_edges, random_edges=0, virtual_vertices=0,
                            deduplicate=True, directed=True, verbose=False, squared=True, **kwargs):
    """
    Creates graph embedding from an object-feature matrix,
    initialize weights with squared euclidian distances |x_i - x_j|^2_2

    The graph consists of three types of edges:
        * knn edges - connecting vertices to their nearest neighbors
        * random edges - connecting random pairs of vertices to get smallworld property
        * edges to virtual_vertices - adds synthetic vertices to task and connect with all other vertices
                                     (init with k-means)

    :param X: task matrix[num_vertors, vector_dim]
    :param knn_edges: connects vertex to this many nearest neighbors
    :param random_edges: adds this many random edges per vertex (long edges for smallworld property)
    :param virtual_vertices: adds this many new vertices connected to all points, initialized as centroids
    :param deduplicate: if enabled(default), removes all duplicate edges
        (e.g. if the edge was first added via :m:, and then added again via :random_rate:
    :param directed: if enabled, treats (i, j) and (j, i) as the same edge
    :param verbose: if enabled, prints progress into stdout
    :param squared: if True, uses squared euclidian distance, otherwise normal euclidian distance
    :param kwargs: other keyword args sent to :GraphEmbedding.__init__:
    :rtype: GraphEmbedding
    """
    num_vectors, vector_dim = X.shape
    X = np.require(check_numpy(X), dtype=np.float32, requirements=['C_CONTIGUOUS'])
    if virtual_vertices != 0:
        if verbose: print("Creating virtual vertices by k-means")
        X_clusters = KMeans(virtual_vertices).fit(X).cluster_centers_
        X = np.concatenate([X, X_clusters])

    if verbose:
        print("Searching for nearest neighbors")
    try:
        from faiss import IndexFlatL2
        index = IndexFlatL2(vector_dim)
        index.add(X)
        neighbor_distances, neighbor_indices = index.search(X, knn_edges + 1)
    except ImportError:
        warn("faiss not found, using slow knn instead")
        neighbor_distances, neighbor_indices = NearestNeighbors(n_neighbors=knn_edges + 1).fit(X).kneighbors(X)

    if verbose:
        print("Adding knn edges")
    edges_from, edges_to, distances = [], [], []
    for vertex_i in np.arange(num_vectors):
        for neighbor_i, distance in zip(neighbor_indices[vertex_i], neighbor_distances[vertex_i]):
            if vertex_i == neighbor_i: continue  # forbid loops
            if neighbor_i == -1: continue  # ANN engine uses -1 for padding
            if not squared: distance **= 0.5
            edges_from.append(vertex_i)
            edges_to.append(neighbor_i)
            distances.append(distance)

    if random_edges != 0:
        if verbose: print("Adding random edges")
        random_from = np.random.randint(0, num_vectors, num_vectors * random_edges)
        random_to = np.random.randint(0, num_vectors, num_vectors * random_edges)
        for vertex_i, neighbor_i in zip(random_from, random_to):
            if vertex_i != neighbor_i:
                distance = np.sum((X[vertex_i] - X[neighbor_i]) ** 2)
                if not squared: distance **= 0.5
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
            len(edges_from), len(edges_from) / float(num_vectors), np.mean(distances)
        ))
    return GraphEmbedding(edges_from, edges_to, initial_weights=distances, directed=directed, **kwargs)
