import numpy as np
from lib import GraphEmbedding


def make_graph_colaborative(matrix, *, knn_edges, knn_drop_rate=0, score_to_distance=lambda x: 1. / x,
                            directed=False, deduplicate=True, eps=1e-10, verbose=False, **kwargs):
    """
    Builds initial graph embedding for colaborative fitlering from sparse user-item matrix
    The initial graph contains three edges:
     * user-user edges between similar users (by cosine distance)
     * item-item edges between similar items (by cosine distance)
     * user-item edges from matrix, edge_weight(i,j) = score_to_distance(matrix[i, j])

    :param matrix: user-iterm matrix[num_users, num_items]
    :param knn_edges: base vertex degree, connects vertex to this many nearest neighbors
    :param knn_drop_rate: drops this fraction of :m: edges at random
    :param score_to_distance: mapping from matrix[i, j] to edge distance, defaults to 1. / matrix[i, j]
    :param directed: if enabled, treats (i, j) and (j, i) as the same edge
    :param deduplicate: if enabled(default), removes all duplicate edges
        (e.g. if the edge was first added via :m:, and then added again via :random_rate:
    :param verbose: if enabled, prints progress into stdout
    :param kwargs: other keyword args sent to :GraphEmbedding.__init__:
    :rtype: GraphEmbedding

    """
    num_users, num_items = matrix.shape

    # user to user, item to item
    if verbose: print("Mining nearest neighbors...")
    edges_from_user, edges_to_user, distances_user = get_knn_edges_cosine(
        matrix.astype(np.float32), k=knn_edges, knn_drop_rate=knn_drop_rate, eps=eps)

    edges_from_item, edges_to_item, distances_item = get_knn_edges_cosine(
        matrix.T.astype(np.float32), k=knn_edges, knn_drop_rate=knn_drop_rate, eps=eps)

    # user to item
    if verbose: print("Assembling user-item edges...")
    usr_from, item_to = matrix.nonzero()
    distances_user_item = np.array([score_to_distance(m_ij) for m_ij in matrix[usr_from, item_to]])

    # assign users and items to common indices, items come first
    edges_from_user += num_items
    edges_to_user += num_items
    usr_from += num_items

    edges_from, edges_to, distances = np.concatenate((edges_from_item, edges_from_user, usr_from)), \
                                      np.concatenate((edges_to_item, edges_to_user, item_to)), \
                                      np.concatenate((distances_item, distances_user, distances_user_item))
    if deduplicate:
        if verbose: print("Deduplicating edges...")
        unique_edges_dict = {}  # {(from_i, to_i) : distance(i, j)}
        for from_i, to_i, distance in zip(edges_from, edges_to, distances):
            edge_iijj = int(from_i), int(to_i)
            if not directed:
                edge_iijj = tuple(sorted(edge_iijj))
            unique_edges_dict[edge_iijj] = distance

        edges_iijj, distances = zip(*unique_edges_dict.items())
        edges_from, edges_to = zip(*edges_iijj)

    edges_from, edges_to, distances = map(np.asanyarray, [edges_from, edges_to, distances])
    distances = np.maximum(eps, distances)
    if verbose:
        print("Total edges: {}, mean edges per vertex: {}, mean distance: {}".format(
            len(edges_from), len(edges_from) / float(num_items + num_users), np.mean(distances)
        ))
    return GraphEmbedding(edges_from, edges_to, initial_weights=distances, directed=directed, **kwargs)


def get_knn_edges_cosine(matrix, *, k, knn_drop_rate=0, eps=1e-10):
    edges_from, edges_to, distances = [], [], []
    num_vectors, vector_size = matrix.shape
    matrix = matrix / (np.square(matrix).sum(-1, keepdims=True) ** 0.5 + eps)
    matrix[np.square(matrix).sum(-1) == 0] = 1. / np.sqrt(vector_size)
    # [num_vectors, vector_size]

    cosines = matrix @ matrix.T
    neighbor_indices = cosines.argsort(axis=-1)[:, -(k + 1):]

    for vertex_i in np.arange(num_vectors):
        for neighbor_i in neighbor_indices[vertex_i]:
            cosine_distance = 1. - cosines[vertex_i, neighbor_i]
            if vertex_i == neighbor_i: continue  # forbid loops
            if knn_drop_rate == 0 or np.random.random() >= knn_drop_rate:
                edges_from.append(vertex_i)
                edges_to.append(neighbor_i)
                distances.append(cosine_distance)
    return np.array(edges_from), np.array(edges_to), np.array(distances)
