"""
Datasets are plaintext collections. The only thing required for dataset is to iterate over raw strings of text.
Good source of datasets: https://github.com/niderhoff/nlp-datasets
"""
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class Dataset:
    def __init__(self, matrix=None, path_csv='', n_neighbors=5,
                 is_faiss=True, metric='minkowski', p=2, algorithm='auto', leaf_size = 30):
        """
        Parameters
        ----------
        matrix: np.array, default None 
                Matrix of elements (nxd)
        path_csv: string, default ''
                If matrix is None, matrix would be read from path 
        n_neighbors : int, optional (default = 5)
                Number of neighbors to use by default
            
        metric : string or callable, default 'minkowski'
                metric to use for distance computation. Any metric from scikit-learn
                or scipy.spatial.distance can be used.

                If metric is a callable function, it is called on each
                pair of instances (rows) and the resulting value recorded. The callable
                should take two arrays as input and return one value indicating the
                distance between them. This works for Scipy's metrics, but is less
                efficient than passing the metric name as a string.
        p : integer, optional (default = 2)
            Parameter for the Minkowski metric from
            sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
            equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        algorithm
            based on the values passed to :meth:`fit` method.

            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.
        leaf_size : int, optional (default = 30)
            Leaf size passed to BallTree or KDTree.
        """
        
        self.n_neighbors = n_neighbors
        
        if path_csv != '':
            matrix = pd.read_csv(path_csv)
            
        elif matrix is None:
            # Plz correct my English
            raise ValueError('Not matrix not path is provided ')
            
        if is_faiss:
            from faiss import IndexFlatL2
            index = IndexFlatL2(matrix.shape[1])
            index.add(matrix)
            self.nearest_distances, self.nearest_indices = index.search(matrix, n_neighbors + 1)
            self.nearest_distances, self.nearest_indices = self.nearest_distances[:, 1:], self.nearest_indices[:, 1:]
        else:
            self.nearest_distances, self.nearest_indices = \
                     NearestNeighbors(n_neighbors=n_neighbors+1, algorithm=algorithm,
                                      metric=metric, leaf_size=leaf_size,
                                      p=p, n_jobs=-1).fit(matrix).kneighbors(matrix)
            self.nearest_distances = self.nearest_distances[:, 1:] ** 2
            self.nearest_indices = self.nearest_indices[:, 1:]

    def __getitem__(self, indices):
        """
        Parameters
        ----------
        indices: np.array, default None 
                array of index (batch_size)
        Return:
        nearest_distances: nearest distances square for each element from indices(batch_size x n_neighbors)
        nearest_indices :  nearest indices for each element from indices(batch_size x n_neighbors)
        ----------
        
        """
        
        return self.nearest_distances[indices, :], self.nearest_indices[indices, :]
