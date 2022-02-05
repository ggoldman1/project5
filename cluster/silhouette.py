import numpy as np
from scipy.spatial.distance import cdist
import utils 

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self._metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        silhouette = []

        # get each centroid and pairwise centroid distance 
        centroid_locs = np.array([np.mean(X[y==c], axis=0) for c in set(y)])

        # pairwise distance across all data points
        pairwise_distance = cdist(X, X)



        for data_idx in range(X.shape[0]):
            within_cluster = within_cluster_dist[y[data_idx]]
            a_i = np.sum(within_cluster)


def zero_out_dist_mat(dist_mat: np.array, idx: np.array) -> np.array:
    """
    Given a symmetrical distance matrix, zero out given data points.
    """
    for i in idx:
        dist_mat[i, :] = np.zeros(dist_mat.shape[0])
        dist_mat[:, i] = np.zeros(dist_mat.shape[0])
    return dist_mat
        


