import numpy as np
from scipy.spatial.distance import cdist

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
        pairwise_distance = cdist(X, X, metric=self._metric)

        # calculate silhouette for each point
        for data_idx in range(X.shape[0]):
            # need to make sure this has the right dimensions for cdist
            data_point = X[data_idx, :].reshape(1, X.shape[1])

            # average within cluster distance of the given point
            a_i = np.mean(cdist(data_point, X[y==y[data_idx]], metric=self._metric))

            centroid_dist = cdist(data_point, centroid_locs, metric=self._metric)[0]
            centroid_dist[y[data_idx]] += np.inf # set this to inf so we can choose the second closest centroid
            closest_centroid = np.argmin(centroid_dist)
            b_i = np.mean(cdist(data_point, X[y==closest_centroid], metric=self._metric))

            silhouette.append((b_i - a_i) / (max(a_i, b_i)))

        return np.array(silhouette)
        
