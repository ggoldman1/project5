import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self._k = k
        self._metric = metric
        self._tol = tol
        self._max_iter = max_iter
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        obs = mat.shape[0]
        feats = mat.shape[1]

        if self._k > obs:
            raise ValueError(f"There must be more data points than clusters. You passed {obs} data points and {self._k}"
                             f"clusters.")

        self._labels = np.zeros(obs)
        mat_min, mat_max = np.min(mat), np.max(mat)
        self.centers = np.random.uniform(mat_min, mat_max, size=(self._k, feats))
        prev = np.inf*np.ones((self._k, feats))

        num_iter = 1
        while not np.all(np.diag(cdist(self.centers, prev, metric=self._metric)) < self._tol):
            if num_iter > self._max_iter:
                print("Max iter exceeded before convergence")
                break

            else:

                # update data assignments to closest centroid
                self._labels = self._assign_points_to_labels(mat)

                # update centeroid to be average of assigned data
                prev = self.centers.copy()
                for c in range(self._k):
                    closest_data = mat[self._labels == c]
                    if closest_data.shape[0] == 0: # no data points assigned to this centroid
                        self.centers[c] = np.random.uniform(mat_min, mat_max, size=(1, feats)) # try moving to new spot
                    else: # take the average value for each component assigned to this centroid
                        self.centers[c] = np.mean(closest_data, axis=0)

                num_iter += 1

        self.mse = self._calculate_mse(mat)

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        return self._assign_points_to_labels(mat)

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.mse

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centers

    def _calculate_mse(self, mat: np.array) -> float:
        """
        Calculate mean-squared error on fit model given training data.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        pred = np.array([list(self.centers[x]) for x in self._labels])
        # get the distance between each point and its centroid, square it, take average across all point-centroid pairs
        return np.average(np.square(np.diag(cdist(mat, pred, metric=self._metric))))

    def _assign_points_to_labels(self, mat: np.array) -> np.array:
        """
        Given data stored in mat (and centers from `self.centers`), assign each point in `mat` to a center.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        returns:
            np.array
                array assigning each data point in `mat` to a center, it is of shape (mat.shape[0], self._k)
        """
        data_centers_dist = cdist(mat, self.centers, metric=self._metric)
        return np.argmin(data_centers_dist, axis=1)