import cluster
import pytest
import numpy as np

def test_kmeans():
    """
    Various tests of my kmeans implementation
    """
    data, _ = cluster.utils.make_clusters(n=5, scale=0.2)

    # if more centroids than data points, raise error
    km = cluster.kmeans.KMeans(k=10)
    with pytest.raises(ValueError):
        km.fit(data)

    # 0 centroids
    km = cluster.kmeans.KMeans(k=0)
    with pytest.raises(ValueError):
        km.fit(data)

    # if same number of centroids and data points, each centroid should match a data point
    data = np.array([[ 1.9296167 ,  4.06830561],
                     [-9.53991766,  9.01554099],
                     [ 6.30386925, -5.86567529],
                     [ -4.44628659, 4.69036832],
                     [ -6.467248  , -2.03567853]])
    km = cluster.kmeans.KMeans(k=5)
    km.fit(data)
    labs = km.predict(data)
    assert set(sorted(labs)) == {0, 1, 2, 3, 4}
    assert km.mse == 0

    # predict on same data (with very small jitter), assert cluster labels don't change
    epsilon = np.array([list(np.random.multivariate_normal([0,0], [[0.001, 0], [0, 0.001]]))
               for x in range(5)])
    assert np.all(km.predict(data + epsilon) == labs)

    # generate a large, high dimensional dataset of very compact clusters
    data, assignments = cluster.utils.make_clusters(n=1000, m=50, k=3, bounds=(-100, 100), scale=0.2)
    # cluster the data
    km = cluster.kmeans.KMeans(k=3)
    km.fit(data)
    labs = km.predict(data)
    fit_groups = {x: list(np.where(labs == x)[0]) for x in range(3)}
    true_groups = {x: list(np.where(assignments == x)[0]) for x in range(3)}
    # assert I am clustering in the right
    assert sorted(fit_groups.values()) == sorted(true_groups.values())


