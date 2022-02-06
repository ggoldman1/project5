import cluster
import pytest

def test_kmeans():
    """
    Various tests of my kmeans implementation
    """
    data, _ = cluster.utils.make_clusters(n=5)
    km = cluster.kmeans.KMeans(k=10)
    with pytest.raises(ValueError):
        km.fit(data)