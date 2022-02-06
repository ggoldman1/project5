from kmeans import KMeans
import pytest

def test_kmeans():
    """
    Various tests of my kmeans implementation
    """
    data, _ = make_clusters(n=5)
    km = KMeans(n=10)
    with pytest.raises(ValueError):
        km.fit(data)