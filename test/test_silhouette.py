import cluster
import pytest
import numpy as np

def test_silhouette():
    """
    Various tests of my silhouette scoring implementation
    """
    # if k == n, each cluster lies on top of a datapoint and should have a silhouette score of 1
    data = np.array([[20.22295601, 41.61446899],
                     [-95.88307694, 93.9817791],
                     [66.48835567, -57.53223409],
                     [11.48842688, -57.53214644],
                     [-33.48843736, -57.53231909]])

    km = cluster.kmeans.KMeans(k=5)
    km.fit(data)
    labs = km.predict(data)

    s = cluster.silhouette.Silhouette()
    assert np.all(s.score(data, labs) == [1, 1, 1, 1, 1])

    # make one tight cluster, one dispersed cluster, assert tight cluster has higher score on average
    tight_cluster = np.array([list(np.random.multivariate_normal([10, 10], [[0.5, 0.5], [0.5, 0.5]]))
                     for x in range(10)])
    dispersed_cluster = np.array([list(np.random.multivariate_normal([-10, -10], [[10, 0.3], [0.3, 10]]))
                         for x in range(10)])
    data = np.stack((tight_cluster, dispersed_cluster), axis=0).reshape(20,2)

    km = cluster.kmeans.KMeans(k=2)
    km.fit(data)
    labs = km.predict(data)

    # scores in dispersed cluster should be lower with a higher standard deviation
    scores = s.score(data, labs)
    assert np.mean(scores[0:10]) > np.mean(scores[10:20])
    assert np.std(scores[0:10]) < np.std(scores[10:20])





