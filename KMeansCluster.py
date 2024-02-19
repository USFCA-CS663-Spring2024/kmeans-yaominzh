import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansCluster:

    def __init__(self, k = 3):
        self.k = k
        self.centroids = None

    # according to slide 13: 
    # it is often attractive to fix the maximum iterations to some finite number — eg. 100
    def fit(self, X, max_iter = 100):
        """
        pseudo code from lesson notes:

        """
        # converge when no x^i changes its cluster
        pass
