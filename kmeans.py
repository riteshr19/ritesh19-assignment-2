import numpy as np
import random

class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.labels = None
        self.iteration = 0
        self.converged = False

    def initialize_centroids(self, X, method='random', manual_centroids=None):
        if method == 'random':
            self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        elif method == 'farthest_first':
            self.centroids = [random.choice(X)]
            for _ in range(1, self.k):
                dist = np.max([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids], axis=0)
                self.centroids.append(X[np.argmax(dist)])
            self.centroids = np.array(self.centroids)
        elif method == 'kmeanso':
            self.centroids = [random.choice(X)]
            for _ in range(1, self.k):
                dist_sq = np.min([np.linalg.norm(X - centroid, axis=1)**2 for centroid in self.centroids], axis=0)
                probs = dist_sq / dist_sq.sum()
                cumulative_probs = probs.cumsum()
                r = random.random()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        self.centroids.append(X[j])
                        break
            self.centroids = np.array(self.centroids)
        elif method == 'manual':
            self.centroids = np.array(manual_centroids)
        else:
            raise ValueError("Invalid initialization method")

    def step(self, X):
        self.labels = np.argmin([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids], axis=0)
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
        self.converged = np.all(new_centroids == self.centroids)
        self.centroids = new_centroids
        self.iteration += 1
        return self.centroids, self.labels, self.converged