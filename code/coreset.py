import torch
import numpy as np

from sklearn.metrics import pairwise_distances

class Coreset_Greedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = np.inf * np.ones(self.dset_size, dtype=np.float32)
        self.already_selected = []

        # reshape
        feature_len = self.all_pts[0].shape[0]
        self.all_pts = self.all_pts.reshape(-1, feature_len)

    def update_dist(self, centers):
        dist = pairwise_distances(
            self.all_pts[centers], self.all_pts, metric='euclidean')
        self.min_distances = np.minimum(
            self.min_distances, np.min(dist, axis=0))

    def sample(self, sample_ratio):
        sample_size = int(self.dset_size * sample_ratio)

        new_batch = []
        for _ in range(sample_size):
            if not self.already_selected:
                ind = np.random.choice(self.dset_size)
            else:
                ind = np.argmax(self.min_distances)

                while ind in self.already_selected:
                    self.min_distances[ind] = 0
                    ind = np.argmax(self.min_distances)

            self.already_selected.append(ind)
            self.update_dist([ind])
            new_batch.append(ind)

        return new_batch
