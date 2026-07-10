"""Batching: hand out a dataset a few examples at a time.

Training on one example at a time is noisy and slow; training on the
whole dataset at once doesn't fit in memory (and generalizes worse).
The DataLoader groups examples into *mini-batches* and, optionally,
re-shuffles them every epoch so the model never sees the same order twice.
"""

import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            batch_images = np.array([self.dataset[i][0] for i in batch_indices])
            batch_labels = np.array([self.dataset[i][1] for i in batch_indices])
            yield batch_images, batch_labels

    def __len__(self):
        """Number of batches per epoch (the last one may be smaller)."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
