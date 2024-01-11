import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            
            # Directly extract images and labels into numpy arrays
            batch_images = np.array([self.dataset[i][0] for i in batch_indices])
            batch_labels = np.array([self.dataset[i][1] for i in batch_indices])
            
            yield batch_images, batch_labels

    def __len__(self):
        return len(self.dataset) // self.batch_size
