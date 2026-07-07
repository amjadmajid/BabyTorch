"""MNIST: 70,000 hand-written digits, the classic starter dataset.

Each example is a 28x28 grayscale image plus a label 0-9.  The files are
downloaded once (about 11 MB) and cached in ``root``.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# The original yann.lecun.com server no longer serves these files without
# authentication; this is the mirror used by PyTorch itself.
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def download_mnist(data_dir='./mnist_data'):
    """Download and decompress the four MNIST files into ``data_dir``."""
    os.makedirs(data_dir, exist_ok=True)

    for name in FILES:
        target = os.path.join(data_dir, name)
        if os.path.exists(target):
            continue
        url = MIRROR + name + ".gz"
        print(f"Downloading {url} ...")
        gz_path = target + ".gz"
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as fin, open(target, 'wb') as fout:
            fout.write(fin.read())
        os.remove(gz_path)


def load_mnist(path, kind='train'):
    """Parse the raw IDX files into (images, labels) NumPy arrays.

    Images come back as float32 in [-1, 1], flattened to 784 values.
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8) \
            .reshape(len(labels), 784).astype(np.float32)
        images = ((images / 255.) - .5) * 2  # normalize to [-1, 1]

    return images, labels


class MNISTDataset:
    """Map-style dataset: ``len(ds)`` examples, ``ds[i] -> (image, label)``.

    ``image`` is a (28, 28) float array in [-1, 1]; ``label`` is an int.
    """

    def __init__(self, root='./mnist_data', train=True, transform=None,
                 download=False):
        self.data_dir = os.path.abspath(root)

        if download:
            download_mnist(self.data_dir)

        missing = [f for f in FILES
                   if not os.path.exists(os.path.join(self.data_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"MNIST files missing in {self.data_dir}: {', '.join(missing)}. "
                f"Pass download=True to fetch them.")

        kind = 'train' if train else 't10k'
        self.data, self.labels = load_mnist(self.data_dir, kind=kind)
        self.data = self.data.reshape(-1, 28, 28)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
