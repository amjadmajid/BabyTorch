import os
import subprocess
import struct
import numpy as np

def download_mnist(data_dir='mnist_data'):
    original_dir = os.getcwd()  # Save the current directory

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)

    print(f"2 {os.getcwd()=}")

    # URLs for the MNIST dataset and corresponding filenames
    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ]
    filenames = [url.split('/')[-1].replace('.gz', '') for url in urls]

    # Check if files have already been downloaded
    if all([os.path.exists(os.path.join(data_dir, fname)) for fname in filenames]):
        print("MNIST data already downloaded.")
        return  # Exit the function

    # Download the files
    for url in urls:
        filename = url.split('/')[-1]
        if not os.path.exists(filename):
            subprocess.run(["wget", url])
        else:
            print(f"{filename} already exists. Skipping download.")

    # Get the list of compressed files
    compressed_files = [f for f in os.listdir(data_dir) if f.endswith('.gz')]
    print("Files to decompress:", compressed_files)

    # Decompress the files
    for file in compressed_files:
        filepath = os.path.join(data_dir, file)
        subprocess.run(["gunzip", filepath])

    os.chdir(original_dir)  # Revert back to the original directory


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    # Load labels
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    # Load images
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2  # Normalize to [-1, 1]
    
    return images, labels



class MNISTDataset:
    def __init__(self, root='./mnist_data', train=True, transform=None, download=False):
        self.data_dir = os.path.abspath(root)  # Get absolute path
        print(f"Data directory set to: {self.data_dir}")

        # Download the dataset if necessary
        if download:
            download_mnist(self.data_dir)
            self._check_files_existence()

        kind = 'train' if train else 't10k'
        self.data, self.labels = load_mnist(self.data_dir, kind=kind)

        # Reshape the data for convenience
        self.data = self.data.reshape(-1, 28, 28)

        # Transformation function
        self.transform = transform

    def _check_files_existence(self):
        """Checks if the required MNIST data files exist in the specified directory."""
        print(f"Checking if MNIST data files exist in the data directory...")

        # Expected files for both 'train' and 't10k' sets
        expected_files = [
            "train-labels-idx1-ubyte",
            "train-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte"
        ]

        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(self.data_dir, f))]

        if missing_files:
            print(f"Warning: Missing files in {self.data_dir} - {', '.join(missing_files)}")
            raise FileNotFoundError(f"Expected MNIST files not found: {', '.join(missing_files)}")
        else:
            print(f"All expected MNIST files are present in {self.data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        
        # If there's any transform function provided, apply it
        if self.transform:
            image = self.transform(image)

        return image, label

    #  Ensure the 'download_mnist' and 'load_mnist' functions from your previous code are still defined before this class