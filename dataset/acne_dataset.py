"""Dataset class."""

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import random


class AcneDataset(Dataset):
    """Dataset class which contains __init__, __getitem__ and __len__ methods."""

    def __init__(self, data_path, img_filename, transform=None):
        """Initialize Dataset object."""
        # Apply 'Path' to use pathlib
        self.img_path = Path(data_path)
        self.transform = transform
        # reading .txt from file in the bucket
        with open(img_filename, "r") as f:
            txt_data = f.read()
        # txt_data is a string
        txt_data = txt_data.split("\n")[:-1]

        # read info about filename, labels and lesions
        self.img_filename = []
        self.labels = []
        self.lesions = []
        for line in txt_data:
            filename, label, lesion = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesions.append(int(lesion))

        self.img_filename = np.array(self.img_filename)
        self.labels = np.array(self.labels)
        self.lesions = np.array(self.lesions)

        if "NNEW_trainval" in img_filename:
            ratio = 1.0
            random.seed(42)
            indexes = []
            for i in range(4):
                index = random.sample(
                    list(np.where(self.labels == i)[0]), int(len(np.where(self.labels == i)[0]) * ratio)
                )
                indexes.extend(index)
            self.img_filename = self.img_filename[indexes]
            self.labels = self.labels[indexes]
            self.lesions = self.lesions[indexes]

    def __getitem__(self, index):
        """Load and return a sample from the dataset at the given index."""
        # read image
        img = Image.open(self.img_path / self.img_filename[index]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(self.labels[index]))
        lesion = torch.from_numpy(np.array(self.lesions[index]))

        return img, label, lesion

    def __len__(self):
        """Return the number of samples in dataset."""
        return len(self.img_filename)
