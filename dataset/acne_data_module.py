"""Acne data module that creates datasets and loaders."""

from torch.utils.data import DataLoader
from dataset.acne_dataset import AcneDataset
from transforms.acne_transforms import AcneTransformsTorch


class AcneDataModule:
    """
    Create dataset and loaders, apply transforms.

    create_loaders - preprocess image data and create data loaders
    """

    def __init__(self, train_file, test_file, data_path, batch_size, batch_size_test):
        """Initialize the module."""
        self.train_file = train_file
        self.test_file = test_file
        self.data_path = data_path
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = 4

    def create_loaders(self):
        """Create loaders both for train and test/validation datasets."""
        # train dataset
        dset_train = AcneDataset(self.data_path, self.train_file, transform=AcneTransformsTorch(train=True))
        # test dataset
        dset_test = AcneDataset(self.data_path, self.test_file, transform=AcneTransformsTorch(train=False))
        # Create loaders
        train_loader = DataLoader(
            dset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

        test_loader = DataLoader(
            dset_test, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        return train_loader, test_loader
