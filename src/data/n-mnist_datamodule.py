from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import tonic


class N_MNISTDataModule(LightningDataModule):
    """ LightningDataModule for the N-MNIST dataset.

    The N-MNIST dataset is a neuromorphic version of the MNIST dataset, which contains
    event-based data generated from the original MNIST dataset.

    """

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `N_MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # TODO: Check out different transformations
        # taken from https://tonic.readthedocs.io/en/latest/getting_started/nmnist.html
        sensor_size = tonic.datasets.NMNIST.sensor_size
        self.transforms = transforms.Compose([
                tonic.transforms.Denoise(filter_time=10000),
                tonic.transforms.ToFrame(sensor_size=sensor_size, num_bins=3)
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    