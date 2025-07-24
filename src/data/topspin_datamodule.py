from typing import Any, Dict, Optional, Tuple

from cv2 import transform
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import tonic
import numpy as np

from src.utils import eventIO, event_represenations


class TopspinDataModule(LightningDataModule):
    """ LightningDataModule for the Topspin dataset.

    The Topspin dataset contains event-based data generated from simulated topspin events.
    """

    def __init__(
        self,
        data_dir: str = "/data/lkolmar/datasets/topspin_fit_to_max/",
        train_val_test_split: Tuple[int, int, int] = (1294, 277, 277),                # n = 1848 (70%, 15%, 15%)
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `TopspinDataModule`.

        :param data_dir: The data directory. Defaults to `"/data/lkolmar/datasets/topspin_fit_to_max/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(1294, 277, 277)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose([
            # Add any necessary transformations here
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    
    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset.

        :return: Number of classes at the moment 6: top and back, with slow, mid, fast spin.
        """
        return 6
    
    def prepare_data(self) -> None:
        pass  # No action needed for this dataset


    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the dataset for training, validation, and testing.

        :param stage: The stage of the data module. Can be 'fit', 'validate', 'test', or 'predict'.
        """
        
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            # Load and split datasets only if not loaded already
            