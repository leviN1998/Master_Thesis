from typing import Any, Dict, Optional, Tuple

from cv2 import transform
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import tonic
import numpy as np

from src.utils import event_representations

from src.utils import eventIO
from src.data.components.TOPSPIN import Hdf5DatasetRegression
from src.data.topspin_datamodule import TopspinDataModule




class RegressionDataModule(TopspinDataModule):
    """ LightningDataModule for the Topspin dataset for regression tasks.
    """


    def __init__(
        self,
        data_dir: str = "/data/lkolmar/datasets/topspin_fit_to_max/",
        time_window: int = 500,  # time window for event sequences
        num_bins: int = 10,  # number of bins for voxel grid
        flip: bool = False,  # whether to flip the event data
        sensor_size: Tuple[int, int] = (100, 100),
        train_val_test_split: Tuple[int, int, int] = (1294, 277, 277),                # n = 1848 (70%, 15%, 15%)
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        max_len: int = 0,
    ) -> None:
        """Initialize a `RegressionDataModule`.

        :param data_dir: The data directory. Defaults to `"/data/lkolmar/datasets/topspin_fit_to_max/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(1294, 277, 277)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(
            data_dir=data_dir,
            time_window=time_window,
            num_bins=num_bins,
            flip=flip,
            sensor_size=sensor_size,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            max_len=max_len,
        
        )

    
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
            n = sum(self.hparams.train_val_test_split)
            indices = np.arange(n)
            dataset = Hdf5DatasetRegression(
                dataset_path=self.hparams.data_dir,
                indices=indices,
                transforms=self.transforms
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )
            print(f"Flipped: {self.hparams.flip}")
            print()
            print("------------------------------------------------------------------------")
            print()
            print("Train indices:", self.data_train.indices)
            print("Val indices:", self.data_val.indices)
            print("Test indices:", self.data_test.indices)
            print()
            print("------------------------------------------------------------------------")  
            print()
