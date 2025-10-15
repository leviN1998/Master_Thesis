from typing import Any, Dict, Optional, Tuple

from cv2 import transform
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import tonic
import numpy as np

from src.utils import event_representations

if __name__ == "__main__":
    import sys
    sys.path.append("src/utils")
    sys.path.append("src/data/components/")
    sys.path.append("src/models/components/")
    import eventIO, event_represenations
    from TOPSPIN import Hdf5Dataset
    import fire_net
else:
    from src.utils import eventIO
    from src.data.components.TOPSPIN import Hdf5Dataset


class TopspinDataModule(LightningDataModule):
    """ LightningDataModule for the Topspin dataset.

    The Topspin dataset contains event-based data generated from simulated topspin events.
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
            lambda ev: event_representations.create_sequence(ev, 
                                                            self.hparams.time_window, self.hparams.num_bins, 
                                                            self.hparams.sensor_size, flip=self.hparams.flip, max_len=self.hparams.max_len),  # create sequences from events
        ])
        # one sample is now [time_bins, num_bins, sensor_size[0], sensor_size[1]]

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
            n = sum(self.hparams.train_val_test_split)
            indices = np.arange(n)
            dataset = Hdf5Dataset(
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

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=pad_collate_fn,  # Custom collate function to handle variable-length sequences
        )
    

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=pad_collate_fn,  # Custom collate function to handle variable-length sequences
        )
    

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=pad_collate_fn,  # Custom collate function to handle variable-length sequences
        )


    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


def pad_collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function to handle variable-length sequences in the batch.

    :param batch: List of tuples containing event data and labels.
    :return: Padded tensor of event data and tensor of labels.
    """
    events, labels = zip(*batch)
    # Stack events and labels
    events = [torch.tensor(ev, dtype=torch.float32) for ev in events]
    lengths = [ev.shape[0] for ev in events]  # Get lengths of each sequence
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences=events, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(np.array(labels))
    
    return padded_sequences, torch.tensor(lengths), labels_tensor


if __name__ == "__main__":
    data_module = TopspinDataModule(
        data_dir="/data/lkolmar/datasets/topspin_fit_to_max/",
        time_window=5000,  # in us
        num_bins=10,
        sensor_size=(100, 100),
        train_val_test_split=(1294, 277, 277),  # n = 1848 (70%, 15%, 15%)
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    data_module.prepare_data()
    data_module.setup()
    batch = data_module.train_dataloader(). __iter__().__next__()
    print(f"Batch size: {len(batch[0])}")
    print(f"Sample shape: {batch[0].shape}")  # Should be [time_bins, num_bins, sensor_size[0], sensor_size[1]]
    print(f"Sample lengths: {batch[1]}")
    print(f"Sample label: {batch[2]}")
    print(f"Number of classes: {data_module.num_classes}")
    print(f"Batch size per device: {data_module.batch_size_per_device}")
    import matplotlib.pyplot as plt
    img = event_representations.get_voxel_grid_as_image(batch[0][0][0].cpu().detach().numpy())
    plt.imshow(img, cmap='gray')
    plt.title(f"Sample 0, label: {batch[2][0]}")
    plt.show()

    import subcomponents
    firenet = fire_net.FireNet(input_channels=10, hidden_channels=16, kernel_size=3, head=subcomponents.ClassificationHead(16, (100, 100), num_classes=10))
    print(firenet)
    output = firenet(batch[0][:5], lengths=batch[1][:5])
    print(f"Output shape: {output.shape}")  # Should be [batch_size, hidden_channels, height, width]
    print(f"Output: {output}")
    print(f"Output label: {batch[2][:5]}")