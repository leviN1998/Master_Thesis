import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# from src.utils import eventIO
import sys
sys.path.append("src/utils")
import eventIO


events_struct = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)]
)

class Hdf5Dataset(Dataset):
    """Custom dataset for loading HDF5 files containing event data.
    
        The dataset is created to behave like the tonic event datasets.
    """

    def __init__(self, dataset_path: str, indices, transforms=None):
        """
        Initialize the dataset.

        :param file_path: Path to the root dir of the dataset folder bsp: ("/data/lkolmar/datasets/topspin_fit_to_max/").
        :param indices: List of indices to account for different splits.
        :param transforms: Optional transforms to apply to the data.
        """
        self.dataset_path = dataset_path
        self.indices = indices
        self.transforms = transforms

        labels_path = dataset_path + "config/labels.csv"
        self.labels = pd.read_csv(labels_path)


    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.

        :param idx: Index of the sample to retrieve.
        :return: Transformed event data and its label. ([x,y,t,p], label)
        """
        index = self.indices[idx] # simulation index (Note: not the same as idx, because splits are taken randomly)
        index_string = str(index).zfill(5)
        events = eventIO.load_hdf5(self.dataset_path + f"data/{index_string}/{index_string}_events.hdf5")
        array = np.empty_like(events.get_x(), dtype=events_struct)
        array["x"] = events.get_x()
        array["y"] = events.get_y()
        array["t"] = events.get_ts()
        array["p"] = events.get_p()
        # print(np.max(np.array([events.get_ts()])))

        label = self.labels.loc[self.labels['index'] == index, 'label'].values[0]
        return array, label
    


if __name__ == "__main__":
    import tonic
    dataset = tonic.datasets.NMNIST(save_to="data/")
    print(dataset[0])
    print(dataset[0][0].shape)

    print("-----------------------------------------------------------------------------------------------------------")
    dataset = Hdf5Dataset("/data/lkolmar/datasets/topspin_fit_to_max/", [0, 1, 2, 3, 4, 5])
    print(dataset[0])
    print(dataset[0][0].shape)