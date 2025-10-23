import torch
import numpy as np
import sys
import os
import pandas as pd
sys.path.append("src/utils")
sys.path.append("src/utils/IEBCS")
import eventIO
from torchvision.transforms import transforms
import event_representations

events_struct = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)]
)


path = "/data/lkolmar/datasets/realistic_topspin/preprocessed/"
target = "/data/lkolmar/datasets/realistic_topspin/pretransformed/"

folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

files = []
for folder in folders:
    folder_path = os.path.join(path, folder)
    folder_files = [f for f in os.listdir(folder_path) if f.endswith(".hdf5")]
    for f in folder_files:
        files.append(folder + "/" + f)

# labels = pd.read_csv("/data/lkolmar/datasets/realistic_topspin/config/labels.csv")

trans = transforms.Compose([
    lambda ev: event_representations.create_sequence(ev, 
                                            time_window=5000, num_bins=10,
                                            sensor_size=(100, 100), flip=False, max_len=0)
])

def get_item(p):
    # idx_string = p.split("/")[0]
    # idx = int(idx_string)
    events = eventIO.load_hdf5(path + p)
    array = np.empty_like(events.get_x(), dtype=events_struct)
    array["x"] = events.get_x()
    array["y"] = events.get_y()
    array["t"] = events.get_ts()
    array["p"] = events.get_p()

    array = trans(array)
    # ende TOPSPIN 
    return array




a = get_item(files[0])
print(a.shape)
name = files[0].split("/")[0]
np.save(target + name + ".npy", a)