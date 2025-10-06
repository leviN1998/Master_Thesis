import h5py
import numpy as np
import hdf5plugin
from matplotlib import pyplot as plt
import time

from scipy import signal
from scipy.ndimage import uniform_filter
import sys
sys.path.append("src/utils/")
import eventIO
import tqdm
import os
import shutil

if __name__ == "__main__":
    path = "/home/lkolmar/Documents/metavision/recordings/finished/"
    output_path = "/home/lkolmar/Documents/metavision/recordings/new_real_dataset/raw_data/"

    for folder in os.listdir(path):
        if not os.path.isdir(path + folder):
            continue
        print(f"Processing folder: {folder}")

        raw_files = [f for f in os.listdir(path + folder) if f.endswith('.hdf5')]
        if len(raw_files) == 0:
            print(f"No .hdf5 files found in {folder}, skipping.")
            continue

        print(f"found {len(raw_files)} .hdf5 files in {folder}")

        for f in raw_files:
            buf = eventIO.load_hdf5_metavision(path + folder + "/" + f)
            eventIO.save_hdf5(buf, output_path + folder + "/" + f, [0,0])
            shutil.copy(path + folder + "/" + f[:-5] + ".mp4", output_path + folder + "/" + f[:-5] + ".mp4")

        