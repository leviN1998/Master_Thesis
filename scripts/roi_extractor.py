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
import pickle

"""

    - Uses the convolution method suggested by David
    - Calculates 20 centers for each recording and then interpolates to have 600 coordinates, to match the simulated data
    - saves coordinates to csv file so we can do preprocessing the same way as for the simulated data 

"""

def find_centers(images, radius, p=0.8, verbose=True):
    r = radius
    kernel = np.zeros((int(2*r)+1, int(2*r)+1))
    for y in range(int(2*r)):
        y1 = y - r
        x1 = r + int(np.sqrt(r**2 - y1**2))
        x2 = r - int(np.sqrt(r**2 - y1**2))
        kernel[x1][y] = 1
        kernel[x2][y] = 1

    if verbose:
        centers = [get_center(k, kernel, p) for k in tqdm.tqdm(images)]
    else:
        centers = [get_center(k, kernel, p) for k in images]
    
    return centers


def get_center(matrix, kernel, p):
    convolved = signal.convolve2d(matrix, kernel, 'same', 'fill', 0)
    convolved2 = convolved>np.max(convolved)*p
    return center_of_ones(convolved2)


def center_of_ones(matrix):
    ys, xs = np.nonzero(matrix)
    if len(xs) == 0:
        return None
    
    x_center = xs.mean()
    y_center = ys.mean()
    return x_center, y_center


def interpolate_positions(known_positions, n_coordinates, known_indices=None):
    known_positions = np.asarray(known_positions, dtype=np.float32)
    if known_indices is None:
        idx = np.round(np.linspace(0, n_coordinates - 1, len(known_positions))).astype(int)
    else:
        idx = np.asarray(known_indices, dtype=np.int32)

    order = np.argsort(idx)
    idx = idx[order]
    known_positions = known_positions[order]

    uniq_idx, inv = np.unique(idx, return_inverse=True)
    if len(uniq_idx) != len(idx):
        agg = np.zeros((len(uniq_idx), 2), dtype=np.float32)
        counts = np.bincount(inv)
        agg[:, 0] = np.bincount(inv, weights=known_positions[:, 0]) / counts
        agg[:, 1] = np.bincount(inv, weights=known_positions[:, 1]) / counts
        idx = uniq_idx
        known_positions = agg

    frames = np.arange(n_coordinates)
    out = np.empty((n_coordinates, 2), dtype=np.float32)
    out[:, 0] = np.interp(frames, idx, known_positions[:, 0])
    out[:, 1] = np.interp(frames, idx, known_positions[:, 1])
    return out



def process_sequence(buffer, n_centers, n_coordinates, verbose=True):
    # make sure ts start at 0
    x = buffer.get_x()
    y = buffer.get_y()
    t = buffer.get_ts()
    t = t - np.min(t)
    n_bins = n_coordinates # for simplicity

    # create images
    width, height = 1280, 720 # always the same for our recordings
    images = np.zeros((n_bins, height, width), dtype=np.float32)
    dt = (np.max(t) - np.min(t)) / n_bins
    bin_indices = np.floor((t - np.min(t)) / dt).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    np.add.at(images, (bin_indices, y, x), 1)

    # select n_centers images
    step = n_bins // n_centers
    if verbose:
        print(f"Selecting every {step} image to get {n_centers} centers.")
    selected_indices = np.round(np.linspace(0, n_bins - 1, n_centers)).astype(int)
    selected_images = images[selected_indices]

    # extract centers
    centers = find_centers(selected_images, radius=30, verbose=verbose)
    
    # interpolate to get n_coordinates coordinates
    full_positions = interpolate_positions(centers, n_coordinates, known_indices=selected_indices)
    return full_positions



if __name__ == "__main__":
    n_centers = 8
    n_coordinates = 140

    #data_path = "/home/lkolmar/Documents/metavision/recordings/new_real_dataset/raw_data/"
    #output_path = "/home/lkolmar/Documents/metavision/recordings/new_real_dataset/roi_coords/"
    data_path = "/data/lkolmar/datasets/spindoe_topspin/data/"
    output_path = "/data/lkolmar/datasets/spindoe_topspin/roi_coords/"

    #folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    #paths = []
    #for folder in folders:
    #    file_names = [f for f in os.listdir(os.path.join(data_path, folder)) if f.endswith('.hdf5')]
    #    for name in file_names:
    #        paths.append(os.path.join(data_path,folder, name))
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    paths = []
    for folder in folders:
        file_names = [f for f in os.listdir(os.path.join(data_path, folder)) if f.endswith('.hdf5')]
        for name in file_names:
            paths.append(os.path.join(data_path, folder, name))
    
    print(f"Found {len(paths)} files.")
    print(paths[0])
    
    
    for path in tqdm.tqdm(paths):    
        name = os.path.basename(path)
        # print(f"Processing {name}...")
        buf = eventIO.load_hdf5(path)

        coords = process_sequence(buf, n_centers=n_centers, n_coordinates=n_coordinates, verbose=False)

        with open(output_path + name.replace(".hdf5", ".pkl"), "wb") as f:
            pickle.dump(coords, f)

        # test = pickle.load(open(output_path + name.replace(".hdf5", ".pkl"), "rb"))
        # print(test)
    