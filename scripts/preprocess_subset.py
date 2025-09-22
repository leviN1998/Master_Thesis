""" Preprocess a subset of the simulation data.

This script takes a subset of the simulation data and applies preprocessing steps
This is necessary because the full simulation takes 6 days.

Coordinate Systems:
- Blender: x-axis points towards camera, y-axis points right, z-axis points up
- SpinDOE: x-axis points right, y-axis points up, z-axis points towards camera

"""

import pandas as pd
import numpy as np
import sys
sys.path.append("src/utils/")
import eventIO

path = "/data/lkolmar/datasets/realistic/"


def extract_roi(path, idx, roi_size):
    """ Taken from Notebook 3.0-lk-preprocess-simulation.ipynb
    
    """
    cfg = pd.read_csv(path + "config/simulation.csv")
    sample = cfg.iloc[idx]
    events = eventIO.load_hdf5(path + sample["path"] + "events.hdf5")
    data = pd.read_csv(path + sample["path"] + "metadata.csv")
    coords = pd.read_csv(path + sample["path"] + "ball_coords.csv")
    total_time_us = data["video_length"].values[0] * 1e6
    frame_time_us = total_time_us / data["total_frames"].values[0]
    t_min = int(events.ts.min())
    xs = np.array([], dtype=np.int32)
    ys = np.array([], dtype=np.int32)
    ts = np.array([], dtype=np.uint64)
    ps = np.array([], dtype=np.int32)
    for f in range(data["total_frames"].values[0]):
        t = f * frame_time_us
        frame_coord = coords.iloc[f]
        idxs_frame = np.where((events.ts >= t) & (events.ts < t + frame_time_us))[0]

        xs_frame, ys_frame, ts_frame, ps_frame = events.x[idxs_frame], events.y[idxs_frame], events.ts[idxs_frame], events.p[idxs_frame]
        idxs_roi = np.where((xs_frame >= frame_coord["screen_x"] - roi_size // 2) & 
                            (xs_frame < frame_coord["screen_x"] + roi_size // 2) &
                            (ys_frame >= frame_coord["screen_y"] - roi_size // 2) & 
                            (ys_frame < frame_coord["screen_y"] + roi_size // 2))
        xs_roi = xs_frame[idxs_roi] - int(np.ceil(frame_coord["screen_x"] - roi_size // 2))
        ys_roi = ys_frame[idxs_roi] - int(np.ceil(frame_coord["screen_y"] - roi_size // 2))
        ts_roi = ts_frame[idxs_roi]
        ps_roi = ps_frame[idxs_roi]
        xs = np.concatenate((xs, xs_roi))
        ys = np.concatenate((ys, ys_roi))
        ts = np.concatenate((ts, ts_roi))
        ps = np.concatenate((ps, ps_roi))

    return xs, ys, ts - ts.min(), ps


if __name__ == "__main__":
    csv_path = path + "config/simulation.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded dataframe with {len(df)} entries.")
    subset = df[df["finished"] == True]
    print(f"Filtered dataframe with {len(subset)} finished entries.")

    for i in range(len(subset)):
        index = subset.iloc[i]["index"]
        # print(f"i: {i}, index: {index}")



        break