""" This file contains a script to preprocess simulated data to be used for machine leraning


    This script can be used on the common dataset structure of the simulator
    it builds the "preprocessed" folder from data in the "data" folder.


    Coordinate Systems:
    - Blender: x-axis points towards camera, y-axis points right, z-axis points up
    - SpinDOE: x-axis points right, y-axis points up, z-axis points towards camera


    ----> lat usage: preprocess subset
"""

import os
import sys
import pandas as pd
import numpy as np
sys.path.append("src/utils")
import eventIO
import tqdm


dataset_path = "/data/lkolmar/datasets/realistic/"
output_path = dataset_path + "preprocessed/"

roi_size = 100  # size of the region of interest in pixels

os.makedirs(output_path, exist_ok=True)

def preprocess(data):
    events = eventIO.load_hdf5(dataset_path + data["path"] + "events.hdf5")
    metadata = pd.read_csv(dataset_path + data["path"] + "metadata.csv")
    coords = pd.read_csv(dataset_path + data["path"] + "ball_coords.csv")
    roi = extract_roi(events, metadata, coords)
    path = data["path"]
    roi_path = path.replace("data/", "", 1) + "roi.hdf5"
    os.makedirs(output_path + roi_path[:5])
    # eventIO.create_video(roi, output_path + roi_path.replace(".hdf5", ".mp4"))
    eventIO.save_hdf5(roi, output_path + roi_path, bias=[0], resolution=(roi_size, roi_size))


def extract_roi(events, metadata, coords):
    total_time_us = metadata["video_length"].values[0] * 1e6
    frame_time_us = total_time_us / metadata["total_frames"].values[0]
    xs = np.array([], dtype=np.int32)
    ys = np.array([], dtype=np.int32)
    ts = np.array([], dtype=np.uint64)
    ps = np.array([], dtype=np.int32)
    for f in range(metadata["total_frames"].values[0]):
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
    ev = eventIO.EventBuffer(0)
    ev.x = xs[:]
    ev.y = ys[:]
    ev.ts = ts[:]
    ev.p = ps[:]
    ev.i = xs.shape[0]
    return ev


if __name__ == "__main__":
    df = pd.read_csv(dataset_path + "config/simulation.csv")
    df = df[df["finished"] == True]
    print(f"Loaded {len(df)} finished simulations.")
    print(df)
    print(df.iloc[0])
    path = df.iloc[0]["path"]
    suffix = path.replace("data", "", 1)
    print(suffix)
    # preprocess(df.iloc[0])

    for i in tqdm.tqdm(range(len(df))):
        preprocess(df.iloc[i])