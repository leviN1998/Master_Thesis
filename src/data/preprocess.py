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
import pickle


dataset_path = "/data/lkolmar/datasets/spindoe_topspin/"
output_path = dataset_path + "preprocessed/"

roi_size = 100  # size of the region of interest in pixels

os.makedirs(output_path, exist_ok=True)

def preprocess(data):
    events = eventIO.load_hdf5(dataset_path + data["path"] + "events.hdf5")
    metadata = pd.read_csv(dataset_path + data["path"] + "metadata.csv")
    coords = pd.read_csv(dataset_path + data["path"] + "ball_coords.csv")
    path = data["path"]
    roi_path = path.replace("data/", "", 1) + "roi.hdf5"
    if not os.path.exists(output_path + roi_path[:5]):
        roi = extract_roi(events, metadata, coords)
        os.makedirs(output_path + roi_path[:5], exist_ok=True)
        # eventIO.create_video(roi, output_path + roi_path.replace(".hdf5", ".mp4"))
        eventIO.save_hdf5(roi, output_path + roi_path, bias=[0], resolution=(roi_size, roi_size))
    else:
        print(f"ROI already exists: {output_path + roi_path}")


def preprocess_real(hdf5_path, coords_path, output_path):
    events = eventIO.load_hdf5(hdf5_path)
    with open(coords_path, "rb") as f:
        coords = pickle.load(f)

    coords_df = pd.DataFrame(coords, columns=["screen_x", "screen_y"])
    #print(coords_df)
    #print(events.get_ts()[-1], events.get_ts()[0])
    ts = events.get_ts() - events.get_ts()[0]
    events.ts = ts
    #print(events.get_ts()[-1], events.get_ts()[0])
    metadata = pd.DataFrame({
        "video_length": [(events.get_ts()[-1]) / 1e6], # in seconds
        "total_frames": [len(coords_df)]
    })
    roi = extract_roi(events, metadata, coords_df)
    eventIO.save_hdf5(roi, output_path, bias=[0], resolution=(roi_size, roi_size))


def extract_roi(events, metadata, coords):
    total_time_us = metadata["video_length"].values[0] * 1e6
    frame_time_us = total_time_us / metadata["total_frames"].values[0]
    #print(f"Total time: {total_time_us} us, Frame time: {frame_time_us} us, passed time {metadata['video_length'].values[0]} s")
    xs = np.array([], dtype=np.int32)
    ys = np.array([], dtype=np.int32)
    ts = np.array([], dtype=np.uint64)
    ps = np.array([], dtype=np.int32)

    # start_pos_x = (((metadata["ball_start_y"].values[0] + 0.45) / 0.9) * 1280) 
    # start_pos_y = (((metadata["ball_start_z"].values[0])))
    # end_pos_x = (((metadata["ball_end_y"].values[0] + 0.45) / 0.9) * 1280)
    # end_pos_y = metadata["ball_end_z"].values[0]

    # print(f"Ball start: {(start_pos_x, start_pos_y)}, Ball end: {(end_pos_x, end_pos_y)}")

    for f in range(metadata["total_frames"].values[0]):
        # temp: only use frames that are between 100 and 240
        # if f < 100 or f > 240:
        #     continue

        t = f * frame_time_us
        matches = coords[coords["frame"] == f]
        if matches.empty:
            continue
        frame_coord = matches.iloc[0]
        #print(f)
        #print(frame_coord)
        #print()
        #print(f"Frame {f}: time {t} us, coord ({frame_coord['screen_x']}, {frame_coord['screen_y']})")
        
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


        # build a 100x100 grid centered at the frame coordinate
        #x0 = int(np.ceil(frame_coord["screen_x"] - roi_size // 2))
        #y0 = int(np.ceil(frame_coord["screen_y"] - roi_size // 2))
#
        #xs_range = np.arange(x0, x0 + roi_size, dtype=np.int32)
        #ys_range = np.arange(y0, y0 + roi_size, dtype=np.int32)
        #xs_mesh, ys_mesh = np.meshgrid(xs_range, ys_range, indexing='xy')
#
        #xs_roi = xs_mesh.ravel()
        #ys_roi = ys_mesh.ravel()
#
        ## assign timestamps and polarities for each synthetic pixel event
        #ts_roi = np.full(xs_roi.shape, int(t), dtype=np.uint64)
        #ps_roi = np.zeros(xs_roi.shape, dtype=np.int32)
#
        #xs_roi = np.concatenate((xs_roi, xs_frame))
        #ys_roi = np.concatenate((ys_roi, ys_frame))
        #ts_roi = np.concatenate((ts_roi, ts_frame))
        #ps_roi = np.concatenate((ps_roi, ps_frame))


        xs = np.concatenate((xs, xs_roi))
        ys = np.concatenate((ys, ys_roi))
        ts = np.concatenate((ts, ts_roi))
        ps = np.concatenate((ps, ps_roi))
    ev = eventIO.EventBuffer(0)
    #ts = ts - ts[0]  # reset time to start at 0
    ev.x = xs[:]
    ev.y = ys[:]
    ev.ts = ts[:]
    ev.p = ps[:]
    ev.i = xs.shape[0]
    
    return ev

def main_sim():
    df_big = pd.read_csv(dataset_path + "config/simulation.csv")
    print(len(df_big))
    df = df_big[(df_big["finished"]) == True]
    print(len(df))
    df = df[df["path"] != "not set"]
    print(len(df))
    
    
    print(f"Loaded {len(df)} finished simulations.")
    print(df)
    print(df.iloc[0])
    path = df.iloc[0]["path"]
    suffix = path.replace("data", "", 1)
    print(suffix)
    # preprocess(df.iloc[0])

    preprocess(df.iloc[2])
    return

    for i in tqdm.tqdm(range(len(df))):
        preprocess(df.iloc[i])

        """
        Check if files exist, if not set finished to False


        data = df.iloc[i]
        if not os.path.exists(dataset_path + data["path"] + "events.hdf5"):
            print(data["path"])
            # print(df_big.loc[df_big["index"] == data["index"]])
            df_big.loc[df_big["index"] == data["index"], "finished"] = False
            df_big.loc[df_big["index"] == data["index"], "path"] = "not set"
            
    df_big.to_csv(dataset_path + "config/simulation.csv", index=False)
    """
        
def main_real():
    base_path = "/home/lkolmar/Documents/metavision/recordings/spindoe_topspin/"
    raw_data_path = base_path + "data/"
    roi_coords_path = base_path + "roi_coords/"
    output_path = base_path + "preprocessed/"

    folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
    files = []
    for folder in folders:
        file_names = [f for f in os.listdir(os.path.join(raw_data_path, folder)) if f.endswith('.hdf5')]
        for name in file_names:
            files.append(os.path.join(folder, name))

    print(f"Found {len(files)} files.")
    print(files[0])

    for file in tqdm.tqdm(files):
        filename = file.replace("program1/", "").replace("program2/", "").replace("program3/", "").replace("program4/", "").replace("program5/", "").replace("program6/", "")

        preprocess_real(
            hdf5_path = os.path.join(raw_data_path, file),
            coords_path = os.path.join(roi_coords_path, filename.replace(".hdf5", ".pkl")),
            output_path = os.path.join(output_path, filename.replace("raw_data/", ""))
        )

if __name__ == "__main__":
    main_real()