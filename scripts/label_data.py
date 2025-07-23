""" Label simulated data


    Topspin dataset spin [x,y,z] from [75, 22, 22] to [-75, -22, -22]       -75 is backspin
"""
import os
import sys
import pandas as pd
import numpy as np
sys.path.append("src/utils")
import eventIO
import rotations
import tqdm


dataset_path = "/data/lkolmar/datasets/topspin_fit_to_max/"
labels = {
    "topspin_slow": 0,
    "topspin_mid": 1,
    "topspin_fast": 2,
    "backspin_slow": 3,
    "backspin_mid": 4,
    "backspin_fast": 5,
}

def label_data(data):
    spin = rotations.Rotation()
    spin.set_axis(data["rotation_x"], data["rotation_y"], data["rotation_z"])
    # print(spin.get_angle())
    topspin = 0 if spin.get_axis()[0] < 0 else 1
    # print(f"Topspin: {topspin}")
    speed = 0
    if spin.get_angle() < 30:
        speed = 0
    elif spin.get_angle() < 55:
        speed = 1
    else:
        speed = 2
    label = labels["topspin_slow"] + topspin * 3 + speed
    # print(f"Label: {label}")
    return label


if __name__ == "__main__":
    data = pd.read_csv(dataset_path + "config/simulation.csv")
    print(data[:5])
    print(data["rotation_x"].max(), data["rotation_y"].max(), data["rotation_z"].max())
    print(data["rotation_x"].min(), data["rotation_y"].min(), data["rotation_z"].min())
    # label_data(data.iloc[0])
    labels = [label_data(data.iloc[i]) for i in tqdm.tqdm(range(len(data)))]
    df = pd.DataFrame({'index': range(len(labels)), 'label': labels})
    print(df[:5])
    df.to_csv(dataset_path + "config/labels.csv", index=False)