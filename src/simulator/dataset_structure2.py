import os
import sys
import numpy as np
import pandas as pd
import shutil

sys.path.append("src/utils/")
import rotations

example_path = "data/datasets/example_dataset/"

def make_folder_structure(path:str) -> None:
    """ Creates the folder structure for the dataset

        Args:
            path (str): Path to the dataset folder
    """
    try:
        os.mkdir(path + "config")
        os.mkdir(path + "data")
        os.mkdir(path + "noise")
        os.mkdir(path + "tmp")
    except FileExistsError:
        print("One of the folders exists already, or couldn't be created. Please fix!")
        sys.exit()

    # create dummy files
    try:
        shutil.copy(example_path + "config/config.yaml", path + "config/config.yaml")
        shutil.copy(example_path + "config/readme.md", path + "config/readme.md")
        shutil.copy(example_path + "config/scene.blend", path + "config/scene_PLACEHOLDER.blend")
        shutil.copy(example_path + "noise/noise_neg_0.1lux.npy", path + "noise/noise_neg_0.1lux.npy")
        shutil.copy(example_path + "noise/noise_neg_3klux.npy", path + "noise/noise_neg_3klux.npy")
        shutil.copy(example_path + "noise/noise_neg_3klux.mat", path + "noise/noise_neg_3klux.mat")
        shutil.copy(example_path + "noise/noise_neg_161lux.npy", path + "noise/noise_neg_161lux.npy")
        shutil.copy(example_path + "noise/noise_pos_0.1lux.npy", path + "noise/noise_pos_0.1lux.npy")
        shutil.copy(example_path + "noise/noise_pos_3klux.npy", path + "noise/noise_pos_3klux.npy")
        shutil.copy(example_path + "noise/noise_pos_3klux.mat", path + "noise/noise_pos_3klux.mat")
        shutil.copy(example_path + "noise/noise_pos_161lux.npy", path + "noise/noise_pos_161lux.npy")

    except Exception as e:
        print(f"One of the files could not be created. Please fix! Error: {e}")


def create_emre_table():
    path = "/data/lkolmar/datasets/emre_dataset/"
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Dataset folder already exists. Please remove it or choose a different path.")
        sys.exit()

    make_folder_structure(path)

    num_samples = 5000

    # samples = np.array([[0.0, -0.45, -0.25, 0.0, 0.45, 0.25],
    #            [0.0, 0.45, 0.25, 0.0, -0.45, -0.25]])
    samples = []

    for _ in range(num_samples):
        z_start = np.random.uniform(-0.25, 0.25)
        z_end = np.random.uniform(-0.25, 0.25)
        samples.append([0.0, -0.45, z_start, 0.0, 0.45, z_end])

    samples = np.array(samples)

    df = pd.DataFrame({
        'index': np.arange(len(samples)),
        'rotation_x': 0,
        'rotation_y': 0,
        'rotation_z': 0,
        'initial_rot_x': 0,
        'initial_rot_y': 0,
        'initial_rot_z': 0,
        'ball_start_x': samples[:, 0],
        'ball_start_y': samples[:, 1],
        'ball_start_z': samples[:, 2],
        'ball_end_x': samples[:, 3],
        'ball_end_y': samples[:, 4],
        'ball_end_z': samples[:, 5],
        'finished': False,
        'path': "not set"
    })
    file_path = path + "config/" + "simulation.csv"
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    create_emre_table()