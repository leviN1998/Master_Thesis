""" Creates the Folder Structure to simulate a Dataset


This script creates the complete structure needed to simulate a dataset.
Most important the table containing all rotations is created. (Necessary to divide and pause creation process)

The structure of the dataset-folder should be as follows:

Parent:
    | 
    |- config
    |   |- config.yaml
    |   |- simulation.csv
    |   |- scene.blend
    |   |- readme.md
    |
    |- data
    |   |- 0001
    |   |   |- 0001_events.hdf5
    |   |   |- 0001_ground_truth.yaml
    |   |   |- 0001_metadata.yaml
    |   |   |- 0001_frames.avi (optional)
    |   |
    |   | - 0002
    |   .   ...
    |
    |- tmp
    |   |- render.log
    |   |- simulation.log
    |   |- image_tmp.png
    |
    |- noise
        |- noise_neg_0.1lux.npy

    Important Files:

        - config.yaml:
            Contains all parameters of the simulation such as the camera configuartion, and all
            relevant simulation parameters and filepaths that are necessary.
            This should also include a short description of the dataset as well as a explaination how to
            label the rotaions if labeling is intended. [Top- / Backspin] eg.

        - simulation.csv:
            Conatains the table that links folders with rotations. Needs to be generated to start thecollecting process.
            With the help of this table the process can be split to multiple machines and the simulation can be stopped without
            losing the whole progress.
            structure:
                idx | rotation | initial orientation | finished? | path

        - scene.blend
            The scene that is used by the simulator

        - ground_truth.yaml
            This file contains the rotation that is performed in the given simulation

        - metadata.yaml
            This file contains all relvent metadata o fthe simulation, such as a table of the ball's positions per frame (pixel coords), 
            the ball and camera positions (world-coords) per frame. Also the ball movement vector, and again the rotation (redundant)

        - render.log
            All console logs that are connected to the blender rendering pipeline

        - simulation.log
            All console logs that have to do with the simulation and the state of the generation progress
"""
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


def create_rotations(n:int, max_speed:float=80, min_speed:float=5) -> np.ndarray:
    """ Creates Rotations in a cubic way, as discussed with David

        This script creates a 3D cube of points that are interpreted as rotaitons (direction and magnitude).
        All rotations are scaled to be inside the range of max_speed and min_speed [rps]

        This script can also be used to check how many rotations would be generated with the current specs

        Args:
            n (int): points per axis. The cube will contain n³ points. The oucoming array will be smaller than that, depending on the other params
            max_speed (float): Maximum speed that should be generated. Unit should be rps
            min_speed (float): Minimum speed that are roation can have. Unit is rps

        Returns:
            rotations (np.ndarray): Numpy array containing all rotations
    """

    lin = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(lin, lin, lin)
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # cut out speeds that are not needed
    distances = np.linalg.norm(points, axis=1)
    # points = points[(distances <= 1) & (distances >= (min_speed / max_speed))]
    points = points[distances <= 1]
    points = points * max_speed
    return points


def is_top_or_back(axis:np.ndarray, threshold_deg:int, reference:np.ndarray=np.array([1,0,0])) -> bool:
    """ Checks if the given axis represents a top or backspin
    """

    axis = np.array(axis)
    axis_normalized = axis / np.linalg.norm(axis)
    reference_normalized = reference / np.linalg.norm(reference)

    cos_angle = np.dot(axis_normalized, reference_normalized)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(abs(cos_angle)))  # use abs() to consider both directions

    return angle_deg < threshold_deg



def create_top_back(n:int, max_speed:float=80, min_speed:float=5) -> np.ndarray:
    """ Creates roations for the first Proof-of concept dataset
    
        Only Top and Backspins are generated. 
        Every rotation is compared to an axis, and if it is close enough, it will be used

        For generation, the original funciton is used

        (1, 0, 0) ~ axis for topspin 
        (-1, 0, 0) ~ axis for backspin
    """

    threshold = 20
    points = create_rotations(n, max_speed, min_speed)
    print(f"Created {points.shape[0]} samples before filtering topspin")
    mask = np.array([is_top_or_back(axis, threshold) for axis in points])
    points = points[mask]
    return points

def create_initial_orientation_topspin(n:int, max_angle: float, min_angle: float) -> np.ndarray:
    """ Creates initial orientations for topspin

        This function creates random initial orientations that are within the given angle range.
        For the top / backspin dataset, the initial orientation axis is totally random, but the angle will be
        between -80 deg and +80 deg. With this configuration the logo is always visible in the video.

        Args:
            n (int): Number of initial orientations to generate
            max_angle (float): Maximum angle in degrees for the initial orientation
            min_angle (float): Minimum angle in degrees for the initial orientation

        Returns:
            np.ndarray: Array of shape (n, 3) containing the initial orientations (The angle is contained as vector length (degrees))
    """
    orientations = np.zeros((n, 3))
    for i in range(n):
        axis = rotations.random_rotation().get_axis()
        while np.linalg.norm(axis) <= 0.01:
            axis = rotations.random_rotation().get_axis()

        angle_deg = np.random.uniform(min_angle, max_angle)
        axis *= angle_deg / np.linalg.norm(axis)
        orientations[i] = axis

    return orientations
    

def create_topspin_table():
    """ Creates a table and folder structure for the topspin dataset

        This function creates a table that contains all rotations and initial orientations for the topspin dataset.
        The table is saved as a CSV file in the config folder of the dataset.
    """
    path = "/data/lkolmar/datasets/topspin/"
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Dataset folder already exists. Please remove it or choose a different path.")
        sys.exit()

    make_folder_structure(path)
    samples = create_top_back(40)
    print(f"Created {samples.shape[0]} samples")
    initial_orientations = create_initial_orientation_topspin(samples.shape[0], 80, -80)
    print(f"Created {initial_orientations.shape[0]} initial orientations")
    
    df = pd.DataFrame({
        'index': np.arange(len(samples)),
        'rotation_x': samples[:, 0],
        'rotation_y': samples[:, 1],
        'rotation_z': samples[:, 2],
        'initial_rot_x': initial_orientations[:, 0],
        'initial_rot_y': initial_orientations[:, 1],
        'initial_rot_z': initial_orientations[:, 2],
        'finished': False,
        'path': "not set"
    })

    file_path = path + "config/" + "simulation.csv"
    df.to_csv(file_path, index=False)


def create_full_table():
    """ Creates table and structure for the full dataset

        This function creates a table that contains all rotations and initial orientations for the full dataset.
        The table is saved as a CSV file in the config folder of the dataset.
    """
    path = "/data/lkolmar/datasets/full_dataset/"
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Dataset folder already exists. Please remove it or choose a different path.")
        sys.exit()

    make_folder_structure(path)
    n = 30  # number of points per axis # 30 000 samples with n=40
    samples = create_rotations(n, max_speed=80, min_speed=5)
    print(f"Created {samples.shape[0]} samples")
    initial_orientations = create_initial_orientation_topspin(samples.shape[0], 180, -180) # 180 deg is the full range (this is a hack to reuse the function)
    print(f"Created {initial_orientations.shape[0]} initial orientations")
    df = pd.DataFrame({
        'index': np.arange(len(samples)),
        'rotation_x': samples[:, 0],
        'rotation_y': samples[:, 1],
        'rotation_z': samples[:, 2],
        'initial_rot_x': initial_orientations[:, 0],
        'initial_rot_y': initial_orientations[:, 1],
        'initial_rot_z': initial_orientations[:, 2],
        'finished': False,
        'path': "not set"
    })

    file_path = path + "config/" + "simulation.csv"
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # create_topspin_table()
    create_full_table()



# --------------------------------------------------------------------------------------------------
# TODO: clean up from here on
# --------------------------------------------------------------------------------------------------
'''

samples = create_top_back(40)
print(f"Created {samples.shape[0]} samples")
dists = []
for s in samples:
    d = np.linalg.norm(s)
    if int(d) not in dists:
        # print(f"Lenght: {np.linalg.norm(s)}")
        dists.append(int(d))

dists.sort()
print(dists)
print(len(dists))

df = pd.DataFrame({
    'index': np.arange(len(samples)),
    'rotation_x': samples[:, 0],
    'rotation_y': samples[:, 1],
    'rotation_z': samples[:, 2],
    'initial_rot_x': 0.0,
    'initial_rot_y': 0.0,
    'initial_rot_z': 0.0,
    'finished': False,
    'path': ""
})

file_path = path + "config/" + "test.csv"
# df.to_csv(file_path, index=False)

# loaded = pd.read_csv(file_path)

'''