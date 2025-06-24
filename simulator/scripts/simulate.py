""" Simulates n datapoints of the specified dataset

    This script needs a path with structure as specified in "create_sim_table.py"
    It loads the configurations and simulates n simulations.
    The class simulator.py is used for every individual run

"""

import numpy as np
import yaml
import pandas as pd
import simulator
import rotations
import logger

path = "../data/test_dataset/"
n = 10

# load configs
with open(path + "config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print("Loaded config:", config)

simulation_df = pd.read_csv(path + "config/simulation.csv")
print(simulation_df[:100])

print(f"{simulation_df.iloc[0]}")
data = simulation_df.iloc[0]

# create simulator instance
initial_orientation = rotations.Rotation()
print(data["initial_rot_x"], data["initial_rot_y"], data["initial_rot_z"])
initial_orientation.set_axis(data["initial_rot_x"], data["initial_rot_y"], data["initial_rot_z"])
spin = rotations.Rotation()
print(data["rotation_x"], data["rotation_y"], data["rotation_z"])
spin.set_axis(data["rotation_x"], data["rotation_y"], data["rotation_z"])
basic_logger = logger.Logger()

sim = simulator.Simulator(config, path, spin, initial_orientation, basic_logger, simulation_nr=0)
sim.run_simulation()