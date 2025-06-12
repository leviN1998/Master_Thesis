""" Simulates n datapoints of the specified dataset

    This script needs a path with structure as specified in "create_sim_table.py"
    It loads the configurations and simulates n simulations.
    The class simulator.py is used for every individual run

"""

import numpy as np
import yaml
import pandas as pd

path = "../data/test_dataset/"
n = 10

# load configs
with open(path + "config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    # print("Loaded config:", config)

simulation_df = pd.read_csv(path + "config/simulation.csv")
print("Loaded simulation table:", simulation_df[:100])