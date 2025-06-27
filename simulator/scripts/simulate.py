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
import sys
import time


path = "../data/test_dataset/"


if __name__ == "__main__":
    n = int(sys.argv[1])              # number of simulations to run
    offset = int(sys.argv[2])         # offset to start from

    with open(path + "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    simulation_df = pd.read_csv(path + "config/simulation.csv")
    print(simulation_df[:100])

    print("Applying offset")
    print(simulation_df[offset:offset + 100])

    basic_logger = logger.Logger(path=path + "tmp/")
    basic_logger.info(f"Running {n} simulations with offset {offset}")
    for i in range(n):
        start_ts = time.time()
        index = i + offset
        basic_logger.info(f"Running simulation {i + 1} of {n} with index {index}")
        data = simulation_df.iloc[index]
        # create simulator instance
        initial_orientation = rotations.Rotation()
        initial_orientation.set_axis(data["initial_rot_x"], data["initial_rot_y"], data["initial_rot_z"])
        spin = rotations.Rotation()
        spin.set_axis(data["rotation_x"], data["rotation_y"], data["rotation_z"])
        if data["finished"] == True:
            basic_logger.info(f"Simulation {index} already finished, skipping.")
            continue

        sim = simulator.Simulator(config, path, spin, initial_orientation, basic_logger, simulation_nr=index)
        sim.run_simulation()
        simulation_df.loc[index, "finished"] = True
        simulation_df.loc[index, "path"] = f"data/{simulator.get_num_string(index)}/{simulator.get_num_string(index)}_"
        simulation_df.to_csv(path + "config/simulation.csv", index=False)
        duration_s = time.time() - start_ts
        estimated_time = (n - i - 1) * duration_s
        estimated_time /= 60
        estimated_time /= 60
        basic_logger.info(f"Simulation {index} finished in {time.time() - start_ts:.2f} seconds. Estimated time left {estimated_time:.2f} hours.")

    basic_logger.info("All simulations finished.")
    basic_logger.close()
    print("Finished!")