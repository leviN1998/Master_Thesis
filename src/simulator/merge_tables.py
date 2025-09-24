import pandas as pd
import numpy as np


path = "/data/lkolmar/datasets/realistic/"

num_threads = 10
index_offset = 0 # does the first thread start at 0 or 1?

def angle_to_axis(vx, vy, vz, zhat):
    norm = np.sqrt(vx*vx + vy*vy + vz*vz)
    if norm == 0:
        return np.nan
    # Skalarprodukt durch Norm => cos(theta)
    cos_theta = (vx*zhat[0] + vy*zhat[1] + vz*zhat[2]) / norm
    # numerische Stabilisierung
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

if __name__ == "__main__":
    dfs = []
    for i in range(num_threads):
        df = pd.read_csv(path + f"config/simulation_pid{i+index_offset}.csv")
        dfs.append(df)

    print(f"Loaded {len(dfs)} dataframes with {len(dfs[0])} entries each.")
    merged_df = dfs[0].copy()
    for i in range(len(dfs[0])):
        data = []
        for j in range(num_threads):
            if dfs[j].iloc[i]["finished"]:
                for d in data:
                    if d["finished"]:
                        print(f"Warning: Simulation {i} has duplicated entries")
                data.append(dfs[j].iloc[i])
                merged_df.iloc[i] = dfs[j].iloc[i]
        
        if not merged_df.iloc[i]["finished"]:
            print(f"Warning: Simulation {i} is not finished in any thread")

    merged_df.to_csv(path + "config/simulation.csv", index=False)
 