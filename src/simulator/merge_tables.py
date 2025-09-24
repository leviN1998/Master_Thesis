import pandas as pd
import numpy as np


path = "/data/lkolmar/datasets/realistic/"

num_threads = 10
index_offset = 0 # does the first thread start at 0 or 1?

def angle_to_axis(vx, vy, vz, zhat):
    axis = np.array((vx, vy, vz))
    axis_norm = axis / np.linalg.norm(axis)
    reference_norm = np.array(zhat) / np.linalg.norm(np.array(zhat))
    cos_angle = np.dot(axis_norm, reference_norm)
    cos_theta = np.clip(cos_angle, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(abs(cos_theta)))

    return angle_degrees

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
    df = pd.read_csv(path + "config/simulation.csv")
    finished_first = 0
    unfinished_first = 0
    finished_second = 0
    unfinished_second = 0
    for i in range(len(df)):
        if i < 5000:
            if df.iloc[i]["finished"]:
                finished_first += 1
            else:
                unfinished_first += 1
        else:
            if df.iloc[i]["finished"]:
                finished_second += 1
            else:
                unfinished_second += 1

    print(f"First half: {finished_first} finished, {unfinished_first} unfinished")
    print(f"Second half: {finished_second} finished, {unfinished_second} unfinished")

    threshold = 20
    for i in range(len(df)):
        sample = df.iloc[i]
        if not sample["finished"]:
            vx, vy, vz = sample["rotation_x"], sample["rotation_y"], sample["rotation_z"]
            pos_x = angle_to_axis(vx, vy, vz, (1, 0, 0))  # backspin
            neg_x = angle_to_axis(vx, vy, vz, (-1, 0, 0)) # topspin

            if min(pos_x, neg_x) >= threshold:
                df.loc[i, "finished"] = True

    df.to_csv(path + "config/simulation.csv", index=False)

    """



    dfs = []
    for i in range(num_threads):
        df = pd.read_csv(path + f"config/simulation_pid{i+index_offset}.csv")
        dfs.append(df)

    print(f"Loaded {len(dfs)} dataframes with {len(dfs[0])} entries each.")
    merged_df = dfs[0].copy()
    for i in range(len(dfs[0])):
        data = []
        for j in range(num_threads):
            if dfs[j].iloc[i]["finished"] and not (dfs[j].iloc[i]["path"] == "not set"):
                # print(f"Simulation {i} finished in thread {j} with path {dfs[j].iloc[i]['path']}")
                # break
                for d in data:
                    if d["finished"]:
                        # print(f"Warning: Simulation {i} has duplicated entries")
                        pass
                data.append(dfs[j].iloc[i])
                merged_df.iloc[i] = dfs[j].iloc[i]
        
        if not merged_df.iloc[i]["finished"] or merged_df.iloc[i]["path"] == "not set":
            print(f"Warning: Simulation {i} is not finished in any thread")
            merged_df.loc[i, "finished"] = False

    merged_df.to_csv(path + "config/simulation.csv", index=False)
 """