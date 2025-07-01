import pandas as pd


path = "/data/lkolmar/datasets/topspin/"
num_threads = 2

if __name__ == "__main__":
    dfs = []
    for i in range(num_threads):
        df = pd.read_csv(path + f"config/simulation_pid{i+1}.csv")
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

    merged_df.to_csv("../data/topspin//simulation.csv", index=False)
