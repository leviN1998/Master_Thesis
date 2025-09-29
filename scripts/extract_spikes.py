import sys
import numpy as np
sys.path.append("src/")
sys.path.append("src/utils/")
import eventIO
import os


path = "/home/lkolmar/Documents/metavision/recordings/"

def load_buffer():
    raw_files = [f for f in os.listdir(path) if f.endswith('.hdf5')]
    buf = eventIO.load_hdf5_metavision(path + raw_files[0])
    buf.ms_to_idx = eventIO.generate_ms_to_idx(buf.get_ts())
    return buf

# ---------------------- 1. ---------------------------------
# spin_values =     [2,  2, 2,  3, 3,  3, 3, 3,  4,  4]
# sidespin_values = [-1, 0, 1, -2, -1, 0, 1, 2, -3, -2]

# ---------------------- 2. ---------------------------------
# spin_values =     [4,  4,  4,  4,  5,  5,  5,  5,  5,  5] 
# sidespin_values = [0,  1,  2,  3, -3, -2, -1,  0,  1,  2] 

# ---------------------- 3. ---------------------------------
# spin_values =     [5,  6,  6,  6,  6,  6,  6,  6,  6,  6] 
# sidespin_values = [3, -4, -3, -2, -1,  0,  1,  2,  3,  4] 

# ---------------------- tmp --------------------------------
spin_values =     [5,  6,  6,  6,  6,  6,  6,  6,  6,  6] 
sidespin_values = [3, -4, -3, -2, -1,  0,  1,  2,  3,  4] 
#                                                      


def convert_recording():
    buf = load_buffer()
    spikes = extract_spikes(buf)

    for i, spike in enumerate(spikes):
        buf_spike = get_events_for_spike(spike, buf)
        bias = [0, 0]
        copy = 0
        for j in range(i):
            if spin_values[j] == spin_values[i] and sidespin_values[j] == sidespin_values[i]:
                copy += 1
                print(f"Found duplicate for spin {spin_values[i]} and sidespin {sidespin_values[i]}. Incrementing index to {copy}.")
        copy = f"({copy})" if copy > 0 else ""
        eventIO.save_hdf5_metavision(buf_spike, path + f"tmp/spin{spin_values[i]}_sidespin{sidespin_values[i]}{copy}.hdf5", bias)
        eventIO.create_video(buf_spike, path + f"tmp/spin{spin_values[i]}_sidespin{sidespin_values[i]}{copy}.mp4", resolution=(1280, 720), fps=30.0, tw=500)
        print(f"Saved spike {i} with {buf_spike.i} events.")



def extract_spikes(buf, threshold=2000):
    event_rates = buf.ms_to_idx[1:] - buf.ms_to_idx[:-1]
    high_rate_idxs = np.where(event_rates > threshold)[0]
    spikes = []
    if len(high_rate_idxs) > 0:
        current_spike = [high_rate_idxs[0]]
        for idx in high_rate_idxs[1:]:
            if idx == current_spike[-1] + 1:
                current_spike.append(idx)
            else:
                spikes.append(current_spike)
                current_spike = [idx]
        spikes.append(current_spike)

    if len(spikes) != len(spin_values):
        if len(spikes) > len(spin_values):
            print(f"Warning: Expected {len(spin_values)} spikes, but found {len(spikes)}. Continuing anyway.")
            spikes = spikes[:len(spin_values)]

        else:
            sys.exit(f"Error: Expected {len(spin_values)} spikes, but found {len(spikes)}")

    return spikes


def get_events_for_spike(spike, buf):
    idx = buf.ms_to_idx[spike[0]], buf.ms_to_idx[spike[-1]]
    x = buf.get_x()[idx[0]:idx[1]]
    y = buf.get_y()[idx[0]:idx[1]]
    p = buf.get_p()[idx[0]:idx[1]]
    ts = buf.get_ts()[idx[0]:idx[1]]

    buf_spike = eventIO.EventBuffer(0)
    buf_spike.x = x
    buf_spike.y = y
    buf_spike.p = p
    buf_spike.ts = ts
    buf_spike.i = x.shape[0]
    return buf_spike


if __name__ == "__main__":
    convert_recording()