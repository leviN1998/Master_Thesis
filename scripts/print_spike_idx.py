import os
import shutil


def get_spike_idxs(src_folder):
    idx_dict = {}
    for filename in os.listdir(src_folder):
        name, ext = os.path.splitext(filename)
        words = name.split('_')
        if len(words) > 3:
            print(f"Error: filename {filename} does not conform to expected format.")
        else:
            if "-" in words[0]:
                spin = words[0][-2:]
            else:
                spin = words[0][-1]
            if "-" in words[1]:
                sidespin = words[1][-2:]
            else:
                sidespin = words[1][-1]
            idx = int(words[2])
            if (spin, sidespin) in idx_dict and idx_dict[(spin, sidespin)] > idx:
                pass
            else:
                idx_dict[(spin, sidespin)] = idx
            

    print("Extracted the following indices:")
    for key, value in idx_dict.items():
        print(f"Spin: {key[0]}, Sidespin: {key[1]} -> Index: {value}")

    print()
    print(f"Total unique (spin, sidespin) pairs: {len(idx_dict)}")
                


if __name__ == "__main__":
    src_folder = "/home/lkolmar/Documents/metavision/recordings/processed/"
    get_spike_idxs(src_folder)