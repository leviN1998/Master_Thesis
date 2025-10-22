import os
import sys
import shutil




path = "/data/lkolmar/datasets/spindoe_topspin/preprocessed/"

for folder in os.listdir(path):
    old_file = folder + "/" + folder +"_events.hdf5"
    new_file = folder + "/" + folder + "_roi.hdf5"
    try:
        shutil.move(path + old_file, path + new_file)
    except Exception as e:
        print(f"Error renaming {old_file} to {new_file}: {e}")