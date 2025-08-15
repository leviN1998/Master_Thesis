import sys
import os
sys.path.append("src/utils")
import eventIO

folder = sys.argv[1] if len(sys.argv) > 1 else "."

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    if os.path.isfile(path):
        buf = eventIO.load_hdf5_metavision(path)
        eventIO.save_hdf5(buf, path.replace(".hdf5", "_converted.hdf5"), bias=[0, 65, 65, 0, 0, 0], sensor="Prophesee EVK4")