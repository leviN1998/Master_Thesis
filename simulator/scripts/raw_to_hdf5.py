import h5py
import hdf5plugin
import sys
import numpy as np
from metavision_core.event_io.raw_reader import RawReader
import matplotlib.pyplot as plt
import argparse
import time
import shutil
import os

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        list: A list of arguments provided by the user.
    """
    parser = argparse.ArgumentParser(description="Process sensor type and paired arguments.")
    parser.add_argument("data", nargs="*", help="Pairs of values")
    return parser.parse_args().data


def save_events_to_hdf5(event_reader, sensor, bias, output_file, chunk_size=10_000_000,
                        compression=hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
                        clevel=1, process=-1):
    """
    Reads events from an event reader and saves them in a compressed HDF5 file.

    Args:
        event_reader: Generator or function that yields (x, y, p, t) as NumPy arrays.
        sensor (str): Type of the sensor.
        bias (list): List of bias values.
        output_file (str): Output HDF5 file path.
        chunk_size (int): Number of events per chunk.
        compression: Compression algorithm settings.
        clevel (int): Compression level.
        process (int): Optional process ID for logging.
    """
    width, height = event_reader.width, event_reader.height
    #print(f"{'Process ' + str(process) + ': ' if process != -1 else ''}Sensor width: {width}, height: {height}")
    
    t00 = time.time()

    with h5py.File(output_file, "w") as h5f:
        event_group = h5f.create_group("events")

        dset_x = event_group.create_dataset("x", shape=(0,), maxshape=(None,), dtype="uint16",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        dset_y = event_group.create_dataset("y", shape=(0,), maxshape=(None,), dtype="uint16",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        dset_p = event_group.create_dataset("p", shape=(0,), maxshape=(None,), dtype="uint8",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        dset_t = event_group.create_dataset("t", shape=(0,), maxshape=(None,), dtype="uint64",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)

        h5f.create_dataset("t_offset", data=[0], maxshape=(None,))
        h5f.attrs.update({"width": width, "height": height, "sensor": sensor})
        h5f.create_dataset("bias", data=bias, maxshape=(None,))

        last_time_stamp = 0
        total_events = 0

        while not event_reader.is_done():
            t0 = time.time()
            events = event_reader.load_n_events(chunk_size)
            num_events = len(events)

            dset_x.resize((total_events + num_events,))
            dset_y.resize((total_events + num_events,))
            dset_p.resize((total_events + num_events,))
            dset_t.resize((total_events + num_events,))

            dset_x[total_events: total_events + num_events] = events["x"]
            dset_y[total_events: total_events + num_events] = events["y"]
            dset_p[total_events: total_events + num_events] = events["p"]
            dset_t[total_events: total_events + num_events] = events["t"]

            total_events += num_events

            #print(f"{'Process ' + str(process) + ': ' if process != -1 else ''}Data saved in {time.time() - t0:.2f} seconds")

        ms_to_idx = generate_ms_to_idx(dset_t[:])
        dset_ms = h5f.create_dataset("ms_to_idx", shape=(len(ms_to_idx),), maxshape=(None,), dtype="uint64")
        dset_ms[:] = ms_to_idx

    #print(f"Saved {total_events} events to {output_file} in {time.time() - t00:.2f} seconds")


def generate_ms_to_idx(timestamps, last_index=0, previous_time_stamps=0):
    """
    Generate an optimized mapping of milliseconds to event indices.

    Args:
        timestamps (np.ndarray): Array of event timestamps.
        last_index (int): Starting index for ms_to_idx array.
        previous_time_stamps (int): Offset for previous timestamps.

    Returns:
        np.ndarray: Array mapping milliseconds to event indices.
    """
    if timestamps.size == 0:
        return np.array([], dtype=np.int64)

    timestamps_ms = timestamps // 1_000
    unique_ms, first_indices = np.unique(timestamps_ms, return_index=True)

    max_time = unique_ms[-1] if unique_ms.size > 0 else last_index
    ms_to_idx = np.zeros(int(max_time + 1 - last_index), dtype=np.int64)

    ms_to_idx[unique_ms - last_index] = first_indices + previous_time_stamps
    return replace_zeros(ms_to_idx)


def replace_zeros(arr):
    """
    Replaces zero values within the timestamps.

    Args:
        arr (np.ndarray): Array mapping milliseconds to event indices.

    Returns:
        np.ndarray: Cleaned array with filled zero entries.
    """
    mask = arr == 0
    if mask.sum() == 0:
        return arr
    if arr.sum() == 0:
        return arr

    mask[0] = 0
    valid_idx = np.where(~mask)[0]
    valid_values = arr[valid_idx]
    next_nonzero_idx = np.searchsorted(valid_idx, np.where(mask)[0])
    arr[mask] = valid_values[next_nonzero_idx]

    return arr


def convert_raw_to_hdf5(raw_file, hdf5_file, bias, sensor_type, process=-1):
    """
    Wrapper to convert RAW file to HDF5.

    Args:
        raw_file (str): Input RAW file path.
        hdf5_file (str): Output HDF5 file path.
        bias (str): Bias values as a string.
        sensor_type (str): Sensor type.
        process (int): Process ID (optional).
    """
    reader = RawReader(raw_file, max_events=1_000_000_000)
    bias_values = [float(value) for value in bias.replace("[", "").replace("]", "").split(",")]
    save_events_to_hdf5(reader, sensor_type, bias_values, hdf5_file, process=process)
    del(reader)
    if os.path.exists(raw_file+".tmp_index"):
        os.remove(raw_file+".tmp_index")


def main():
    """
    Entry point of the script.
    """
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Usage: python3 raw_to_hdf5_converter.py raw_filepath hdf5_filepath biases sensor_type [process_id]")
        sys.exit(1)

    args = parse_arguments()
    if len(args) == 5:
        convert_raw_to_hdf5(args[0], args[1], args[2], args[3], int(args[4]))
    else:
        convert_raw_to_hdf5(args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    main()
