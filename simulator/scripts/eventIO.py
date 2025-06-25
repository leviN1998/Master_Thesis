''' File to contain custom event saving and loading functions

    This file contains functions to handle event-Buffers from IEBCS and
    save/load them as hdf5 files.

    It also contains tools to get information about tghe event content of a buffer,
    such as the total time passed, or the number of events and other useful stuff.

    A function to create a video from the events is also included.
'''

import sys
import h5py
import cv2
import numpy as np
# import the IECBS simulator
sys.path.append("../src")
from dvs_sensor import *


'''

events.h5
├── events
│   ├── polarity       → int8 or bool, shape (N,)
│   ├── x              → uint16 or int, shape (N,)
│   ├── y              → uint16 or int, shape (N,)
│   └── timestamp      → uint64 or float64, shape (N,)
└── metadata (attrs)
    ├── resolution     → (width, height)
    └── source         → "DAVIS346", "CustomCam", etc.

'''


def save_hdf5(event_buffer: EventBuffer, filename: str):
    """ Save the event buffer as hdf5 file
        Args:
            event_buffer: EventBuffer to save
            filename: name of the file to save to
    """
    with h5py.File(filename, 'w') as f:
        grp = f.create_group('events')
        grp.create_dataset('p', data=event_buffer.get_p())
        grp.create_dataset('x', data=event_buffer.get_x())
        grp.create_dataset('y', data=event_buffer.get_y())
        grp.create_dataset('t', data=event_buffer.get_ts())

        f.attrs['resolution'] = (5, 5) # TODO: add good attributes



def load_hdf5(filename: str) -> EventBuffer:
    """ Load the event buffer from an hdf5 file
        Args:
            filename: name of the file to load from
        Returns:
            EventBuffer with the loaded events
    """
    buf = EventBuffer(0)
    with h5py.File(filename, 'r') as f:
        data = f['events']
        buf.x = data['x'][:]
        buf.y = data['y'][:]
        buf.p = data['p'][:]
        buf.ts = data['t'][:]
        buf.i = buf.ts.shape[0]
    return buf



def print_event_info(event_buffer: EventBuffer):
    """ Print information about the event buffer
        Args:
            event_buffer: EventBuffer to print information about
    """
    print("Event Buffer Information:")
    print(f"Number of events: {event_buffer.i}")
    print(f"Time range [us]: {event_buffer.ts[0]} - {event_buffer.ts[-1]}")
    print(f"Total time [s]: {(event_buffer.ts[-1] - event_buffer.ts[0]) / 1e6:.6f}")
    total_rounds = 2.0
    print(f"Recalculated RPS (using {total_rounds} total rounds): {total_rounds / ((event_buffer.ts[-1] - event_buffer.ts[0]) / 1e6):.2f} RPS")
    print(f"Resolution: {np.max(event_buffer.x) + 1} x {np.max(event_buffer.y) + 1}")
    print(f"Polarity values: {np.unique(event_buffer.p)}")
    print(f"Events per pixel: {event_buffer.i / ((np.max(event_buffer.x) + 1) * (np.max(event_buffer.y) + 1)):.2f}")
    print(f"Sample events (first 10):")
    for i in range(min(10, event_buffer.i)):
        print(f"Event {i}: t={event_buffer.ts[i]}, x={event_buffer.x[i]}, y={event_buffer.y[i]}, p={event_buffer.p[i]}")



def create_video(events: EventBuffer, save_filename: str, resolution=(1280, 720), fps=30.0, tw=1000):
    """ Create a video from the events in an EventBuffer
        Args:
            events: EventBuffer with the events
            save_filename: name of the file to save the video to
            resolution: resolution of the video (width, height)
            fps: frames per second for the video
            tw: time window for each frame in ms
    """
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_filename, fourcc, fps, (resolution[1], resolution[0]))
    
    ts = events.get_ts()
    x = events.get_x()
    y = events.get_y()
    p = events.get_p()

    res = [resolution[1], resolution[0]]
    out = cv2.VideoWriter(save_filename, fourcc, fps, (res[1], res[0]))
    img = np.zeros((res[0], res[1]), dtype=np.uint8)
    tsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
    indsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
    
    for t in range(ts[0], ts[-1], tw):
        ind = np.where((ts > t)&(ts < t + tw))
        tsurface[:, :] = 0
        tsurface[y[ind], x[ind]] = t + tw
        indsurface[y[ind], x[ind]] = p[ind]
        ind = np.where(tsurface > 0)
        img[:, :] = 125
        img[ind] = 125 + (2 * indsurface[ind] - 1) * np.exp(-(t + tw - tsurface[ind].astype(np.float32))/ (tw/30)) * 125
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_VIRIDIS)
        
        out.write(img_c)
    out.release()

if __name__ == "__main__":
    ev = load_hdf5("../data/output/spinning_ball.hdf5")
    print_event_info(ev)
    create_video(ev, "../data/output/spinning_ball_events.avi", resolution=(1280, 720), fps=20.0, tw=50)