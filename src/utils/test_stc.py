import stc_filter
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_base import EventCD

import sys
sys.path.append("IEBCS/")
from player import VideoPlayer
import eventIO
import numpy as np
import event_representations
import matplotlib.pyplot as plt

buf = eventIO.load_hdf5("/home/lkolmar/Documents/metavision/recordings/dataset_full_ball-gun/hdf5/" + "spin_0_rec1_converted.hdf5")


stc_filter_mv = SpatioTemporalContrastAlgorithm(1280, 720, 5000, True)
event_buf = stc_filter_mv.get_empty_output_buffer()
dtype = np.dtype([
            ("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.int16)
        ])
evs = np.empty(buf.get_x().shape[0], dtype=EventCD)
evs["x"] = buf.get_x()
evs["y"] = buf.get_y()
evs["p"] = buf.get_p()
evs["t"] = buf.get_ts()

print(evs.shape)
# print(evs[0])
# stc_filter_mv.process_events(evs, event_buf)
stc = stc_filter.SpatioTemporalContrastAlgorithm(1280, 720, 5000, True)
out = stc.process_events(evs)



out = event_buf.numpy(True).copy()
print(out.shape)
# print(evs.shape)

buf = eventIO.EventBuffer(0)

buf.x = out["x"]
buf.y = out["y"]
buf.p = out["p"]
buf.ts = out["t"]
buf.i = out.shape[0]
buf.ms_to_idx = eventIO.generate_ms_to_idx(buf.get_ts())
print(buf.i)


event_rates = buf.ms_to_idx[1:] - buf.ms_to_idx[:-1]

threshold = 600  # Set a threshold for event rates [ev/ms]
high_rate_idxs = np.where(event_rates > threshold)[0]

# Group consecutive indices into individual spikes
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

print(len(spikes))
print(spikes[0][0], spikes[0][-1])


idx_spike1 = buf.ms_to_idx[spikes[0][0]], buf.ms_to_idx[spikes[0][-1]]
x_spike_1 = buf.get_x()[idx_spike1[0]:idx_spike1[1]]
y_spike_1 = buf.get_y()[idx_spike1[0]:idx_spike1[1]]
p_spike_1 = buf.get_p()[idx_spike1[0]:idx_spike1[1]]
ts_spike_1 = buf.get_ts()[idx_spike1[0]:idx_spike1[1]]

buf_spike1 = eventIO.EventBuffer(0)
buf_spike1.x = x_spike_1
buf_spike1.y = y_spike_1
buf_spike1.p = p_spike_1
buf_spike1.ts = ts_spike_1
buf_spike1.i = x_spike_1.shape[0]

timeframe_us = 1000
start_ts = buf_spike1.get_ts().min()
end_ts = buf_spike1.get_ts().max()
end_ts += timeframe_us
end_ts = int(end_ts)
frames = []
for i in range(start_ts, end_ts, timeframe_us):
    idxs = np.where((buf_spike1.get_ts() >= i) & (buf_spike1.get_ts() < i + timeframe_us))[0]
    # print(idxs.shape)
    if len(idxs) == 0:
        frames.append(np.zeros((720, 1280), dtype=np.uint8))  # Append an empty frame if no events
        print(f"Empty frame at {i}")
    else:
        frame = event_representations.events_to_image(buf_spike1.get_x()[idxs], buf_spike1.get_y()[idxs], buf_spike1.get_ts()[idxs], buf_spike1.get_p()[idxs])
        frames.append(frame)

print(f"Generated {len(frames)} frames")

# Map background to dark blue and event values to light blue
frames_img = []
for frame in frames:
    norm_frame = ((frame / 12) * 255).clip(0, 255).astype(np.uint8)
    img = np.zeros((norm_frame.shape[0], norm_frame.shape[1], 3), dtype=np.uint8)
    # Set background to dark blue
    img[:, :, 2] = 40  # B
    img[:, :, 1] = 0   # G
    img[:, :, 0] = 0   # R
    # Set event pixels to light blue
    mask = norm_frame > 0
    # print(norm_frame.shape, frame.shape, mask.shape, img.shape)
    img[mask, 2] = 220  # B
    img[mask, 1] = 180  # G
    img[mask, 0] = 100  # R
    frames_img.append(img)


app = VideoPlayer(frames_img, fps=30)
app.mainloop()