import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import os

import sys
import numpy as np
sys.path.append("../")
sys.path.append("../src")
sys.path.append("../src/models")
sys.path.append("../src/data")
sys.path.append("../src/utils")
sys.path.append("../src/data/components/")
sys.path.append("../src/models/components/")
sys.path.append("../src/utils/IEBCS")
sys.path.append("../src/utils/IEBCS/representations")
import eventIO
import event_representations
from player import VideoPlayer
import cv2
from event_buffer import EventBuffer

ArrayLikeImage = np.ndarray
Point = Tuple[float, float]


def _ensure_bgr(img: ArrayLikeImage) -> ArrayLikeImage:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError("Unsupported image format")

def _load_images(images: List[Union[str, ArrayLikeImage]]) -> List[ArrayLikeImage]:
    loaded = []
    for im in images:
        if isinstance(im, str):
            arr = cv2.imread(im, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(f"Could not read image: {im}")
            loaded.append(_ensure_bgr(arr))
        elif isinstance(im, np.ndarray):
            loaded.append(_ensure_bgr(im))
        else:
            raise TypeError("images must be list of file paths or numpy arrays")
    return loaded

def _linear_interpolate_positions(pts: List[Optional[Point]]) -> np.ndarray:
    N = len(pts)
    out = np.full((N, 2), np.nan, dtype=np.float32)
    known = [i for i, p in enumerate(pts) if p is not None]
    if not known:
        return out
    for i in known:
        out[i] = pts[i]
    for a, b in zip(known[:-1], known[1:]):
        xa, ya = out[a]
        xb, yb = out[b]
        span = b - a
        for k in range(1, span):
            t = k / span
            out[a + k, 0] = (1 - t) * xa + t * xb
            out[a + k, 1] = (1 - t) * ya + t * yb
    return out


def annotate_every_kth_and_interpolate(
    images: List[Union[str, ArrayLikeImage]],
    step: int = 10,
    window_name: str = "Annotation Tool",
    circle_radius: int = 6,
    font_scale: float = 0.5
) -> np.ndarray:
    imgs = _load_images(images)
    N = len(imgs)
    if N == 0:
        raise ValueError("No images provided.")
    if step <= 0:
        raise ValueError("step must be >= 1")

    sample_idxs = list(range(0, N, step))
    if sample_idxs[-1] != N - 1:
        sample_idxs.append(N - 1)

    positions: List[Optional[Point]] = [None] * N
    has_point = {i: False for i in sample_idxs}

    state = {"sp": 0, "await_second_click": False, "redraw": True, "running": True}

    def on_mouse(event, x, y, flags, param):
        if not state["running"]:
            return
        sp = state["sp"]
        frame_idx = sample_idxs[sp]
        if event == cv2.EVENT_LBUTTONDOWN:
            if not has_point[frame_idx]:
                positions[frame_idx] = (float(x), float(y))
                has_point[frame_idx] = True
                state["await_second_click"] = True
                state["redraw"] = True
            else:
                if state["await_second_click"]:
                    if sp < len(sample_idxs) - 1:
                        state["sp"] += 1
                        state["await_second_click"] = has_point[sample_idxs[state["sp"]]]
                        state["redraw"] = True
                    else:
                        state["running"] = False

    def draw_current():
        sp = state["sp"]
        frame_idx = sample_idxs[sp]
        img = imgs[frame_idx].copy()
        help_lines = [
            f"Frame {frame_idx+1}/{N} (Sample {sp+1}/{len(sample_idxs)})",
            "Links-Klick: Punkt setzen / weiter",
            "SPACE: zurueck | ESC: beenden"
        ]
        y0 = 25
        for i, line in enumerate(help_lines):
            cv2.putText(img, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        p = positions[frame_idx]
        if p is not None:
            cx, cy = int(round(p[0])), int(round(p[1]))
            cv2.circle(img, (cx, cy), circle_radius, (0,255,0), -1, lineType=cv2.LINE_AA)
        cv2.imshow(window_name, img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while state["running"]:
        if state["redraw"]:
            draw_current()
            state["redraw"] = False
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if state["sp"] > 0:
                state["sp"] -= 1
                state["await_second_click"] = has_point[sample_idxs[state["sp"]]]
                state["redraw"] = True
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    # --- Interpolation ---
    interpolated = _linear_interpolate_positions(positions)

    # --- Visualisierung auf schwarzem Bild ---
    H, W = imgs[0].shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for i, p in enumerate(interpolated):
        if np.isnan(p[0]) or np.isnan(p[1]):
            continue
        cx, cy = int(round(p[0])), int(round(p[1]))
        if i in sample_idxs and has_point[i]:   # manuell
            color = (0, 0, 255)  # Rot
        else:                                    # interpoliert
            color = (0, 255, 0)  # Grün
        cv2.circle(canvas, (cx, cy), circle_radius, color, -1, lineType=cv2.LINE_AA)

    cv2.imshow("Final Positions", canvas)
    cv2.waitKey(0)
    cv2.destroyWindow("Final Positions")

    return interpolated


# --- Beispielnutzung ---
if __name__ == "__main__":
    # Beispiel: Liste von Bildpfaden
    # images = [f"/path/to/frames/frame_{i:05d}.png" for i in range(0, 300)]
    # Oder: bereits geladene Arrays (np.ndarray), gemischt ist auch ok.

    # Ergebnis: (N, 2)-Array mit x,y; NaN an Rändern ohne Extrapolation.
    # positions = annotate_every_kth_and_interpolate(images, step=10)
    # print(positions.shape, positions[:15])
    filepath = f"/home/lkolmar/Documents/metavision/recordings/dataset_full_ball-gun/cut/"
    filenames = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
    
    for i in range(len(filenames)):
        print(f"Sample {i+1}/{len(filenames)}")
        cur_filename = filenames[i]
        print(cur_filename)
        buf = eventIO.load_hdf5(filepath + cur_filename)
        timeframe_us = 1000
        frames = eventIO.buffer_to_video(buf, tw_us=timeframe_us)

        positions = annotate_every_kth_and_interpolate(frames, step=10)
        print(positions.shape, positions[:5])
        np.save(f"/home/lkolmar/Documents/metavision/recordings/dataset_full_ball-gun/annotations/{cur_filename.replace('.hdf5', '')}.npy", positions)