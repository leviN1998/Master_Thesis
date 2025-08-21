from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np


@dataclass
class FieldNames:
    """Names of the fields in the event array.

    The class assumes events are provided as a NumPy structured array (or record array)
    with at least the fields for x, y, t (timestamp in microseconds), and p (polarity in {0,1}).
    If your field names differ, customize them here.
    """
    x: str = "x"
    y: str = "y"
    t: str = "t"  # timestamp, integer microseconds preferred
    p: str = "p"  # polarity: 0 or 1


class SpatioTemporalContrastAlgorithm:
    """
    Metavision‑free reimplementation of the Spatio‑Temporal‑Contrast (STC) filter.

    Semantics follow the Metavision SDK description:
      • An event is forwarded only if it is preceded, at the same pixel & same polarity,
        by another event within a configurable time window (``threshold_us``). This keeps
        the *second* event of a burst and filters isolated events.
      • If ``cut_trail=True`` ("STC_CUT_TRAIL"), once such an event passes, all
        *subsequent* events of the same polarity at that pixel are *suppressed until a
        polarity change at that pixel*. If ``cut_trail=False`` ("STC_KEEP_TRAIL"), these
        subsequent events may pass provided they are still within the time window.

    Notes
    -----
    • The filter is stateful per pixel and polarity and is intended to be used in a
      streaming fashion (i.e., call ``process_events`` for successive, time‑sorted chunks).
    • Timestamps should be monotonically non‑decreasing within each processed buffer.
    • This implementation does not depend on Metavision and is "drop‑in similar" in API.
    """

    def __init__(
        self,
        width: int,
        height: int,
        threshold_us: int,
        cut_trail: bool = True,
        field_names: Optional[FieldNames] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if threshold_us < 0:
            raise ValueError("threshold_us must be non‑negative")

        self.width = int(width)
        self.height = int(height)
        self._threshold = int(threshold_us)
        self._cut_trail = bool(cut_trail)
        self.fields = field_names or FieldNames()

        # Per‑pixel last timestamps for each polarity (shape: H x W x 2)
        # Use int64; initialize so that first dt is > threshold → first events are dropped
        self._last_ts = np.full((self.height, self.width, 2), -2**62, dtype=np.int64)
        # Per‑pixel cut‑state: -1 = no cut; 0 = cutting polarity 0; 1 = cutting polarity 1
        self._cut_state = np.full((self.height, self.width), -1, dtype=np.int8)

    # --- API parity helpers -------------------------------------------------
    def get_threshold(self) -> int:
        return self._threshold

    def set_threshold(self, threshold_us: int) -> None:
        if threshold_us < 0:
            raise ValueError("threshold_us must be non‑negative")
        self._threshold = int(threshold_us)

    def get_cut_trail(self) -> bool:
        return self._cut_trail

    def set_cut_trail(self, v: bool) -> None:
        self._cut_trail = bool(v)

    def reset(self) -> None:
        """Reset the internal per‑pixel state."""
        self._last_ts.fill(-2**62)
        self._cut_state.fill(-1)

    # --- Core processing ----------------------------------------------------
    def process_events(self, events: np.ndarray) -> np.ndarray:
        """Filter a time‑sorted buffer of events.

        Parameters
        ----------
        events : np.ndarray
            Structured array with fields (x, y, t, p). Timestamps should be int‑like
            (preferably microseconds). Polarity should be in {0,1} (bool/uint8/int8).

        Returns
        -------
        np.ndarray
            Array of the same dtype containing only the events that pass STC.
        """
        if events.size == 0:
            return events[:0]

        # Validate fields exist
        for fname in (self.fields.x, self.fields.y, self.fields.t, self.fields.p):
            if fname not in events.dtype.names:
                raise ValueError(
                    f"Input events dtype must have field '{fname}'. Present fields: {events.dtype.names}"
                )

        out = np.empty_like(events)  # worst‑case allocate; we will slice at the end
        out_count = 0

        xs = events[self.fields.x]
        ys = events[self.fields.y]
        ts = events[self.fields.t]
        ps = events[self.fields.p]

        H, W = self.height, self.width
        thr = np.int64(self._threshold)
        cut_trail = self._cut_trail

        # Process sequentially for correctness with per‑pixel state
        for i in range(events.shape[0]):
            x = int(xs[i])
            y = int(ys[i])
            t = np.int64(ts[i])
            p = int(ps[i]) & 1

            # Bounds check (defensive; can be removed for speed once trusted)
            if not (0 <= x < W and 0 <= y < H):
                continue  # silently discard out‑of‑range

            # If cut is active for the *other* polarity, encountering a polarity change ends it
            if self._cut_state[y, x] != -1 and self._cut_state[y, x] != p:
                self._cut_state[y, x] = -1

            # If cut is active for this polarity → drop immediately
            if cut_trail and self._cut_state[y, x] == p:
                # still update last_ts so huge gaps don’t cause overflow elsewhere
                self._last_ts[y, x, p] = t
                continue

            last = self._last_ts[y, x, p]
            dt = t - last

            if dt <= thr:
                # This is at least the 2nd event in a burst → forward
                out[out_count] = events[i]
                out_count += 1
                self._last_ts[y, x, p] = t
                if cut_trail:
                    # Start cutting this polarity until a polarity change occurs
                    self._cut_state[y, x] = p
            else:
                # First event of a (new) burst → drop but remember timestamp
                self._last_ts[y, x, p] = t

        return out[:out_count]


# ----------------------------- Convenience ---------------------------------

def make_events(
    xs: Iterable[int], ys: Iterable[int], ts: Iterable[int], ps: Iterable[int],
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """Create a standard structured event array for quick testing.

    Default dtype is (x:uint16, y:uint16, t:int64, p:uint8).
    """
    if dtype is None:
        dtype = np.dtype([
            ("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.uint8)
        ])
    arr = np.empty(len(list(xs)), dtype=dtype)
    arr["x"] = np.fromiter(xs, dtype=np.uint16, count=arr.shape[0])
    arr["y"] = np.fromiter(ys, dtype=np.uint16, count=arr.shape[0])
    arr["t"] = np.fromiter(ts, dtype=np.int64, count=arr.shape[0])
    arr["p"] = np.fromiter(ps, dtype=np.uint8, count=arr.shape[0])
    return arr


if __name__ == "__main__":
    # Minimal sanity test
    # Pixel (10,10), polarity 1 emits events at t=[0, 200, 300, 10000, 10100] us
    # With threshold=1000 us:
    #  - t=0 (first in burst) is dropped
    #  - t=200 (second) is forwarded
    #  - t=300: with CUT_TRAIL=True, it is cut; with CUT_TRAIL=False, forwarded
    #  - t=10000: dt=9700>thr → first of new burst → dropped
    #  - t=10100: second → forwarded
    events = make_events(
        xs=[10, 10, 10, 10, 10],
        ys=[10, 10, 10, 10, 10],
        ts=[0, 200, 300, 10000, 10100],
        ps=[1, 1, 1, 1, 1],
    )

    stc_keep = SpatioTemporalContrastAlgorithm(640, 480, threshold_us=1000, cut_trail=False)
    stc_cut = SpatioTemporalContrastAlgorithm(640, 480, threshold_us=1000, cut_trail=True)

    out_keep = stc_keep.process_events(events)
    out_cut = stc_cut.process_events(events)

    print("KEEP_TRAIL →", out_keep[["t", "x", "y", "p"]].tolist())
    print("CUT_TRAIL  →", out_cut[["t", "x", "y", "p"]].tolist())