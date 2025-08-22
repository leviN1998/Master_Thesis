import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import h5py
import eventIO
import event_representations

from src.utils.IEBCS import event_buffer

mv_dtype = {'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}

class VideoPlayer(tk.Tk):
    def __init__(self, frames, fps=30):
        super().__init__()
        self.title("NumPy Video Player")
        self.frames = [self._to_rgb_uint8(f) for f in frames]
        self.num_frames = len(self.frames)
        if self.num_frames == 0:
            raise ValueError("frames-Liste ist leer.")
        self.fps_base = fps
        self.speed = 1.0
        self.playing = False
        self.idx = 0
        self._after_id = None
        self._imgtk = None
        self._updating_seek = False      # <-- Guard gegen Reentrancy
        self._destroyed = False

        # --- UI ---
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, columnspan=8, padx=8, pady=(8,4))

        # Seekbar
        self.seek = ttk.Scale(self, from_=0, to=self.num_frames-1,
                              orient="horizontal", command=self._on_seek,
                              length=600)
        self.seek.grid(row=1, column=0, columnspan=8, padx=8, pady=4, sticky="ew")

        # Play/Pause
        self.play_btn = ttk.Button(self, text="▶ Play", command=self.play)
        self.pause_btn = ttk.Button(self, text="⏸ Pause", command=self.pause)
        self.play_btn.grid(row=2, column=0, padx=4, pady=8, sticky="ew")
        self.pause_btn.grid(row=2, column=1, padx=4, pady=8, sticky="ew")

        # Geschwindigkeiten
        speeds = [0.1, 0.2, 0.5, 1, 2, 5]
        for j, spd in enumerate(speeds):
            ttk.Button(self, text=f"x{spd}", command=lambda s=spd: self.set_speed(s))\
                .grid(row=2, column=2+j, padx=4, pady=8, sticky="ew")

        for c in range(8):
            self.grid_columnconfigure(c, weight=1)

        # erstes Bild
        self._show_frame(self.idx)

        # Shortcuts
        self.bind("<space>", lambda e: (self.pause() if self.playing else self.play()))
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(+1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Hilfsfunktionen ----------
    def _to_rgb_uint8(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:
            rgb = arr[..., :3].astype(np.float32)
            a = arr[..., 3:4].astype(np.float32)/255.0
            arr = (rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
        else:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _show_frame(self, i):
        frame = self.frames[i]
        img = Image.fromarray(frame)
        self._imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.configure(image=self._imgtk)

        # Seekbar-Wert sicher setzen (Guard + after_idle vermeidet Reentrancy)
        def _set_seek():
            if self._destroyed:
                return
            self._updating_seek = True
            try:
                self.seek.set(i)
            finally:
                self._updating_seek = False
        self.after_idle(_set_seek)

    def _tick(self):
        if not self.playing:
            return
        self.idx += 1
        if self.idx >= self.num_frames:
            self.idx = 0  # Loop
        self._show_frame(self.idx)
        delay = max(1, int(1000 / (self.fps_base * self.speed)))
        self._after_id = self.after(delay, self._tick)

    def _on_seek(self, value):
        # Wird durch Benutzerbewegung der Seekbar aufgerufen.
        if self._updating_seek:
            return  # programmatische Änderung -> ignoriere Callback
        try:
            i = int(float(value))
        except Exception:
            return
        i = max(0, min(self.num_frames - 1, i))
        self.idx = i
        self._show_frame(self.idx)

    # ---------- Controls ----------
    def play(self):
        if self.playing:
            return
        self.playing = True
        self._tick()

    def pause(self):
        self.playing = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def set_speed(self, s):
        self.speed = float(s)

    def step(self, delta):
        self.pause()
        self.idx = (self.idx + delta) % self.num_frames
        self._show_frame(self.idx)

    def _on_close(self):
        self._destroyed = True
        self.pause()
        self.destroy()


def img_from_array(arr):
    ev_it = eventIO.EventIterator(arr, tw_us=1000)
    imgs = []
    for evs in ev_it:
        x, y, p, ts = evs
        frame = event_representations.events_to_image(x, y, ts, p)
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
        imgs.append(img)
        
    return imgs


def img_from_file(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['events']
        x = data['x'][:]
        y = data['y'][:]
        p = data['p'][:]
        ts = data['t'][:]
        
    data = np.array(list(zip(x, y, p, ts)), dtype=mv_dtype)
    return img_from_array(data)


# ---------- Demo ----------
if __name__ == "__main__":
    H, W = 240, 320
    frames = []
    for t in range(120):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        r = 30
        cx = int((W/2) + (W/3)*np.sin(2*np.pi*t/60))
        cy = int((H/2) + (H/4)*np.cos(2*np.pi*t/45))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx-cx)**2 + (yy-cy)**2 <= r*r
        img[..., 1] = 40
        img[mask] = (255, 120, 0)
        img[10:20, (t*3) % W: ((t*3) % W)+60] = (0, 200, 255)
        frames.append(img)

    app = VideoPlayer(frames, fps=30)
    app.mainloop()
