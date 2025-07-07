""" File for manipulating raw event recordings and transforming them into different representations.

    Inputs are not EventBuffers, because the DataLoader shoul dbe able to work on Hdf5 for dynamic loading

    Inspired by https://github.com/TimoStoff/event_utils
"""

import numpy as np
import cv2


def events_to_image(xs, ys, ts, ps, sensor_size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """Convert an event buffer to an image representation.

    Args:
        xs (np.ndarray): The x-coordinates of the events.
        ys (np.ndarray): The y-coordinates of the events.
        ts (np.ndarray): The timestamps of the events.
        ps (np.ndarray): The polarities of the events. (could also be counts) -> this part will be visualized
        sensor_size (tuple[int, int]): The size of the sensor (width, height). Defaults to (1280, 720).

    Returns:
        np.ndarray: [x, y] The image representation of the events, where each pixel value corresponds to the normalized
    """
    assert len(xs) == len(ys) == len(ts) == len(ps), "All event arrays must have the same length."
    # Create an empty image
    image = np.zeros(sensor_size, dtype=np.float32)
    max_p = ps.max()
    min_p = ps.min()
    # Normalize polarities to the range [0, 255]
    ps_normalized = ((ps - min_p) / (max_p - min_p) * 255).astype(np.uint8)
    # Map events to the image
    for x, y, p in zip(xs, ys, ps_normalized):
        image[y, x] += p
    return image


def get_voxel_grid_as_image(voxelgrid: np.ndarray) -> np.ndarray:
    """ Convert a voxel grid to an image representation.
    
        Args:
            voxelgrid (np.ndarray): [B, x, y] The voxel grid to convert.
    
        Returns:
            np.ndarray: The image representation of the voxel grid.
    """
    images = []
    splitter = np.ones((voxelgrid.shape[1], 2))*np.max(voxelgrid)
    for time_bin in voxelgrid:
        images.append(time_bin / np.max(time_bin))
        images.append(splitter)
    
    images.pop()
    sidebyside = np.hstack(images)
    sidebyside = cv2.normalize(sidebyside, None, 0, 255, cv2.NORM_MINMAX)
    return sidebyside



def events_to_voxel(xs, ys, ts, ps, num_bins: int = 10, sensor_size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """ Convert an event buffer to a voxel representation.
    
        Important: This function converts all events ointo a single voxel grid.
        For a whole simulation / video this should be used on every frame
        This will be the input to the model at every recurrent time step.

        -> It is a good question how the individual voxels should look like
           In this case we just add all events and dont take their polarity into account.
    
    Args:
        xs (np.ndarray): The x-coordinates of the events. (All events need to be sorted by time)
        ys (np.ndarray): The y-coordinates of the events.
        ts (np.ndarray): The timestamps of the events.
        ps (np.ndarray): The polarities of the events.
        num_bins (int): The number of bins for the voxel grid. Defaults to 10.
        sensor_size (tuple[int, int]): The size of the sensor (width, height). Defaults to (1280, 720).
    Returns:
        np.ndarray: [B, x, y] The voxel grid representation of the events.
    """
    assert len(xs) == len(ys) == len(ts) == len(ps), "All event arrays must have the same length."

    # Create the voxel grid
    voxel_grid = np.zeros((num_bins, sensor_size[1], sensor_size[0]),dtype=np.float32)
    t0 = ts.min()
    t1 = ts.max()
    bin_size = (t1 - t0) / num_bins
    img_size = (sensor_size[0]+1, sensor_size[1]+1)
    
    for b in range(num_bins):
        # Get the events in the current bin
        mask = (ts >= t0 + b * bin_size) & (ts < t0 + (b + 1) * bin_size)
        xs_bin = xs[mask]
        ys_bin = ys[mask]
        ps_bin = ps[mask]

        coords = np.stack((ys_bin, xs_bin))
        try:
            abs_coords = np.ravel_multi_index(coords, img_size)
        except ValueError:
            print("Issue with input arrays! minx={}, maxx={}, miny={}, maxy={}, coords.shape={}, \
                    sum(coords)={}, sensor_size={}".format(np.min(xs), np.max(xs), np.min(ys), np.max(ys),
                    coords.shape, np.sum(coords), img_size))
            raise ValueError
        v = np.bincount(abs_coords, minlength=img_size[0] * img_size[1])
        v = v.reshape(img_size)
        voxel_grid[b] = v[0:sensor_size[0], 0:sensor_size[1]]

    return voxel_grid


# TODO: think about neg/pos voxels
# TODO: check if dataloading is fast enought -> TimoStoff has some torch dataloading stuff