import json
from pathlib import Path

import h5py
import numpy as np
import scipy as sp


def create_waterfall(data_filename, out_filename, axis=1):
    """ Reads the data and creates a waterfall by adding all the pixels in the vertical direction."""
    input_path = Path(data_filename)
    input_filename = input_path.name

    if not input_path.is_file():
        raise Exception(f'The specified path {data_filename} does not exist')
    output_path = Path(out_filename)
    directory = output_path.parents[0]
    directory.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_filename, 'r') as data:
        metadata = json.loads(data['data']['metadata'][()].decode())
        timelapse = data['data']['timelapse']
        movie_data = timelapse[:, :, :metadata['frames']]
        waterfall = np.sum(movie_data, axis)

    with h5py.File(out_filename, "a") as f:
        group = f.create_group(input_filename)
        group.create_dataset('waterfall', data=waterfall.T)


def calculate_waterfall_background(waterfall, axis=1, sigma=50):
    """ Calculates the background of the waterfall using a gaussian filter in 1D
    along the temporal axis.
    """
    raw_bkg = sp.ndimage.gaussian_filter1d(waterfall, axis=axis, sigma=sigma)
    return raw_bkg