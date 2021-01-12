import json
from pathlib import Path

import h5py
import numpy as np
import scipy as sp


def create_waterfall(data_filename, transpose=False, axis=1):
    """ Reads the data and creates a waterfall by adding all the pixels in the vertical direction."""
    input_path = Path(data_filename)

    if not input_path.is_file():
        raise Exception(f'The specified path {data_filename} does not exist')

    with h5py.File(data_filename, 'r') as data:
        metadata = json.loads(data['data']['metadata'][()].decode())
        timelapse = data['data']['timelapse']
        movie_data = timelapse[:, :, :metadata['frames']]
        if transpose:
            movie_data = movie_data.T
        waterfall = np.sum(movie_data, axis)

    return waterfall.T


def save_waterfall(waterfall, filename, dataset='waterfall'):
    with h5py.File(filename, 'a') as f:
        if dataset in f:
            raise KeyError('The waterfall already exists, chose a different name.')
        f.create_dataset(dataset, data=waterfall)


def calculate_waterfall_background(waterfall, axis=1, sigma=50):
    """ Calculates the background of the waterfall using a gaussian filter in 1D
    along the temporal axis.
    """
    raw_bkg = sp.ndimage.gaussian_filter1d(waterfall, axis=axis, sigma=sigma)
    return raw_bkg