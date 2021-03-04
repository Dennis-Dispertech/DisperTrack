import atexit
import json
import numpy as np
from datetime import datetime
from shutil import copy2

import h5py

from dispertrack import config_path


class MakeWaterfall:
    def __init__(self):
        self.config_file_path = config_path / 'waterfall_config.dat'
        self.contextual_data = {
            'last_run': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        if not self.config_file_path.is_file():
            with open(self.config_file_path, 'w') as f:
                json.dump(self.contextual_data, f)
        else:
            try:
                with open(self.config_file_path, 'r') as f:
                    self.contextual_data = json.load(f)
            except Exception:
                Warning('There is something wrong with the config file, creating a new one and backing up '
                                 'the old one', UserWarning)

                copy2(self.config_file_path, config_path / '_bkg_waterfall.dat')
                with open(self.config_file_path, 'w') as f:
                    json.dump(self.contextual_data, f)

        self.movie_metadata = None
        self.movie_data = None
        self.waterfall = None

        atexit.register(self.finalize)

    def load_movie(self, filename):
        """ Loads the movie into memory.

        .. warning:: This is a temporary solution, if the file is too large it will not fit into the RAM of the
        computer.

        :param Path filename: Path object pointing to the file
        """
        if not filename.is_file():
            raise FileNotFoundError(f'The specified path {filename} does not exist')

        with h5py.File(filename, 'r') as data:
            metadata = json.loads(data['data']['metadata'][()].decode())
            timelapse = data['data']['timelapse']
            movie_data = timelapse[:, :, :metadata['frames']]
            self.movie_data = movie_data
            self.movie_metadata = metadata

    def calculate_waterfall(self, transpose=False, axis=1):
        if self.movie_data is None:
            raise KeyError('There is no movie loaded. First load a movie.')

        if transpose:
            self.waterfall = np.sum(self.movie_data.T, axis)
        else:
            self.waterfall = np.sum(self.movie_data, axis)

    def save_waterfall(self, filename, dataset='waterfall'):
        """ Saves the calculated waterfall to the given filename.

        :param Path filename: Path object where to store the waterfall.
        """
        if self.waterfall is None:
            raise Exception('Waterfall was not yet calculated')

        with h5py.File(filename, 'a') as f:
            if dataset in f:
                raise KeyError('The waterfall already exists, choose a different name.')
            f.create_dataset(dataset, data=self.waterfall)

    def unload_movie(self):
        print('Unloading movie')
        del self.movie_data
        del self.movie_metadata
        del self.waterfall
        
        self.movie_metadata = None
        self.movie_data = None
        self.waterfall = None

    def finalize(self):
        """ This method will be called when finalizing the class, it may be useful to do some clean up, clsoe the
        opened files, etc."""
        pass