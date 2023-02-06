import atexit
import json
import numpy as np
from datetime import datetime
from shutil import copy2

import h5py

from dispertrack import config_path
from dispertrack.model.util import fitgaussian


class MakeWaterfall:
    def __init__(self):
        self.config_file_path = config_path / 'movie_config.dat'
        last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.contextual_data = {
            'make_waterfall_last_run': last_run,
            'waterfall_output_dir': None,
            }

        self.waterfall_metadata = {}

        self.data_file = None

        if not self.config_file_path.is_file():
            with open(self.config_file_path, 'w') as f:
                json.dump(self.contextual_data, f)
        else:
            try:
                with open(self.config_file_path, 'r') as f:
                    self.contextual_data = json.load(f)
                    self.contextual_data.update({'make_waterfall_last_run': last_run})
            except Exception:
                Warning('There is something wrong with the config file, creating a new one and backing up '
                                 'the old one', UserWarning)

                copy2(self.config_file_path, config_path / '_bkp_movie.dat')
                with open(self.config_file_path, 'w') as f:
                    json.dump(self.contextual_data, f)

        self.movie_metadata = None
        self.movie_data = None
        self.movie_background = None
        self.waterfall = None
        self.roi = None  # Tuple to hold the min/max pixel to calculate the waterfall

        atexit.register(self.finalize)

    def load_movie(self, filename):
        """ Loads the movie into memory.

        :param Path filename: Path object pointing to the file
        """
        if not filename.is_file():
            raise FileNotFoundError(f'The specified path {filename} does not exist')

        self.data_file = h5py.File(filename, 'r')
        self.movie_metadata = json.loads(self.data_file['data']['metadata'][()].decode())
        self.movie_data = self.data_file['data']['timelapse']

        self.contextual_data.update({
            'last_movie_file': str(filename),
            'last_movie_directory': str(filename.parents[0])
            })

        self.waterfall_metadata.update(self.movie_metadata)

    def calculate_region_of_interest(self, index=0, frames=10, sigmas=1):
        """ Calculates the region of interest by adding all the pixels along the fiber direction, fitting a gaussian
        to the profile and returning the min and max pixel that fall within the number of sigmas.

        Parameters
        ----------
        index : int
            Frame from which to start calculating
        frames : int
            Number of frames used for averaging
        sigmas : int
            Number of sigmas around the gaussian peak to consider for the region of interest

        Returns
        -------
        min_pix : int
            Minimum pixel line of the ROI
        max_pix : int
            Maximum pixel line of the ROI
        """

        if self.movie_data is None:
            raise Exception("Movie not loaded")

        if not 'frames' in self.movie_metadata:
            self.movie_metadata['frames'] = self.movie_data.shape[2]

        if index + frames > self.movie_metadata['frames']:
            raise Exception("Not enough frames to perform this operation")

        cross_section = np.sum(np.mean(self.movie_data[:, :, index:index+frames], 2), 0)
        p0 = [
            np.max(cross_section)-np.min(cross_section),
            np.argwhere(cross_section == np.max(cross_section))[0][0],
            len(cross_section)/10,
            np.min(cross_section)]
        params = fitgaussian(cross_section, p0)

        self.roi = np.int(params[1]-sigmas*params[2]), np.int(params[1]+sigmas*params[2])
        self.waterfall_metadata.update({'roi': self.roi})
        return np.int(params[1]-sigmas*params[2]), np.int(params[1]+sigmas*params[2])

    def calculate_movie_background(self, index=0, frames=10):
        """  Calculates the background for the movie frames using a simple method of the median on a sliding window.

        Parameters
        ----------
        index : int
            Frame index around which to calculate the background
        frames : int
            Number of frames used to calculate the background

        Returns
        -------
        background : numpy.array
            The calculated background is a 2D numpy array that can be subtracted from the image
        """
        if self.movie_data is None:
            return

        if index > frames/2:
            start_frame = index-int(frames/2)
        else:
            start_frame = index

        if index + frames/2 < self.movie_metadata['frames']:
            end_frame = index + int(frames/2)
        else:
            end_frame = self.movie_metadata['frames']

        background = np.median(self.movie_data[:, :, start_frame:end_frame], 2)
        return background

    def calculate_waterfall(self, transpose=False, axis=1, roi=None):
        """
        .. warning:: This is a temporary solution, if the file is too large it will not fit into the RAM of the
        computer.

        Parameters
        ----------
        transpose : bool
            Whether the movie data is transposed before calculating the waterfall
        axis : int
            The axis that will be used in numpy sum (this is related to whether images are stored in colum-major or
            row-major order
        roi : tuple
            The min and max pixel to crop the movie data before calculating the waterfall.
            If not specified it will use the ROI stored in self.roi

        Returns
        -------

        """
        if self.movie_data is None:
            raise KeyError('There is no movie loaded. First load a movie.')

        if roi is None:
            if (roi := self.roi) is None:
                raise KeyError('You need to either supply a ROI or calculate one using calculate_region_of_interest')
        movie_data = self.movie_data[:, roi[0]:roi[1], :self.movie_metadata['frames']]
        if transpose:
            self.waterfall = np.sum(movie_data.T, axis)
        else:
            self.waterfall = np.sum(movie_data, axis)

    def save_waterfall(self, filename, group='data', dataset='timelapse'):
        """ Saves the calculated waterfall to the given filename.

        :param Path filename: Path object where to store the waterfall.
        """
        if self.waterfall is None:
            raise Exception('Waterfall was not yet calculated')

        folder = filename.parents[0]
        folder.mkdir(parents=True, exist_ok=True)
        with h5py.File(filename, 'a') as f:
            if group in f:
                raise KeyError(f'The specified group {group} for the waterfall already exists, choose a different '
                               f'name.')

            g = f.create_group(group)
            g.create_dataset(dataset, data=self.waterfall)
            g.create_dataset('metadata', data=json.dumps(self.waterfall_metadata).encode("utf-8", "ignore"))
            f.flush()

        self.contextual_data.update({
                'waterfall_output_dir': str(folder)
            })

    def unload_movie(self):
        self.movie_metadata = None
        self.movie_data = None
        self.waterfall = None

    def finalize(self):
        """ This method will be called when finalizing the class, it may be useful to do some clean up, clsoe the
        opened files, etc. """
        self.data_file.close()
        with open(self.config_file_path, 'w') as f:
            json.dump(self.contextual_data, f)