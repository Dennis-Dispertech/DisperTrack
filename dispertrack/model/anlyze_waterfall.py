import atexit
import json
from datetime import datetime
from shutil import copy2

import numpy as np
import scipy as sp
import scipy.ndimage

import h5py

from dispertrack import config_path, home_path
from dispertrack.model.exceptions import WrongDataFormat
from dispertrack.model.find import find_peaks1d
from dispertrack.model.refine import refine_positions


class AnalyzeWaterfall:
    def __init__(self):
        self.waterfall = None
        self.bkg = None
        self.corrected_data = None

        self.metadata = {
            'start_frame': None,
            'end_frame': None,
            'bkg_axis': None,
            'bkg_sigma': None,
            'exposure_time': None,
            'sample_description': None,
            'fps': None,
            }
        self.file = None

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

        atexit.register(self.finalize)

    def load_waterfall(self, filename, mode='a'):
        file = h5py.File(filename, mode=mode)
        if 'waterfall' in file.keys():
            self.waterfall = file['waterfall'][()]
        else:
            for group in file.keys():
                if 'waterfall' in file[group]:
                    self.waterfall = file[group]['waterfall'][()]
                    break
            else:
                raise WrongDataFormat(f'The selected file {filename.name} does not contain waterfall data')

        for key in self.metadata.keys():
            if key in file.keys():
                self.metadata[key] = file[key][()]

        if mode == 'a':
            self.file = file

    def transpose_waterfall(self):
        if self.waterfall is None:
            return
        self.waterfall = self.waterfall.T

    def crop_waterfall(self, start, stop):
        """ Selects the range of frames that will be analyzed, this is handy to remove unwanted data from memory and
        it helps speed up the GUI.
        """
        print(start, stop)
        self.waterfall = self.waterfall[:, start:stop]

    def calculate_background(self, axis=1, sigma=25):
        self.bkg = sp.ndimage.gaussian_filter1d(self.waterfall, axis=axis, sigma=sigma)
        self.corrected_data = (self.waterfall.astype(np.float) - self.bkg).clip(0, 2 ** 16 - 1).astype(np.uint16)

    def calculate_slice(self, start, stop, width):
        data = self.corrected_data if self.corrected_data is not None else self.waterfall
        slope = -data.shape[0]/(stop-start)
        offset = data.shape[0]
        cropped_data = np.zeros((2*width, stop-start))
        for i in range(stop-start):
            center = int(i*slope) + offset
            if width > center: continue
            if center > data.shape[0] - width: continue
            d = data[center-width:center+width, start+i]
            cropped_data[:, i] = d

        return cropped_data.T

    def calculate_intensities_cropped(self, data, separation=15, radius=5, threshold=1):
        """Calculates the intensity in each frame of a cropped image. It assumes there is
        only one particle present.

        Parameters
        ----------
        data : numpy.array
            It should be a rectangular image, resulting from cropping the waterfall around a bright peak.
        """

        frames = np.max(data.shape)
        intensities = np.zeros(frames)
        positions = np.zeros(frames)
        for i in range(frames):
            pos = find_peaks1d(data[i, :], separation=separation, threshold=threshold)
            pos = refine_positions(data[i, :], pos, radius)
            if len(pos) != 1: continue
            intensities[i] = pos[0][1]
            positions[i] = pos[0][0]

        return intensities, positions

    def finalize(self):
        with open(self.config_file_path, 'w') as f:
            json.dump(self.contextual_data, f)
        if self.file is not None:
            self.file.close()
