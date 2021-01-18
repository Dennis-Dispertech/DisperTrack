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
        self.waterfall = self.find_waterfall(file).T

        for key in self.metadata.keys():
            if key in file.keys():
                self.metadata[key] = file[key][()]

        self.contextual_data.update({
            'last_file': str(filename),
            })

        if mode == 'a':
            self.file = file

    def transpose_waterfall(self):
        if self.waterfall is None:
            return
        self.waterfall = self.waterfall.T

    @staticmethod
    def find_waterfall(file):
        """ Find and retrieve the waterfall in a given opened HDF5 file"""
        if 'waterfall' in file.keys():
            return file['waterfall'][()]
        else:
            for group in file.keys():
                if 'waterfall' in file[group]:
                    return file[group]['waterfall'][()]
            else:
                raise WrongDataFormat(f'The selected file {file} does not contain waterfall data')

    def clear_crop(self):
        if self.file is not None:
            self.waterfall = self.find_waterfall(self.file).T

    def crop_waterfall(self, start, stop):
        """ Selects the range of frames that will be analyzed, this is handy to remove unwanted data from memory and
        it helps speed up the GUI.
        """
        self.waterfall = self.waterfall[:, start:stop]
        self.metadata['start_frame'] = start
        self.metadata['end_frame'] = stop

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

    def save_particle_data(self, data, metadata, particle_num=None):
        """ Save the particle data to the same file from which the waterfall was taken. The data to be saved is a 2D
        array that contains the position and intensity at each frame, or 0 if no information is available (for
        instance if the peak was not detected. Metadata stores the parameters needed to re-acquire the same data,
        such as start, stop frames, width, threshold, separation and radius. Especially start stop and width allow to
        reconstruct the absolut position of the particle in the waterfall (to characterize size based on diffusion).
        """

        # Check if the file already has information on particle
        if not 'particles' in self.file.keys():
            self.file.create_group('particles')

        particles = self.file['particles']
        if particle_num is None:
            pcle_num = len(particles.keys())
            pcle = particles.create_group(str(pcle_num))
            pcle.create_dataset('data', data=data)
            pcle.create_dataset('metadata', data=json.dumps(metadata))
            return pcle_num
        else:
            if str(particle_num) not in particles.keys():
                raise ValueError('That particle does not exist in the waterfall file')
            del particles[str(particle_num)]
            pcle = particles.create_group(str(particle_num))
            pcle.create_dataset('data', data=data)
            pcle.create_dataset('metadata', data=json.dumps(metadata))
            return particle_num

    def finalize(self):
        with open(self.config_file_path, 'w') as f:
            json.dump(self.contextual_data, f)
        if self.file is not None:
            for key, value in self.metadata.items():
                if key in self.file.keys():
                    del self.file[key]
                if value is not None:
                    self.file.create_dataset(key, data=value)
            self.file.close()
