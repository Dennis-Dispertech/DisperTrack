import atexit
import json
import pickle
from datetime import datetime
from shutil import copy2

import numpy as np
import scipy as sp
import scipy.ndimage

import pandas as pd

import h5py
from scipy.ndimage import correlate1d
from skimage import measure, morphology

from dispertrack import config_path
from dispertrack.model.displacement import msd_iter
from dispertrack.model.exceptions import WrongDataFormat
from dispertrack.model.find import find_peaks1d
from dispertrack.model.refine import refine_positions
from dispertrack.model.util import gaussian_kernel


class AnalyzeWaterfall:
    def __init__(self):
        self.waterfall = None
        self.bkg = None
        self.corrected_data = None
        self.mask = None
        self.filtered_props = []
        self.coupled_intensity = None

        self.metadata = {
            'start_frame': None,
            'end_frame': None,
            'bkg_axis': None,
            'bkg_sigma': None,
            'exposure_time': None,
            'sample_description': None,
            'fps': None,
            'mask_threshold': None,
            'mask_max_gap': None,
            'mask_min_size': None,
            'mask_min_length': None,
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
        path = file.visit(self.find_waterfall)
        self.waterfall = file[path][()].T

        for key in self.metadata.keys():
            if key in file.keys():
                self.metadata[key] = file[key][()]

        self.contextual_data.update({
            'last_file': str(filename),
            })

        if mode == 'a':
            self.file = file

    def load_mask(self):
        if 'mask' in self.file.keys():
            self.mask = self.file['mask']['mask'][()]
            self.filtered_props = []

            for i in self.file['mask']['props'].keys():
                self.filtered_props.append(self.file['mask']['props'][i][()])
        else:
            Warning('Trying to load mask but the file does not specify one')

    def transpose_waterfall(self):
        if self.waterfall is None:
            return
        self.waterfall = self.waterfall.T

    @staticmethod
    def find_waterfall(name):
        """ Callable to use with HDF visit method. It returns the first element it encounters with the text waterfall.
        """
        if 'waterfall' in name:
            return name

    def clear_crop(self):
        if self.file is not None:
            path = self.file.visit(self.find_waterfall)
            self.waterfall = self.file[path][()].T

    def crop_waterfall(self, start, stop):
        """ Selects the range of frames that will be analyzed, this is handy to remove unwanted data from memory and
        it helps speed up the GUI.
        """
        self.waterfall = self.waterfall[:, start:stop]
        self.metadata['start_frame'] = start
        self.metadata['end_frame'] = stop

    def calculate_coupled_intensity(self, min_pixel=0, max_pixel=-1):
        self.metadata['coupled_intensity_min_pixel'] = min_pixel
        self.metadata['coupled_intensity_max_pixel'] = max_pixel

        self.coupled_intensity = np.mean(self.waterfall[:, min_pixel:max_pixel])

    def calculate_background(self, axis=1, sigma=25):
        self.bkg = sp.ndimage.gaussian_filter1d(self.waterfall, axis=axis, sigma=sigma)
        self.corrected_data = (self.waterfall.astype(np.float) - self.bkg).clip(0, 2 ** 16 - 1).astype(np.uint16)

    def denoise(self, sigma=(0, 1), truncate=4.0):
        if self.corrected_data is not None:
            to_denoise = self.corrected_data
        else:
            to_denoise = self.waterfall

        result = np.array(to_denoise, dtype=np.float)
        for axis, _sigma in enumerate(sigma):
            if _sigma > 0:
                correlate1d(result, gaussian_kernel(_sigma, truncate), axis, output=result, mode='constant', cval=0.0)
        self.corrected_data = result

    def calculate_slice(self, start, stop, width):
        """ Calculates the slice of data given a start and end frame with a width. It links both sides of the
        waterfall with a straight line and calculates a sliding window accordingly.

        .. warning:: This only works for getting slices across the entire image, partial slices require another approach
        """
        data = self.corrected_data if self.corrected_data is not None else self.waterfall
        slope = -data.shape[0]/(stop-start)
        offset = data.shape[0]
        cropped_data = np.zeros((2*width+1, stop-start))
        for i in range(stop-start):
            center = int(i*slope) + offset
            cropped_data[0, :] = center
            if width > center:
                cropped_data[1:, i] = np.nan
                continue
            if center > data.shape[0] - width:
                cropped_data[1:, i] = np.nan
                continue
            d = data[center-width:center+width, start+i]
            cropped_data[1:, i] = d

        return cropped_data.T

    def calculate_slice_from_label(self, coord, width=25):
        """ Given a set of pixels as tose returned by regionprops, return the data around the center pixel.
        This method complements :meth:~calculate_slice and can be used with the resulting properties from
        :meth:~label_mask
        """
        data = self.corrected_data if self.corrected_data is not None else self.waterfall

        min_frame = np.min(coord[:, 1])
        max_frame = np.max(coord[:, 1])
        cropped_data = np.zeros((2*width+1, max_frame-min_frame))
        for i, f in enumerate(range(min_frame, max_frame)):
            pixels = coord[coord[:, 1] == f, 0]
            if len(pixels) > 1:
                center = np.mean(pixels).astype(np.int)
                cropped_data[0, i] = center
                if width > center:
                    cropped_data[1:, i] = np.nan
                    continue
                if center > data.shape[0] - width:
                    cropped_data[1:, i] = np.nan
                    continue
                d = data[center - width:center + width, f]
                cropped_data[1:, i] = d

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
            if len(pos) != 1:
                intensities[i] = np.nan
                positions[i] = np.nan
            else:
                intensities[i] = pos[0][1]
                positions[i] = pos[0][0]

        return intensities, positions

    def calculate_diffusion(self, center, position):

        # Transform to 'absolute' position
        position = position+center

        X = np.arange(len(center))
        fit = np.polyfit(X, center, 1)

        # Remove the drift by subtracting a first-order fit to the center position.

        position = position - np.polyval(fit, X)
        lagtimes = np.arange(1, int(len(position)/2))
        MSD = pd.DataFrame(msd_iter(position, lagtimes=lagtimes), columns=['MD', 'MSD'], index=lagtimes)
        return MSD

    def calculate_mask(self, threshold, min_size=0, max_gap=0):
        self.metadata['mask_threshold'] = threshold
        self.metadata['mask_min_size'] = min_size
        self.metadata['mask_max_gap'] = min_size

        self.mask = self.corrected_data > threshold
        if min_size > 0:
            self.mask = morphology.remove_small_objects(self.mask, min_size)

        if max_gap > 0:
            self.mask = morphology.remove_small_holes(self.mask, max_gap)

    def label_mask(self, min_len=100):
        """ Label the mask and remove those tracks that have fewer than a minimum number of frames in them.
        """
        self.metadata['mask_min_length'] = min_len

        labels = measure.label(self.mask, background=False)
        props = measure.regionprops(labels, intensity_image=self.corrected_data)

        filtered_props = []
        for p in props:
            if np.max(p.coords[:, 1]) - np.min(p.coords[:, 1]) >= min_len:
                filtered_props.append(p.coords)

        self.filtered_props = filtered_props

    def save_particle_label(self, data, metadata, particle_num):
        if self.mask is not None:
            if 'mask' not in self.file.keys():
                mask = self.file.create_group('mask')
                particles = mask.create_group('particles')
                print('Creating mask and particle gourps')
            elif 'particles' not in self.file['mask'].keys():
                particles = self.file['mask'].create_group('particles')
                print('Creating particles group')
            else:
                particles = self.file['mask']['particles']
                print('Retrieving particles group')
        else:
            Warning('Can\'t save label data if there\'s no mask defined in the model')
            return

        if str(particle_num) in particles.keys():
            print(f'Deleting particle {particle_num}')
            del particles[str(particle_num)]

        dset = self.file['mask/particles'].create_dataset(str(particle_num), data=data)
        print(dset)
        print(f'Creating pcle: {particle_num}')
        for key, value in metadata.items():
            dset.attrs[key] = value

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
            self.file.flush()
            self.file.close()
