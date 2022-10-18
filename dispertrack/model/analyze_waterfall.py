import atexit
import json
from datetime import datetime
from shutil import copy2

import h5py
import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage
from numpy.linalg import LinAlgError
from scipy.ndimage import correlate1d, median_filter
from skimage import measure, morphology

from dispertrack import config_path
from dispertrack.model.displacement import msd_iter
from dispertrack.model.find import find_peaks1d
from dispertrack.model.refine import refine_positions
from dispertrack.model.util import gaussian_kernel

import sympy

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
        self.meta = json.loads(file['data']['metadata'][()].decode())
        self.waterfall = self.find_waterfall(file)

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
    def find_waterfall(file):
        path = file.visit(lambda name: name if 'waterfall' in name else None)
        if path is None:
            path = file.visit(lambda name: name if 'timelapse' in name else None)
        return file[path][()]

    def clear_crop(self):
        if self.file is not None:
            self.waterfall = self.find_waterfall(self.file)

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

    # def calculate_background(self, axis=1, sigma=10):
    #     """ Defines the background as the median intensity over a number of frames prior to the current frame. In
    #     this way we do not subtract the signal of the particle from the particle itself. If particles are too close
    #     to each other, however, this may give some problems. Using the median instead of the mean overcomes some of
    #     the problems of using particle data to define background.
    #
    #     """
    #
    #     # self.bkg = sp.ndimage.gaussian_filter1d(self.waterfall, axis=axis, sigma=sigma)
    #     self.bkg = np.zeros_like(self.waterfall)
    #     for i in range(self.waterfall.shape[axis]):
    #         self.bkg[:, i] = sp.ndimage.median_filter(self.waterfall[:, i], sigma)
    #     self.bkg = np.roll(self.bkg, sigma, axis=0)
    #     self.corrected_data = self.waterfall - self.bkg
    #     self.corrected_data[self.waterfall < self.bkg] = 0

    def calculate_background(self, axis=None, sigma=10):
        self.bkg = np.zeros_like(self.waterfall)
        # for column in range(sigma, self.waterfall.shape[0]):
        #     self.bkg[column, :] = sp.ndimage.median_filter(self.waterfall[column, :], sigma)
        for row in range(sigma, self.waterfall.shape[1]):
            self.bkg[:, row] = np.median(self.waterfall[:, row-sigma:row], axis=1)
        self.bkg[:, :sigma] = np.tile(np.median(self.waterfall[:, :sigma], axis=1)[:,None], (1, sigma))
        self.corrected_data = self.waterfall - self.bkg
        self.corrected_data[self.waterfall < self.bkg] = 0


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
        slope = -data.shape[0] / (stop - start)
        offset = data.shape[0]
        cropped_data = np.zeros((2 * width + 1, stop - start))
        for i in range(stop - start):
            center = int(i * slope) + offset
            cropped_data[0, :] = center
            if width > center:
                cropped_data[1:, i] = np.nan
                continue
            if center > data.shape[0] - width:
                cropped_data[1:, i] = np.nan
                continue
            d = data[center - width:center + width, start + i]
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
        cropped_data = np.zeros((2 * width + 1, max_frame - min_frame))
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
        position = position + center
        position = position[~np.isnan(position)]
        try:
            X = np.arange(len(position))
            fit = np.polyfit(X, position, 3)
        except LinAlgError:
            print(position)
            print(center)
            raise


        # Remove the drift by subtracting a first-order fit to the center position.

        position = position - np.polyval(fit, X)  # position is still in pixels I think
        fiber_width_pixl = 412
        fiber_width = 180E-6
        pixl = fiber_width / fiber_width_pixl
        position = position*pixl
        lagtimes = np.arange(1, int(len(position) / 2))  # delta t in frames
        MSD = pd.DataFrame(msd_iter(position, lagtimes=lagtimes), columns=['MD', 'MSD'], index=lagtimes)
        return MSD

    def calculate_mask(self, threshold, min_size=0, max_gap=0):
        self.metadata['mask_threshold'] = threshold
        self.metadata['mask_min_size'] = min_size
        self.metadata['mask_max_gap'] = max_gap

        self.mask = self.corrected_data > threshold
        if min_size > 0:
            self.mask = morphology.remove_small_objects(self.mask, min_size)

        if max_gap > 0:
            self.mask = morphology.remove_small_holes(self.mask, max_gap)
        self.mask = self.mask.astype(np.bool8)

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
                print('Creating mask and particle groups')
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

    def calculate_particle_properties(self):
        bkg_intensity = np.mean(self.waterfall[:, :10])
        self.particles = {}

        fps = self.meta['fps']

        fiber_width_pixl = 412
        fiber_width = 180E-6
        pixl = fiber_width / fiber_width_pixl  # ~ 0.44 um per pixel
        pixl_um = pixl * 1E6

        # lagtimes = 5E-3 * np.arange(1, 5)
        lagtimes = np.arange(1, 5) / fps

        for particle in self.file['mask/particles'].keys():
            data = self.file['mask/particles'][particle][:]
            MSD = self.calculate_diffusion(data[2, :], data[0, :])
            fit = np.polyfit(lagtimes, MSD['MSD'].array[:len(lagtimes)], 2)
            # fit = np.polyfit(lagtimes, pixl**2 * MSD['MSD'].array[:len(lagtimes)], 1)
            attrs = self.file['mask/particles'][particle].attrs
            if attrs['valid'] == 0:
                continue
            m_i = np.nanmean(np.power(data[1, :] / bkg_intensity, 1 / 6))
            self.particles[particle] = {
                'data': data,
                'mean_intensity': m_i,
                'D': fit,
                }

        mean_intensity = np.zeros(len(self.particles))
        diffusion = np.zeros(len(self.particles))
        # self.d = np.ones_like(diffusion) * 20E-9
        # d = sympy.symbols('d')
        self.r = np.ones_like(diffusion) * 20E-9  # starting values for gradient descent fitting
        r = sympy.symbols('r')
        C = 560E-9  # core diameter
        T = 300
        k_b = 1.380649E-23
        eta = 0.0009532
        self.F_ERR=[]
        for i, values in enumerate(self.particles.values()):
            mean_intensity[i] = values['mean_intensity']
            diffusion[i] = values['D'][1] / 2
            # f = (1 - d/C)**2 * (1 - 2.104*(d/C) + 2.09*(d/C)**3 - 0.95*(d/C)**5) - (d * diffusion[i] * 3*np.pi*eta/(k_b*T) )
            # f_deriv = f.diff(d)
            f = (1 - 2*r / C) ** 2 * (1 - 2.104 * (2*r / C) + 2.09 * (2*r / C) ** 3 - 0.95 * (2*r / C) ** 5) - (2*r * diffusion[i] * 3 * np.pi * eta / (k_b * T))
            f_deriv = f.diff(r)
            f_err = []
            for j in range(25):  # gradient descent fitting
                # self.d[i] -= np.float64(f.evalf(subs={d: self.d[i]})) / np.float64(f_deriv.evalf(subs={d: self.d[i]}))
                self.r[i] -= np.float64(f.evalf(subs={r: self.r[i]})) / np.float64(f_deriv.evalf(subs={r: self.r[i]}))
                f_err.append(f.evalf(subs={r: self.r[i]}))
            self.F_ERR.append(f_err)

        # self.d = k_b * T / (3 * np.pi * eta * diffusion[diffusion > 0]/H(60/560))
        # self.r = 2 * k_b * T / (6 * np.pi * eta * diffusion[diffusion > 0] / H(70 / 560))
        self.mean_intensity = mean_intensity[diffusion > 0]
        self.diffusion_coefficient = diffusion[diffusion > 0]

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


k_b = 1.38E-23
T = 300
eta = 0.0009532


def H(λ):
    return (1 - λ) ** 2 * (1 - 2.104 * λ + 2.09 * λ ** 3 - 0.95 * λ ** 5)


x = np.linspace(5 / 670, 200 / 670, 400)
y = H(x)



if __name__ == '__main__':
    print('working...')
    a = AnalyzeWaterfall()
    a.load_waterfall(r'C:\Users\aron\Documents\NanoCET\data\test.h5')
    a.calculate_coupled_intensity(10, 6000)
    a.crop_waterfall(8300, 28300) # 64300)
    a.calculate_background(None, 20)
    a.calculate_mask(200, 20, 100)
    a.label_mask(200)
    print('done')