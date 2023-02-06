import atexit
import json
from datetime import datetime
from shutil import copy2

import h5py
import numpy as np
import pandas as pd
import scipy as sp
import yaml
from numpy.linalg import LinAlgError
from scipy.ndimage import correlate1d, median_filter
from scipy.optimize import root_scalar
from skimage import measure, morphology

from dispertrack import config_path, logger
from dispertrack.model.displacement import msd_iter
from dispertrack.model.find import find_peaks1d
from dispertrack.model.refine import refine_positions
from dispertrack.model.util import d_r, gaussian_kernel, H, r_d


class AnalyzeWaterfall:
    def __init__(self):
        self.waterfall = None
        self.bkg = None
        self.corrected_data = None
        self.mask = None
        self.filtered_props = []
        self.coupled_intensity = None
        self.pcle_data = {}

        self.metadata = {
            'start_frame': None,
            'end_frame': None,
            'bkg_axis': None,
            'bkg_sigma': None,
            'exposure': None,
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
            if key in self.meta.keys():
                self.metadata[key] = self.meta[key]

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
        self.metadata['start_frame'] = int(start)
        self.metadata['end_frame'] = int(stop)

    def calculate_coupled_intensity(self, min_pixel=0, max_pixel=-1):
        self.metadata['coupled_intensity_min_pixel'] = min_pixel
        self.metadata['coupled_intensity_max_pixel'] = max_pixel

        self.coupled_intensity = np.mean(self.waterfall[:, min_pixel:max_pixel])
        self.metadata['coupled_intensity'] = self.coupled_intensity

    def calculate_background(self, sigma=10):
        """ Defines the background as the median intensity over a number of frames prior to the current frame. In
        this way we do not subtract the signal of the particle from the particle itself. If particles are too close
        to each other, however, this may give some problems. Using the median instead of the mean overcomes some of
        the problems of using particle data to define background.

        """
        self.metadata.update({'bkg_sigma': sigma})
        bkg = sp.ndimage.median_filter(self.waterfall, size=(1, sigma))
        bkg[bkg > self.waterfall] = self.waterfall[bkg > self.waterfall]
        self.bkg = np.copy(bkg)
        self.corrected_data = self.waterfall - self.bkg

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

        It also gives the base coordinate for the window, which can be used to get the absolute position in the image.

        Parameters
        ----------
        coord : np.array
            Coordinates of pixels used for slicing. They are normally the result of using regionprops.

        width (optional) : int
            The width of the slice

        Returns
        -------
        cropped_data : np.array
            The extracted data with an extra column (cropped_data[:, 0]) that holds the coordinate of the sliced window
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

    def calculate_intensities_cropped(self, data, separation=15, radius=5, threshold=None):
        """Calculates the intensity in each frame of a cropped image. It assumes there is
        only one particle present.

        Parameters
        ----------
        data : numpy.array
            It should be a rectangular image, resulting from cropping the waterfall around a bright peak.
        separation : int
            value passed directly to `find_peaks`
        radius : int
            value passed directly to `refine_positions`
        threshold : int
            value passed directly to `find_peaks`

        Returns
        -------
        intensities : np.array
            the output of `refine_positions`
        positions : np.array
            the output of `refine_positions
        """

        frames = np.max(data.shape)
        intensities = np.zeros(frames)
        positions = np.zeros(frames)
        for i in range(frames):
            pos = find_peaks1d(data[i, :], separation=separation, threshold=threshold, precise=False)
            if pos.size == 0:
                intensities[i] = np.nan
                positions[i] = np.nan
                continue
            pos = refine_positions(data[i, :], pos, radius)
            if len(pos) != 1:
                intensities[i] = np.nan
                positions[i] = np.nan
            else:
                intensities[i] = pos[0][1]
                positions[i] = pos[0][0]

        return intensities, positions

    @staticmethod
    def calculate_diffusion(center, position):
        # Transform to 'absolute' position
        position = position + center
        X = np.arange(len(position))
        X = X[~np.isnan(position)]
        clean_position = position[~np.isnan(position)]
        if len(X) < 2:
            return
        try:
            fit = np.polyfit(X, clean_position, 1)
        except LinAlgError:
            logger.warning('Problem calculating drift correction')
            return

        # Remove the drift by subtracting a first-order fit to the center position.

        position = position - np.polyval(fit, np.arange(len(position)))

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
        The data of the label is stored as `filtered_props`
        """
        self.metadata['mask_min_length'] = min_len

        labels = measure.label(self.mask, background=False)
        props = measure.regionprops(labels, intensity_image=self.corrected_data)

        filtered_props = []
        for p in props:
            if np.max(p.coords[:, 1]) - np.min(p.coords[:, 1]) >= min_len:
                filtered_props.append(p.coords)

        self.filtered_props = filtered_props[1:]

    def analyze_traces(self, width=50, radius=7, separation=20, threshold=50):
        if self.filtered_props is None:
            raise Exception('First particles must be labelled')

        calibration = self.meta.get('calibration', 440E-9)  # m/px uses a default value in case it was not defined
        print(f'Using {calibration}m/px as calibration')

        self.pcle_data = {}
        for i, data in enumerate(self.filtered_props):
            sliced_data = self.calculate_slice_from_label(data, width)
            intensities, positions = self.calculate_intensities_cropped(sliced_data[:, 1:], separation, radius, threshold)
            positions = positions * calibration
            MSD = self.calculate_diffusion(sliced_data[:, 0]*calibration, positions)
            self.pcle_data.update({
                i: {'intensity': intensities, 'position': positions, 'MSD': MSD, 'mean_intensity': np.nan,
                    'intensity_std': np.nan, 'D': np.nan}
                })

    def calculate_particle_properties(self):
        C = self.meta.get('channel_diameter', '560')
        self.metadata.update({'Channel diameter (nm)': C})
        C = int(C)*1E-9
        bkg_intensity = np.mean(self.waterfall[:, :10])

        fps = self.meta['fps']
        lagtimes = np.arange(1, 5) / fps

        for p, pcle_data in self.pcle_data.items():
            r = np.nan
            try:
                MSD = pcle_data['MSD']
                fit = np.polyfit(lagtimes, MSD['MSD'].array[:len(lagtimes)], 1)
                m_i = np.nanmean(pcle_data['intensity'] / bkg_intensity)
                d_i = np.nanstd(pcle_data['intensity'] / bkg_intensity)
                self.pcle_data[p].update({
                    'mean_intensity': m_i,
                    'intensity_std': d_i,
                    'D': fit,
                    })
            except LinAlgError as e:
                logger.error(f'Particle {p} has a problem.', e)
                continue
            except TypeError as e:
                logger.error(f'Particle {p} has no MSD defined.', e)
                continue

            def to_minimize(X):
                return d_r(X) - fit[0] / H(X, C)
            try:
                r = root_scalar(to_minimize, bracket=(1E-9, C/2)).root
                self.pcle_data[p].update({'r': r})
            except:
                print(f'Problem with pcle {p}')

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

    def save_particle_data(self):
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
        for p, pcle_data in self.pcle_data.items():
            if p in particles:
                del particles[p]
            pcle = particles.create_group(str(p))
            pcle.create_dataset('data', data=json.dumps(pcle_data))

    def export_particle_data(self, filename):
        """ Exports the information of the particles to a CSV file.
        """
        self.contextual_data = {
            'last_export_folder': str(filename)
            }
        data = {p: [pp.get('D'), pp.get('mean_intensity'), pp.get('r'), pp.get('intensity_std')]
                for p, pp in self.pcle_data.items()}

        data_df = pd.DataFrame.from_dict(data, orient='index', columns=['Diffusion coefficient',
                                                                        'Intensity',
                                                                        'Diameter',
                                                                        'Intensity standard deviation'])
        data_df.to_csv(filename)

    def export_metadata(self, filename):
        with open(filename, 'w') as outfile:
            yaml.dump(self.metadata, outfile)

    def load_particle_data(self):
        if not 'particles' in self.file.keys():
            raise KeyError('There is no particle data in this file')

        particles = self.file['particles']
        for p in particles:
            self.pcle_data.update({
                p: json.loads(particles[p]['data'][()])
                })

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


if __name__ == '__main__':
    a = AnalyzeWaterfall()
    a.load_waterfall(r'C:\Users\aron\Documents\NanoCET\data\test.h5')
    a.calculate_coupled_intensity(10, 6000)
    a.crop_waterfall(8300, 28300) # 64300)
    a.calculate_background(None, 20)
    a.calculate_mask(200, 20, 100)
    a.label_mask(200)
    print('done')