import atexit
import json
from datetime import datetime
from shutil import copy2

import h5py

from dispertrack import config_path, home_path
from dispertrack.model.exceptions import WrongDataFormat


class AnalyzeWaterfall:
    def __init__(self):
        self.waterfall = None
        self.metadata = {
            'start_frame': None,
            'end_frame': None,
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

    def finalize(self):
        with open(self.config_file_path, 'w') as f:
            json.dump(self.contextual_data, f)
