from pathlib import Path
from threading import Thread

import numpy as np

from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from dispertrack import home_path
from dispertrack.model.analyze_waterfall import AnalyzeWaterfall
from dispertrack.view import view_folder

import pyqtgraph as pg

from dispertrack.view.movie_window import MovieWindow
from dispertrack.view.particle_window import ParticleWindow

import dispertrack.view.GUI.resources_rc

import matplotlib.pyplot as plt

from dispertrack.view.util import error_message


class WaterfallWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(view_folder / 'GUI' / 'waterfall_analysis.ui', self)
        self.setWindowIcon(QIcon(str(view_folder / 'GUI'/ 'favicon.png')))
        self.analyze_model = AnalyzeWaterfall()
        self.action_open.triggered.connect(self.open_waterfall)
        self.action_analyze_particles.triggered.connect(self.analyze_particles)
        self.action_crop.triggered.connect(self.crop_waterfall)
        self.action_background.triggered.connect(self.calculate_background)
        self.action_denoise.triggered.connect(self.denoise_image)
        self.action_create_mask.triggered.connect(self.calculate_mask)
        self.action_show_labels.triggered.connect(self.display_labels)
        self.action_load_mask.triggered.connect(self.load_mask)
        self.action_select_intensity.triggered.connect(self.calculate_coupled_intensity)
        self.action_open_movie.triggered.connect(self.open_movie_window)
        self.action_view_histogram.triggered.connect(self.show_histogram_window)
        self.action_save.triggered.connect(self.analyze_model.save_particle_data)
        self.action_load.triggered.connect(self.analyze_model.load_particle_data)
        self.action_export_data.triggered.connect(self.export_data)

        self.button_clear_roi.clicked.connect(self.clear_crop)
        self.button_show_mask.clicked.connect(self.toggle_show_mask)

        self.slider_frame_selection.valueChanged.connect(self.update_sliding_window)

        self.waterfall_image = pg.ImageView()
        self.waterfall_image.setPredefinedGradient('thermal')

        self.ROI_line = None
        self.showing_mask = False

        self.hline1 = pg.InfiniteLine(angle=0, movable=True, hoverPen={'color': "#FF0", 'width': 4})
        self.hline2 = pg.InfiniteLine(angle=0, movable=True, hoverPen={'color': "#FF0", 'width': 4})

        self.first_waterfall_update = True

        self.particle_windows = []
        self.open_movie_windows = []

        plot_layout = self.plot_widget.layout()
        plot_layout.addWidget(self.waterfall_image)

    def update_sliding_window(self, index=0):
        if self.analyze_model.waterfall is None:
            return

        if self.showing_mask:
            to_display = self.analyze_model.mask[:, index:index+self.analyze_model.mask.shape[0]]
        else:
            if self.analyze_model.corrected_data is not None:
                to_display = self.analyze_model.corrected_data[:, index:index+self.analyze_model.corrected_data.shape[0]]
            else:
                to_display = self.analyze_model.waterfall[:, index:index+self.analyze_model.waterfall.shape[0]]

        self.update_image(to_display)

    def open_movie_window(self):
        self.open_movie_windows.append(MovieWindow())
        self.open_movie_windows[-1].show()

    @error_message
    def open_waterfall(self, _):
        last_dir = self.analyze_model.contextual_data.get('last_dir', home_path)
        file = QFileDialog.getOpenFileName(self, 'Open Waterfall data', str(last_dir), filter='*.h5')[0]
        if file != '':
            file = Path(file)
        else:
            return
        self.analyze_model.load_waterfall(file)

        self.setWindowTitle(f'Single Particle Analysis: {file.name}')
        self.update_image(self.analyze_model.waterfall)
        self.analyze_model.contextual_data.update({'last_dir': str(file.parent)})
        if ind := self.analyze_model.metadata['bkg_axis'] is not None:
            self.combo_bkg_axis.setCurrentIndex(ind)

        if (sigma := self.analyze_model.metadata['bkg_sigma']) is not None:
            self.line_bkg.setText(str(sigma))
        else:
            self.line_bkg.setText('')

        if (start_frame := self.analyze_model.metadata['start_frame']) is not None:
            self.line_start_frame.setText(str(start_frame))
            self.hline1.setValue(int(start_frame))
        else:
            self.line_start_frame.setText('')

        if (end_frame := self.analyze_model.metadata['end_frame']) is not None:
            self.line_end_frame.setText(str(end_frame))
            self.hline2.setValue((int(end_frame)))
        else:
            self.line_end_frame.setText('')

        if (mask_threshold := self.analyze_model.metadata.get('mask_threshold', None)) is not None:
            self.line_mask_threshold.setText(str(mask_threshold))

        if (mask_min_size := self.analyze_model.metadata.get('mask_min_size', None)) is not None:
            self.line_mask_min_size.setText(str(mask_min_size))

        if (mask_max_gap := self.analyze_model.metadata.get('mask_max_gap', None)) is not None:
            self.line_mask_max_gap.setText(str(mask_max_gap))

        if (mask_min_length := self.analyze_model.metadata.get('mask_min_length', None)) is not None:
            self.line_mask_min_len.setText(str(mask_min_length))

        self.slider_frame_selection.setRange(0, self.analyze_model.meta['frames']-self.analyze_model.waterfall.shape[0])

    def load_mask(self):
        try:
            self.analyze_model.load_mask()
            self.update_image(self.analyze_model.mask)
        except Exception as e:
            mb = QMessageBox(self)
            mb.setText('Something went wrong loading the mask')
            mb.setDetailedText(str(e))
            mb.exec()
            return

    def update_image(self, image):
        self.waterfall_image.setImage(image)
        self.hline1.setValue(0)
        self.hline2.setValue(image.shape[1])
        self.hline1.setBounds((0, image.shape[1]))
        self.hline2.setBounds((0, image.shape[1]))
        if self.first_waterfall_update:
            view = self.waterfall_image.getView()
            view.addItem(self.hline1)
            view.addItem(self.hline2)
            self.first_waterfall_update = False
        if self.ROI_line is None:
            self.ROI_line = pg.LineSegmentROI([(0, 0), (image.shape[0], 0)])

        self.waterfall_image.autoRange()
        if image.dtype == bool:
            self.waterfall_image.autoLevels()
        else:
            self.waterfall_image.setLevels(image.min(), np.percentile(image, 99))

    def calculate_coupled_intensity(self):
        x = [self.hline1.value(), self.hline2.value()]
        x = np.sort(x).astype(np.int)
        self.analyze_model.calculate_coupled_intensity(min_pixel=x[0], max_pixel=x[1])

    def clear_crop(self):
        self.analyze_model.clear_crop()
        self.update_image(self.analyze_model.waterfall)

    def crop_waterfall(self):
        x = [self.hline1.value(), self.hline2.value()]
        x = np.sort(x).astype(np.int)
        self.analyze_model.crop_waterfall(x[0], x[1])
        self.update_image(self.analyze_model.waterfall)
        self.line_start_frame.setText(str(x[0]))
        self.line_end_frame.setText(str(x[1]))

    def calculate_background(self):
        def calculate_bkg_thread(self, sigma):
            self.statusbar.showMessage('Calculating Background')
            if sigma is not None:
                self.analyze_model.calculate_background(sigma=sigma)
            else:
                self.analyze_model.calculate_background()

            self.update_image(self.analyze_model.corrected_data)
            self.statusbar.showMessage('')

        if (sigma := self.line_bkg.text()) != '':
            sigma = int(sigma)
        else:
            sigma = None

        t = Thread(target=calculate_bkg_thread, args=(self, sigma))
        t.start()

    def denoise_image(self):
        def denoise(self):
            self.statusbar.showMessage('Denoising image')
            self.analyze_model.denoise()
            self.update_image(self.analyze_model.corrected_data)
            self.statusbar.showMessage('')

        t = Thread(target=denoise, args=(self, ))
        t.start()

    def calculate_mask(self):
        self.statusbar.showMessage('Calculating mask')
        threshold = int(self.line_mask_threshold.text())
        min_size = int(self.line_mask_min_size.text())
        max_gap = int(self.line_mask_max_gap.text())
        min_len = int(self.line_mask_min_len.text())
        self.analyze_model.calculate_mask(threshold=threshold, min_size=min_size, max_gap=max_gap)
        self.analyze_model.label_mask(min_len=min_len)
        if self.showing_mask:
            self.update_image(self.analyze_model.mask)
        self.statusbar.showMessage('')

    def toggle_show_mask(self):
        if self.showing_mask:
            self.update_image(self.analyze_model.corrected_data)
            self.button_show_mask.setText('Show Mask')
            self.showing_mask = False
        else:
            self.update_image(self.analyze_model.mask)
            self.button_show_mask.setText('Show Waterfall')
            self.showing_mask = True

    def analyze_particles(self):
        self.statusbar.showMessage('Analyzing particles')
        self.analyze_model.metadata.update({'Temperature (C)': float(self.line_temperature.text())})
        self.analyze_model.analyze_traces()
        self.analyze_model.calculate_particle_properties()
        self.statusbar.showMessage('')

    def display_labels(self):
        self.particle_windows.append(ParticleWindow(self.analyze_model, props=self.analyze_model.filtered_props,
                                                    particle_number=0))
        self.particle_windows[-1].show()

    def show_histogram_window(self):
        d = [1E9*2*self.analyze_model.pcle_data[i].get('r', np.nan) for i in self.analyze_model.pcle_data.keys()]
        i = [self.analyze_model.pcle_data[i].get('mean_intensity', np.nan) for i in self.analyze_model.pcle_data.keys()]
        i = np.power(i, 1/6)
        plt.figure()
        plt.plot(d, i, 'o')
        plt.xlabel('Diameter (nm)', fontsize=16)
        plt.ylabel('Normalized Intensity (a.u.) ', fontsize=16)
        plt.show()

        plt.figure()
        plt.hist(i, 40)
        plt.xlabel('Normalized Intensity (a.u.)', fontsize=16)
        plt.show()

        plt.figure()
        plt.hist(d, 40)
        plt.xlabel('Diameter (nm)', fontsize=16)
        plt.show()

    def export_data(self):
        last_dir = self.analyze_model.contextual_data.get('last_export_folder', home_path)
        file = QFileDialog.getSaveFileName(self, 'Export data to file', str(last_dir), filter='*.csv')[0]
        self.analyze_model.export_particle_data(file)
        p = Path(file)
        meta_file = str(p.parent / (p.name + '.yml'))
        self.analyze_model.export_metadata(meta_file)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    win = WaterfallWindow()
    win.show()
    app.exec()
    