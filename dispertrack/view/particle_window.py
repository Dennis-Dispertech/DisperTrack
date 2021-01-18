from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QMessageBox

import numpy as np
import pyqtgraph as pg

from dispertrack.view import view_folder


class ParticleWindow(QMainWindow):
    def __init__(self, analyze_model, slice):
        super().__init__()
        uic.loadUi(view_folder / 'GUI' / 'particle_window.ui', self)
        self.analyze_model = analyze_model

        self.data = None
        self.intensities = None
        self.positions = None

        self.widget_image.setPredefinedGradient('thermal')

        self.line_slice_start.setText(str(slice.start))
        self.line_slice_stop.setText(str(slice.stop))

        self.calculate_particle_data()

        self.button_apply.clicked.connect(self.calculate_particle_data)
        self.button_save.clicked.connect(self.save_data)
        self.slider_frames.valueChanged.connect(self.update_1d_plot)

        self.saved = False

        self.particle_num = None

    def calculate_particle_data(self):
        """ Calculates particle data based on some parameters such as the coordinates of the
        slice and """
        start = int(self.line_slice_start.text())
        stop = int(self.line_slice_stop.text())
        (start, stop) = np.sort((start, stop))
        width = int(self.line_slice_width.text())
        separation = int(self.line_position_separation.text())
        radius = int(self.line_position_radius.text())
        threshold = int(self.line_position_threshold.text())
        self.data = self.analyze_model.calculate_slice(start, stop, width)
        self.intensities, self.positions = self.analyze_model.calculate_intensities_cropped(
            self.data,
            separation=separation,
            radius=radius,
            threshold=threshold,
            )
        self.update_image()
        self.update_intensities()
        self.update_positions()

        self.slider_frames.setRange(0, self.data.shape[0]-1)

    def update_image(self):
        self.widget_image.setImage(self.data)
        self.widget_image.autoLevels()
        self.widget_image.autoRange()
        self.widget_1d_plot.setYRange(0, np.max(self.data), padding=0)

    def update_positions(self):
        pi = self.widget_position.getPlotItem()
        pi.clear()
        x = np.arange(0, len(self.positions))
        pi.plot(x[self.positions > 0], self.positions[self.positions > 0])

    def update_intensities(self):
        pi = self.widget_intensity.getPlotItem()
        pi.clear()
        x = np.arange(0, len(self.intensities))
        pi.plot(x[self.intensities > 0], self.intensities[self.intensities > 0])

    def update_1d_plot(self, frame):
        self.widget_1d_plot.clear()
        self.widget_1d_plot.plot(self.data[frame])
        self.widget_1d_plot.plot([self.positions[frame], ], [self.data[frame][round(self.positions[frame])],],
                                 symbolBrush=pg.mkBrush(0, 0, 255, 255),
                                 symbolSize=7)

    def save_data(self):
        if self.particle_num is not None:
            overwrite_message = QMessageBox.question(
                self,
                'Overwrite saved particle data?',
                'This particle was already saved. Overwrite?',
                )
            if overwrite_message != QMessageBox.Yes:
                self.particle_num = None

        data = np.stack((self.positions, self.intensities))
        metadata = {
            'start': int(self.line_slice_start.text()),
            'stop': int(self.line_slice_stop.text()),
            'width': int(self.line_slice_width.text()),
            'threshold': int(self.line_position_threshold.text()),
            'separation': int(self.line_position_separation.text()),
            'radius': int(self.line_position_radius.text()),
            }

        self.particle_num = self.analyze_model.save_particle_data(data, metadata, self.particle_num)

        self.saved = True

    def closeEvent(self, event):
        if self.saved:
            event.accept()
        else:
            save = QMessageBox.question(
                self,
                'Save before quitting?',
                'Data not yet saved, save before close?'
                )
            if save == QMessageBox.Yes:
                self.save_data()
                event.accept()
            elif save == QMessageBox.No:
                event.accept()
            else:
                event.ignore()