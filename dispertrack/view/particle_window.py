from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QMessageBox

import numpy as np
import pyqtgraph as pg

from dispertrack.view import view_folder


class ParticleWindow(QMainWindow):
    def __init__(self, analyze_model, slice_data=None, props=None, particle_number=None):
        super().__init__()
        uic.loadUi(view_folder / 'GUI' / 'particle_window.ui', self)

        if particle_number is not None:
            self.setWindowTitle(f'Single-Particle Analysis - pcle {particle_number}')
            self.line_current_particle.setText(str(particle_number))

        self.analyze_model = analyze_model

        self.data = None
        self.intensities = None
        self.positions = None
        self.centers = None

        self.widget_image.setPredefinedGradient('thermal')

        self.standard_slice = False
        self.props_slice = False

        if slice_data is not None:
            self.line_slice_start.setText(str(slice_data.start))
            self.line_slice_stop.setText(str(slice_data.stop))
            self.standard_slice = True
        elif props is not None:
            self.props = props
            self.line_slice_start.setText('None')
            self.line_slice_stop.setText('None')
            self.props_slice = True
        else:
            raise Exception('Either slice_data is specified or label is specified')

        self.button_apply.clicked.connect(self.calculate_particle_data)
        self.button_save.clicked.connect(self.save_data)
        self.slider_frames.valueChanged.connect(self.update_1d_plot)
        self.button_next.clicked.connect(self.next_particle)
        self.button_previous.clicked.connect(self.previous_particle)

        self.saved = False
        self.particle_num = particle_number

        self.calculate_particle_data()

    def next_particle(self):
        if self.particle_num is None:
            self.button_next.setEnabled(False)
            return
        if self.particle_num + 1 <= len(self.props):
            self.save_label_data()
            self.particle_num += 1
            self.calculate_particle_data()
            self.setWindowTitle(f'Single-Particle Analysis - pcle {self.particle_num}')
            self.line_current_particle.setText(str(self.particle_num))
            if self.particle_num + 1 > len(self.props):
                self.button_next.setEnabled(False)
        else:
            self.button_next.setEnabled(False)

    def previous_particle(self):
        if self.particle_num is None:
            self.button_previous.setEnabled(False)
            return
        if self.particle_num - 1 >= 0:
            self.save_label_data()
            self.particle_num -= 1
            self.calculate_particle_data()
            self.setWindowTitle(f'Single-Particle Analysis - pcle {self.particle_num}')
            self.line_current_particle.setText(str(self.particle_num))
            if self.particle_num == 0:
                self.button_previous.setEnabled(False)
        else:
            self.button_previous.setEnabled(False)

    def calculate_particle_data(self):
        """ Calculates particle data based on some parameters such as the coordinates of the
        slice and """
        width = int(self.line_slice_width.text())
        separation = int(self.line_position_separation.text())
        radius = int(self.line_position_radius.text())
        threshold = int(self.line_position_threshold.text())

        if self.standard_slice:
            start = int(self.line_slice_start.text())
            stop = int(self.line_slice_stop.text())
            (start, stop) = np.sort((start, stop))

            sliced_data = self.analyze_model.calculate_slice(start, stop, width)
        elif self.props_slice:
            props = self.props[self.particle_num]
            sliced_data = self.analyze_model.calculate_slice_from_label(props, width=width)
        else:
            return

        self.data = sliced_data[:, 1:]
        self.centers = sliced_data[:, 0]

        self.intensities, self.positions = self.analyze_model.calculate_intensities_cropped(
            self.data,
            separation=separation,
            radius=radius,
            threshold=threshold,
            )
        self.MSD = self.analyze_model.calculate_diffusion(self.centers, self.positions)
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

        msd_plot = self.widget_diffusion.getPlotItem()
        msd_plot.clear()
        x = 5E-3*np.arange(1, int(len(self.positions)/2))
        msd_plot.plot(x, self.MSD['MSD'].array, pen=None, symbol='o')
        msd_plot.setLogMode(True, True)

    def update_intensities(self):
        pi = self.widget_intensity.getPlotItem()
        pi.clear()
        x = np.arange(0, len(self.intensities[self.intensities > 0]))
        pi.plot(x, self.intensities[self.intensities > 0])

        hi = self.widget_intensity_histogram.getPlotItem()
        hi.clear()
        bins = np.linspace(np.nanmin(self.intensities), np.nanmax(self.intensities), 20)
        y, x = np.histogram(self.intensities[~np.isnan(self.intensities)], bins=bins)
        hi.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 125, 125, 150))

    def update_1d_plot(self, frame):
        self.widget_1d_plot.clear()
        self.widget_1d_plot.plot(self.data[frame])
        if np.isnan(self.data[frame][round(self.positions[frame])]):
            return

        self.widget_1d_plot.plot([self.positions[frame], ], [self.data[frame][round(self.positions[frame])],],
                                 symbolBrush=pg.mkBrush(0, 0, 255, 255),
                                 symbolSize=7)

    def save_label_data(self):
        data = np.stack((self.positions, self.intensities, self.centers))
        metadata = {
            'width': int(self.line_slice_width.text()),
            'threshold': int(self.line_position_threshold.text()),
            'separation': int(self.line_position_separation.text()),
            'radius': int(self.line_position_radius.text()),
            }
        self.analyze_model.save_particle_data(data, metadata, self.particle_num)

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