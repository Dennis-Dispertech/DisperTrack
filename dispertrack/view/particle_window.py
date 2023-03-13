from PyQt5 import uic
from PyQt5.QtCore import QTimer
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
            if props is None:
                self.line_current_particle.setText(str(particle_number))
            else:
                self.line_current_particle.setText(f'{particle_number}/{len(props)}')

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

        self.action_next.triggered.connect(self.next_particle)
        self.action_previous.triggered.connect(self.previous_particle)
        self.action_valid.triggered.connect(self.toggle_valid)
        self.button_apply.clicked.connect(self.calculate_particle_data)
        self.button_save.clicked.connect(self.save_data)
        self.slider_frames.valueChanged.connect(self.update_1d_plot)
        self.button_next.clicked.connect(self.next_particle)
        self.button_previous.clicked.connect(self.previous_particle)

        self.saved = False
        self.particle_num = particle_number

        self.calculate_particle_data()

        self.button_save_all.clicked.connect(self.save_all)

        self.save_all_timer = QTimer()
        self.save_all_timer.timeout.connect(self.next_particle)

    def toggle_valid(self):
        self.check_valid.setCheckState(not self.check_valid.checkState())

    def next_particle(self):
        if self.particle_num is None:
            self.button_next.setEnabled(False)
            self.save_all_timer.stop()
            return
        if self.particle_num + 1 < len(self.props):
            try:
                self.save_label_data()
            except ValueError:
                print(f'Problem saving particle {self.particle_num}')
            self.particle_num += 1
            self.check_valid.setCheckState(True)
            self.calculate_particle_data()
            self.setWindowTitle(f'Single-Particle Analysis - pcle {self.particle_num}')
            # self.line_current_particle.setText(str(self.particle_num))
            self.line_current_particle.setText(f'{self.particle_num}/{len(self.props)}')
            if self.particle_num + 1 > len(self.props):
                self.button_next.setEnabled(False)
        else:
            self.button_next.setEnabled(False)
            self.save_all_timer.stop()

    def previous_particle(self):
        if self.particle_num is None:
            self.button_previous.setEnabled(False)
            return
        if self.particle_num - 1 >= 0:
            self.save_label_data()
            self.particle_num -= 1
            self.calculate_particle_data()
            self.setWindowTitle(f'Single-Particle Analysis - pcle {self.particle_num}')
            # self.line_current_particle.setText(str(self.particle_num))
            self.line_current_particle.setText(f'{self.particle_num}/{len(self.props)}')
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
        pi.setTitle('position')

        msd_plot = self.widget_diffusion.getPlotItem()
        msd_plot.clear()
        x = 5E-3*np.arange(1, int(len(self.positions)/2))

        fiber_width_pixl = 412
        fiber_width = 180E-6
        pixl = fiber_width / fiber_width_pixl   # ~ 0.44 um per pixel
        pixl_um = pixl * 1E6
        fps = self.analyze_model.meta['fps']

        # msd_plot.plot(x, self.MSD['MSD'].array, pen=None, symbol='o')
        msd_plot.plot(x / fps, pixl_um**2 * self.MSD['MSD'].array, pen=None, symbol='o')
        msd_plot.setLogMode(True, True)
        msd_plot.setTitle('diffusion')

        # fit = np.polyfit(x[:10], self.MSD['MSD'][:10], 2)
        fit = np.polyfit(x[:10] / fps, pixl_um ** 2 * self.MSD['MSD'][:10], 2)
        info = f'D: {fit[1]/2:2.2f}um^2/s\n V: {np.sqrt(fit[0])}um/s\n O: {fit[2]}'
        self.diffusion_information.setText(info)

    def update_intensities(self):
        pi = self.widget_intensity.getPlotItem()
        pi.clear()
        x = np.arange(0, len(self.intensities[self.intensities > 0]))
        pi.plot(x, self.intensities[self.intensities > 0])
        pi.setTitle('intensity')

        hi = self.widget_intensity_histogram.getPlotItem()
        hi.clear()
        intensity = self.intensities
        if self.analyze_model.coupled_intensity is not None:
            intensity /= self.analyze_model.coupled_intensity
            print('normalized')
        intensity = np.power(intensity[~np.isnan(intensity)], 1/6)
        bins = np.linspace(np.nanmin(intensity), np.nanmax(intensity), 20)
        y, x = np.histogram(intensity, bins=bins)

        hi.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 125, 125, 150))
        hi.setTitle('intensity histogram')

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
            'valid': self.check_valid.checkState(),
            'width': int(self.line_slice_width.text()),
            'threshold': int(self.line_position_threshold.text()),
            'separation': int(self.line_position_separation.text()),
            'radius': int(self.line_position_radius.text()),
            }
        self.analyze_model.save_particle_label(data, metadata, self.particle_num)

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
        # line_slice_start and line_slice_stop ar None sometimes
        # I don't know what it is, so for now I just work around it
        try:
            line_slice_start = int(self.line_slice_start.text())
        except:
            line_slice_start = 0
        try:
            line_slice_stop = int(self.line_slice_stop.text())
        except:
            line_slice_stop = 0
        metadata = {
            'start': line_slice_start,
            'stop': line_slice_stop,
            'width': int(self.line_slice_width.text()),
            'threshold': int(self.line_position_threshold.text()),
            'separation': int(self.line_position_separation.text()),
            'radius': int(self.line_position_radius.text()),
            }

        self.particle_num = self.analyze_model.save_particle_data(data, metadata, self.particle_num)
        # this method only takes ONE argument ??????????????????

        self.saved = True

    def save_all(self):
        self.save_all_timer.start(200)

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