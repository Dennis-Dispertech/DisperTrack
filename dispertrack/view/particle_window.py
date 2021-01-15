from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

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

        self.particle_image = pg.ImageView()
        self.particle_image.setPredefinedGradient('thermal')
        self.widget_image.layout().addWidget(self.particle_image)

        self.intensity_plot = pg.PlotWidget()
        self.widget_intensity.layout().addWidget(self.intensity_plot)

        self.position_plot = pg.PlotWidget()
        self.widget_position.layout().addWidget(self.position_plot)

        self.frame_cut_plot = pg.PlotWidget()
        self.widget_1d_plot.layout().addWidget(self.frame_cut_plot)

        self.line_slice_start.setText(str(slice.start))
        self.line_slice_stop.setText(str(slice.stop))

        self.calculate_particle_data()

        self.button_apply.clicked.connect(self.calculate_particle_data)
        self.slider_frames.valueChanged.connect(self.update_1d_plot)

    def calculate_particle_data(self):
        """ Calculates particle data based on some parameters such as the coordinates of the
        slice and """
        start = int(self.line_slice_start.text())
        stop = int(self.line_slice_stop.text())
        (start, stop) = np.sort((start, stop))
        width = int(self.line_slice_width.text())
        separation = int(self.line_position_separation.text())
        radius = int(self.line_position_radius.text())
        self.data = self.analyze_model.calculate_slice(start, stop, width)
        self.intensities, self.positions = self.analyze_model.calculate_intensities_cropped(
            self.data,
            separation=separation,
            radius=radius
            )
        self.update_image()
        self.update_intensities()
        self.update_positions()

        self.slider_frames.setRange(0, self.data.shape[0])

    def update_image(self):
        self.particle_image.setImage(self.data)
        self.particle_image.autoLevels()
        self.particle_image.autoRange()

    def update_positions(self):
        pi = self.position_plot.getPlotItem()
        pi.clear()
        x = np.arange(0, len(self.positions))
        pi.plot(x[self.positions > 0], self.positions[self.positions > 0])

    def update_intensities(self):
        pi = self.intensity_plot.getPlotItem()
        pi.clear()
        x = np.arange(0, len(self.intensities))
        pi.plot(x[self.intensities > 0], self.intensities[self.intensities > 0])

    def update_1d_plot(self, frame):
        pi = self.frame_cut_plot.getPlotItem()
        pi.clear()
        pi.plot(self.data[frame])
