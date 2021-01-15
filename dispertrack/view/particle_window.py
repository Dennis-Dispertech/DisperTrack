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
        self.slice = slice
        self.data = None

        self.particle_image = pg.ImageView()
        self.widget_image.layout().addWidget(self.particle_image)

        self.intensity_plot = pg.PlotWidget()
        self.widget_intensity.layout().addWidget(self.intensity_plot)

        self.position_plot = pg.PlotWidget()
        self.widget_position.layout().addWidget(self.position_plot)

        self.calculate_particle_data()

    def calculate_particle_data(self):
        """ Calculates particle data based on some parameters such as the coordinates of the
        slice and """
        start = self.slice.start
        stop = self.slice.stop
        (start, stop) = np.sort((start, stop))
        width = int(self.line_slice_width.text())
        self.data = self.analyze_model.calculate_slice(start, stop, width)
        print(self.data.shape)
        self.update_image()

    def update_image(self):
        self.particle_image.setImage(self.data)
