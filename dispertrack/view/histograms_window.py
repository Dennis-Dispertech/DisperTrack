import numpy as np

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

from dispertrack.view import view_folder
import matplotlib.pyplot as plt


class HistogramWindow(QMainWindow):
    def __init__(self, analyze_model):
        super(HistogramWindow, self).__init__()
        uic.loadUi(view_folder / 'GUI' / 'histograms_sizes.ui', self)
        self.analyze_model = analyze_model
        self.d = [2*self.analyze_model.pcle_data[i].get('r', np.nan) for i in self.analyze_model.pcle_data.keys()]
        self.i = [self.analyze_model.pcle_data[i].get('mean_intensity', np.nan) for i in self.analyze_model.pcle_data.keys()]


        self.update_histogram_diffusion()
        self.update_histogram_intensities()
        self.update_plot_diffusion_intensity()

    def update_histogram_diffusion(self):
        plt.figure()
        plt.hist(self.d, 40)
        plt.show()

    def update_histogram_intensities(self):
        plt.figure()
        plt.hist(self.i, 40)
        plt.show()

    def update_plot_diffusion_intensity(self):
        plt.figure()
        plt.plot(self.d, self.i)
        plt.show()