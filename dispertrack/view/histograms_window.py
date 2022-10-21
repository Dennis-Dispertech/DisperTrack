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
        self.analyze_model.calculate_particle_properties()

        self.update_histogram_diffusion()
        self.update_histogram_intensities()
        self.update_plot_diffusion_intensity()

    def update_histogram_diffusion(self):
        d = 2 * self.analyze_model.r / 1e-9
        # y, x = np.histogram(self.analyze_model.r, bins=np.linspace(np.min(self.analyze_model.r),
        #                                                            np.max(self.analyze_model.r), 40))
        y, x = np.histogram(d, bins=np.linspace(np.min(d), np.max(d), 40))
        # y, x = np.histogram(self.analyze_model.d, bins=np.linspace(np.min(self.analyze_model.d),
        #                                                            np.max(self.analyze_model.d), 40))
        self.analyze_model.x_y = (x, y)
        self.diffusion_histogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))  # this shows incorrect x-axis
        # self.diffusion_histogram.plot(x, y, stepMode='center', fillLevel=0, brush=(0,0,255,150))  # this gives an error
        # self.diffusion_histogram.plot(np.repeat(x, 2)[1:-1], np.repeat(y, 2), fillLevel=0, brush=(0, 0, 255, 150))  # making it "manually" results in the exact same incorrect x-axis
        # self.diffusion_histogram.plot((x[:-1] + x[1:]) / 2, y, fillLevel=0, brush=(0, 0, 255, 150))  # making it "manually" results in the exact same incorrect x-axis
        # plt.hist(d, 40)
        self.diffusion_histogram.setTitle('particle size histogram')

    def update_histogram_intensities(self):
        y, x = np.histogram(self.analyze_model.mean_intensity, bins=np.linspace(np.min(
            self.analyze_model.mean_intensity),
                            np.max(self.analyze_model.mean_intensity), 40))

        self.intensity_histogram.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
        self.intensity_histogram.setTitle('intensity histogram')

    def update_plot_diffusion_intensity(self):
        self.diffusion_intensity_plot.plot(self.analyze_model.mean_intensity, self.analyze_model.r, symbol ='o')
        self.diffusion_intensity_plot.setTitle('"diffusion intensity plot"')