from pathlib import Path
from threading import Thread

import numpy as np

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog, QMainWindow, QMessageBox

from dispertrack import home_path
from dispertrack.model.anlyze_waterfall import AnalyzeWaterfall
from dispertrack.view import view_folder

import pyqtgraph as pg


class WaterfallWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(view_folder / 'GUI' / 'waterfall_analysis.ui', self)
        self.analyze_model = AnalyzeWaterfall()
        self.action_open.triggered.connect(self.open_waterfall)
        self.action_transpose.triggered.connect(self.transpose_waterfall)
        self.action_setup_roi.triggered.connect(self.setup_roi_line)
        self.action_apply_roi.triggered.connect(self.display_ROI)
        self.action_crop.triggered.connect(self.crop_waterfall)
        self.action_background.triggered.connect(self.calculate_background)

        self.line_angle.textEdited.connect(self.change_line_angle)
        self.slider_angle.valueChanged.connect(self.change_slider_angle)

        self.waterfall_image = pg.ImageView()

        self.ROI_line = pg.LineROI((0, 0), (1, 0), width=2)

        self.hline1 = pg.InfiniteLine(angle=0, movable=True, hoverPen={'color': "FF0", 'width': 4})
        self.hline2 = pg.InfiniteLine(angle=0, movable=True, hoverPen={'color': "FF0", 'width': 4})

        self.first_waterfall_update = True

        plot_layout = self.plot_widget.layout()
        plot_layout.addWidget(self.waterfall_image)

    def open_waterfall(self):
        last_dir = self.analyze_model.contextual_data.get('last_dir', home_path)
        file = QFileDialog.getOpenFileName(self, 'Open Waterfall data', str(last_dir), filter='*.h5')[0]
        if file != '':
            file = Path(file)
        else:
            return
        try:
            self.analyze_model.load_waterfall(file)
        except Exception as e:
            mb = QMessageBox(self)
            mb.setText('Something went wrong opening the file')
            mb.setDetailedText(str(e))
            mb.exec()
            return

        self.update_image(self.analyze_model.waterfall)
        self.analyze_model.contextual_data.update({'last_dir': str(file.parent)})
        if ind := self.analyze_model.metadata['bkg_axis'] is not None:
            self.combo_bkg_axis.setCurrentIndex(ind)

        if sigma := self.analyze_model.metadata['bkg_sigma'] is not None:
            self.line_bkg.setText(str(sigma))
        else:
            self.line_bkg.setText('')

    def update_image(self, image):
        self.waterfall_image.setImage(image, autoLevels=True, autoRange=True)
        self.hline1.setValue(0)
        self.hline2.setValue(image.shape[0])
        self.hline1.setBounds((0, image.shape[1]))
        self.hline2.setBounds((0, image.shape[1]))
        if self.first_waterfall_update:
            view = self.waterfall_image.getView()
            view.addItem(self.hline1)
            view.addItem(self.hline2)
            self.first_waterfall_update = False

    def crop_waterfall(self):
        x = [self.hline1.value(), self.hline2.value()]
        x = np.sort(x).astype(np.int)
        self.analyze_model.crop_waterfall(x[0], x[1])
        self.update_image(self.analyze_model.waterfall)

    def calculate_background(self):
        def calculate_bkg_thread(self, axis, sigma):
            self.statusbar.showMessage('Calculating Background')
            if sigma is not None:
                self.analyze_model.calculate_background(axis=axis, sigma=sigma)
            else:
                self.analyze_model.calculate_background(axis=axis)

            self.update_image(self.analyze_model.corrected_data)
            self.statusbar.showMessage('')

        axis = int(self.combo_bkg_axis.currentIndex())
        if sigma := self.line_bkg.text() != '':
            sigma = int(sigma)
        else:
            sigma = None

        t = Thread(target=calculate_bkg_thread, args=(self, axis, sigma))
        t.start()

    def transpose_waterfall(self):
        self.analyze_model.transpose_waterfall()
        self.update_image(self.analyze_model.waterfall)

    def setup_roi_line(self):
        view = self.waterfall_image.getView()
        view.addItem(self.ROI_line)

    def display_ROI(self):
        data = self.ROI_line.getArrayRegion(self.analyze_model.waterfall, self.waterfall_image.getImageItem())
        self.waterfall_image.setImage(data)

    def change_slider_angle(self, value):
        self.line_angle.setText(str(value))
        self.ROI_line.setAngle(value)

    def change_line_angle(self, value):
        self.slider_angle.setValue(int(value))
        self.ROI_line.setAngle(int(value))


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    win = WaterfallWindow()
    win.show()
    app.exec()