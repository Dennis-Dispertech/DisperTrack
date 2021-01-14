from pathlib import Path

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog, QMainWindow, QMessageBox

from dispertrack import home_path
from dispertrack.model.anlyze_waterfall import AnalyzeWaterfall
from dispertrack.view import view_folder

import pyqtgraph as pg


class WaterfallWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(view_folder / 'GUI' / 'form.ui', self)
        self.analyze_model = AnalyzeWaterfall()
        self.action_open.triggered.connect(self.open_waterfall)
        self.action_transpose.triggered.connect(self.transpose_waterfall)
        self.action_setup_roi.triggered.connect(self.setup_roi_line)
        self.action_apply_roi.triggered.connect(self.display_ROI)

        self.line_angle.textEdited.connect(self.change_line_angle)
        self.slider_angle.valueChanged.connect(self.change_slider_angle)

        self.waterfall_image = pg.ImageView()

        self.ROI_line = pg.LineROI((0, 0), (1, 0), width=2)

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
        self.waterfall_image.setImage(self.analyze_model.waterfall)
        self.analyze_model.contextual_data.update({'last_dir': str(file.parent)})

    def transpose_waterfall(self):
        self.analyze_model.transpose_waterfall()
        self.waterfall_image.setImage(self.analyze_model.waterfall)

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