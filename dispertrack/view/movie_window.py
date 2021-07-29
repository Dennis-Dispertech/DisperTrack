from pathlib import Path

from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from dispertrack import home_path
from dispertrack.model.make_waterfall import MakeWaterfall
from dispertrack.view import view_folder


class MovieWindow(QMainWindow):
    """Main window to do the pre-processing of the movie data, before making it a waterfall.
    It allows to select the ROI, transpose or not, etc.
    """
    def __init__(self):
        super(MovieWindow, self).__init__()
        uic.loadUi(view_folder / 'GUI' / 'open_video.ui', self)

        self.movie_model = MakeWaterfall()

        if 'last_waterfall_directory' in self.movie_model.contextual_data:
            self.line_output_dir.setText(self.movie_model.contextual_data['last_waterfall_directory'])
        else:
            self.line_output_dir.setText(str(home_path))

        self.action_open_video.triggered.connect(self.open_video)
        self.action_calculate_waterfall.triggered.connect(self.calculate_waterfall)

        self.slider_frames.valueChanged.connect(self.update_movie_image)
        self.button_save.clicked.connect(self.save_waterfall)
        self.button_select_folder.clicked.connect(self.select_output_folder)

    def select_output_folder(self):
        last_dir = self.movie_model.contextual_data.get('waterfall_output_dir', home_path)
        output_dir = QFileDialog.getExistingDirectory(self,
                                                      'Select output waterfall directory',
                                                      str(last_dir))
        self.movie_model.contextual_data['waterfall_output_dir'] = str(output_dir)
        self.line_output_dir.setText(output_dir)

    def open_video(self):
        """ Triggers a dialog to select the file to be analyzed.
        It opens the file but does not load it to memory.
        """
        last_dir = self.movie_model.contextual_data.get('last_movie_directory', home_path)
        file = QFileDialog.getOpenFileName(self, 'Open movie data', str(last_dir), filter='*.h5')[0]

        if file != '':
            file = Path(file)
        else:
            return
        try:
            self.movie_model.load_movie(file)
        except Exception as e:
            mb = QMessageBox(self)
            mb.setText('Something went wrong opening the file')
            mb.setDetailedText(str(e))
            mb.exec()
            return

        self.setWindowTitle(f'Analysis of movie {file.name}')
        self.update_movie_image()
        self.slider_frames.setRange(0, self.movie_model.movie_metadata['frames'])
        self.line_filename.setText(file.name)

    def update_movie_image(self, index=0):
        """ Updates the image being displayed on the image viewport.
        """
        if self.movie_model.movie_data is None:
            return
        bkg = self.movie_model.calculate_movie_background(index)
        auto_levels = False
        if index == 0:
            auto_levels = True
        self.video_images.update_image(self.movie_model.movie_data[:, :, index]-bkg, auto_histogram_range=auto_levels)

    def calculate_waterfall(self):
        transpose = self.checkbox_transpose.checkState()
        axis_choice = [1, 0]
        axis = axis_choice[self.combo_axis.currentIndex()]
        roi_x, roi_y = self.video_images.get_roi_values()
        ROI = (roi_y[0], roi_y[0]+roi_y[1])
        self.movie_model.calculate_waterfall(transpose=transpose, axis=axis, roi=ROI)
        self.waterfall_image.update_image(self.movie_model.waterfall)

    def save_waterfall(self):
        try:
            waterfall_directory = Path(self.line_output_dir.text())
            waterfall_name = self.line_filename.text()
            self.movie_model.save_waterfall(waterfall_directory/waterfall_name)
        except Exception as e:
            mb = QMessageBox(self)
            mb.setText('Something went wrong opening the file')
            mb.setDetailedText(str(e))
            mb.exec()
            return

    def closeEvent(self, *args, **kwargs):
        self.movie_model.unload_movie()
        super().closeEvent(*args, **kwargs)
