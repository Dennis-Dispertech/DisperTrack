from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
import os

from dispertrack.view import view_folder

os.environ['QT_MAC_WANTS_LAYER'] = '1'
from dispertrack.view.waterfall_window import WaterfallWindow


def start_analysis():
    app = QApplication([])
    app.setWindowIcon(QIcon(str(view_folder / 'GUI'/ 'favicon.png')))
    win = WaterfallWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    # start_analysis()
    app = QApplication([])
    win = WaterfallWindow()
    win.show()
    app.exec()