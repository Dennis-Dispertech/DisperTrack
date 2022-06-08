from PyQt5.QtWidgets import QApplication
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
from dispertrack.view.waterfall_window import WaterfallWindow


def start_analysis():
    app = QApplication([])
    win = WaterfallWindow()
    win.show()
    app.exec()


if __name__ == '__main__':
    start_analysis()