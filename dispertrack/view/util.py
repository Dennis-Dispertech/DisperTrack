from functools import wraps

from PyQt5.QtWidgets import QMessageBox


def error_message(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            mb = QMessageBox(args[0])
            mb.setText('Something went wrong')
            mb.setDetailedText(str(e))
            mb.exec()
            return
    return func_wrapper