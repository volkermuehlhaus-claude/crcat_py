"""Small PySide6 window around :mod:`crcat.model` (command ``crcat-gui``).

The GUI only calls the core; the core never imports this package.
"""

import sys


def main(argv=None):
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.exit('crcat-gui needs PySide6 and matplotlib: pip install "crcat[gui]"')
    from .main_window import MainWindow
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    from PySide6.QtCore import QSettings
    win = MainWindow(QSettings('crcat', 'crcat-gui'))
    if len(argv) > 1:
        win.set_file(argv[1])
    else:
        win.restore_last_file()
    win.show()
    return app.exec()
