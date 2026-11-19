"""Runs the model extraction in a thread, so the window stays responsive."""

from PySide6.QtCore import QThread, Signal

from .. import model


class ExtractWorker(QThread):
    progress = Signal(float, str)
    done = Signal(object)            # crcat.model.Model
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, network, options, parent=None):
        super().__init__(parent)
        self.network, self.options = network, options
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            m = model.extract(self.network, self.options, progress=self.progress.emit,
                              cancel=lambda: self._cancel)
        except model.Cancelled:
            self.cancelled.emit()
        except Exception as ex:      # shown in the window, not a crash of the thread
            self.failed.emit(f'{type(ex).__name__}: {ex}')
        else:
            self.done.emit(m)
