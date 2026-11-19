"""GUI smoke test, offscreen: load, extract, switch an alternative, save, cancel."""

import os
import time

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication  # noqa: E402

from crcat.gui.main_window import MainWindow  # noqa: E402
from test_model import TWO_PORT, make_network  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, condition, timeout=120):
    end = time.time() + timeout
    while not condition():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() < end, 'timeout'


def test_gui_extract_and_save(app, tmp_path):
    src = tmp_path / 'dut.s2p'
    make_network(2, TWO_PORT).write_touchstone(str(src.with_suffix('')))
    win = MainWindow()
    win.run()                                        # no file yet: a message, no dialog
    assert 'file first' in win.status.text() and win.worker is None
    assert not win.set_file(tmp_path / 'missing.s2p') and 'Cannot read' in win.status.text()
    assert win.set_file(src) and '2 port(s)' in win.status.text()
    win.pick.setCurrentIndex(win.pick.findData('best'))
    win.run()
    assert not win.run_button.isEnabled()
    wait_for(app, lambda: win.model is not None and win.worker is None)
    assert win.table.rowCount() == 3 and win.save_button.isEnabled()
    assert win.model.error_db() < -150 and 'Model: 5 elements' in win.status.text()
    assert '.SUBCKT dut_model p1 p2' in win.netlist_view.toPlainText()
    # another candidate for the series branch: table, netlist and error follow
    combo = win.table.cellWidget(2, 1)
    assert combo.count() > 1
    before = win.netlist_view.toPlainText()
    combo.setCurrentIndex(combo.count() - 1)
    assert win.netlist_view.toPlainText() != before and win.model.error_db() > -150
    combo.setCurrentIndex(0)
    assert win.netlist_view.toPlainText() == before
    win.format.setCurrentIndex(win.format.findData('spectre'))
    assert 'subckt dut_model (p1 p2)' in win.netlist_view.toPlainText()
    files = win.save_to(tmp_path / 'out')
    assert {p.name for p in files} == {'dut_crcat.scs', 'dut_crcat.s2p', 'dut_crcat.json',
                                       'dut_crcat.png'}
    win.close()


def test_gui_cancel(app, tmp_path):
    src = tmp_path / 'dut.s2p'
    make_network(2, TWO_PORT).write_touchstone(str(src.with_suffix('')))
    win = MainWindow()
    win.set_file(src)
    win.run()
    win.cancel()
    wait_for(app, lambda: win.worker is None)
    assert win.status.text() == 'Cancelled.' and win.model is None
    assert win.run_button.isEnabled() and not win.cancel_button.isEnabled()
    win.close()
