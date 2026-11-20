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
    assert win.model.error_db() < -150 and 'Model (pi form): 5 elements' in win.status.text()
    assert '.SUBCKT dut_model p1 p2' in win.netlist_view.toPlainText()
    # the schematic shows the selected row
    def shown():
        return win.schematic.axes[0].get_title()
    assert win.schematic_row() is win.model.rows()[0] and shown() == 'p1 - gnd: CR3 C'
    win.table.setCurrentCell(2, 0)
    assert shown() == 'p1 - p2: CR7 SerRL'
    assert 'L\n400pH' in [t.get_text() for t in win.schematic.axes[0].texts]
    # another candidate for the series branch: table, netlist and error follow
    combo = win.table.cellWidget(2, 1)
    assert combo.count() > 1
    before = win.netlist_view.toPlainText()
    combo.setCurrentIndex(combo.count() - 1)
    assert win.netlist_view.toPlainText() != before and win.model.error_db() > -150
    assert shown() != 'p1 - p2: CR7 SerRL'                 # the schematic follows too
    combo.setCurrentIndex(0)
    assert win.netlist_view.toPlainText() == before and shown() == 'p1 - p2: CR7 SerRL'
    # plot views: S-parameters, error, and L/R/Q for a 2-port; all data on request
    assert [win.view.itemData(k) for k in range(win.view.count())] == ['s', 'error', 'lrq']
    assert [ax.get_ylabel() for ax in win.figure.axes] == ['magnitude (dB)', 'phase (deg)']
    win.view.setCurrentIndex(win.view.findData('lrq'))
    assert [ax.get_ylabel() for ax in win.figure.axes] == ['series L (pH)', 'series R (ohm)',
                                                           'Q at port 1']
    win.view.setCurrentIndex(win.view.findData('error'))
    assert win.figure.axes[0].get_ylabel() == '|S model - S data| (dB)'
    assert 'shaded' not in win.figure.axes[0].get_title()
    win.all_data.setChecked(True)
    assert 'fit band shaded' in win.figure.axes[0].get_title()
    # marker: readout of all curves, kept across a change of view, moved by steps
    win.view.setCurrentIndex(win.view.findData('s'))
    text = win.set_marker(12.0)
    assert text.startswith('f = ') and 'S21: data' in text
    assert win.readout.toPlainText() == text
    assert all(len(ax.lines) for ax in win.figure.axes) and len(win.marker_lines) == 2
    f0 = win.marker_f
    win.step_marker(1)
    assert win.marker_f > f0
    win.view.setCurrentIndex(win.view.findData('lrq'))
    assert win.marker_f > f0 and 'series L: data' in win.readout.toPlainText()
    assert len(win.marker_lines) == 3
    assert win.table.columnCount() == 6 and win.table.item(2, 3).text().endswith('%')
    assert win.table.item(2, 4).text().endswith('%')
    win.format.setCurrentIndex(win.format.findData('spectre'))
    assert 'subckt dut_model (p1 p2)' in win.netlist_view.toPlainText()
    files = win.save_to(tmp_path / 'out')
    # another file: old results and the old band are gone; the same file keeps the band
    other = tmp_path / 'other.s2p'
    make_network(2, TWO_PORT).write_touchstone(str(other.with_suffix('')))
    win.fmax.setValue(20)
    assert win.set_file(src) and win.fmax.value() == 20
    assert win.set_file(other) and win.fmax.value() == 0 and win.fmax.text() == 'data'
    assert win.table.rowCount() == 0 and win.netlist_view.toPlainText() == ''
    assert win.marker_f is None and win.readout.toPlainText() == ''
    assert not win.figure.axes and not win.save_button.isEnabled()
    win.fmin.setValue(5)
    win.full_range.click()
    assert win.fmin.value() == 0 and win.options().fmin is None and win.options().fmax is None
    assert {p.name for p in files} == {
        'dut_crcat.scs', 'dut_crcat.s2p', 'dut_crcat.json', 'dut_crcat.png',
        'dut_crcat_error.png', 'dut_crcat_lrq.png', 'dut_crcat_schematic.png'}
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
