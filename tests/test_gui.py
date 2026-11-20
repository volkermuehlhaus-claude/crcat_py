"""GUI smoke test, offscreen: load, extract, switch an alternative, save, cancel."""

import os
import time

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication  # noqa: E402

from crcat import model  # noqa: E402
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
    assert win.verdict.text() == 'suited' and 'inductor' in win.status.text()
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
    assert [win.view.itemData(k) for k in range(win.view.count())] == ['s', 'error', 'lrq', 'size']
    # data check and verdict: an inductor opens with L, R, Q; no suggestions for a good model
    assert win.data_check.kind == 'inductor' and win.view.currentData() == 'lrq'
    assert win.verdict.text() == 'good' and not win.suggest_buttons
    assert win.status.text().startswith('0.5 to 40 GHz: 5 elements')
    win.copy_button.click()
    assert app.clipboard().text() == win.netlist_view.toPlainText()
    # size against accuracy: a click picks that model; the limit field re-picks at once
    win.view.setCurrentIndex(win.view.findData('size'))
    points = win.figure.crcat_tradeoff
    win.use_size(points[1])
    assert win.model.n_elements() == points[1].n_elements < 5 and win.verdict.text() != 'good'
    win.use_size(points[-1])
    assert win.model.error_db() < -150
    win.max_elements.setValue(1)
    assert all(b.chosen.n_elements <= 1 for b in win.model.branches) and win.worker is None
    win.max_elements.setValue(5)
    assert win.model.error_db() < -150 and win.table.rowCount() == 3
    win.view.setCurrentIndex(win.view.findData('s'))
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
        'dut_crcat.scs', 'dut_crcat_tb.scs', 'dut_crcat.s2p', 'dut_crcat.json', 'dut_crcat.png',
        'dut_crcat_error.png', 'dut_crcat_lrq.png', 'dut_crcat_schematic.png'}
    win.close()


def test_gui_suggestion_button(app, tmp_path):
    import numpy as np
    from test_easy import line
    src = tmp_path / 'line.s2p'
    line(50.0, 6e-12, np.linspace(0.5e9, 100e9, 120), loss=0.01).write_touchstone(
        str(src.with_suffix('')))
    win = MainWindow()
    win.coupling.setChecked(False)
    assert win.set_file(src) and win.verdict.text() == 'limited'
    assert [b.text()[:7] for b in win.suggest_buttons] == ['Fit to ']   # from the data check
    win.suggest_buttons[0].click()                                      # sets the band and runs
    assert 10 < win.fmax.value() < 35 and win.worker is not None
    wait_for(app, lambda: win.model is not None and win.worker is None)
    assert win.model.freq[-1] <= win.fmax.value() * 1e9 and win.verdict.text() in ('good', 'fair')
    win.close()


def test_gui_quality_and_advanced(app, tmp_path):
    src = tmp_path / 'dut.s2p'
    make_network(2, TWO_PORT).write_touchstone(str(src.with_suffix('')))
    win = MainWindow()
    win.show()
    # the simple window: band, quality; everything else folded away
    assert win.quality.currentData() == 'standard' and win.advanced.isHidden()
    assert win.advanced_note.text() == ''
    win.advanced_button.click()
    assert not win.advanced.isHidden()
    win.advanced_button.click()
    assert win.advanced.isHidden()
    # a preset sets the fields behind it
    win.quality.setCurrentIndex(win.quality.findData('small'))
    assert (win.target.value(), win.reactance_percent.value(), win.loss_percent.value()) == (-30, 30, 50)
    assert win.options().pick == 'simplest' and model.quality_of(win.options()) == 'small'
    win.quality.setCurrentIndex(win.quality.findData('accurate'))
    assert win.options().pick == 'best' and win.target.value() == -40
    # a field changed by hand: Custom; back to the values of a preset: that preset
    win.target.setValue(-45)
    assert win.quality.currentData() is None and win.advanced.isHidden()
    win.target.setValue(-40)
    assert win.quality.currentData() == 'accurate'
    # hidden settings that differ from the defaults are named
    win.require.setCurrentIndex(win.require.findData('positive-r'))
    assert win.line_mode.currentData() == 'auto' and not win.delay.isEnabled()
    assert win.dc.currentData() == 'data'
    win.dc.setCurrentIndex(win.dc.findData('no-ground'))
    assert win.options().dc == 'no-ground' and 'no DC path to ground' in win.advanced_note.text()
    win.dc.setCurrentIndex(win.dc.findData('data'))
    win.line_mode.setCurrentIndex(win.line_mode.findData('off'))
    assert 'no line at the ports' in win.advanced_note.text() and win.options().delay == 'off'
    win.line_mode.setCurrentIndex(win.line_mode.findData('fixed'))
    assert win.delay.isEnabled() and win.line_z0.isEnabled()
    win.delay.setValue(1.5)
    win.line_z0.setValue(70)
    assert win.options().delay == 1.5e-12 and win.options().line_z0 == 70
    win.line_z0.setValue(0)
    win.coupling.setChecked(False)
    assert win.advanced_note.text() == ('In use: R positive, L and C any sign, '
                                        'line 1.5 ps at each port, no coupled coils')
    win.reset_advanced()
    assert win.advanced_note.text() == '' and win.quality.currentData() == 'standard'
    assert win.options() == model.Options()
    # choosing Custom opens the advanced part
    win.quality.setCurrentIndex(win.quality.findData(None))
    assert not win.advanced.isHidden()
    win.reset_advanced()
    # with a model, a preset picks again at once, without a new search
    assert win.set_file(src)
    win.run()
    wait_for(app, lambda: win.model is not None and win.worker is None)
    n = win.model.n_elements()
    win.quality.setCurrentIndex(win.quality.findData('small'))
    assert win.worker is None and win.model.options.target_db == -30
    assert win.model.n_elements() <= n
    # a setting that needs a new search takes the model away
    for change in (lambda: win.form.setCurrentIndex(win.form.findData('pi')),
                   lambda: win.fmax.setValue(20), lambda: win.refine.setChecked(False),
                   lambda: win.require.setCurrentIndex(win.require.findData('any'))):
        win.run()
        assert not win.advanced.isEnabled() and not win.fmax.isEnabled()
        wait_for(app, lambda: win.model is not None and win.worker is None)
        assert win.advanced.isEnabled() and win.table.rowCount() == 3
        change()
        assert win.model is None and win.table.rowCount() == 0
        assert not win.save_button.isEnabled() and not win.copy_button.isEnabled()
        assert win.netlist_view.toPlainText() == '' and not win.figure.axes
        assert 'the model was removed' in win.status.text()
        assert win.verdict.text() == 'suited'            # the data check again
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
