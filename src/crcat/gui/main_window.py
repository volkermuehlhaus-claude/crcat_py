"""Main window: file and options at the top, branches and netlist left, plot right."""

import pathlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
                               QGroupBox, QProgressBar, QPushButton, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

from .. import __version__, advice, check, data, model, netlist
from ..cli_netlist import write_outputs
from ..plots import VIEWS, compare, views_for
from ..schematic import draw_row
from .worker import ExtractWorker

MAX_ALTERNATIVES = 25
REQUIRE_TEXT = {'positive': 'All positive', 'positive-r': 'R positive, L and C any sign',
                'passive': 'Passive, any sign', 'any': 'Any sign'}
PICK_TEXT = {'simplest': 'Simplest within the error limit', 'best': 'Lowest error'}
QUALITY_TEXT = {'small': 'Small', 'standard': 'Standard', 'accurate': 'Most accurate'}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'crcat {__version__}: S-parameters to lumped-element netlist')
        self.resize(1250, 760)
        self.network = None
        self.model = None
        self.worker = None
        defaults = model.Options()

        # --- file ---
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText('Touchstone file (.sNp)')
        self.file_edit.editingFinished.connect(self._load_from_edit)
        browse = QPushButton('Browse...')
        browse.clicked.connect(self.browse)
        row_file = QHBoxLayout()
        row_file.addWidget(QLabel('Data:'))
        row_file.addWidget(self.file_edit, 1)
        row_file.addWidget(browse)

        # --- options ---
        self.fmin = QDoubleSpinBox(decimals=3, maximum=1e4, suffix=' GHz')
        self.fmax = QDoubleSpinBox(decimals=3, maximum=1e4, suffix=' GHz')
        for w in (self.fmin, self.fmax):
            w.setSpecialValueText('data')            # 0 = limit of the data
            w.setToolTip('0 (shown as "data") uses the limit of the data')
        self.full_range = QPushButton('Full range')
        self.full_range.setToolTip('Fit all frequencies of the data')
        self.full_range.clicked.connect(self.reset_band)
        self.require = QComboBox()
        for key in model.REQUIRE:
            self.require.addItem(REQUIRE_TEXT[key], key)
        self.pick = QComboBox()
        for key in model.PICK:
            self.pick.addItem(PICK_TEXT[key], key)
        self.target = QDoubleSpinBox(decimals=0, minimum=-120, maximum=-10, suffix=' dB')
        self.target.setValue(defaults.target_db)
        self.target.setToolTip('Error limit per branch: RMS difference of the S-parameters '
                               'that the branch model may cause')
        self.reactance_percent = QDoubleSpinBox(decimals=0, minimum=0, maximum=100, suffix=' %')
        self.reactance_percent.setValue(defaults.reactance_percent)
        self.reactance_percent.setSpecialValueText('off')
        self.reactance_percent.setToolTip(
            'Error limit of the port reactance: what a branch does to Im Yii of the ports '
            '(other ports shorted), relative to |Yii|')
        self.loss_percent = QDoubleSpinBox(decimals=0, minimum=0, maximum=100, suffix=' %')
        self.loss_percent.setValue(defaults.loss_percent)
        self.loss_percent.setSpecialValueText('off')
        self.loss_percent.setToolTip(
            'Error limit of the port loss: what a branch does to Re Yii of the ports, relative '
            'to Re Yii. Keeps R and Q right where the loss is a small part of the admittance '
            '(high Q).')
        self.refine = QCheckBox('Refine values')
        self.refine.setChecked(defaults.refine)
        self.refine.setToolTip('Refine the element values of the best networks by a bounded '
                               'least-squares fit on all error measures')
        self.max_elements = QSpinBox(minimum=1, maximum=5)
        self.max_elements.setValue(defaults.max_elements)
        self.form = QComboBox()
        for key, text in (('auto', 'Automatic'), ('pi', 'Pi'), ('tee', 'Tee (2-port)')):
            self.form.addItem(text, key)
        self.form.setToolTip('Pi: a branch from each port to ground and between the ports. '
                             'Tee: two series legs and one leg to ground. Automatic tries '
                             'the tee form if a pi branch misses the error limit.')
        self.coupling = QCheckBox('Coupled coils')
        self.coupling.setChecked(defaults.coupling)
        self.coupling.setToolTip('Replace branches that need a negative inductance by two '
                                 'coils with a coupling factor')
        self.delay = QDoubleSpinBox(decimals=2, minimum=0, maximum=1000, singleStep=0.1,
                                    suffix=' ps')
        self.delay.setSpecialValueText('none')
        self.delay.setToolTip(
            'A piece of line (reference impedance, this delay) at each port, removed from the '
            'data before the fit and put back in the netlist as a line element. For data '
            'with a known feed line.')
        self.sequential = QCheckBox('Sequential')
        self.sequential.setChecked(defaults.sequential)
        self.sequential.setToolTip(
            'Pi form: fit the branches between ports first, then each branch to ground to the '
            'port admittance minus those models (not minus their data), so it makes up for '
            'what they missed')
        self.format = QComboBox()
        for key in netlist.FORMATS:
            self.format.addItem(key, key)
        self.format.currentIndexChanged.connect(self._show_netlist)
        # model quality: one choice for the error limits and the choice of network
        self.quality = QComboBox()
        for key in model.QUALITY:
            self.quality.addItem(QUALITY_TEXT[key], key)
        self.quality.addItem('Custom (see Advanced)', None)
        self.quality.setCurrentIndex(self.quality.findData(model.quality_of(defaults)))
        self.quality.setToolTip(
            'Small: fewer elements, more error allowed (-30 dB, reactance 30 %, loss 50 %).\n'
            'Standard: fewest elements within -40 dB, reactance 20 %, loss 30 %.\n'
            'Most accurate: the lowest error, whatever the number of elements.\n'
            'With a model at hand, a change takes effect at once, without a new search.')
        self.quality.currentIndexChanged.connect(self._quality_chosen)
        self.advanced_button = QToolButton()
        self.advanced_button.setText('Advanced')
        self.advanced_button.setCheckable(True)
        self.advanced_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_button.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_button.setToolTip('Element signs, error limits, form, line at the ports, '
                                        'coupled coils and more')
        self.advanced_button.toggled.connect(self.show_advanced)
        self.advanced_note = QLabel('')          # what is set in the hidden part
        row_simple = QHBoxLayout()
        row_simple.addWidget(QLabel('Fit band from'))
        row_simple.addWidget(self.fmin)
        row_simple.addWidget(QLabel('to'))
        row_simple.addWidget(self.fmax)
        row_simple.addWidget(self.full_range)
        row_simple.addSpacing(16)
        row_simple.addWidget(QLabel('Model:'))
        row_simple.addWidget(self.quality)
        row_simple.addSpacing(16)
        row_simple.addWidget(self.advanced_button)
        row_simple.addWidget(self.advanced_note, 1)
        form2, form3 = QFormLayout(), QFormLayout()
        form2.addRow('Element values', self.require)
        form2.addRow('Choice', self.pick)
        form2.addRow('Elements per branch, up to', self.max_elements)
        form3.addRow('Error limit, S-parameters', self.target)
        form3.addRow('Error limit, reactance', self.reactance_percent)
        form3.addRow('Error limit, loss', self.loss_percent)
        form4 = QFormLayout()
        form4.addRow('Form', self.form)
        form4.addRow('Line at each port', self.delay)
        form5 = QVBoxLayout()
        for w in (self.coupling, self.refine, self.sequential):
            form5.addWidget(w)
        form5.addStretch(1)
        self.advanced = QGroupBox()
        row_opt = QHBoxLayout(self.advanced)
        for form in (form2, form3, form4, form5):
            row_opt.addLayout(form)
            row_opt.addSpacing(12)
        row_opt.addStretch(1)
        self.advanced.setVisible(False)

        # --- run ---
        self.run_button = QPushButton('Extract model')
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.save_button = QPushButton('Save netlist and results...')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.status = QLabel('Choose a Touchstone file.')
        self.status.setWordWrap(True)
        self.verdict = QLabel('')                # one word on a colored patch
        self.verdict.setVisible(False)
        self.suggest_row = QHBoxLayout()         # buttons that apply a suggestion
        self.suggest_row.addWidget(self.verdict)
        self.suggest_row.addStretch(1)
        self.suggest_buttons = []
        self.data_check = None
        self.assessment = None
        self.copy_button = QPushButton('Copy netlist')
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_netlist)
        row_run = QHBoxLayout()
        row_run.addWidget(self.run_button)
        row_run.addWidget(self.cancel_button)
        row_run.addWidget(self.bar, 1)
        row_run.addWidget(QLabel('Netlist:'))
        row_run.addWidget(self.format)
        row_run.addWidget(self.copy_button)
        row_run.addWidget(self.save_button)

        # --- results ---
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Branch', 'Network', 'S error', 'X error',
                                              'Loss error', 'Element values'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.netlist_view = QPlainTextEdit(readOnly=True)
        mono = QFont('Consolas')
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.netlist_view.setFont(mono)
        self.netlist_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.currentCellChanged.connect(lambda *_: self._show_schematic())
        self.schematic = Figure(figsize=(6, 2.4))
        self.schematic_canvas = FigureCanvasQTAgg(self.schematic)
        self.schematic_canvas.setMinimumHeight(170)
        self.schematic_canvas.setToolTip('Schematic of the selected row')
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.table)
        left.addWidget(self.schematic_canvas)
        left.addWidget(self.netlist_view)
        left.setSizes([190, 230, 200])
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.view = QComboBox()
        self.view.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.view.setToolTip('S-parameters; the error of each S-parameter against frequency; '
                             'or series L (or C), series R and Q of a 2-port')
        self.view.currentIndexChanged.connect(lambda *_: self._show_plot())
        self.all_data = QCheckBox('All data')
        self.all_data.setToolTip('Show all frequencies of the data, not only the fit band')
        self.all_data.toggled.connect(lambda *_: self._show_plot())
        row_plot = QHBoxLayout()
        row_plot.addWidget(QLabel('Plot:'))
        row_plot.addWidget(self.view)
        row_plot.addWidget(self.all_data)
        row_plot.addStretch(1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(row_plot)
        right_layout.addWidget(self.canvas, 1)
        self.readout = QPlainTextEdit(readOnly=True)
        self.readout.setFont(mono)
        self.readout.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.readout.setFixedHeight(96)
        self.readout.setPlaceholderText('Click in the plot for a marker; left / right arrow keys '
                                        'move it by one frequency point.')
        right_layout.addWidget(self.readout)
        self.marker_f = None                     # marker frequency (GHz), kept across redraws
        self.marker_lines = []
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.canvas.mpl_connect('key_press_event', self._on_key)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([620, 630])

        layout = QVBoxLayout()
        layout.addLayout(row_file)
        layout.addLayout(row_simple)
        layout.addWidget(self.advanced)
        layout.addLayout(row_run)
        layout.addLayout(self.suggest_row)
        layout.addWidget(self.status)
        for w in (self.target, self.reactance_percent, self.loss_percent, self.max_elements):
            w.valueChanged.connect(self._reselect)
        self.pick.currentIndexChanged.connect(self._reselect)
        for w in (self.target, self.reactance_percent, self.loss_percent, self.max_elements,
                  self.delay):
            w.valueChanged.connect(self._settings_changed)
        for w in (self.pick, self.require, self.form):
            w.currentIndexChanged.connect(self._settings_changed)
        for w in (self.coupling, self.refine, self.sequential):
            w.toggled.connect(self._settings_changed)
        self.reset_button = QPushButton('Defaults')
        self.reset_button.setToolTip('All advanced settings back to their defaults')
        self.reset_button.clicked.connect(self.reset_advanced)
        form5.addWidget(self.reset_button)
        layout.addWidget(split, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # --- file ---------------------------------------------------------------------------

    def browse(self):
        start = str(pathlib.Path(self.file_edit.text()).parent) if self.file_edit.text() else ''
        path, _ = QFileDialog.getOpenFileName(self, 'Touchstone file', start,
                                              'Touchstone (*.s*p *.y*p *.z*p);;All files (*)')
        if path:
            self.set_file(path)

    def _load_from_edit(self):
        if self.file_edit.text() and (self.network is None
                                      or self.file_edit.text() != getattr(self, '_loaded', '')):
            self.set_file(self.file_edit.text())

    def set_file(self, path):
        """Load a Touchstone file; returns True if it worked."""
        path = str(path)
        self.file_edit.setText(path)
        try:
            self.network = data.read(path)
        except Exception as ex:
            self.network = None
            self._loaded = ''
            self.status.setText(f'Cannot read {path}: {ex}')
            return False
        new_file = path != getattr(self, '_loaded', '')
        self._loaded = path
        self.model = None
        self.save_button.setEnabled(False)
        self.clear_results()
        if new_file:                                 # the band of another file means nothing here
            self.reset_band()
        self.look_at_data()
        return True

    # --- data check, verdict, suggestions ---------------------------------------------------

    def look_at_data(self):
        """The data check for the band in the fields: verdict and a band suggestion."""
        try:
            c = check.check(self.network, self.fmin.value() * 1e9 or None,
                            self.fmax.value() * 1e9 or None, self.target.value())
        except ValueError as ex:
            self.data_check = None
            self.status.setText(str(ex))
            self._show_verdict('', [])
            return
        self.data_check = c
        self.status.setText(pathlib.Path(self._loaded).name + ': ' + '\n'.join(c.lines()))
        suggestions = []
        if c.fmax:
            suggestions.append(advice.Suggestion(f'Fit to {c.fmax / 1e9:.3g} GHz',
                                                 'above that the part is distributed', fmax=c.fmax))
        self._show_verdict({'suited': 'good', 'limited': 'fair', 'not suited': 'poor'}[c.verdict],
                           suggestions, c.verdict)

    def _show_verdict(self, level, suggestions, word=None):
        colors = {'good': '#2e7d32', 'fair': '#b26a00', 'poor': '#b71c1c'}
        self.verdict.setVisible(bool(level))
        if level:                                # text and background both set: dark mode safe
            self.verdict.setText(word or level)
            self.verdict.setStyleSheet(f'color: white; background: {colors[level]}; '
                                       f'padding: 2px 8px; border-radius: 3px; font-weight: bold;')
        for b in self.suggest_buttons:
            self.suggest_row.removeWidget(b)
            b.deleteLater()
        self.suggest_buttons = []
        for s in suggestions:
            b = QPushButton(s.label)
            b.setToolTip(s.why)
            b.clicked.connect(lambda _=False, s=s: self.apply_suggestion(s))
            self.suggest_row.insertWidget(self.suggest_row.count() - 1, b)
            self.suggest_buttons.append(b)

    def apply_suggestion(self, suggestion):
        """Set the fields a suggestion names and extract again."""
        for key, value in suggestion.changes.items():
            if key == 'fmax':
                self.fmax.setValue(value / 1e9)
            else:
                combo = {'require': self.require, 'pick': self.pick, 'form': self.form}[key]
                combo.setCurrentIndex(combo.findData(value))
        self.model = None
        self.run()

    def copy_netlist(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.netlist_view.toPlainText())
        self.copy_button.setText('Copied')

    # --- model quality, advanced settings ----------------------------------------------------

    def show_advanced(self, on):
        self.advanced.setVisible(on)
        self.advanced_button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        if self.advanced_button.isChecked() != on:
            self.advanced_button.setChecked(on)

    def _quality_chosen(self, *_):
        """A preset sets the limits and the choice; one new pick from the candidates."""
        values = model.QUALITY.get(self.quality.currentData())
        if values is None:                       # Custom: the fields are in the advanced part
            self.show_advanced(True)
            return
        widgets = (self.target, self.reactance_percent, self.loss_percent, self.pick)
        for w in widgets:
            w.blockSignals(True)
        self.target.setValue(values['target_db'])
        self.reactance_percent.setValue(values['reactance_percent'])
        self.loss_percent.setValue(values['loss_percent'])
        self.pick.setCurrentIndex(self.pick.findData(values['pick']))
        for w in widgets:
            w.blockSignals(False)
        self._reselect()

    def _settings_changed(self, *_):
        """Keep the quality field and the note on hidden settings in step with the fields."""
        opt = self.options()
        self.quality.blockSignals(True)
        self.quality.setCurrentIndex(self.quality.findData(model.quality_of(opt)))
        self.quality.blockSignals(False)
        d = model.Options()
        notes = []
        if opt.require != d.require:
            notes.append(REQUIRE_TEXT[opt.require])
        if opt.max_elements != d.max_elements:
            notes.append(f'up to {opt.max_elements} elements per branch')
        if opt.form != d.form:
            notes.append(f'{opt.form} form')
        if opt.delay != d.delay:
            notes.append(f'line {opt.delay * 1e12:.4g} ps at each port')
        if opt.coupling != d.coupling:
            notes.append('no coupled coils')
        if opt.refine != d.refine:
            notes.append('values not refined')
        if opt.sequential != d.sequential:
            notes.append('sequential')
        self.advanced_note.setText('In use: ' + ', '.join(notes) if notes else '')

    def reset_advanced(self):
        """All advanced settings back to their defaults (the band stays)."""
        d = model.Options()
        self.require.setCurrentIndex(self.require.findData(d.require))
        self.max_elements.setValue(d.max_elements)
        self.form.setCurrentIndex(self.form.findData(d.form))
        self.delay.setValue(0)
        self.coupling.setChecked(d.coupling)
        self.refine.setChecked(d.refine)
        self.sequential.setChecked(d.sequential)
        self.quality.setCurrentIndex(self.quality.findData(model.quality_of(d)))

    def _reselect(self, *_):
        """Limits or choice changed with a model at hand: choose again from the candidates
        tried, without a new search."""
        if self.model is None or self.worker is not None:
            return
        self.model.reselect(self.options())
        self._fill_table()
        self._refresh()

    def reset_band(self):
        """Fit band back to all frequencies of the data."""
        self.fmin.setValue(0)
        self.fmax.setValue(0)

    def clear_results(self):
        """Empty table, schematic, netlist and plot (a new file or no model)."""
        self.table.setRowCount(0)
        self.netlist_view.clear()
        self.view.clear()
        self.copy_button.setEnabled(False)
        self.assessment = None
        self.readout.clear()
        self.marker_f = None
        self.marker_lines = []
        for fig, canvas in ((self.schematic, self.schematic_canvas), (self.figure, self.canvas)):
            fig.clear()
            canvas.draw_idle()

    # --- run ----------------------------------------------------------------------------

    def options(self):
        return model.Options(
            fmin=self.fmin.value() * 1e9 or None, fmax=self.fmax.value() * 1e9 or None,
            require=self.require.currentData(), pick=self.pick.currentData(),
            target_db=self.target.value(), reactance_percent=self.reactance_percent.value(),
            loss_percent=self.loss_percent.value(), refine=self.refine.isChecked(),
            max_elements=self.max_elements.value(),
            form=self.form.currentData(), coupling=self.coupling.isChecked(),
            sequential=self.sequential.isChecked(),
            delay=self.delay.value() * 1e-12 or 'off')

    def run(self):
        if self.worker is not None:
            return
        if self.network is None and not (self.file_edit.text() and self.set_file(self.file_edit.text())):
            self.status.setText('Choose a Touchstone file first.')
            return
        self.worker = ExtractWorker(self.network, self.options(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(lambda: self._finish('Cancelled.'))
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText('Searching the catalog...')
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def _on_progress(self, fraction, text):
        self.bar.setValue(int(1000 * fraction))
        self.status.setText(f'Searching the catalog: {text}')

    def _finish(self, text):
        if self.worker is not None:
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.bar.setValue(0)
        self.status.setText(text)

    def _on_failed(self, text):
        self._finish(f'Failed: {text}')

    def _on_done(self, m):
        m.source = pathlib.Path(self._loaded).name
        self.model = m
        self._finish('')
        self.save_button.setEnabled(True)
        self._fill_table()
        first = self.view.count() == 0
        self._refresh()
        if first and self.data_check is not None and self.data_check.kind in ('inductor', 'capacitor'):
            self.view.setCurrentIndex(self.view.findData('lrq'))     # what such a part is judged by

    # --- results ------------------------------------------------------------------------

    def _fill_table(self):
        m = self.model
        self.table.clearContents()
        rows = m.rows()
        self.table.setRowCount(len(rows))
        for row, r in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(r.name))
            if isinstance(r, model.Branch):
                combo = QComboBox()
                for c in r.candidates[:MAX_ALTERNATIVES]:
                    mark = '' if c.score <= r.limit else '  (above limit)'
                    combo.addItem(f'{c.label}, {c.n_elements} el., {c.error_db:.1f} dB, '
                                  f'X {c.reactance_text()}, loss {c.loss_text()}{mark}')
                combo.setToolTip('Other networks that were tried for this branch')
                combo.currentIndexChanged.connect(
                    lambda index, b=m.branches.index(r): self._choose(b, index))
                self.table.setCellWidget(row, 1, combo)
            else:
                self.table.setItem(row, 1, QTableWidgetItem(r.label))
            for col in (2, 3, 4, 5):
                self.table.setItem(row, col, QTableWidgetItem(''))
        self.table.resizeColumnToContents(0)
        self.table.setColumnWidth(1, 300)
        self.table.setCurrentCell(0, 0)

    def _choose(self, row, index):
        if self.model is not None and index >= 0:
            self.model.choose(row, index)
            self._refresh()

    def _refresh(self):
        m = self.model
        for row, r in enumerate(m.rows()):
            shown = r.chosen if isinstance(r, model.Branch) else r
            self.table.item(row, 2).setText(f'{shown.error_db:.1f} dB')
            branch = isinstance(r, model.Branch)
            self.table.item(row, 3).setText(shown.reactance_text() if branch else '-')
            self.table.item(row, 4).setText(shown.loss_text() if branch else '-')
            self.table.item(row, 5).setText(shown.values_text())
        warnings = m.warnings()
        text = (f'Model ({m.form} form): {m.n_elements()} elements, S-parameter error {m.error_db():.1f} dB rms '
                f'in the fit band {m.freq[0] / 1e9:.6g} to {m.freq[-1] / 1e9:.6g} GHz.')
        if m.delay:
            text += f' Line at each port: {m.delay * 1e12:.4g} ps.'
        if warnings:
            text += '\nWarning: ' + '\nWarning: '.join(warnings)
        self.assessment = a = advice.assess(m, self.network, self.data_check)
        self.status.setText(a.text + '\n' + text)
        self._show_verdict(a.level, a.suggestions)
        self.copy_button.setEnabled(True)
        self.copy_button.setText('Copy netlist')
        self._show_netlist()
        self._show_schematic()
        self._show_plot()

    def _show_plot(self):
        if self.model is None:
            return
        if self.view.count() != len(views_for(self.model)):   # views depend on the port count
            self.view.blockSignals(True)
            self.view.clear()
            for key in views_for(self.model):
                self.view.addItem(VIEWS[key], key)
            self.view.blockSignals(False)
        compare(self.model, self.network, self.figure, self.view.currentData(),
                self.all_data.isChecked())
        self.marker_lines = []
        if self.marker_f is not None:
            self.set_marker(self.marker_f)
        else:
            self.readout.clear()
        self.canvas.draw_idle()

    # --- marker -------------------------------------------------------------------------

    def set_marker(self, f_ghz):
        """Put the marker at the data point nearest to f_ghz; returns the readout text."""
        readout = getattr(self.figure, 'crcat_readout', None)
        if readout is None or not readout.f.size:
            return ''
        i = readout.index(f_ghz)
        self._marker_index = i
        self.marker_f = float(readout.f[i])
        if not self.marker_lines:
            self.marker_lines = [ax.axvline(self.marker_f, color='k', lw=0.9, ls=':', zorder=5)
                                 for ax in self.figure.axes]
        for line in self.marker_lines:
            line.set_xdata([self.marker_f, self.marker_f])
        text = readout.text(i)
        self.readout.setPlainText(text)
        self.canvas.draw_idle()
        return text

    def step_marker(self, steps):
        readout = getattr(self.figure, 'crcat_readout', None)
        if readout is None or self.marker_f is None:
            return
        i = min(max(self._marker_index + steps, 0), readout.f.size - 1)
        self.set_marker(readout.f[i])

    def _on_click(self, event):
        if event.inaxes is None or event.button != 1 or event.xdata is None:
            return
        points = getattr(self.figure, 'crcat_tradeoff', None)
        if points:
            self.use_size(min(points, key=lambda p: abs(p.n_elements - event.xdata)))
        else:
            self.set_marker(event.xdata)

    def use_size(self, point):
        """Use the model of a point of the size-against-accuracy view."""
        self.model.apply(point)
        self._fill_table()
        self._refresh()

    def _on_drag(self, event):
        if getattr(self.figure, 'crcat_tradeoff', None):
            return
        if event.inaxes is not None and event.button == 1 and event.xdata is not None:
            self.set_marker(event.xdata)

    def _on_key(self, event):
        if event.key in ('left', 'right'):
            self.step_marker(-1 if event.key == 'left' else 1)

    def schematic_row(self):
        """The result row whose schematic is shown (the selected one), or None."""
        if self.model is None:
            return None
        rows = self.model.rows()
        k = self.table.currentRow()
        return rows[k] if 0 <= k < len(rows) else rows[0]

    def _show_schematic(self):
        row = self.schematic_row()
        self.schematic.clear()
        if row is not None:
            draw_row(self.schematic.subplots(), row)
            self.schematic.tight_layout()
        self.schematic_canvas.draw_idle()

    def subckt_name(self):
        return netlist.subckt_name(pathlib.Path(self._loaded).stem + '_model')

    def _show_netlist(self):
        if self.model is not None:
            self.netlist_view.setPlainText(
                netlist.render(self.model, self.format.currentData(), self.subckt_name()))

    # --- save ---------------------------------------------------------------------------

    def save(self):
        folder = QFileDialog.getExistingDirectory(self, 'Folder for the results',
                                                  str(pathlib.Path(self._loaded).parent))
        if folder:
            self.save_to(folder)

    def save_to(self, folder):
        """Write netlist, model response, JSON and plot into ``folder``."""
        files = write_outputs(self.model, self.network, folder, pathlib.Path(self._loaded).stem,
                              (self.format.currentData(),), self.subckt_name(),
                              all_data=self.all_data.isChecked())
        self.status.setText('Wrote ' + ', '.join(p.name for p in files) + f' in {folder}')
        return files

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait()
        super().closeEvent(event)
