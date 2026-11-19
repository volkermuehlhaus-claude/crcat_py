"""Main window: file and options at the top, branches and netlist left, plot right."""

import pathlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__, data, model, netlist
from ..cli_netlist import write_outputs
from ..plots import VIEWS, compare, views_for
from ..schematic import draw_row
from .worker import ExtractWorker

MAX_ALTERNATIVES = 25
REQUIRE_TEXT = {'positive': 'All positive', 'positive-r': 'R positive, L and C any sign',
                'passive': 'Passive, any sign', 'any': 'Any sign'}
PICK_TEXT = {'simplest': 'Simplest within the error limit', 'best': 'Lowest error'}


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
        self.branch_percent = QDoubleSpinBox(decimals=0, minimum=0, maximum=100, suffix=' %')
        self.branch_percent.setValue(defaults.branch_percent)
        self.branch_percent.setSpecialValueText('off')
        self.branch_percent.setToolTip(
            'Error limit of the branch itself: RMS of |Y model - Y data| / |Y data|. Keeps '
            'R and Q right for parts that are nearly a through or an open in the '
            'reference impedance. Off: S-parameters only.')
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
        self.format = QComboBox()
        for key in netlist.FORMATS:
            self.format.addItem(key, key)
        self.format.currentIndexChanged.connect(self._show_netlist)
        form1, form2, form3 = QFormLayout(), QFormLayout(), QFormLayout()
        form1.addRow('Fit band from', self.fmin)
        form1.addRow('to', self.fmax)
        form1.addRow('', self.full_range)
        form2.addRow('Element values', self.require)
        form2.addRow('Choice', self.pick)
        form3.addRow('Error limit, S-parameters', self.target)
        form3.addRow('Elements per branch, up to', self.max_elements)
        form4 = QFormLayout()
        form4.addRow('Error limit, branch', self.branch_percent)
        form4.addRow('Form', self.form)
        form4.addRow('', self.coupling)
        row_opt = QHBoxLayout()
        for form in (form1, form2, form3, form4):
            row_opt.addLayout(form)
        row_opt.addStretch(1)

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
        row_run = QHBoxLayout()
        row_run.addWidget(self.run_button)
        row_run.addWidget(self.cancel_button)
        row_run.addWidget(self.bar, 1)
        row_run.addWidget(QLabel('Netlist:'))
        row_run.addWidget(self.format)
        row_run.addWidget(self.save_button)

        # --- results ---
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Branch', 'Network', 'S error', 'Branch error',
                                              'Element values'])
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
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([620, 630])

        layout = QVBoxLayout()
        layout.addLayout(row_file)
        layout.addLayout(row_opt)
        layout.addLayout(row_run)
        layout.addWidget(self.status)
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
        f = self.network.f
        self.status.setText(f'{pathlib.Path(path).name}: {self.network.nports} port(s), '
                            f'{f[0] / 1e9:.6g} to {f[-1] / 1e9:.6g} GHz, {f.size} points.')
        return True

    def reset_band(self):
        """Fit band back to all frequencies of the data."""
        self.fmin.setValue(0)
        self.fmax.setValue(0)

    def clear_results(self):
        """Empty table, schematic, netlist and plot (a new file or no model)."""
        self.table.setRowCount(0)
        self.netlist_view.clear()
        for fig, canvas in ((self.schematic, self.schematic_canvas), (self.figure, self.canvas)):
            fig.clear()
            canvas.draw_idle()

    # --- run ----------------------------------------------------------------------------

    def options(self):
        return model.Options(
            fmin=self.fmin.value() * 1e9 or None, fmax=self.fmax.value() * 1e9 or None,
            require=self.require.currentData(), pick=self.pick.currentData(),
            target_db=self.target.value(), branch_percent=self.branch_percent.value(),
            max_elements=self.max_elements.value(),
            form=self.form.currentData(), coupling=self.coupling.isChecked())

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
        self._refresh()

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
                                  f'{c.relative_text()}{mark}')
                combo.setToolTip('Other networks that were tried for this branch')
                combo.currentIndexChanged.connect(
                    lambda index, b=m.branches.index(r): self._choose(b, index))
                self.table.setCellWidget(row, 1, combo)
            else:
                self.table.setItem(row, 1, QTableWidgetItem(r.label))
            for col in (2, 3, 4):
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
            self.table.item(row, 3).setText(
                shown.relative_text() if isinstance(r, model.Branch) else '-')
            self.table.item(row, 4).setText(shown.values_text())
        warnings = m.warnings()
        text = (f'Model ({m.form} form): {m.n_elements()} elements, S-parameter error {m.error_db():.1f} dB rms '
                f'in the fit band {m.freq[0] / 1e9:.6g} to {m.freq[-1] / 1e9:.6g} GHz.')
        if warnings:
            text += '\nWarning: ' + '\nWarning: '.join(warnings)
        self.status.setText(text)
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
        self.canvas.draw_idle()

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
