"""A plain text field for a frequency in GHz: type a number, empty means not set."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLineEdit

ACCEPTED = 'A number in GHz, for example 28 or 0.5. Empty: not set.'


def parse_ghz(text):
    """GHz from what was typed: '28', '28,5', '30 GHz', '30G'. '' gives 0 (not set); None
    if it is not a number or negative."""
    t = text.strip().lower().replace(',', '.')
    for unit in ('ghz', 'g'):
        if t.endswith(unit):
            t = t[:-len(unit)].strip()
            break
    if not t:
        return 0.0
    try:
        value = float(t)
    except ValueError:
        return None
    return value if value >= 0 and value == value and value != float('inf') else None


class FrequencyEdit(QLineEdit):
    """Text field with the small interface of a spin box: ``value()`` (GHz, 0 when empty),
    ``setValue()`` and ``valueChanged``. The signal comes when the field is left or Enter
    is pressed, and at once when the field is emptied, not on every keystroke."""

    valueChanged = Signal(float)

    def __init__(self, placeholder='', tooltip='', parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._tooltip = tooltip
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setMaximumWidth(110)
        self.setToolTip(self._tip())
        self.editingFinished.connect(self._commit)
        self.textChanged.connect(self._on_text)

    def _tip(self, problem=''):
        return '\n'.join(t for t in (problem, self._tooltip, ACCEPTED) if t)

    def describe(self, tooltip):
        """Set the explanation shown as tooltip (above the note on what can be typed)."""
        self._tooltip = tooltip
        self.setToolTip(self._tip())

    def value(self):
        return self._value

    def setValue(self, ghz):
        self._show(float(ghz))
        self._set(float(ghz))

    def _show(self, ghz):
        text = f'{ghz:g}' if ghz else ''
        if text != self.text():
            self.blockSignals(True)
            self.setText(text)
            self.blockSignals(False)

    def _set(self, ghz):
        if ghz != self._value:
            self._value = ghz
            self.valueChanged.emit(ghz)

    def _on_text(self, text):
        if text == '':                           # emptied (clear button): not set, at once
            self._set(0.0)

    def _commit(self):
        ghz = parse_ghz(self.text())
        if ghz is None:                          # not a frequency: back to the last value
            self.setToolTip(self._tip(f'"{self.text()}" is not a frequency.'))
            self._show(self._value)
            return
        self.setToolTip(self._tip())
        self._show(ghz)
        self._set(ghz)
