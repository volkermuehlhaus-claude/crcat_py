"""A verdict on an extracted model in plain words, with suggestions that can be applied.

:func:`assess` returns an :class:`Assessment`: ``level`` ('good', 'fair', 'poor'), ``text``
(one line), ``details`` (lines) and ``suggestions``, each a :class:`Suggestion` with a
``label`` for a button and ``changes``, a dict of Options fields to set before extracting
again. No GUI here.
"""

import math

import numpy as np

from .plots import series_quantities

__all__ = ['assess', 'Assessment', 'Suggestion']

GOOD_DB = 3          # 'good': model error within this of the S-parameter limit
FAIR_DB = 12         # 'fair': within this
Q_GOOD = 0.2         # median relative error of Q for 'good' (2-port inductors, capacitors)
Q_FAIR = 0.5


class Suggestion:
    def __init__(self, label, why, **changes):
        self.label, self.why, self.changes = label, why, changes

    def __repr__(self):
        return f'<{self.label}: {self.changes}>'


class Assessment:
    def __init__(self, level, text, details, suggestions, quantities, target=''):
        self.level, self.text, self.details = level, text, details
        self.suggestions, self.quantities = suggestions, quantities
        self.target = target                     # the model at the target frequency

    def lines(self):
        return [f'{self.level}: {self.text}'] + ([self.target] if self.target else []) + \
            self.details + \
            [f'suggestion: {s.label} ({s.why})' for s in self.suggestions]


def _series_errors(model, network):
    """Median relative errors of the series L or C, series R and Q of a 2-port in the fit
    band: (name of the reactive quantity, its error, R error, Q error)."""
    keep = (network.f >= model.freq[0]) & (network.f <= model.freq[-1]) & (network.f > 0)
    f = network.f[keep]
    w = 2 * np.pi * f
    (zs_d, zin_d), (zs_m, zin_m) = (series_quantities(s, model.z0)
                                    for s in (network.s[keep], model.s_matrix(f)))
    with np.errstate(all='ignore'):
        cap = np.median(zs_d.imag) < 0
        xd, xm = (-1 / (w * zs_d.imag), -1 / (w * zs_m.imag)) if cap else \
            (zs_d.imag / w, zs_m.imag / w)
        qd, qm = np.abs(zin_d.imag) / zin_d.real, np.abs(zin_m.imag) / zin_m.real

        def med(a, b):
            r = np.abs(a - b) / np.abs(b)
            r = r[np.isfinite(r)]
            return float(np.median(r)) if r.size else math.nan
        return ('C' if cap else 'L'), med(xm, xd), med(zs_m.real, zs_d.real), med(qm, qd)


def at_frequency(model, network, freq):
    """The model against the data at the data point nearest to ``freq`` (Hz): a line of
    text and the numbers. For a 2-port the series L or C and Q of model and data, for any
    part the S-parameter error near that frequency."""
    from .component import format_value
    f_all = network.f
    i = int(np.argmin(np.abs(f_all - freq)))
    f = f_all[i:i + 1]
    near = 20 * math.log10(max(model.error_near(freq), 1e-30))
    out = {'frequency': float(f[0]), 'error_db': near}
    text = f'at {f[0] / 1e9:.4g} GHz: '
    if model.nports == 2:
        w = 2 * np.pi * f[0]
        (zs_d, zin_d), (zs_m, zin_m) = (series_quantities(s, model.z0)
                                        for s in (network.s[i:i + 1], model.s_matrix(f)))
        with np.errstate(all='ignore'):
            cap = zs_d.imag[0] < 0
            xd, xm = ((-1 / (w * zs_d.imag[0]), -1 / (w * zs_m.imag[0])) if cap
                      else (zs_d.imag[0] / w, zs_m.imag[0] / w))
            qd, qm = (abs(z.imag[0]) / z.real[0] for z in (zin_d, zin_m))
        name = 'C' if cap else 'L'
        out.update({name: (float(xm), float(xd)), 'Q': (float(qm), float(qd))})
        if all(math.isfinite(v) for v in (xm, xd, qm, qd)):
            text += (f'{name} {format_value(xm, name)} (data {format_value(xd, name)}), '
                     f'Q {qm:.3g} (data {qd:.3g}), ')
    return text + f'S-parameters within {near:.1f} dB there', out


def _good_up_to(model):
    """Highest frequency up to which the RMS error of the model stays within the
    S-parameter limit (Hz), or None if it never does."""
    d = np.sqrt(np.mean(np.abs(model.s_matrix(model.freq) - model.s_data) ** 2, axis=(1, 2)))
    running = np.sqrt(np.cumsum(d ** 2) / np.arange(1, d.size + 1))
    ok = np.flatnonzero(running <= model.options.target)
    return float(model.freq[ok[-1]]) if ok.size else None


def assess(model, network, check=None):
    """Judge a model against its data. ``check``: the DataCheck of the data, if at hand
    (adds what it knows about lines and unphysical branches to the suggestions)."""
    opt = model.options
    err = model.error_db()
    warnings = model.warnings()
    level = 'good' if err <= opt.target_db + GOOD_DB and not warnings else \
        'fair' if err <= opt.target_db + FAIR_DB else 'poor'
    band = f'{model.freq[0] / 1e9:.4g} to {model.freq[-1] / 1e9:.4g} GHz'
    parts = [f'{model.n_elements()} elements, S-parameters within {err:.1f} dB']
    quantities = {'error_db': err}
    details = []
    if model.nports == 2:
        name, ex, er, eq = _series_errors(model, network)
        quantities.update({name: ex, 'R': er, 'Q': eq})
        if math.isfinite(ex) and math.isfinite(eq):
            parts.append(f'{name} within {100 * ex:.2g} %, Q within {100 * eq:.2g} % (median)')
            if eq > Q_FAIR:
                level = 'poor' if level == 'poor' else 'fair'
                details.append(f'Q is off by {100 * eq:.0f} % (median): the loss is not well modelled')
            elif eq > Q_GOOD and level == 'good':
                level = 'fair'
    text = f'{band}: ' + ', '.join(parts)
    details += warnings

    suggestions = []
    if level != 'good':
        fok = _good_up_to(model)
        span = model.freq[-1] - model.freq[0]
        if fok and model.freq[0] + 0.25 * span < fok < model.freq[0] + 0.9 * span:
            suggestions.append(Suggestion(f'Fit to {fok / 1e9:.3g} GHz',
                                          'the model follows the data up to there', fmax=fok))
        elif check is not None and check.fmax and check.fmax < 0.9 * model.freq[-1]:
            suggestions.append(Suggestion(f'Fit to {check.fmax / 1e9:.3g} GHz',
                                          'above that the part is distributed', fmax=check.fmax))
        missed = [b for b in model.branches if b.warning()]
        if opt.require == 'positive' and any(b.negative_real_part > 0.1 or b.has_negative_element()
                                             for b in missed) or \
                (opt.require == 'positive' and model.nports > 1 and missed and not suggestions):
            suggestions.append(Suggestion('Allow negative L and C',
                                          'a branch needs them; resistors stay positive',
                                          require='positive-r'))
        if opt.pick == 'simplest' and not missed:
            suggestions.append(Suggestion('Use more elements', 'lowest error, not fewest elements',
                                          pick='best'))
        if model.nports == 2 and opt.form == 'auto' and missed and model.form == 'pi' and \
                check is not None and check.form == 'tee':
            suggestions.append(Suggestion('Try the tee form', 'fewer unphysical branches in the data',
                                          form='tee'))
    target = ''
    if opt.ftarget:
        target, numbers = at_frequency(model, network, opt.ftarget)
        quantities['target'] = numbers
    return Assessment(level, text, details, suggestions[:3], quantities, target)
