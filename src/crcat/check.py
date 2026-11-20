"""A look at the data before any fit: what kind of part it is, how large it is against the
wavelength, which form suits it, and whether a pi or tee of small RLC branches can model it
at all. Takes a fraction of a second; no catalog search, no GUI.

:func:`check` returns a :class:`DataCheck`; ``DataCheck.lines()`` is the report as text.
"""

import math

import numpy as np

from . import model as _model

__all__ = ['check', 'DataCheck', 'BranchFacts']

# Electrical length, from fits of an ideal and of a measured 50 ohm line: a lumped model
# reaches the -40 dB limit up to about 45 deg, -36 dB at 60 deg, -24 dB at 90 deg.
LUMPED_DEG = 45          # up to here a few lumped elements follow a line well
DISTRIBUTED_DEG = 90     # from here on the part counts as distributed
# A distributed 2-port gets a line at each port that takes all but this much of its
# electrical length; the lumped elements model the rest. From fits of a simulated 50 ohm
# line and of ideal 35, 50 and 70 ohm lines (216 to 230 deg): 15 to 45 deg left gave -36 to
# -45 dB, 60 deg -34 to -38 dB, nothing left -26 to -42 dB.
LINE_REST_DEG = 30
NEGATIVE_SHARE = 0.3     # a branch is unphysical if its real part is negative this often
K_SHOWN = 0.05           # smallest coupling factor reported


class BranchFacts:
    """A branch of the data: share of the band with a negative real part, number of
    resonances (sign changes of the susceptance), whether leaving it out stays within the
    S-parameter limit."""

    def __init__(self, name, negative, resonances, negligible):
        self.name, self.negative = name, negative
        self.resonances, self.negligible = resonances, negligible

    @property
    def unphysical(self):
        return not self.negligible and self.negative > NEGATIVE_SHARE


class DataCheck:
    """Result of :func:`check`. Main fields: ``kind`` ('inductor', 'capacitor', 'line',
    'two-port', 'one-port', 'multi-port'), ``verdict`` ('suited', 'limited', 'not suited'),
    ``reasons``, ``form`` (suggested), ``fmax`` (suggested top frequency in Hz, or None),
    ``srf`` (first self-resonance in Hz, or None), ``line`` (dict with L, C, z0, delay,
    degrees at the band end, or None), ``port_line`` (the line at each port that suits a
    distributed 2-port: dict with z0 and delay, or None), ``branches`` ({form: [BranchFacts]}), ``couplings``
    ([(ports, k)]), ``reciprocity`` and ``passive``."""

    def __init__(self):
        self.kind = self.verdict = self.form = None
        self.reasons, self.couplings, self.branches = [], [], {}
        self.fmax = self.srf = self.line = self.port_line = None
        self.nports = self.points = 0
        self.band = (0.0, 0.0)
        self.reciprocity = 0.0
        self.passive = True

    def unphysical(self, form=None):
        return [b for b in self.branches.get(form or self.form, []) if b.unphysical]

    def significant(self, form=None):
        return [b for b in self.branches.get(form or self.form, []) if not b.negligible]

    def lines(self):
        """The check as a few lines of text."""
        out = [f'{self.nports} port(s), {self.band[0] / 1e9:.6g} to {self.band[1] / 1e9:.6g} GHz, '
               f'{self.points} points: {self.kind}'
               + (f', self-resonance at {self.srf / 1e9:.4g} GHz' if self.srf else '')]
        if self.line:
            ln = self.line
            out.append(f'as a line: {ln["z0"]:.3g} ohm, {ln["delay"] * 1e12:.3g} ps, '
                       f'{ln["degrees"]:.0f} deg at the top of the band')
        out.append(f'{self.verdict}: ' + '; '.join(self.reasons))
        return out


def _resonances(y):
    b = y.imag
    scale = np.max(np.abs(b)) or 1.0
    sign = np.sign(np.where(np.abs(b) > 1e-6 * scale, b, 0))
    sign = sign[sign != 0]
    return int(np.sum(sign[1:] != sign[:-1]))


def _facts(branches, s_error, target):
    out = []
    for name, y, open_error in branches:
        negative = float(np.mean(y.real < -0.02 * np.abs(y)))
        out.append(BranchFacts(name, negative, _resonances(y), open_error <= target))
    return out


def check(network, fmin=None, fmax=None, target_db=-40.0):
    """Look at a scikit-rf Network (or Touchstone file name) in the band fmin..fmax (Hz)."""
    if not hasattr(network, 's'):
        from . import data
        network = data.read(network)
    f = network.f.astype(float)
    keep = f > 0
    if fmin is not None:
        keep &= f >= fmin
    if fmax is not None:
        keep &= f <= fmax
    if keep.sum() < 12:
        raise ValueError(f'only {int(keep.sum())} frequencies in the band, 12 needed')
    f = f[keep]
    w = 2 * np.pi * f
    n = network.nports
    z0 = float(np.real(network.z0[0, 0])) or 50.0
    s = network.s[keep]
    target = 10 ** (target_db / 20)
    eye = np.eye(n)
    c = DataCheck()
    c.nports, c.points, c.band = n, f.size, (float(f[0]), float(f[-1]))
    c.reciprocity = _model._rms(s - np.transpose(s, (0, 2, 1))) / 2
    c.passive = bool(np.max(np.linalg.svd(s, compute_uv=False)) <= 1 + 1e-3)
    with np.errstate(all='ignore'):
        try:
            y = np.linalg.solve(z0 * (eye + s), eye - s)
        except np.linalg.LinAlgError:
            sc = s * (1 - 1e-12)
            y = np.linalg.solve(z0 * (eye + sc), eye - sc)
    y = (y + np.transpose(y, (0, 2, 1))) / 2
    s_ref = _model._y2s(y, z0)

    # --- branches of the pi form, and of the tee form for a 2-port
    pi = []
    for ports, yb in _model.branch_admittances(y).items():
        yy = y.copy()
        _model._stamp(yy, ports, -yb)
        name = f'p{ports[0]} - ' + (f'p{ports[1]}' if len(ports) == 2 else 'gnd')
        pi.append((name, yb, _model._rms(_model._y2s(yy, z0) - s_ref)))
    c.branches['pi'] = _facts(pi, None, target)
    if n == 2:
        with np.errstate(all='ignore'):
            try:
                z = np.linalg.inv(y)
                legs = [1 / (z[:, 0, 0] - z[:, 0, 1]), 1 / (z[:, 1, 1] - z[:, 0, 1]), 1 / z[:, 0, 1]]
                if all(np.all(np.isfinite(v)) for v in legs):
                    tee = []
                    for k, name in enumerate(('p1 - c', 'p2 - c', 'c - gnd')):
                        v = list(legs)
                        v[k] = np.full_like(v[k], _model.SHORT_SIEMENS) if k < 2 else 0 * v[k]
                        e = _model._rms(_model._tee_s(1 / v[0], 1 / v[1], v[2], z0) - s_ref)
                        tee.append((name, legs[k], e))
                    c.branches['tee'] = _facts(tee, None, target)
            except np.linalg.LinAlgError:
                pass
    c.form = min(c.branches, key=lambda form: (len(c.unphysical(form)),
                                               len(c.significant(form)), form != 'pi'))

    # --- coupling between branches (pi form): the quick test of the extraction
    low = f <= f[0] + (f[-1] - f[0]) / 3
    yb = _model.branch_admittances(y)
    for (i, j) in [p for p in yb if len(p) == 2]:
        for k in [0] + [p for p in range(1, n + 1) if p not in (i, j)]:
            pij, pik, pjk = _model._pair_ports((i, j, k))
            with np.errstate(all='ignore'):
                det = (yb[pik] + yb[pij]) * (yb[pjk] + yb[pij]) - yb[pij] ** 2
                l1, l2, m = (float(np.mean(v.imag[low] / w[low])) for v in
                             ((yb[pjk] + yb[pij]) / det, (yb[pik] + yb[pij]) / det, yb[pij] / det))
            if l1 > 0 and l2 > 0 and math.isfinite(m) and K_SHOWN <= abs(m) / math.sqrt(l1 * l2) <= 1:
                c.couplings.append(((i, j, k), m / math.sqrt(l1 * l2)))

    # --- kind of part, self-resonance, size against the wavelength
    lowb = slice(0, max(3, f.size // 10))
    with np.errstate(all='ignore'):
        zin = 1 / y[:, 0, 0]
        if n == 2:
            zs = -1 / y[:, 0, 1]
            l_series = float(np.median(zs.imag[lowb] / w[lowb]))
            c_shunt = float(np.median((y[:, 0, 0] + y[:, 1, 1] + 2 * y[:, 0, 1]).imag[lowb] / w[lowb]))
        x = zin.imag
    inductive = float(np.median(x[lowb])) > 0
    flips = np.flatnonzero(np.sign(x[1:]) != np.sign(x[:-1]))
    if flips.size:
        c.srf = float(f[flips[0] + 1])
    if n == 2 and l_series > 0 and c_shunt > 0:
        delay = math.sqrt(l_series * c_shunt)
        c.line = {'L': l_series, 'C': c_shunt, 'z0': math.sqrt(l_series / c_shunt), 'delay': delay,
                  'degrees': math.degrees(w[-1] * delay)}
    if n > 2:
        c.kind = 'multi-port'
    elif n == 1:
        c.kind = 'one-port, ' + ('inductive' if inductive else 'capacitive')
    elif (l_series > 0 and c.line and c.line['degrees'] > DISTRIBUTED_DEG
            and 0.5 < c.line['z0'] / z0 < 2):
        c.kind = 'line'
    elif l_series > 0:
        c.kind = 'inductor'
    elif l_series < 0:
        c.kind = 'capacitor'
    else:
        c.kind = 'two-port'

    # --- verdict
    unphys, signif = c.unphysical(), c.significant()
    coupled = {}
    for (i, j, k), kf in c.couplings:            # coupling that can explain a branch i - j
        coupled.setdefault(f'p{i} - p{j}', kf)
    # a branch between ports that needs explaining: a negative real part, or a negative
    # inductance (susceptance positive and falling with frequency)
    odd = {b.name for b in unphys}
    for ports, v in yb.items():
        if len(ports) == 2 and np.median(v.imag[low]) > 0 and v.imag[low][-1] < 0.9 * np.max(v.imag[low]):
            odd.add(f'p{ports[0]} - p{ports[1]}')
    c.couplings = [(p, kf) for p, kf in c.couplings if f'p{p[0]} - p{p[1]}' in odd]
    left = [b for b in unphys if b.name not in coupled]
    degrees = c.line['degrees'] if c.line else 0.0
    if c.line and degrees > DISTRIBUTED_DEG:
        c.fmax = float(LUMPED_DEG / degrees * f[-1])
        c.port_line = {'z0': c.line['z0'],
                       'delay': c.line['delay'] * (1 - LINE_REST_DEG / degrees) / 2}
    if signif and len(unphys) > max(1, len(signif) // 3):
        c.verdict = 'not suited'
        c.reasons.append(f'{len(unphys)} of {len(signif)} branches of the data have a negative '
                         f'real part (in the {c.form} form), which no network of positive R, L, C '
                         f'gives: a structure of coupled or distributed conductors')
    elif degrees > DISTRIBUTED_DEG:
        c.verdict = 'limited'
        c.reasons.append(f'the part is {degrees:.0f} deg long at {f[-1] / 1e9:.4g} GHz, distributed: '
                         f'a few lumped elements follow it well up to about '
                         f'{c.fmax / 1e9:.3g} GHz ({LUMPED_DEG} deg); for the whole band a line of '
                         f'{c.port_line["z0"]:.3g} ohm and {c.port_line["delay"] * 1e12:.3g} ps at each '
                         f'port is tried')
    elif left:
        c.verdict = 'limited'
        c.reasons.append('branch ' + ', '.join(b.name for b in left) + ' of the data has a negative '
                         'real part; a narrower band or negative L and C may be needed')
    else:
        c.verdict = 'suited'
        c.reasons.append(f'{len(signif)} branch(es) to model in the {c.form} form')
        for name in sorted({f'p{p[0]} - p{p[1]}' for p, _ in c.couplings}):
            c.reasons.append(f'magnetic coupling at branch {name} (k about {coupled[name]:.2f}): '
                             f'coupled coils will be tried')
    many = [b.name for b in signif if b.resonances > 2]
    if many and c.verdict != 'not suited':
        c.reasons.append('branch ' + ', '.join(many) + ' has more than two resonances in the band: '
                         'five elements may not be enough')
    if not c.passive:
        c.reasons.append('the data are not passive')
    return c
