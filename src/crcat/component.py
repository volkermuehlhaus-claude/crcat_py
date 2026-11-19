"""Components: catalog networks with element values (port of CRComp.m).

Units: any consistent set, e.g. Hz, H, F, ohm or GHz, nH, nF, ohm. The admittance is
evaluated at s = j*2*pi*f.
"""

import mpmath
import numpy as np

from . import catalog as _catalog
from .fit import DEFAULT_DPS, crfit

__all__ = ['Component', 'format_value', 'use_ascii_units']

_PREFIXES = [(1e-12, 1e15, 'f'), (1e-9, 1e12, 'p'), (1e-6, 1e9, 'n'), (1e-3, 1e6, 'μ'),
             (1, 1e3, 'm'), (1e3, 1, ''), (1e6, 1e-3, 'k')]
_UNITS = {'R': 'Ω', 'L': 'H', 'C': 'F'}
_ASCII = {'μ': 'u', 'Ω': 'Ohm'}


def use_ascii_units(on=True):
    """Write micro and ohm as 'u' and 'Ohm' (for consoles without these characters)."""
    global _use_ascii
    _use_ascii = on


_use_ascii = False


def format_value(value, element='', digits=4):
    """Value with SI prefix and unit for an element name (R..., L..., C...), as CRCat
    prints SI element values: 3e-12 for 'Cd' -> '3pF'."""
    v = complex(value)
    if abs(v.imag) * 1e5 < abs(v):
        v = v.real
    unit = _UNITS.get(element[:1], '')
    if v == 0:
        return f'0{unit}'
    scale, prefix = 1e-6, 'M'
    for limit, sc, pre in _PREFIXES:
        if abs(v) < limit:
            scale, prefix = sc, pre
            break
    x = v * scale
    text = f'{x:.{digits}g}' if isinstance(x, float) else f'({x:.{digits}g})'
    if _use_ascii:
        prefix, unit = _ASCII.get(prefix, prefix), _ASCII.get(unit, unit)
    return f'{text}{prefix}{unit}'


def _to_number(x):
    """mpmath value -> float, or complex if the imaginary part matters."""
    c = complex(x)
    if abs(c.imag) <= 1e-12 * abs(c.real):
        return c.real
    return c


class Component:
    """A catalog network with one or more sets of element values (solutions).

    ``values`` is one list of element values in ``network.symVn`` order or a list of such
    lists; default: the catalog's test values.
    """

    def __init__(self, network, catalog=None, values=None):
        if not isinstance(network, _catalog.Network):
            network = (catalog or _catalog.load())[network]
        self.network = network
        if values is None:
            values = network.test_values
        self.set_values(values)

    def __repr__(self):
        vals = '; '.join(self.format(i) for i in range(len(self.values)))
        return f'<Component CR{self.network.id} {self.network.name}: {vals or "no values"}>'

    @property
    def name(self):
        return self.network.name

    def set_values(self, values):
        """Set element values: one list (symVn order) or a list of lists (solutions)."""
        values = [] if values is None else list(values)
        if values and not isinstance(values[0], (list, tuple, np.ndarray)):
            values = [values]
        n = len(self.network.symVn)
        for row in values:
            if len(row) != n:
                raise ValueError(f'{self.name} has {n} elements, {len(row)} values given')
        self.values = [[_to_number(v) for v in row] for row in values]

    # --- evaluation -----------------------------------------------------------------

    def eval(self, freq, solution=None):
        """Admittance at frequencies ``freq`` (complex numpy array).

        Shape (solutions, frequencies), or (frequencies,) for one ``solution`` index.
        """
        freq = np.asarray(freq, dtype=float).ravel()
        rows = self.values if solution is None else [self.values[solution]]
        out = np.empty((len(rows), freq.size), dtype=complex)
        s = 2j * np.pi * freq
        for i, row in enumerate(rows):
            a, b = self.network.polynomials(row)
            a = np.array([complex(x) for x in a])
            b = np.array([complex(x) for x in b])
            with np.errstate(all='ignore'):
                out[i] = np.polyval(a, s) / np.polyval(b, s)
        return out[0] if solution is not None else out

    # --- fitting --------------------------------------------------------------------

    def fit(self, y, freq, fixed=None, dps=DEFAULT_DPS, coefficients=None):
        """Fit to admittance ``y`` at ``freq``; return the RMS error of each solution.

        Port of CRComp.Fit: CRFit with the network's coefficient signature, then the
        synthesis equations. ``fixed`` gives values for the elements without synthesis
        equations (default: the current values). All solutions are kept in ``values``
        (sorted by error); complex element values are allowed, as in CRCat. Returns
        None (and clears ``values``) if the network cannot be fitted.
        ``coefficients`` may pass a FitResult for this signature to skip the fit.
        """
        net = self.network
        y = np.asarray(y, dtype=complex).ravel()
        freq = np.asarray(freq, dtype=float).ravel()
        if not net.can_synthesize:
            self.values = []
            return None
        if net.fixed:
            if fixed is None:
                if not self.values:
                    raise ValueError(f'{self.name}: values needed for {", ".join(net.fixed)}')
                fixed = [self.values[0][net.symVn.index(v)] for v in net.fixed]
            fixed = list(fixed)
        else:
            if fixed:
                raise ValueError(f'{self.name} has no fixed elements')
            fixed = []
        r = coefficients if coefficients is not None else \
            crfit(y, freq, net.flag_a, net.flag_b, dps)
        if not r.ok:
            self.values = []
            return None
        coef = [c for c, f in zip(r.a, net.flag_a) if f] + \
               [c for c, f in zip(r.b, net.flag_b) if f]
        with mpmath.workdps(dps):
            rows = net.synthesize(coef, fixed)
        if not rows:
            self.values = []
            return None
        self.values = [[_to_number(v) for v in row] for row in rows]
        err = np.array([np.sqrt(np.mean(np.abs(y - yf) ** 2)) for yf in self.eval(freq)])
        good = np.isfinite(err)
        order = np.argsort(np.where(good, err, np.inf))
        self.values = [self.values[i] for i in order if good[i]]
        err = err[order][good[order]]
        if not self.values:
            return None
        return err

    # --- transforms -----------------------------------------------------------------

    def transform(self, to, catalog=None):
        """Equivalent component of network ``to`` (Id or name), from the first solution.

        Returns None if there are no transform equations (CRComp.Transform).
        """
        cat = catalog or self.network.catalog
        rows = self.network.transform(to, self.values[0])
        if not rows:
            return None
        return Component(cat[to], values=[[_to_number(v) for v in r] for r in rows])

    # --- passivity ------------------------------------------------------------------

    def gain_frequencies(self, solution=0):
        """Radian frequencies where Re Y changes sign (port of CRComp.IsGain).

        Empty if the real part of the admittance is >= 0 at all frequencies (passive);
        [0.0] if it is negative everywhere. Where the real part only touches zero, it does
        not count. Decided by the sign of the real part between its zeros, not by the
        derivative test of CRCat (the polynomial spans many decades in SI units).
        """
        with mpmath.workdps(50):
            a, b = self.network.polynomials(self.values[solution])
            # Re{A(jw) * B(-jw)}: same sign as Re Y(jw); polynomial in w, highest power first
            p = [mpmath.re(c) for c in _polymul(_jw(a, 1), _jw(b, -1))]
            return _sign_changes(p)

    def is_passive(self, solution=0):
        return not self.gain_frequencies(solution)

    # --- output ---------------------------------------------------------------------

    def format(self, solution=0, digits=4):
        """Element values as text with SI prefixes ('La=500pH Lb=400pH ...')."""
        if not self.values:
            return 'no element values'
        return ' '.join(f'{v}={format_value(x, v, digits)}'
                        for v, x in zip(self.network.symVn, self.values[solution]))


def _sign_changes(p):
    """Positive w where the real polynomial p(w) changes sign; [0.0] if negative for all
    w > 0; [] if p(w) >= 0 for all w > 0."""
    p = list(p)
    if all(c == 0 for c in p):
        return []                                  # lossless: Re Y = 0
    # Only exact zeros are dropped: coefficients of different powers of w cannot be
    # compared by size (they span many decades in SI units).
    while p[0] == 0:
        p = p[1:]
    while p[-1] == 0:                              # factors of w
        p = p[:-1]
    if len(p) == 1:
        return [] if p[0] > 0 else [0.0]
    # w = w0 * u with w0 chosen so the coefficients are of similar size
    deg = len(p) - 1
    w0 = (abs(p[-1]) / abs(p[0])) ** (mpmath.mpf(1) / deg)
    q = [c * w0 ** (deg - k) for k, c in enumerate(p)]
    qmax = max(abs(c) for c in q)
    q = [c / qmax for c in q]
    try:
        roots = mpmath.polyroots(q, maxsteps=500, extraprec=300)
    except mpmath.libmp.NoConvergence as ex:
        roots = ex.args[0] if ex.args and isinstance(ex.args[0], list) else             [mpmath.mpc(r) for r in np.roots([float(c) for c in q])]
    cand = []
    for r in sorted(mpmath.re(r) for r in roots
                    if mpmath.re(r) > 0 and abs(mpmath.im(r)) <= 1e-6 * abs(r)):
        if not cand or r - cand[-1] > 1e-12 * r:    # a multiple root splits numerically
            cand.append(r)
    # test points below, between and above the candidate zeros
    tests = [cand[0] / 2] if cand else [mpmath.mpf(1)]
    tests += [mpmath.sqrt(x * y) for x, y in zip(cand, cand[1:])]
    if cand:
        tests.append(cand[-1] * 2)
    signs = [mpmath.sign(mpmath.polyval(q, u)) for u in tests]
    if all(sg >= 0 for sg in signs):
        return []
    if all(sg <= 0 for sg in signs):
        return [0.0]
    return [float(cand[i] * w0) for i in range(len(cand)) if signs[i] * signs[i + 1] < 0]


def _jw(coef, sign):
    """Coefficients of P(sign*j*w) in w, for P given in s (highest power first)."""
    n = len(coef) - 1
    return [c * (sign * 1j) ** (n - k) for k, c in enumerate(coef)]


def _polymul(p, q):
    out = [mpmath.mpc(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out
