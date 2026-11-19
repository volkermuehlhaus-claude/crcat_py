"""From network data to a lumped model: one catalog network per branch of a pi network.

A reciprocal n-port is exactly a set of branches: one from each port to ground (admittance
= sum of a column of Y) and one between each pair of ports (admittance = -Ymn). Each branch
is a one-port admittance, which the catalog search can model with up to five R, L, C.

For every branch the candidates are judged by what they do to the S-parameters: the branch
admittance in the measured Y-matrix is replaced by the candidate's, and the RMS difference
of all S-parameters is the candidate's error. The default choice is the network with the
fewest elements, all positive, whose error is within the limit; "open" (no elements) is a
candidate too.

No printing, no plotting, no GUI here: :func:`extract` returns a :class:`Model`, and takes
``progress`` and ``cancel`` callbacks.
"""

import dataclasses
import math

import numpy as np

from . import catalog as _catalog
from .component import Component
from .fit import DEFAULT_DPS, crfit, evaluate
from .seed import Cancelled

__all__ = ['Options', 'Model', 'Branch', 'Candidate', 'extract', 'branch_admittances',
           'Cancelled']

REQUIRE = ('positive', 'positive-r', 'passive', 'any')
PICK = ('simplest', 'best')
EQUAL_ERROR = 1e-10      # RMS |S difference| below which two candidates count as equal


@dataclasses.dataclass
class Options:
    """Settings of :func:`extract`.

    fmin, fmax: fit band in Hz (default: all data; a 0 Hz point is never used).
    require: 'positive' (all element values > 0), 'positive-r' (resistors > 0, L and C of
        either sign: still passive, since only resistors dissipate), 'passive' (Re Y >= 0,
        any signs) or 'any' (any real element values).
    pick: 'simplest' (fewest elements within the error limit) or 'best' (lowest error;
        the fewest elements among practically equal errors).
    target_db: error limit for 'simplest', RMS |S difference| in dB (-40 dB = 0.01).
    tolerance: if no candidate reaches target_db, the limit is tolerance x the lowest error.
    max_elements: largest number of elements per branch (1..5).
    full_search: synthesize every network instead of only those that can be chosen
        (slower, same result; fills the list of alternatives completely).
    dps: working precision of the fits, decimal digits.
    """
    fmin: float = None
    fmax: float = None
    require: str = 'positive'
    pick: str = 'simplest'
    target_db: float = -40.0
    tolerance: float = 1.25
    max_elements: int = 5
    full_search: bool = False
    dps: int = DEFAULT_DPS

    def check(self):
        if self.require not in REQUIRE:
            raise ValueError(f'require must be one of {REQUIRE}')
        if self.pick not in PICK:
            raise ValueError(f'pick must be one of {PICK}')
        if self.tolerance < 1:
            raise ValueError('tolerance must be >= 1')
        if not 1 <= self.max_elements <= 5:
            raise ValueError('max_elements must be 1..5')
        if self.target_db >= 0:
            raise ValueError('target_db must be negative')


class Candidate:
    """A network with element values for one branch. ``error``: RMS |S difference| when
    this candidate replaces the branch in the data; ``error_y``: RMS error of the branch
    admittance (S)."""

    def __init__(self, component, error, error_y):
        self.component = component
        self.error = float(error)
        self.error_y = float(error_y)

    @property
    def n_elements(self):
        return len(self.component.network.symVn) if self.component is not None else 0

    @property
    def is_open(self):
        return self.component is None

    @property
    def label(self):
        if self.is_open:
            return 'open'
        net = self.component.network
        return f'CR{net.id} {net.name}'

    @property
    def error_db(self):
        return 20 * math.log10(max(self.error, 1e-300))

    def values_text(self):
        return 'no elements' if self.is_open else self.component.format()

    def __repr__(self):
        return f'<{self.label}: {self.error_db:.1f} dB, {self.values_text()}>'


class Branch:
    """One branch of the pi network between ``node1`` and ``node2`` ('p1'.., '0')."""

    def __init__(self, ports, freq, y):
        self.ports = ports                       # (i,) shunt at port i; (i, j) between ports
        self.node1 = f'p{ports[0]}'
        self.node2 = f'p{ports[1]}' if len(ports) == 2 else '0'
        self.tag = 'p' + 'p'.join(str(p) for p in ports)
        self.freq, self.y = freq, y
        self.candidates = []                     # acceptable ones, best choice first
        self.chosen = None
        self.limit = None                        # error limit used for the choice
        self.target = None                       # target error (RMS |S difference|)

    @property
    def name(self):
        return f'{self.node1} - {"gnd" if self.node2 == "0" else self.node2}'

    @property
    def target_missed(self):
        """True if the chosen candidate does not reach the target error."""
        return self.chosen is not None and self.chosen.error > self.target

    @property
    def negative_real_part(self):
        """Share (0..1) of the fit band where the branch admittance of the data has a
        clearly negative real part: there, no network of positive elements can fit it."""
        return float(np.mean(self.y.real < -0.02 * np.abs(self.y)))

    def warning(self):
        if not self.target_missed:
            return ''
        text = f'{self.name}: error limit not reached'
        if self.negative_real_part > 0.1:
            text += (f'; the data of this branch have a negative real part in '
                     f'{100 * self.negative_real_part:.0f} % of the band, which positive '
                     f'resistors cannot give (try a narrower band)')
        return text

    def admittance(self, freq):
        """Admittance of the chosen candidate at ``freq`` (Hz)."""
        freq = np.asarray(freq, dtype=float)
        if self.chosen is None or self.chosen.is_open:
            return np.zeros(freq.size, dtype=complex)
        return self.chosen.component.eval(freq, 0)

    def elements(self):
        """Chosen elements as (name, kind, node1, node2, value)."""
        if self.chosen is None or self.chosen.is_open:
            return []
        comp = self.chosen.component
        out = []
        for (sym, n1, n2), value in zip(comp.network.connections, comp.values[0]):
            nodes = [self.node1 if n == 1 else self.node2 if n == 0 else f'{self.tag}_n{n}'
                     for n in (n1, n2)]
            out.append((f'{sym}_{self.tag}', sym[0], nodes[0], nodes[1], value))
        return out


class Model:
    """The extracted model: branches with their chosen networks."""

    def __init__(self, nports, z0, branches, freq, s_data, options, source='',
                 reciprocity_error=0.0):
        self.nports, self.z0 = nports, z0
        self.branches = branches
        self.freq, self.s_data = freq, s_data    # fit band
        self.options = options
        self.source = source
        self.reciprocity_error = reciprocity_error

    def choose(self, branch, candidate):
        """Use another candidate (index or Candidate) for a branch (index)."""
        b = self.branches[branch]
        b.chosen = b.candidates[candidate] if isinstance(candidate, int) else candidate

    def y_matrix(self, freq):
        freq = np.asarray(freq, dtype=float).ravel()
        y = np.zeros((freq.size, self.nports, self.nports), dtype=complex)
        for b in self.branches:
            _stamp(y, b.ports, b.admittance(freq))
        return y

    def s_matrix(self, freq):
        return _y2s(self.y_matrix(freq), self.z0)

    def to_network(self, freq=None):
        """The model's response as a scikit-rf Network (default: the fit band)."""
        import skrf
        freq = self.freq if freq is None else np.asarray(freq, dtype=float).ravel()
        return skrf.Network(frequency=skrf.Frequency.from_f(freq, unit='hz'),
                            s=self.s_matrix(freq), z0=self.z0)

    def error(self):
        """RMS |S difference| between model and data in the fit band."""
        return _rms(self.s_matrix(self.freq) - self.s_data)

    def error_db(self):
        return 20 * math.log10(max(self.error(), 1e-300))

    def max_error(self):
        return float(np.abs(self.s_matrix(self.freq) - self.s_data).max())

    def elements(self):
        return [e for b in self.branches for e in b.elements()]

    def n_elements(self):
        return len(self.elements())

    def to_dict(self):
        """Plain data (JSON-ready) of the result."""
        return {
            'source': self.source, 'ports': self.nports, 'z0': self.z0,
            'band_hz': [float(self.freq[0]), float(self.freq[-1])],
            'points': int(self.freq.size),
            'options': dataclasses.asdict(self.options),
            'error_rms': self.error(), 'error_db': self.error_db(),
            'error_max': self.max_error(), 'reciprocity_error': self.reciprocity_error,
            'branches': [{
                'name': b.name, 'node1': b.node1, 'node2': b.node2,
                'network': None if b.chosen.is_open else b.chosen.component.network.name,
                'id': None if b.chosen.is_open else b.chosen.component.network.id,
                'error_rms': b.chosen.error, 'error_db': b.chosen.error_db,
                'limit_rms': b.limit, 'warning': b.warning(),
                'elements': [{'name': n, 'kind': k, 'node1': n1, 'node2': n2, 'value': v}
                             for n, k, n1, n2, v in b.elements()],
            } for b in self.branches],
        }

    def summary(self):
        """Text table of the branches."""
        lines = []
        for b in self.branches:
            c = b.chosen
            lines.append(f'{b.name:10s} {c.label:20s} {c.error_db:7.1f} dB  {c.values_text()}')
        lines += ['warning: ' + b.warning() for b in self.branches if b.warning()]
        lines.append(f'model: {self.n_elements()} elements, S-parameter error '
                     f'{self.error_db():.1f} dB rms, {20 * math.log10(max(self.max_error(), 1e-300)):.1f} dB max')
        return '\n'.join(lines)


# --- helpers ----------------------------------------------------------------------------

def _rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _y2s(y, z0):
    n = y.shape[-1]
    eye = np.eye(n)
    return np.einsum('fij,fjk->fik', eye - z0 * y, np.linalg.inv(eye + z0 * y))


def _stamp(y, ports, yb):
    """Add branch admittance yb (per frequency) to the Y-matrices y."""
    if len(ports) == 1:
        i = ports[0] - 1
        y[:, i, i] += yb
    else:
        i, j = ports[0] - 1, ports[1] - 1
        y[:, i, i] += yb
        y[:, j, j] += yb
        y[:, i, j] -= yb
        y[:, j, i] -= yb


def branch_admittances(y):
    """Pi decomposition of symmetric Y-matrices (frequency, n, n): {ports: admittance}."""
    n = y.shape[-1]
    out = {}
    for i in range(n):
        out[(i + 1,)] = y[:, :, i].sum(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            out[(i + 1, j + 1)] = -y[:, i, j]
    return out


def _acceptable(comp, sol, require):
    vals = comp.values[sol]
    if not all(isinstance(v, float) and math.isfinite(v) for v in vals):
        return False                             # complex values cannot be built
    if require == 'positive':
        return all(v > 0 for v in vals)
    if require == 'positive-r':
        return all(v > 0 for v, sym in zip(vals, comp.network.symVn) if sym[0] == 'R')
    if require == 'passive':
        return all(v > 0 for v in vals) or comp.is_passive(sol)
    return True


class _BranchSearch:
    """Candidates of one branch, synthesized only as far as the choice needs."""

    def __init__(self, branch, y_sym, s_ref, z0, cat, opt, tick):
        self.b, self.opt, self.tick = branch, opt, tick
        self.y_sym, self.s_ref, self.z0 = y_sym, s_ref, z0
        self.done = {}                           # network id -> list of Candidate
        nets = [n for n in cat if n.can_synthesize and not n.fixed
                and len(n.symVn) <= opt.max_elements]
        self.sigs = {}
        for n in nets:
            self.sigs.setdefault(n.flag_sig, []).append(n)
        # the rational fit of each signature and its error: a network with synthesis
        # equations realizes the fitted function, so this is its error before synthesis
        self.fit, self.sig_error = {}, {}
        for sig, members in self.sigs.items():
            r = crfit(branch.y, branch.freq, members[0].flag_a, members[0].flag_b, opt.dps)
            self.fit[sig] = r
            if r.ok:
                self.sig_error[sig] = self.s_error(evaluate(r.a, r.b, branch.freq, opt.dps))
            tick()
        self.open = Candidate(None, self.s_error(np.zeros_like(branch.y)), _rms(branch.y))

    def s_error(self, yb):
        """RMS |S difference| when yb replaces the branch admittance in the data."""
        if not np.all(np.isfinite(yb)):
            return math.inf
        y = self.y_sym.copy()
        _stamp(y, self.b.ports, yb - self.b.y)
        try:
            return _rms(_y2s(y, self.z0) - self.s_ref)
        except np.linalg.LinAlgError:
            return math.inf

    def candidates(self, net):
        if net.id not in self.done:
            self.tick()
            out = []
            comp = Component(net)
            try:
                errs = comp.fit(self.b.y, self.b.freq, dps=self.opt.dps,
                                coefficients=self.fit[net.flag_sig])
            except (ValueError, ZeroDivisionError):
                errs = None
            if errs is not None:
                for sol in range(len(comp.values)):
                    if _acceptable(comp, sol, self.opt.require):
                        c = Component(net, values=comp.values[sol])
                        out.append(Candidate(c, self.s_error(c.eval(self.b.freq, 0)), errs[sol]))
            self.done[net.id] = out
        return self.done[net.id]

    def run(self):
        opt = self.opt
        target = 10 ** (opt.target_db / 20)
        order = sorted(self.sig_error, key=self.sig_error.get)
        if opt.full_search:
            for sig in order:
                for net in self.sigs[sig]:
                    self.candidates(net)
        # lowest error: signatures in order of error until one has a candidate
        best = self.open
        for sig in order:
            if self.sig_error[sig] >= best.error and best is not self.open:
                break
            found = [c for net in self.sigs[sig] for c in self.candidates(net)]
            if found:
                best = min(found + [best], key=lambda c: c.error)
        if opt.pick == 'simplest':
            limit = max(target, opt.tolerance * best.error)
        else:
            # lowest error; among practically equal errors the fewest elements, so that
            # exact data do not give a larger network with meaningless extra elements
            limit = 1.01 * best.error + EQUAL_ERROR
        chosen = best
        if self.open.error <= limit:
            chosen = self.open
        else:
            # networks that can reach the limit, fewest elements first
            nets = [n for sig in order if self.sig_error[sig] <= limit
                    for n in self.sigs[sig]]
            for count in range(1, best.n_elements + 1):
                found = [c for n in nets if len(n.symVn) == count
                         for c in self.candidates(n) if c.error <= limit]
                if found:
                    chosen = min(found, key=lambda c: c.error)
                    break
        allc = [c for cs in self.done.values() for c in cs] + [self.open]
        allc.sort(key=lambda c: (c.error > limit, c.n_elements if c.error <= limit else 0,
                                 c.error))
        self.b.candidates = [chosen] + [c for c in allc if c is not chosen]
        self.b.chosen = chosen
        self.b.limit = limit
        self.b.target = target


def extract(network, options=None, catalog=None, progress=None, cancel=None):
    """Lumped model of a scikit-rf Network (or Touchstone file name): see the module text.

    ``progress(fraction, text)`` is called during the work; if ``cancel()`` returns True,
    :class:`Cancelled` is raised. Returns a :class:`Model`.
    """
    opt = options or Options()
    opt.check()
    source = ''
    if not hasattr(network, 's'):
        source = str(network)
        from . import data
        network = data.read(network)
    cat = catalog or _catalog.load('base')
    f = network.f.astype(float)
    keep = f > 0
    if opt.fmin is not None:
        keep &= f >= opt.fmin
    if opt.fmax is not None:
        keep &= f <= opt.fmax
    if keep.sum() < 12:
        raise ValueError(f'only {int(keep.sum())} frequencies in the fit band, 12 needed')
    f = f[keep]
    z0 = float(np.real(network.z0[0, 0])) or 50.0
    if not np.allclose(network.z0, z0):
        network = network.copy()
        network.renormalize(z0)
    y = network.y[keep]
    y_sym = (y + np.transpose(y, (0, 2, 1))) / 2
    s_data = network.s[keep]
    s_ref = _y2s(y_sym, z0)
    recip = _rms(s_ref - s_data)
    yb = branch_admittances(y_sym)
    branches = [Branch(ports, f, yb[ports]) for ports in yb]

    nsig = len({n.flag_sig for n in cat if n.can_synthesize})
    state = {'k': 0, 'branch': 0}
    total = len(branches) * (nsig + 60)          # rough number of steps, for the bar

    def tick():
        if cancel is not None and cancel():
            raise Cancelled()
        state['k'] += 1
        if progress is not None:
            b = branches[state['branch']]
            progress(min(state['k'] / total, 0.99), f'branch {b.name}')

    for k, b in enumerate(branches):
        state['branch'] = k
        state['k'] = max(state['k'], k * (nsig + 60))
        _BranchSearch(b, y_sym, s_ref, z0, cat, opt, tick).run()
    if progress is not None:
        progress(1.0, 'done')
    return Model(network.nports, z0, branches, f, s_data, opt, source, recip)


def describe(model):
    """Lines for a report: data, options, branches, total error."""
    o = model.options
    lines = [f'data: {model.source or "network"}, {model.nports} port(s), '
             f'fit band {model.freq[0] / 1e9:.6g} to {model.freq[-1] / 1e9:.6g} GHz, '
             f'{model.freq.size} points',
             f'choice: {o.pick}, elements {o.require}, limit {o.target_db:g} dB or '
             f'{o.tolerance:g} x lowest error, up to {o.max_elements} elements per branch']
    if model.reciprocity_error > 1e-6:
        lines.append(f'data are not reciprocal: RMS |S - S^T|/2 = '
                     f'{20 * math.log10(model.reciprocity_error):.1f} dB (the model is reciprocal)')
    lines.append(model.summary())
    return lines

