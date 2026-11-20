"""From network data to a lumped model: one catalog network per branch.

**Pi form.** A reciprocal n-port is exactly a set of branches: one from each port to ground
(admittance = sum of a column of Y) and one between each pair of ports (admittance = -Ymn).
Each branch is a one-port, which the catalog search models with up to five R, L, C.

**Tee form** (2-ports): two series legs from the ports to a center node and one leg from
the center to ground (Z11 - Z21, Z22 - Z21, Z21). A device with a common path to ground is
a tee by nature; in the pi form its loss can end up in a branch with a negative real part.

**Coupled coils.** Magnetic coupling between two branches shows in the pi form as a
negative inductance (and resistance) in the branch between them. Such a triangle of
branches is replaced by two coils with a coupling factor, and a common resistor if needed.

For every branch the candidates are judged in three ways. (1) What they do to the
S-parameters: the branch in the data is replaced by the candidate, and the RMS difference
of all S-parameters is the candidate's error. (2) and (3), what it does to the input
admittance Yii of each port (all other ports shorted): reactance, the RMS of Im dYii / |Yii|,
and loss, the RMS of Re dYii / |Re Yii|; the port with the largest error counts. The loss
error compares a branch's loss with the loss of the whole port, so an error in the series
resistance of an inductor counts fully however high its Q, while the loss of a small
parasitic branch hardly counts. Port losses below |Yii| / 100 count as none. The port
measures are needed for parts that are nearly a through or an open in the reference
impedance, and for the loss of high-Q parts: neither shows in the S-parameters. A
candidate's score is the largest of the three errors, each divided by its limit.

Candidates get their element values from the closed-form synthesis of a rational fit,
which does not aim at the loss. With ``refine`` the best few networks of each element count
are then refined: a bounded least-squares fit of their element values on all three
measures, signs kept. The default choice is the network with the fewest elements, all
positive, whose score is within 1; "open" (or "short" for a series leg) is a candidate too
and is judged by the S-parameters only, so negligible branches drop out.

No printing, no plotting, no GUI here: :func:`extract` returns a :class:`Model`, and takes
``progress`` and ``cancel`` callbacks.
"""

import dataclasses
import math

import numpy as np

from . import catalog as _catalog
from .component import Component, format_value
from .fit import DEFAULT_DPS, crfit, evaluate
from .seed import Cancelled

__all__ = ['Options', 'Model', 'Branch', 'Candidate', 'Coupled', 'extract', 'shift_ports',
           'TradeoffPoint', 'pick_candidate', 'score_of',
           'branch_admittances', 'Cancelled']

REQUIRE = ('positive', 'positive-r', 'passive', 'any')
PICK = ('simplest', 'best')
FORM = ('auto', 'pi', 'tee')
# Model quality: one word for the error limits and the choice. 'standard' is the default of
# Options; 'small' accepts more error for fewer elements; 'accurate' takes the lowest error.
QUALITY = {
    'small': dict(target_db=-30.0, reactance_percent=30.0, loss_percent=50.0, pick='simplest'),
    'standard': dict(target_db=-40.0, reactance_percent=20.0, loss_percent=30.0, pick='simplest'),
    'accurate': dict(target_db=-40.0, reactance_percent=20.0, loss_percent=30.0, pick='best'),
}


def quality_of(options):
    """Name of the quality preset that ``options`` are equal to, or None."""
    for name, values in QUALITY.items():
        if all(getattr(options, key) == value for key, value in values.items()):
            return name
    return None
EQUAL_ERROR = 1e-10      # RMS |S difference| below which two candidates count as equal
EQUAL_SCORE = 1e-3       # scores closer than this (1 = at the limits) count as equal
SHORT_OHM = 1e-6         # resistor written for a shorted leg that cannot be merged away
SHORT_SIEMENS = 1e12     # admittance standing in for a series leg of zero impedance
LOSS_FLOOR_Q = 100       # losses below |Z| / LOSS_FLOOR_Q count as none (loss error)
REFINE_SPAN = 1e3        # refinement keeps each element within this factor of its start
REFINE_TRY = 6           # networks synthesized per element count for the refinement
REFINE_BEST = 3          # of those, the best ones are refined
K_MAX = 0.999            # largest coupling factor accepted
K_MIN = 0.05             # smaller coupling factors are not worth a coupled-coil model


@dataclasses.dataclass
class Options:
    """Settings of :func:`extract`.

    fmin, fmax: fit band in Hz (default: all data; a 0 Hz point is never used).
    require: 'positive' (all element values > 0), 'positive-r' (resistors > 0, L and C of
        either sign: still passive, since only resistors dissipate), 'passive' (Re Y >= 0,
        any signs) or 'any' (any real element values).
    pick: 'simplest' (fewest elements within the error limit) or 'best' (lowest error;
        the fewest elements among practically equal errors).
    target_db: limit of the S-parameter error, RMS |S difference| in dB (-40 dB = 0.01).
    reactance_percent: limit of the port reactance error of a branch, in percent; 0 = off.
    loss_percent: limit of the port loss error of a branch, in percent; 0 = off.
    refine: refine the element values of the best networks (bounded least squares).
    tolerance: if no candidate reaches both limits, candidates up to tolerance x the best
        score are accepted.
    max_elements: largest number of elements per branch (1..5).
    form: 'pi', 'tee' (2-ports only) or 'auto': pi, and for a 2-port also the tee form if
        a pi branch misses the error limit; the form with the lower error is kept.
    coupling: replace a branch that misses the limit (or has a negative element) and its
        two neighbours by coupled coils, if that lowers the error (pi form).
    sequential: pi form: fit the branches between ports first; the target of each branch
        to ground is then the port admittance Yii minus the models of those branches, not
        minus their data, so it makes up for what they missed (J. Rautio's suggestion).
        Off by default: what a model missed is often not a physical admittance, and the
        branch to ground then misses its limits (see the README for the comparison).
    delay: a piece of line (reference impedance, this delay) at each port, removed from
        the data before the fit and put back in the netlist as a line element: 'off' or
        the delay in seconds. For data with a known feed line; a line that is short
        against the wavelength needs none of this, the lumped elements cover it.
    full_search: synthesize every network instead of only those that can be chosen
        (slower, same result; fills the list of alternatives completely).
    dps: working precision of the fits, decimal digits.
    """
    fmin: float = None
    fmax: float = None
    require: str = 'positive'
    pick: str = 'simplest'
    target_db: float = -40.0
    reactance_percent: float = 20.0
    loss_percent: float = 30.0
    refine: bool = True
    tolerance: float = 1.25
    max_elements: int = 5
    form: str = 'auto'
    coupling: bool = True
    sequential: bool = False
    delay: object = 'off'
    full_search: bool = False
    dps: int = DEFAULT_DPS

    def check(self):
        if self.require not in REQUIRE:
            raise ValueError(f'require must be one of {REQUIRE}')
        if self.pick not in PICK:
            raise ValueError(f'pick must be one of {PICK}')
        if self.form not in FORM:
            raise ValueError(f'form must be one of {FORM}')
        if self.tolerance < 1:
            raise ValueError('tolerance must be >= 1')
        if not 1 <= self.max_elements <= 5:
            raise ValueError('max_elements must be 1..5')
        if self.target_db >= 0:
            raise ValueError('target_db must be negative')
        if not (self.delay == 'off' or (isinstance(self.delay, (int, float))
                                        and not isinstance(self.delay, bool)
                                        and self.delay >= 0)):
            raise ValueError("delay must be 'off' or a delay in seconds (>= 0)")
        if self.reactance_percent < 0 or self.loss_percent < 0:
            raise ValueError('reactance_percent and loss_percent must be >= 0')

    @property
    def target(self):
        return 10 ** (self.target_db / 20)


def _db(x):
    return 20 * math.log10(max(x, 1e-300))


class Candidate:
    """A network with element values for one branch. ``error``: RMS |S difference| when
    this candidate replaces the branch in the data; ``error_y``: RMS error of the branch
    admittance (S). ``reactance``, ``loss``: relative errors of the port admittances (see
    the module text). ``score``: the largest error over its limit. ``refined``:
    element values from the refinement. ``trivial``: 'open' or 'short' for the candidates
    without elements."""

    def __init__(self, component, error, error_y, trivial=None, score=None, reactance=None,
                 loss=None, refined=False):
        self.component = component
        self.error = float(error)
        self.error_y = float(error_y)
        self.trivial = trivial
        self.reactance, self.loss = reactance, loss
        self.refined = refined
        self.score = float(error if score is None else score)   # <= 1: within the limits

    @staticmethod
    def _percent(x):
        return '-' if x is None else 'inf' if not math.isfinite(x) else f'{100 * x:.3g} %'

    def reactance_text(self):
        return self._percent(self.reactance)

    def loss_text(self):
        return self._percent(self.loss)

    @property
    def n_elements(self):
        return len(self.component.network.symVn) if self.component is not None else 0

    @property
    def is_trivial(self):
        return self.component is None

    @property
    def is_open(self):
        return self.trivial == 'open'

    @property
    def label(self):
        if self.is_trivial:
            return self.trivial
        net = self.component.network
        return f'CR{net.id} {net.name}' + (' refined' if self.refined else '')

    @property
    def error_db(self):
        return _db(self.error)

    def values_text(self):
        return 'no elements' if self.is_trivial else self.component.format()

    def __repr__(self):
        return (f'<{self.label}: {self.error_db:.1f} dB, X {self.reactance_text()}, '
                f'loss {self.loss_text()}, {self.values_text()}>')


def _node_name(node):
    return 'gnd' if node == '0' else node


class Branch:
    """One branch between ``node1`` and ``node2`` ('p1'.., 'c' = tee center, '0' = ground).

    ``y``: admittance of the branch in the data. ``ports``: for pi branches, (i,) or (i, j).
    ``group``: the :class:`Coupled` group that replaces this branch, if any.
    """

    def __init__(self, node1, node2, tag, freq, y, trivial='open', ports=None):
        self.node1, self.node2, self.tag = node1, node2, tag
        self.ports = ports
        self.freq, self.y = freq, y
        self.trivial = trivial
        self.candidates = []                     # best choice first
        self.chosen = None
        self.limit = None                        # largest score accepted for the choice
        self.group = None

    @property
    def name(self):
        return f'{self.node1} - {_node_name(self.node2)}'

    @property
    def target_missed(self):
        """True if the chosen candidate does not reach the error limits."""
        return self.chosen is not None and self.chosen.score > 1 + 1e-9

    @property
    def negative_real_part(self):
        """Share (0..1) of the fit band where the branch of the data has a clearly negative
        real part: there, no network with positive resistors can fit it."""
        return float(np.mean(self.y.real < -0.02 * np.abs(self.y)))

    def has_negative_element(self):
        return (self.chosen is not None and not self.chosen.is_trivial
                and any(v < 0 for v in self.chosen.component.values[0]))

    def warning(self):
        if self.group is not None or not self.target_missed:
            return ''
        text = f'{self.name}: error limit not reached'
        if self.negative_real_part > 0.1:
            text += (f'; the data of this branch have a negative real part in '
                     f'{100 * self.negative_real_part:.0f} % of the band, which positive '
                     f'resistors cannot give (try a narrower band)')
        return text

    def admittance(self, freq):
        """Admittance of the chosen candidate at ``freq`` (Hz); inf for a short."""
        freq = np.asarray(freq, dtype=float)
        c = self.chosen
        if c is None or c.trivial == 'open':
            return np.zeros(freq.size, dtype=complex)
        if c.trivial == 'short':
            return np.full(freq.size, np.inf, dtype=complex)
        return c.component.eval(freq, 0)

    def elements(self, alias=None):
        """Chosen elements as (name, kind, node1, node2, value)."""
        if self.chosen is None or self.chosen.is_trivial:
            return []
        alias = alias or {}
        return _component_elements(self.chosen.component, alias.get(self.node1, self.node1),
                                   alias.get(self.node2, self.node2), self.tag)


def _component_elements(comp, node1, node2, tag):
    out = []
    for (sym, n1, n2), value in zip(comp.network.connections, comp.values[0]):
        nodes = [node1 if n == 1 else node2 if n == 0 else f'{tag}_n{n}' for n in (n1, n2)]
        out.append((f'{sym}_{tag}', sym[0], nodes[0], nodes[1], value))
    return out


def series_inductor(comp):
    """Index of the largest positive inductor that carries the whole current of the
    one-port (removing it separates the two terminals), or None."""
    conns = comp.network.connections
    best = None
    for k, ((sym, _, _), value) in enumerate(zip(conns, comp.values[0])):
        if sym[0] != 'L' or not value > 0:
            continue
        reach, grew = {1}, True
        while grew:
            grew = False
            for m, (_, a, b) in enumerate(conns):
                if m != k and (a in reach) != (b in reach):
                    reach |= {a, b}
                    grew = True
        if 0 not in reach and (best is None or value > comp.values[0][best]):
            best = k
    return best


class Coupled:
    """Two coupled coils from ports i and j to a common node, which is node k (a port or
    ground) or, with a common resistor ``rc``, a node joined to k by that resistor."""

    def __init__(self, ports, coils, rc, mutual, error):
        self.ports = ports                       # (i, j, k); k = 0 for ground
        self.coils = coils                       # two Components
        self.rc, self.mutual = rc, mutual
        self.error = float(error)
        self.index = [series_inductor(c) for c in coils]
        la, lb = (c.values[0][n] for c, n in zip(coils, self.index))
        self.k = mutual / math.sqrt(la * lb)
        i, j, k = ports
        self.tag = f'k{i}{j}'
        self.node_k = f'p{k}' if k else '0'
        self.node_m = f'{self.tag}_m' if rc > 0 else self.node_k

    @property
    def name(self):
        i, j, _ = self.ports
        return f'p{i}, p{j} - {_node_name(self.node_k)}'

    @property
    def label(self):
        return 'coupled coils: ' + ' + '.join(f'CR{c.network.id} {c.network.name}'
                                              for c in self.coils)

    @property
    def error_db(self):
        return _db(self.error)

    def values_text(self):
        text = ' | '.join(c.format() for c in self.coils) + f' | k={self.k:.4g}'
        if self.rc > 0:
            text += f' Rc={format_value(self.rc, "R")}'
        return text

    def impedances(self, freq):
        """Z11, Z22, Z12 of the pair, seen from ports i and j against node k."""
        freq = np.asarray(freq, dtype=float)
        with np.errstate(all='ignore'):
            z1, z2 = (1 / c.eval(freq, 0) for c in self.coils)
        return z1 + self.rc, z2 + self.rc, self.rc + 2j * np.pi * freq * self.mutual

    def stamp(self, y, freq):
        _stamp_pair(y, self.ports, *self.impedances(freq))

    def elements(self):
        i, j, _ = self.ports
        out, names = [], []
        for port, comp, n, letter in zip((i, j), self.coils, self.index, 'ab'):
            part = _component_elements(comp, f'p{port}', self.node_m, self.tag + letter)
            names.append(part[n][0])
            out += part
        if self.rc > 0:
            out.append((f'Rc_{self.tag}', 'R', self.node_m, self.node_k, self.rc))
        out.append((f'K_{self.tag}', 'K', names[0], names[1], self.k))
        return out


class Model:
    """The extracted model: branches with their chosen networks, and coupled-coil groups."""

    def __init__(self, nports, z0, branches, freq, s_data, options, form='pi', source='',
                 reciprocity_error=0.0):
        self.nports, self.z0 = nports, z0
        self.branches = branches
        self.couplings = []
        self.form = form
        self.delay = 0.0                         # line at each port, seconds
        self.freq, self.s_data = freq, s_data    # fit band
        self.options = options
        self.source = source
        self.reciprocity_error = reciprocity_error

    def choose(self, branch, candidate):
        """Use another candidate (index or Candidate) for a branch (index)."""
        b = self.branches[branch]
        b.chosen = b.candidates[candidate] if isinstance(candidate, int) else candidate

    def reselect(self, options):
        """Choose again from the candidates already tried, for other error limits, choice
        rule or element count in ``options`` (an Options). No new search: instant, but
        only as good as the candidates at hand; a much stricter limit may need a new
        extraction. Band, element signs and form stay as they are."""
        options.check()
        for b in self.free_branches():
            pool = [c for c in b.candidates if c.n_elements <= options.max_elements]
            for c in pool:
                c.score = score_of(c, options)
            chosen, limit = pick_candidate(pool, options)
            rest = sorted((c for c in b.candidates if c is not chosen),
                          key=lambda c: (c.score > limit,
                                         c.n_elements if c.score <= limit else 0, c.score))
            b.candidates, b.chosen, b.limit = [chosen] + rest, chosen, limit
        for name in ('target_db', 'reactance_percent', 'loss_percent', 'pick', 'tolerance',
                     'max_elements'):
            setattr(self.options, name, getattr(options, name))

    def tradeoff(self):
        """Models from few to many elements, from the candidates already tried: a list of
        TradeoffPoint, each with a lower error than the one before. Starts with the
        smallest candidate in every branch, then repeatedly makes the one change to a
        larger candidate that lowers the error of the whole model most per added element."""
        free = [(k, b) for k, b in enumerate(self.branches) if b.group is None]
        keep = {k: b.chosen for k, b in free}
        best = {}                                # per branch: element count -> candidate
        for k, b in free:
            for c in b.candidates:
                if c.n_elements not in best.setdefault(k, {}) or c.score < best[k][c.n_elements].score:
                    best[k][c.n_elements] = c

        def error(selection):
            for k, c in selection.items():
                self.branches[k].chosen = c
            e = self.error()
            return e if math.isfinite(e) else math.inf

        fixed = self.n_elements() - sum(c.n_elements for c in keep.values())   # coils, lines

        def size(selection):
            return fixed + sum(c.n_elements for c in selection.values())

        try:
            sel = {k: best[k][min(best[k])] for k, _ in free}
            points = [TradeoffPoint(size(sel), error(sel), dict(sel))]
            while True:
                here = points[-1]
                step = None
                for k, _ in free:
                    for count, c in best[k].items():
                        added = count - sel[k].n_elements
                        if added <= 0:
                            continue
                        e = error({**sel, k: c})
                        gain = (_db(here.error) - _db(e)) / added
                        if e < here.error and (step is None or gain > step[0]):
                            step = (gain, k, c, e)
                if step is None:
                    break
                sel[step[1]] = step[2]
                points.append(TradeoffPoint(size(sel), step[3], dict(sel)))
        finally:
            for k, c in keep.items():
                self.branches[k].chosen = c
        return points

    def apply(self, point):
        """Use the selection of a TradeoffPoint."""
        for k, c in point.selection.items():
            self.branches[k].chosen = c

    def free_branches(self):
        """Branches that are not replaced by a coupled-coil group."""
        return [b for b in self.branches if b.group is None]

    def rows(self):
        """Result rows, each a Branch or a Coupled (both have name, error and values)."""
        return self.free_branches() + self.couplings

    def y_matrix(self, freq):
        if self.form != 'pi':
            raise ValueError('the tee form has no Y-matrix if a leg is open: use s_matrix')
        freq = np.asarray(freq, dtype=float).ravel()
        y = np.zeros((freq.size, self.nports, self.nports), dtype=complex)
        for b in self.free_branches():
            _stamp(y, b.ports, b.admittance(freq))
        for c in self.couplings:
            c.stamp(y, freq)
        return y

    def s_matrix(self, freq):
        freq = np.asarray(freq, dtype=float).ravel()
        if self.form == 'tee':
            a, b, c = (br.admittance(freq) for br in self.branches)
            with np.errstate(all='ignore'):
                s = _tee_s(1 / a, 1 / b, c, self.z0)
        else:
            s = _y2s(self.y_matrix(freq), self.z0)
        return shift_ports(s, freq, self.delay) if self.delay else s

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
        return _db(self.error())

    def max_error(self):
        return float(np.abs(self.s_matrix(self.freq) - self.s_data).max())

    def sections(self):
        """Netlist sections: list of (comment, elements); elements are
        (name, kind, node1, node2, value), for kind 'K' (inductor1, inductor2, k)."""
        alias, shorted = {}, []
        if self.form == 'tee':                   # a shorted leg merges the center node
            for b in self.branches[:2]:
                if b.chosen.trivial == 'short':
                    if 'c' not in alias:
                        alias['c'] = b.node1
                    else:
                        shorted.append((f'Rshort_{b.tag}', 'R', b.node1, alias['c'], SHORT_OHM))
        out = []
        for b in self.free_branches():
            out.append((f'branch {b.name}: {b.chosen.label}, error {b.chosen.error_db:.1f} dB',
                        b.elements(alias)))
        if shorted:
            out.append(('short between the ports', shorted))
        for c in self.couplings:
            out.append((f'{c.name}: {c.label}, error {c.error_db:.1f} dB', c.elements()))
        if self.delay:                           # the lumped part sits behind the lines
            core = {f'p{i + 1}': f'p{i + 1}_c' for i in range(self.nports)}
            out = [(text, [e if e[1] == 'K' else
                           (e[0], e[1], core.get(e[2], e[2]), core.get(e[3], e[3]), e[4])
                           for e in part]) for text, part in out]
            out.append((f'line of {self.delay * 1e12:.4g} ps ({self.z0:g} ohm) at each port',
                        [(f'T_{p}', 'T', p, c, self.delay) for p, c in core.items()]))
        return out

    def elements(self):
        return [e for _, part in self.sections() for e in part]

    def n_elements(self):
        return len(self.elements())

    def warnings(self):
        return [b.warning() for b in self.branches if b.warning()]

    def to_dict(self):
        """Plain data (JSON-ready) of the result."""
        def elements(part):
            return [{'name': n, 'kind': k, 'node1': n1, 'node2': n2, 'value': v}
                    for n, k, n1, n2, v in part]
        return {
            'source': self.source, 'ports': self.nports, 'z0': self.z0, 'form': self.form,
            'delay_s': self.delay,
            'band_hz': [float(self.freq[0]), float(self.freq[-1])],
            'points': int(self.freq.size),
            'options': dataclasses.asdict(self.options),
            'error_rms': self.error(), 'error_db': self.error_db(),
            'error_max': self.max_error(), 'reciprocity_error': self.reciprocity_error,
            'branches': [{
                'name': b.name, 'node1': b.node1, 'node2': b.node2,
                'network': None if b.chosen.is_trivial else b.chosen.component.network.name,
                'id': None if b.chosen.is_trivial else b.chosen.component.network.id,
                'choice': b.chosen.label,
                'error_rms': b.chosen.error, 'error_db': b.chosen.error_db,
                'error_reactance': b.chosen.reactance, 'error_loss': b.chosen.loss,
                'refined': b.chosen.refined, 'score': b.chosen.score,
                'score_limit': b.limit, 'warning': b.warning(),
                'elements': elements(b.elements()),
            } for b in self.free_branches()],
            'couplings': [{
                'name': c.name, 'networks': [k.network.name for k in c.coils],
                'k': c.k, 'mutual': c.mutual, 'rc': c.rc,
                'error_rms': c.error, 'error_db': c.error_db,
                'elements': elements(c.elements()),
            } for c in self.couplings],
        }

    def summary(self):
        """Text table of the result."""
        lines = []
        for r in self.rows():
            label = r.chosen.label if isinstance(r, Branch) else r.label
            err = r.chosen.error_db if isinstance(r, Branch) else r.error_db
            text = r.chosen.values_text() if isinstance(r, Branch) else r.values_text()
            if isinstance(r, Branch):
                x, loss = r.chosen.reactance_text(), r.chosen.loss_text()
            else:
                x = loss = '-'
            lines.append(f'{r.name:10s} {label:28s} {err:7.1f} dB  X {x:>8s}  loss {loss:>8s}  '
                         f'{text}')
        lines += ['warning: ' + w for w in self.warnings()]
        if self.delay:
            lines.append(f'line at each port: {self.delay * 1e12:.4g} ps, {self.z0:g} ohm')
        lines.append(f'model ({self.form} form): {self.n_elements()} elements, S-parameter '
                     f'error {self.error_db():.1f} dB rms, {_db(self.max_error()):.1f} dB max')
        return '\n'.join(lines)


# --- helpers ----------------------------------------------------------------------------

def _rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _y2s(y, z0):
    n = y.shape[-1]
    eye = np.eye(n)
    return np.einsum('fij,fjk->fik', eye - z0 * y, np.linalg.inv(eye + z0 * y))


def shift_ports(s, freq, delay):
    """S-parameters with a matched line of ``delay`` seconds added at every port
    (negative delay: removed)."""
    d = np.exp(-2j * np.pi * np.asarray(freq, dtype=float) * delay)
    return s * (d * d)[:, None, None]


def _tee_s(za, zb, yc, z0):
    """S-parameters of a tee: series za, shunt yc, series zb (ABCD chain, so a shorted leg
    and an open center are allowed)."""
    a = 1 + za * yc
    b = za + zb + za * zb * yc
    d = 1 + zb * yc
    den = a + b / z0 + yc * z0 + d
    s = np.empty((np.size(den), 2, 2), dtype=complex)
    s[:, 0, 0] = (a + b / z0 - yc * z0 - d) / den
    s[:, 1, 1] = (-a + b / z0 - yc * z0 + d) / den
    s[:, 0, 1] = s[:, 1, 0] = 2 / den
    return s


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


def _pair_ports(ports):
    """Pi branches of the triangle i, j, k (k = 0: ground): (i-j, i-k, j-k)."""
    i, j, k = ports
    return (tuple(sorted((i, j))), tuple(sorted((i, k))) if k else (i,),
            tuple(sorted((j, k))) if k else (j,))


def _pair_admittances(z11, z22, z12):
    """Branch admittances (i-j, i-k, j-k) of a coupled pair given by its impedances."""
    det = z11 * z22 - z12 * z12
    return z12 / det, (z22 - z12) / det, (z11 - z12) / det


def _stamp_pair(y, ports, z11, z22, z12):
    with np.errstate(all='ignore'):
        for p, yb in zip(_pair_ports(ports), _pair_admittances(z11, z22, z12)):
            _stamp(y, p, yb)


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


def netlist_admittance(connections, values, s):
    """Admittance between node 1 and ground (node 0) of an R, L, C netlist at complex s
    (numpy, for many frequencies at once; independent of the catalog equations)."""
    nodes = sorted({n for _, a, b in connections for n in (a, b)} - {0})
    pos = {n: k for k, n in enumerate(nodes)}
    s = np.asarray(s, dtype=complex)
    y = np.zeros((s.size, len(nodes), len(nodes)), dtype=complex)
    with np.errstate(all='ignore'):
        for (sym, a, b), v in zip(connections, values):
            kind = sym[0]
            g = 1 / v + 0 * s if kind == 'R' else 1 / (s * v) if kind == 'L' else s * v
            for p, q in ((a, b), (b, a)):
                if p:
                    y[:, pos[p], pos[p]] += g
                    if q:
                        y[:, pos[p], pos[q]] -= g
        rhs = np.zeros((s.size, len(nodes)), dtype=complex)
        rhs[:, pos[1]] = 1
        try:
            v1 = np.linalg.solve(y, rhs[..., None])[:, pos[1], 0]
        except np.linalg.LinAlgError:
            return np.full(s.size, np.nan, dtype=complex)
        return 1 / v1


def score_of(cand, opt):
    """Score of a candidate for the limits of ``opt``: the largest error over its limit."""
    score = cand.error / opt.target
    if opt.reactance_percent > 0 and cand.reactance is not None:
        score = max(score, 100 * cand.reactance / opt.reactance_percent)
    if opt.loss_percent > 0 and cand.loss is not None:
        score = max(score, 100 * cand.loss / opt.loss_percent)
    return score


def pick_candidate(pool, opt):
    """Choose from the pool by the rules of Options.pick; returns (chosen, limit)."""
    best = min(pool, key=lambda c: c.score)
    if opt.pick == 'simplest':
        limit = max(1.0, opt.tolerance * best.score)
    else:
        # best score; among practically equal ones the fewest elements, so that exact data
        # do not give a larger network with meaningless extra elements
        limit = 1.01 * best.score + EQUAL_SCORE
    ok = [c for c in pool if c.score <= limit]
    count = min(c.n_elements for c in ok)
    chosen = min((c for c in ok if c.n_elements == count), key=lambda c: c.score)
    return chosen, limit


class TradeoffPoint:
    """A model of a given size: ``n_elements`` in the branches that can be chosen,
    ``error`` (RMS |S difference|), ``selection`` ({branch index: Candidate})."""

    def __init__(self, n_elements, error, selection):
        self.n_elements, self.error, self.selection = n_elements, error, selection

    @property
    def error_db(self):
        return _db(self.error)


class _BranchSearch:
    """Candidates of one branch, synthesized only as far as the choice needs.

    ``s_diff(yb)``: S-parameter differences (frequency, n, n) when admittance yb replaces
    the branch in the data. ``data``: the _Data (reference S and Y). ``accept(component)``:
    extra condition on a candidate.
    """

    def __init__(self, branch, s_diff, data, cat, opt, tick, accept=None):
        self.b, self.opt, self.tick = branch, opt, tick
        self.diff_of, self.accept = s_diff, accept
        self.data = data
        self.done = {}                           # network id -> list of Candidate
        yii = np.diagonal(data.y, axis1=1, axis2=2)          # port admittances of the data
        self.den_x = np.abs(yii)
        self.den_r = np.maximum(np.abs(yii.real), np.abs(yii) / LOSS_FLOOR_Q)
        nets = [n for n in cat if n.can_synthesize and not n.fixed
                and len(n.symVn) <= opt.max_elements]
        self.sigs = {}
        for n in nets:
            self.sigs.setdefault(n.flag_sig, []).append(n)
        # the rational fit of each signature and its score: a network with synthesis
        # equations realizes the fitted function, so this is its score before synthesis
        self.fit, self.sig_error = {}, {}
        for sig, members in self.sigs.items():
            r = crfit(branch.y, branch.freq, members[0].flag_a, members[0].flag_b, opt.dps)
            self.fit[sig] = r
            if r.ok:
                self.sig_error[sig] = self.measure(evaluate(r.a, r.b, branch.freq, opt.dps))[0]
            tick()
        self.trivial = None
        if branch.trivial in ('open', 'short'):
            yb = (np.zeros_like if branch.trivial == 'open' else
                  lambda v: np.full_like(v, SHORT_SIEMENS))(branch.y)
            score, e, ex, el = self.measure(yb)
            self.trivial = Candidate(None, e, _rms(branch.y) if branch.trivial == 'open'
                                     else math.inf, branch.trivial, score=score,
                                     reactance=ex, loss=el)

    # --- measures -----------------------------------------------------------------------

    def diffs(self, yb):
        """Per frequency, when yb replaces the branch: the RMS over all S-parameters of the
        difference to the data (frequency,), and the relative differences of the port
        admittances Yii (other ports shorted), reactance and loss (frequency, ports):
        Im dYii / |Yii| and Re dYii / max(|Re Yii|, |Yii| / LOSS_FLOOR_Q)."""
        n = self.b.freq.size
        with np.errstate(all='ignore'):
            try:
                d = self.diff_of(yb)
                s = self.data.s_ref + d
                eye = np.eye(s.shape[-1])
                y = np.linalg.solve(self.data.z0 * (eye + s), eye - s)
            except np.linalg.LinAlgError:
                inf = np.full(n, np.inf)
                return inf, np.full(self.den_x.shape, np.inf), np.full(self.den_x.shape, np.inf)
            sd = np.sqrt(np.mean(np.abs(d) ** 2, axis=(1, 2)))
            dyii = np.diagonal(y - self.data.y, axis1=1, axis2=2)
            dx = dyii.imag / self.den_x
            dr = dyii.real / self.den_r
        sd = np.where(np.isfinite(sd), sd, np.inf)
        bad = ~(np.isfinite(dx) & np.isfinite(dr))
        dx[bad], dr[bad] = np.inf, np.inf
        return sd, dx, dr

    def s_error(self, yb):
        e = float(np.sqrt(np.mean(self.diffs(yb)[0] ** 2)))
        return e if math.isfinite(e) else math.inf

    def measure(self, yb):
        """(score, S error, reactance error, loss error) of admittance yb; the port errors
        are the RMS over frequency at the port where they are largest."""
        opt = self.opt
        sd, dx, dr = self.diffs(yb)
        e = float(np.sqrt(np.mean(sd ** 2)))
        e = e if math.isfinite(e) else math.inf
        ex = float(np.max(np.sqrt(np.mean(dx ** 2, axis=0))))
        el = float(np.max(np.sqrt(np.mean(dr ** 2, axis=0))))
        score = e / opt.target
        if opt.reactance_percent > 0:
            score = max(score, 100 * ex / opt.reactance_percent)
        if opt.loss_percent > 0:
            score = max(score, 100 * el / opt.loss_percent)
        return score, e, ex, el

    def make(self, comp, error_y, refined=False):
        score, e, ex, el = self.measure(comp.eval(self.b.freq, 0))
        return Candidate(comp, e, error_y, None, score=score, reactance=ex, loss=el,
                         refined=refined)

    # --- candidates ---------------------------------------------------------------------

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
                    if not _acceptable(comp, sol, self.opt.require):
                        continue
                    c = Component(net, values=comp.values[sol])
                    if self.accept is None or self.accept(c):
                        out.append(self.make(c, errs[sol]))
            self.done[net.id] = out
        return self.done[net.id]

    def refine(self, cand):
        """Bounded least-squares refinement of a candidate's element values on all three
        measures together; signs stay as they are. Returns a new Candidate, or None if it
        is not better or an element runs to its bound (not determined by the data)."""
        from scipy.optimize import least_squares
        opt = self.opt
        comp = cand.component
        conns = comp.network.connections
        vals = np.array(comp.values[0], dtype=float)
        if not np.all(np.isfinite(vals)) or np.any(vals == 0):
            return None
        if cand.score <= EQUAL_SCORE:            # as good as exact: nothing to refine
            return None
        sign, x0 = np.sign(vals), np.log(np.abs(vals))
        s = 2j * np.pi * self.b.freq
        n = self.b.freq.size
        wx = 100 / opt.reactance_percent if opt.reactance_percent > 0 else 0.0
        wl = 100 / opt.loss_percent if opt.loss_percent > 0 else 0.0

        def residuals(x):
            yb = netlist_admittance(conns, sign * np.exp(x), s)
            sd, dx, dr = self.diffs(yb)
            r = np.concatenate([sd / opt.target, wx * dx.ravel(), wl * dr.ravel()]) / math.sqrt(n)
            return np.where(np.isfinite(r), r, 1e6)

        span = math.log(REFINE_SPAN)
        try:
            fit = least_squares(residuals, x0, bounds=(x0 - span, x0 + span), method='trf',
                                max_nfev=100 * len(x0))
        except (ValueError, np.linalg.LinAlgError):
            return None
        if np.any(np.abs(fit.x - x0) > 0.99 * span):
            return None
        new = Component(comp.network, values=list(sign * np.exp(fit.x)))
        if self.accept is not None and not self.accept(new):
            return None
        c = self.make(new, cand.error_y, refined=True)
        return c if c.score < 0.99 * cand.score else None

    # --- the choice ---------------------------------------------------------------------

    def _pick(self, pool):
        return pick_candidate(pool, self.opt)

    def run(self):
        """Choose; returns the chosen Candidate (None if there is no candidate at all)."""
        opt = self.opt
        order = sorted(self.sig_error, key=self.sig_error.get)
        if opt.full_search:
            for sig in order:
                for net in self.sigs[sig]:
                    self.candidates(net)
        # best score: signatures in order of their score until one has a candidate
        best, found_any = self.trivial, False
        for sig in order:
            if found_any and self.sig_error[sig] >= best.score:
                break
            found = [c for net in self.sigs[sig] for c in self.candidates(net)]
            if found:
                found_any = True
                best = min(found + ([best] if best is not None else []), key=lambda c: c.score)
        if best is None:
            return None
        pool = [c for cs in self.done.values() for c in cs]
        if self.trivial is not None:
            pool.append(self.trivial)
        chosen, limit = self._pick(pool)
        if not chosen.is_trivial:
            # networks with fewer elements that can reach the limit
            nets = [n for sig in order if self.sig_error[sig] <= limit for n in self.sigs[sig]
                    if len(n.symVn) < chosen.n_elements]
            for n in nets:
                self.candidates(n)
            pool = [c for cs in self.done.values() for c in cs] + \
                ([self.trivial] if self.trivial is not None else [])
            chosen, limit = self._pick(pool)
        if opt.refine and not (self.trivial is not None and chosen is self.trivial):
            pool += self._refined_pool()
            chosen, limit = self._pick(pool)
        pool.sort(key=lambda c: (c.score > limit, c.n_elements if c.score <= limit else 0,
                                 c.score))
        self.b.candidates = [chosen] + [c for c in pool if c is not chosen]
        self.b.chosen = chosen
        self.b.limit = limit
        return chosen

    def _refined_pool(self):
        """Refine the best few networks of each element count (synthesizing them first,
        by the score of their signature); returns the refined candidates."""
        per_count = {}
        for sig in sorted(self.sig_error, key=self.sig_error.get):
            for net in self.sigs[sig]:
                per_count.setdefault(len(net.symVn), []).append(net)
        out = []
        for count, nets in sorted(per_count.items()):
            found = []
            for net in nets[:REFINE_TRY]:
                found += self.candidates(net)
            found.sort(key=lambda c: c.score)
            for c in found[:REFINE_BEST]:
                self.tick()
                r = self.refine(c)
                if r is not None:
                    out.append(r)
        return out


# --- the forms --------------------------------------------------------------------------

class _Data:
    """The data in the fit band and what the searches need."""

    def __init__(self, f, y_sym, s_data, z0, cat, opt, tick):
        self.f, self.y, self.s_data, self.z0 = f, y_sym, s_data, z0
        self.s_ref = _y2s(y_sym, z0)
        self.cat, self.opt, self.tick = cat, opt, tick
        self.nports = y_sym.shape[-1]
        self.at = lambda fraction: None          # progress: where the work stands (0..1)

    def y_diff(self, y):
        return _y2s(y, self.z0) - self.s_ref

    def y_error(self, y):
        return _rms(self.y_diff(y))


def _extract_pi(d):
    yb = branch_admittances(d.y)
    branches = []
    for ports in yb:
        node2 = f'p{ports[1]}' if len(ports) == 2 else '0'
        branches.append(Branch(f'p{ports[0]}', node2, 'p' + 'p'.join(str(p) for p in ports),
                               d.f, yb[ports], 'open', ports))
    # The branches between ports first. With Options.sequential the branches to ground
    # then make up for what those models missed: the target of the branch at port i is
    # Yii minus the *models* of the branches at that port (not minus their data), and its
    # candidates are judged with those models in place. So Yii of the model follows the data.
    order = [b for b in branches if len(b.ports) == 2] + [b for b in branches if len(b.ports) == 1]
    context = d.y                                # the data, later with the pair models
    for n, b in enumerate(order):
        d.at(0.6 * n / len(order))
        y0 = b.y                                 # what this branch is in `context`
        if d.opt.sequential and len(b.ports) == 1:
            if context is d.y:
                context = d.y.copy()
                for pair in order:
                    if len(pair.ports) == 2:
                        _stamp(context, pair.ports, pair.admittance(d.f) - pair.y)
            i = b.ports[0]
            b.y = y0 + sum(pair.y - pair.admittance(d.f) for pair in order
                           if len(pair.ports) == 2 and i in pair.ports)

        def s_error(ycand, b=b, y0=y0, context=context):
            y = context.copy()
            _stamp(y, b.ports, ycand - y0)
            return d.y_diff(y)
        _BranchSearch(b, s_error, d, d.cat, d.opt, lambda b=b: d.tick(f'branch {b.name}')).run()
    return Model(d.nports, d.z0, branches, d.f, d.s_data, d.opt, 'pi')


def _extract_tee(d):
    """Tee form of a 2-port, or None if the data have no Z-matrix."""
    try:
        with np.errstate(all='ignore'):
            z = np.linalg.inv(d.y)
            legs = [1 / (z[:, 0, 0] - z[:, 0, 1]), 1 / (z[:, 1, 1] - z[:, 0, 1]), 1 / z[:, 0, 1]]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(legs[2])):
        return None
    for v in legs[:2]:                           # a series leg of exactly zero impedance
        v[~np.isfinite(v)] = SHORT_SIEMENS
    branches = [Branch('p1', 'c', 'p1c', d.f, legs[0], 'short'),
                Branch('p2', 'c', 'p2c', d.f, legs[1], 'short'),
                Branch('c', '0', 'c', d.f, legs[2], 'open')]
    for k, b in enumerate(branches):
        d.at(0.9 + 0.03 * k)

        def s_error(ycand, k=k):
            v = list(legs)
            v[k] = ycand
            return _tee_s(1 / v[0], 1 / v[1], v[2], d.z0) - d.s_ref
        _BranchSearch(b, s_error, d, d.cat, d.opt,
                      lambda b=b: d.tick(f'tee form, branch {b.name}')).run()
    return Model(2, d.z0, branches, d.f, d.s_data, d.opt, 'tee')


def _has_inductor(branch):
    c = branch.chosen
    return (c is not None and not c.is_trivial
            and any(sym[0] == 'L' for sym in c.component.network.symVn))


def _try_coupling(d, model, ports, before=math.inf):
    """Coupled coils for the triangle ports = (i, j, k).

    The branches i-k and j-k become the coils. What the coils and their coupling do not
    explain of the branch i-j (the capacitance between the coil ends, typically) stays a
    branch of its own. Returns (Coupled, that branch, error of the three) or None.
    """
    opt = d.opt
    by_ports = {b.ports: b for b in model.branches}
    tri = [by_ports[p] for p in _pair_ports(ports)]
    yij, yik, yjk = (b.y for b in tri)
    i, j, _ = ports
    tag = f'k{i}{j}'
    w = 2 * np.pi * d.f
    low = d.f <= d.f[0] + (d.f[-1] - d.f[0]) / 3     # where the coils are inductive

    def diff(z1, z2, rc, mutual, y_rest):
        """S difference when the three data branches are replaced by the pair (coil
        impedances z1, z2) and the branch y_rest between i and j."""
        y = d.y.copy()
        for t in tri:
            _stamp(y, t.ports, -t.y)
        _stamp_pair(y, ports, z1 + rc, z2 + rc, rc + 1j * w * mutual)
        _stamp(y, tri[0].ports, y_rest)
        return d.y_diff(y)

    # A cheap test first: the coils of the data with one coupling factor. Without real
    # coupling, or if even this does not beat the three branches as they are, no search.
    with np.errstate(all='ignore'):
        det = (yik + yij) * (yjk + yij) - yij * yij
        q11, q22, q12 = (yjk + yij) / det, (yik + yij) / det, yij / det
        l1, l2, m0 = (float(np.mean(v.imag[low] / w[low])) for v in (q11, q22, q12))
    if not (all(np.all(np.isfinite(v)) for v in (q11, q22, q12)) and l1 > 0 and l2 > 0):
        return None
    if not K_MIN <= abs(m0) / math.sqrt(l1 * l2) <= 1:
        return None
    if _rms(diff(q11, q22, 0.0, m0, np.zeros_like(yij))) >= before:
        return None

    def fit(use_rc):
        y_rest = np.zeros_like(yij)
        result = None
        for _ in range(2):                       # second pass: coils without the rest branch
            with np.errstate(all='ignore'):
                yc = yij - y_rest
                det = (yik + yc) * (yjk + yc) - yc * yc
                z11, z22, z12 = (yjk + yc) / det, (yik + yc) / det, yc / det
            if not all(np.all(np.isfinite(v)) for v in (z11, z22, z12)):
                return result
            # Coupling and common resistor: start from the low band, where the coils are
            # inductive, then take the values with the lowest S-parameter error (the
            # apparent mutual inductance grows towards the resonance).
            m0 = float(np.mean(z12.imag[low] / w[low]))
            r0 = max(float(np.mean(z12.real[low])), 0.0)
            if m0 == 0 or (use_rc and r0 == 0):
                return result
            m_grid = m0 * np.linspace(0.6, 1.6, 21)
            r_grid = r0 * np.linspace(0.0, 3.0, 13) if use_rc else np.zeros(1)
            for _ in range(8):                   # coarse grid, then finer ones around it
                _, mutual, rc = min(
                    ((_rms(diff(z11 - r, z22 - r, r, m, y_rest)), m, r)
                     for m in m_grid for r in r_grid), key=lambda t: t[0])
                dm = abs(m_grid[1] - m_grid[0])
                dr = max(r_grid[1:] - r_grid[:-1]) if use_rc else 0
                m_grid = mutual + dm * np.linspace(-1, 1, 11)
                r_grid = np.maximum(rc + dr * np.linspace(-1, 1, 11), 0) if use_rc else r_grid
            mutual, rc = float(mutual), float(rc)
            coils, z_model = [], [z11 - rc, z22 - rc]
            for n, port in enumerate((i, j)):
                coil = Branch(f'p{port}', 'm', tag + 'ab'[n], d.f, 1 / z_model[n], None)

                def s_error(ycand, n=n):
                    z = list(z_model)
                    z[n] = 1 / ycand
                    return diff(z[0], z[1], rc, mutual, y_rest)
                c = _BranchSearch(coil, s_error, d, d.cat, opt,
                                  lambda: d.tick(f'coupled coils {tag}'),
                                  accept=lambda comp: series_inductor(comp) is not None).run()
                if c is None:
                    return result
                coils.append(c.component)
                with np.errstate(all='ignore'):
                    z_model[n] = 1 / c.component.eval(d.f, 0)
            g = Coupled(ports, coils, rc, mutual, 0.0)
            if abs(g.k) > K_MAX:
                return result
            # the rest of the branch between the two ports
            y_pair = _pair_admittances(z_model[0] + rc, z_model[1] + rc, rc + 1j * w * mutual)[0]
            rest = Branch(tri[0].node1, tri[0].node2, tri[0].tag, d.f, yij - y_pair, 'open',
                          tri[0].ports)
            chosen = _BranchSearch(
                rest, lambda ycand: diff(z_model[0], z_model[1], rc, mutual, ycand),
                d, d.cat, opt, lambda: d.tick(f'branch {rest.name} beside coupled coils')).run()
            g.error = chosen.error
            if result is None or g.error < result[2]:
                result = (g, rest, g.error)
            y_rest = rest.admittance(d.f)
        return result

    best = fit(False)                            # without the common resistor first
    if best is None or best[2] > opt.target or opt.pick == 'best':
        with_rc = fit(True)
        if with_rc is not None and (best is None or with_rc[2] * opt.tolerance < best[2] - EQUAL_ERROR):
            best = with_rc
    return best


def _add_couplings(d, model):
    """Replace triangles around a failing branch between two ports by coupled coils."""
    todo = list(model.branches)
    for n, b in enumerate(todo):
        d.at(0.6 + 0.3 * n / len(todo))
        if b not in model.branches or len(b.ports) != 2 or b.group is not None:
            continue
        if not (b.target_missed or b.has_negative_element()):
            continue
        i, j = b.ports
        by_ports = {x.ports: x for x in model.branches}
        best = None
        for k in [0] + [p for p in range(1, model.nports + 1) if p not in (i, j)]:
            tri = [by_ports[p] for p in _pair_ports((i, j, k))]
            if any(t.group is not None for t in tri) or not all(map(_has_inductor, tri[1:])):
                continue
            y = d.y.copy()                       # error of the three branches as chosen
            for t in tri:
                _stamp(y, t.ports, t.admittance(d.f) - t.y)
            before = d.y_error(y)
            r = _try_coupling(d, model, (i, j, k), before)
            if r is not None and r[2] < before and (best is None or r[2] < best[2]):
                best = r
        if best is not None:
            g, rest, _ = best
            model.couplings.append(g)
            model.branches[model.branches.index(b)] = rest
            for t in (by_ports[p] for p in _pair_ports(g.ports)[1:]):
                t.group = g


def _missed(model):
    return bool(model.warnings())


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
    if opt.form == 'tee' and network.nports != 2:
        raise ValueError('the tee form is for 2-port data only')
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
    s_data = network.s[keep]
    n = network.nports
    eye = np.eye(n)
    done = [0.0]                                 # progress: fraction of the work done

    def tick(text=''):
        if cancel is not None and cancel():
            raise Cancelled()
        if progress is not None:
            progress(done[0], text)

    def at(fraction):
        done[0] = max(done[0], min(fraction, 0.99))
        if progress is not None:
            progress(done[0], 'coupled coils' if fraction >= 0.6 else '')

    def run(delay):
        s_core = shift_ports(s_data, f, -delay) if delay else s_data
        try:
            y = np.linalg.solve(z0 * (eye + s_core), eye - s_core)
        except np.linalg.LinAlgError:            # ports shorted together: no Y-matrix;
            sc = s_core * (1 - 1e-12)            # a hair of loss gives a very large one
            y = np.linalg.solve(z0 * (eye + sc), eye - sc)
        y_sym = (y + np.transpose(y, (0, 2, 1))) / 2
        d = _Data(f, y_sym, s_core, z0, cat, opt, tick)
        d.at = at
        model = None
        if opt.form in ('auto', 'pi'):
            model = _extract_pi(d)
            if opt.coupling:
                _add_couplings(d, model)
        if opt.form == 'tee' or (opt.form == 'auto' and n == 2 and _missed(model)):
            tee = _extract_tee(d)
            if model is None:
                if tee is None:
                    raise ValueError('the data have no Z-matrix: the tee form cannot be used')
                model = tee
            elif tee is not None and tee.error() < model.error():
                model = tee
        model.source = source
        model.delay = delay
        model.s_data = s_data                    # the data as given, with the lines
        model.reciprocity_error = _rms(d.s_ref - s_core)
        return model

    model = run(0.0 if opt.delay == 'off' else float(opt.delay))
    if progress is not None:
        progress(1.0, 'done')
    return model


def _delay_text(delay):
    return delay if isinstance(delay, str) else f'{delay * 1e12:g} ps'


def describe(model):
    """Lines for a report: data, options, result, total error."""
    o = model.options
    lines = [f'data: {model.source or "network"}, {model.nports} port(s), '
             f'fit band {model.freq[0] / 1e9:.6g} to {model.freq[-1] / 1e9:.6g} GHz, '
             f'{model.freq.size} points',
             f'choice: {o.pick}, elements {o.require}, limits {o.target_db:g} dB (S-parameters), '
             f'{o.reactance_percent:g} % (reactance), {o.loss_percent:g} % (loss), '
             f'else {o.tolerance:g} x the best, refine {"on" if o.refine else "off"}, '
             f'up to {o.max_elements} elements per branch, delay {_delay_text(o.delay)}, '
             f'sequential {"on" if o.sequential else "off"}, '
             f'form {o.form}, coupled coils {"on" if o.coupling else "off"}']
    if model.reciprocity_error > 1e-6:
        lines.append(f'data are not reciprocal: RMS |S - S^T|/2 = '
                     f'{_db(model.reciprocity_error):.1f} dB (the model is reciprocal)')
    lines.append(model.summary())
    return lines
