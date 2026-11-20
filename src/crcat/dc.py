"""What a model does at DC: which networks conduct, which branches of the data are open,
the DC resistances of a finished netlist, and its nodes without a DC path.

At DC a capacitor is open, an inductor and a line are a short. A model fitted only at high
frequency can get a resistor where the part has none at DC, for example a resistor parallel
to the capacitor of a branch to ground: a leakage path for bias. The safeguard
(``Options.dc``) keeps such networks out of branches that are open at DC.
"""

import math

import numpy as np

__all__ = ['conducts', 'dc_class', 'dc_class_at_dc', 'low_cut', 'resistances', 'floating_nodes',
           'SHORT_KINDS']

SHORT_KINDS = 'LT'           # elements that are a short at DC
LOW_START = 0.1              # the data must start below this share of the top frequency
CAP_SLOPE = 0.6              # |Y| must rise at least this fast with frequency (1 for a C)
LOSS_SMALL = 0.02            # Re Y below this share of |Y| counts as none
LOSS_SLOPE = 1.3             # else Re Y must rise at least this fast (2 for R in series with C)
LOW_CUT = 0.1                # a change of S below this share of the error limit is no change
LOW_CUT_MIN = 3              # fewer points than this are not a DC region
MIN_POINTS = 12              # points a fit needs
DC_SIEMENS = 1e-6            # conductance at DC above which a branch conducts (1 Mohm)


def conducts(network):
    """True if a catalog network has a DC path between its two terminals (through R and L
    only)."""
    reach, grew = {1}, True
    while grew:
        grew = False
        for sym, a, b in network.connections:
            if sym[0] != 'C' and (a in reach) != (b in reach):
                reach |= {a, b}
                grew = True
    return 0 in reach


def dc_class(freq, y):
    """What the admittance ``y`` of a branch says about DC: 'open' (capacitive towards low
    frequency, no conduction), 'path' (it conducts), or None if the data start too high to
    tell."""
    freq, y = np.asarray(freq, dtype=float), np.asarray(y, dtype=complex)
    mag = np.abs(y)
    if not np.all(np.isfinite(mag)):
        return 'path'                            # a short
    if mag.max() < 1e-12:
        return 'open'
    if freq[0] > LOW_START * freq[-1]:
        return None
    low = slice(0, max(5, freq.size // 10))
    f, v = freq[low], y[low]
    m = np.abs(v)
    if np.median(v.imag) <= 0 or m.min() <= 0:
        return 'path'                            # inductive or resistive
    slope = np.polyfit(np.log(f), np.log(m), 1)[0]
    if slope < CAP_SLOPE:
        return 'path'
    re = np.abs(v.real)
    if np.median(re / m) < LOSS_SMALL:
        return 'open'
    if re.min() <= 0:
        return 'open' if np.median(re / m) < 5 * LOSS_SMALL else 'path'
    return 'open' if np.polyfit(np.log(f), np.log(re), 1)[0] >= LOSS_SLOPE else 'path'


def low_cut(freq, s, target):
    """Number of points at the low end to leave out of a fit: the DC region.

    Far below RF the S-parameters of a part do not change with frequency: they show its DC
    resistances and nothing of its L and C, whose reactances are too small to be read.
    Points there carry no information for the fit, and what is read from them as L or C is
    not physical. The region ends at the first point whose S-parameters differ from those
    of the lowest point by ``LOW_CUT`` times ``target`` (the S-parameter error limit,
    linear). 0 if there is no such region, if nothing changes at all (a resistor), or if
    too few points would be left."""
    freq, s = np.asarray(freq, dtype=float), np.asarray(s)
    change = np.abs(s - s[0]).reshape(freq.size, -1).max(axis=1)
    above = np.flatnonzero(change >= LOW_CUT * target)
    if above.size == 0:
        return 0
    k = int(above[0])
    if k < LOW_CUT_MIN or freq.size - k < MIN_POINTS:
        return 0
    return k


def dc_class_at_dc(y_dc):
    """'path' or 'open' for a branch whose admittance in the DC region is ``y_dc``."""
    return 'path' if np.real(y_dc) > DC_SIEMENS else 'open'


class _Net:
    """The netlist at DC: nodes joined by shorts, resistors between them."""

    def __init__(self, elements):
        self.parent = {}
        self.resistors = []
        for _, kind, n1, n2, value in elements:
            if kind == 'K':
                continue
            self.find(n1)
            self.find(n2)
            if kind in SHORT_KINDS or (kind == 'R' and value == 0):
                self.parent[self.find(n1)] = self.find(n2)
        for _, kind, n1, n2, value in elements:
            if kind == 'R' and value != 0:
                self.resistors.append((self.find(n1), self.find(n2), float(value)))

    def find(self, n):
        self.parent.setdefault(n, n)
        while self.parent[n] != n:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def island(self, start):
        """All (merged) nodes with a DC path to ``start``."""
        reach, grew = {self.find(start)}, True
        while grew:
            grew = False
            for a, b, _ in self.resistors:
                if (a in reach) != (b in reach):
                    reach |= {a, b}
                    grew = True
        return reach

    def resistance(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return 0.0
        nodes = self.island(a)
        if b not in nodes:
            return math.inf
        index = {n: k for k, n in enumerate(sorted(nodes - {b}))}
        g = np.zeros((len(index), len(index)))
        for n1, n2, r in self.resistors:
            if n1 in nodes and n1 != n2:
                for p, q in ((n1, n2), (n2, n1)):
                    if p in index:
                        g[index[p], index[p]] += 1 / r
                        if q in index:
                            g[index[p], index[q]] -= 1 / r
        rhs = np.zeros(len(index))
        rhs[index[a]] = 1.0
        try:
            return float(np.linalg.solve(g, rhs)[index[a]])
        except np.linalg.LinAlgError:
            return math.inf


def resistances(elements, ports):
    """DC resistance in ohm between each pair of ``ports`` and from each port to ground
    ('0'), all other ports open: {(a, b): ohm}, math.inf without a path."""
    net = _Net(elements)
    out = {}
    for i, a in enumerate(ports):
        for b in list(ports[i + 1:]) + ['0']:
            out[(a, b)] = net.resistance(a, b)
    return out


def floating_nodes(elements, ports):
    """Internal nodes without a DC path to a port or to ground: one node of each such
    island (a simulator needs a DC path there to find the operating point)."""
    net = _Net(elements)
    anchored = set()
    for n in list(ports) + ['0']:
        anchored |= net.island(n)
    out, seen = [], set(anchored)
    for n in sorted(net.parent):
        root = net.find(n)
        if root not in seen:
            seen |= net.island(root)
            out.append(n)
    return out
