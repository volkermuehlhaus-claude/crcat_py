"""The catalog of RLC networks (port of CRCat's @CRCat class, without catalog generation).

The data are the JSON files in ``crcat/data``, converted from CRCat's MATLAB catalogs by
``tools/convert_mat.py``. Networks are numbered from 1 like CRCat's Ids ("CR500").
"""

import functools
import json
import pathlib

import mpmath

from .symbolic import Compiled, compile_rows

__all__ = ['Catalog', 'Network', 'load']

DATA = pathlib.Path(__file__).parent / 'data'

# relationship names, as in CRCat.m
SELF, SELF_NO_SYNTH = 'Self', 'SelfNoSynth'
EQUIV, EQUIV_NO_EQN = 'Equiv', 'EquivNoEqn'
DEGEN, DEGEN_NO_EQN = 'Degen', 'DegenNoEqn'
REGEN, REGEN_NO_EQN = 'Regen', 'RegenNoEqn'
COMPOUND, COMPOUND_NO_EQN = 'Compound', 'CompoundNoEqn'
NEED_SYNTH = 'NeedSynth'
TRIVIAL, NO_XFRM, NIL, EQN_NOT_FOUND = 'Trivial', 'NoXfrm', 'Nil', 'EqnNotFound'
JUST_SAME_SIG, NO_RELATION = 'JustSameSig', 'NoRelation'
SHOULD_NOT_BE_POSSIBLE = 'ShouldNotBePossible'


def _flags(sig):
    a, b = sig.split('-')
    return [c == '1' for c in a], [c == '1' for c in b]


class Network:
    """One catalog network: netlist, admittance A(s)/B(s), synthesis and transform equations."""

    def __init__(self, d, catalog, id_offset=0):
        self.catalog = catalog
        self.id = d['id'] + id_offset
        self.name = d['name']
        self.symV = d['symV']          # elements with synthesis equations
        self.symVn = d['symVn']        # all elements
        self.symC = d['symC']          # fitted coefficients used by the synthesis equations
        self.flag_sig = d['flagSig']
        self.flag_a, self.flag_b = _flags(d['flagSig'])
        self.nodes = d['nodes']
        self.test_values = d['testEL']
        self.A, self.B = d['A'], d['B']
        self._acoef, self._bcoef = d['Acoef'], d['Bcoef']
        self._el = d['EL']
        self.same = [dict(t, to=t['to'] + id_offset) for t in d['same']]

    def __repr__(self):
        return f'<Network CR{self.id} {self.name}: {" ".join(self.symVn)}>'

    @property
    def fixed(self):
        """Elements without synthesis equations; their values must be given for a fit."""
        return [v for v in self.symVn if v not in self.symV]

    @property
    def has_equations(self):
        """True if the network has synthesis equations (CRCat's non-empty symEL)."""
        return self._el is not None

    @property
    def can_synthesize(self):
        """True if the synthesis equations are usable (35 base networks have only zeros)."""
        return self._el is not None and not all(
            t == '0' for row in self._el['rows'] for t in row)

    @property
    def connections(self):
        """List of (element, node1, node2); port between node 1 and ground (node 0)."""
        return [(v, self.nodes[2 * i], self.nodes[2 * i + 1]) for i, v in enumerate(self.symVn)]

    # --- compiled expressions -------------------------------------------------------

    @functools.cached_property
    def _poly(self):
        return Compiled(self.symVn, [self._acoef, self._bcoef])

    @functools.cached_property
    def _synthesis(self):
        return compile_rows(self.symC + self.fixed, self._el)

    @functools.lru_cache(maxsize=None)
    def _transform(self, to):
        for t in self.same:
            if t['to'] == to and 'rows' in t:
                return compile_rows(self.symVn, t)
        return None

    def polynomials(self, values):
        """Coefficients of A(s) and B(s), highest power first, for element values (symVn)."""
        a, b = self._poly(*[mpmath.mpmathify(v) for v in values])[0]
        return a, b

    def admittance(self, values, s):
        """Y(s) = A(s)/B(s) for element values (symVn order) at complex s (mpmath)."""
        a, b = self.polynomials(values)
        return mpmath.polyval(a, s) / mpmath.polyval(b, s)

    def synthesize(self, coefficients, fixed=()):
        """Element values from rational-polynomial coefficients.

        ``coefficients`` are the values of ``symC`` (the flagged numerator coefficients,
        then the flagged denominator ones, highest power first, as returned by a fit).
        ``fixed`` gives values for the ``fixed`` elements. Returns a list of solutions,
        each a list of values in ``symVn`` order (mpmath numbers, possibly complex).
        """
        if not self.can_synthesize:
            return []
        fixed = [mpmath.mpmathify(v) for v in fixed]
        if len(fixed) != len(self.fixed):
            raise ValueError(f'{self.name}: {len(self.fixed)} fixed element values needed '
                             f'({", ".join(self.fixed)}), {len(fixed)} given')
        args = [mpmath.mpmathify(c) for c in coefficients] + fixed
        if len(args) != len(self.symC) + len(fixed):
            raise ValueError(f'{self.name}: {len(self.symC)} coefficients needed')
        out = []
        for rows in self._synthesis(*args):
            for row in rows:
                vals = dict(zip(self.symV, row))
                vals.update(zip(self.fixed, fixed))
                full = [vals[v] for v in self.symVn]
                if not any(mpmath.isnan(x) or mpmath.isinf(x) for x in full):
                    out.append(full)
        return _unique(out)

    def transform(self, to, values):
        """Element values of network ``to`` (Id or name) equivalent to ``values`` here.

        Returns a list of solutions (symVn order of the other network), empty if there are
        no transform equations.
        """
        to = self.catalog.id(to)
        f = self._transform(to)
        if f is None:
            return []
        out = []
        for rows in f(*[mpmath.mpmathify(v) for v in values]):
            for row in rows:
                if not any(mpmath.isnan(x) or mpmath.isinf(x) for x in row):
                    out.append(list(row))
        return _unique(out)


def _unique(rows, tol=1e-12):
    out = []
    for r in rows:
        if not any(all(abs(x - y) <= tol * max(abs(x), abs(y), 1e-300) for x, y in zip(r, o))
                   for o in out):
            out.append(r)
    return out


class Catalog:
    """A list of networks; index by Id (int, from 1) or name."""

    def __init__(self, networks, units):
        self.networks = networks
        self.units = units
        self._by_name = {}
        for n in networks:
            if n.name in self._by_name:
                raise ValueError(f'duplicate network name {n.name}')
            self._by_name[n.name] = n.id

    def __len__(self):
        return len(self.networks)

    def __iter__(self):
        return iter(self.networks)

    def __getitem__(self, key):
        return self.networks[self.id(key) - 1]

    def __repr__(self):
        return f'<Catalog: {len(self)} networks>'

    def id(self, key):
        """Id for an Id or a name (KeyError if unknown)."""
        if isinstance(key, Network):
            return key.id
        if isinstance(key, str):
            if key.upper().startswith('CR') and key[2:].isdigit():
                key = int(key[2:])
            else:
                return self._by_name[key]
        key = int(key)
        if not 1 <= key <= len(self.networks):
            raise KeyError(f'no network with Id {key}')
        return key

    def name(self, key):
        return self[key].name

    # --- relations (CRCat.m) ------------------------------------------------------------

    def find_order(self, key):
        """Smallest number of elements of any network this one transforms to (FindOrder)."""
        net = self[key]
        order = len(net.symVn)
        if net.has_equations:
            return order
        for t in net.same:
            n = len(self[t['to']].symVn)
            if 'rows' in t or t.get('status') == EQN_NOT_FOUND:
                order = min(order, n)
        return order

    def _same_entry(self, frm, to):
        for t in self[frm].same:
            if t['to'] == to:
                return t
        return None

    def relationship(self, frm, to):
        """Relationship from network ``frm`` to ``to`` (port of CRCat.Relationship).

        One of Self, SelfNoSynth, Equiv, EquivNoEqn, Degen, DegenNoEqn, Regen, RegenNoEqn,
        Compound, CompoundNoEqn, NeedSynth, Rvrs... (reverse direction), Trivial,
        RvrsTrivial, NoXfrm, JustSameSig, NoRelation, ShouldNotBePossible.
        """
        frm, to = self.id(frm), self.id(to)
        a, b = self[frm], self[to]
        if a.flag_sig != b.flag_sig:
            return NO_RELATION
        fwd, rev = self._same_entry(frm, to), self._same_entry(to, frm)
        if fwd is None or rev is None:
            return NO_RELATION
        fwd_eqn, rev_eqn = 'rows' in fwd, 'rows' in rev
        fwd_exists = fwd_eqn or fwd.get('status') == EQN_NOT_FOUND
        rev_exists = rev_eqn or rev.get('status') == EQN_NOT_FOUND
        n_from, n_to = len(a.symVn), len(b.symVn)
        o_from, o_to = self.find_order(frm), self.find_order(to)

        def pick(eqn, yes, no):
            return yes if eqn else no

        if frm == to:
            return pick(fwd_eqn, SELF, SELF_NO_SYNTH)
        if fwd_exists and rev_exists:
            return pick(fwd_eqn, EQUIV, EQUIV_NO_EQN) if n_from == n_to else SHOULD_NOT_BE_POSSIBLE
        if fwd_exists:
            if o_from > o_to:
                return SHOULD_NOT_BE_POSSIBLE
            if o_from == o_to and o_from == n_from:
                return pick(fwd_eqn, EQUIV, EQUIV_NO_EQN)
            if o_from == o_to:
                return pick(fwd_eqn, DEGEN, DEGEN_NO_EQN)
            if o_from == n_from:
                return pick(fwd_eqn, REGEN, REGEN_NO_EQN)
            return pick(fwd_eqn, COMPOUND, COMPOUND_NO_EQN)
        if rev_exists:
            if o_to > o_from:
                return SHOULD_NOT_BE_POSSIBLE
            if o_from == o_to and o_to == n_to:
                return NEED_SYNTH
            if o_to == o_from:
                return 'Rvrs' + pick(rev_eqn, DEGEN, DEGEN_NO_EQN)
            if o_to == n_to:
                return 'Rvrs' + pick(rev_eqn, REGEN, REGEN_NO_EQN)
            return 'Rvrs' + pick(rev_eqn, COMPOUND, COMPOUND_NO_EQN)
        fs, rs = fwd.get('status'), rev.get('status')
        if fs == TRIVIAL:
            return TRIVIAL
        if rs == TRIVIAL:
            return 'Rvrs' + TRIVIAL
        if NO_XFRM in (fs, rs):
            return NO_XFRM
        if NIL in (fs, rs):
            return JUST_SAME_SIG
        return SHOULD_NOT_BE_POSSIBLE

    def relations(self, key):
        """Related networks as (Id, relationship), without NoXfrm, Nil, JustSameSig,
        NoRelation (CRCat.NetworkRelations)."""
        frm = self.id(key)
        out = []
        for t in self[frm].same:
            r = self.relationship(frm, t['to'])
            if r not in (NO_XFRM, NIL, JUST_SAME_SIG, NO_RELATION):
                out.append((t['to'], r))
        return out


def _read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


@functools.lru_cache(maxsize=None)
def load(which='default'):
    """Load a catalog: 'base' (623 networks), 'custom' (8 examples) or 'default' (both;
    custom Ids follow the base ones, 624...). Transforms between base and custom networks
    are not included in 'default' (CRCat recomputes them, this port does not)."""
    nets = []
    cat = Catalog.__new__(Catalog)
    units = None
    if which in ('base', 'default'):
        d = _read('catalog_base.json')
        units = d['units']
        nets += [Network(n, cat) for n in d['networks']]
    if which in ('custom', 'default'):
        d = _read('catalog_custom.json')
        units = units or d['units']
        offset = len(nets)
        nets += [Network(n, cat, offset) for n in d['networks']]
    if not nets:
        raise ValueError(f"unknown catalog {which!r}: use 'default', 'base' or 'custom'")
    Catalog.__init__(cat, nets, units)
    return cat
