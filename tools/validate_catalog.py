"""Check the converted catalog against its own netlists.

Usage:
    python tools/validate_catalog.py [catalog.json ...] [--quick]

For every network:
1. A/B: the admittance A(s)/B(s) equals a nodal analysis of the netlist (random element
   values, random complex s).
2. Synthesis round trip: the coefficients of A and B for the test element values (testEL),
   put into the synthesis equations (EL), give testEL back (in at least one solution row).
3. Transforms: element values from a transform give the same admittance as the source
   network (at least one solution row).
4. Counts, to compare with the paper (491 networks with synthesis equations).

Exits with status 1 if any check fails.
"""

import argparse
import collections
import json
import pathlib
import random
import re
import sys
import time

import mpmath
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / 'src'))
from crcat.symbolic import all_rows, compile_rows, compile_scalar, first_value  # noqa: E402

__version__ = '0.1'
DPS = 40
TOL = mpmath.mpf('1e-25')        # A/B against nodal analysis
TOL_EL = 1e-8                    # synthesis round trip and transforms (relative)


def nodal_admittance(names, nodes, values, s):
    """Input admittance between node 1 and ground (0) of an RLC netlist."""
    pairs = list(zip(nodes[0::2], nodes[1::2]))
    idx = sorted({n for p in pairs for n in p if n != 0})
    pos = {n: i for i, n in enumerate(idx)}
    y = mpmath.zeros(len(idx))
    for name, (n1, n2), v in zip(names, pairs, values):
        kind = name[0]
        g = 1 / v if kind == 'R' else 1 / (s * v) if kind == 'L' else s * v
        for a, b in ((n1, n2), (n2, n1)):
            if a:
                y[pos[a], pos[a]] += g
                if b:
                    y[pos[a], pos[b]] -= g
    z = mpmath.inverse(y)
    return 1 / z[pos[1], pos[1]]


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), mpmath.mpf('1e-300'))


class Net:
    def __init__(self, d):
        self.d = d
        self.vn = d['symVn']
        self.A = compile_scalar(self.vn + ['s'], d['A'])
        self.B = compile_scalar(self.vn + ['s'], d['B'])

    def y(self, values, s):
        return first_value(self.A, *values, s) / first_value(self.B, *values, s)


def check_ab(net, rng):
    for trial in range(3):
        vals = [mpmath.mpf(rng.uniform(0.1, 10)) for _ in net.vn] if trial else \
            [mpmath.mpf(v) for v in net.d['testEL']]
        s = mpmath.mpc(rng.uniform(0.05, 3), rng.uniform(0.05, 3))
        if rel(net.y(vals, s), nodal_admittance(net.vn, net.d['nodes'], vals, s)) > TOL:
            return False
    return True


def coefficients(net, values):
    """Numerical coefficients a_k (of s**k in A) and b_k (in B) for given element values."""
    s = sp.Symbol('s')
    loc = {n: sp.Symbol(n) for n in net.vn}
    loc['s'] = s
    sub = {loc[n]: sp.Float(str(v), DPS) for n, v in zip(net.vn, values)}
    out = {}
    for letter, text in (('a', net.d['A']), ('b', net.d['B'])):
        poly = sp.Poly(sp.sympify(text, locals=loc).subs(sub), s)
        for (k,), c in poly.terms():
            out[f'{letter}{k}'] = mpmath.mpf(str(sp.Float(c, DPS)))
    return out


def el_args(net):
    fixed = [n for n in net.vn if n not in net.d['symV']]
    return net.d['symC'], fixed


def check_el(net, compiled):
    """Synthesis round trip with the network's test values."""
    test = net.d['testEL']
    coef = coefficients(net, test)
    symc, fixed = el_args(net)
    args = [coef.get(c, mpmath.mpf(0)) for c in symc] + \
        [mpmath.mpf(test[net.vn.index(f)]) for f in fixed]
    target = [test[net.vn.index(v)] for v in net.d['symV']]
    try:
        rows = all_rows(compiled, *args)
    except (ZeroDivisionError, ValueError):
        return False
    for row in rows:
        if len(row) == len(target) and all(
                abs(complex(x) - t) <= TOL_EL * max(abs(t), 1e-30) for x, t in zip(row, target)):
            return True
    return False


def check_transform(src, dst, compiled, rng):
    test = [mpmath.mpf(v) for v in src.d['testEL']]
    if dst.vn != dst.d['symV']:
        return None                      # destination has fixed elements: not checkable
    try:
        rows = all_rows(compiled, *test)
    except (ZeroDivisionError, ValueError):
        return False
    pts = [mpmath.mpc(rng.uniform(0.05, 3), rng.uniform(0.05, 3)) for _ in range(3)]
    for row in rows:
        try:
            if all(rel(dst.y(row, s), src.y(test, s)) < TOL_EL for s in pts):
                return True
        except ZeroDivisionError:
            continue
    return False


def is_zero(entry):
    return all(t == '0' for row in entry['rows'] for t in row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data = pathlib.Path(__file__).parents[1] / 'src' / 'crcat' / 'data'
    ap.add_argument('catalogs', nargs='*', type=pathlib.Path,
                    default=[data / 'catalog_base.json', data / 'catalog_custom.json'])
    ap.add_argument('--quick', action='store_true', help='skip the transform check')
    ap.add_argument('--version', action='version', version=__version__)
    args = ap.parse_args()
    mpmath.mp.dps = DPS
    rng = random.Random(1)
    print(f'validate_catalog {__version__}: catalogs {[str(c) for c in args.catalogs]}, '
          f'dps {DPS}, quick {args.quick}')
    failed = 0
    for path in args.catalogs:
        t0 = time.time()
        cat = json.loads(path.read_text(encoding='utf-8'))
        nets = {d['id']: Net(d) for d in cat['networks']}
        stats = collections.Counter()
        for net in nets.values():
            d = net.d
            stats['networks'] += 1
            if not check_ab(net, rng):
                print(f'  A/B differs from nodal analysis: {d["id"]} {d["name"]}')
                stats['A/B failed'] += 1
            if d['EL'] is None:
                stats['no EL'] += 1
            elif is_zero(d['EL']):
                stats['EL all zero'] += 1
            else:
                stats['EL'] += 1
                symc, fixed = el_args(net)
                if check_el(net, compile_rows(symc + fixed, d['EL'])):
                    stats['EL round trip ok'] += 1
                else:
                    print(f'  synthesis round trip failed: {d["id"]} {d["name"]}')
                    stats['EL failed'] += 1
            for tr in d['same']:
                if 'status' in tr:
                    stats['transform status ' + tr['status']] += 1
                    continue
                stats['transform equations'] += 1
                if tr['to'] == d['id']:
                    stats['transform to self'] += 1
                if args.quick:
                    continue
                ok = check_transform(net, nets[tr['to']], compile_rows(d['symVn'], tr), rng)
                if ok is None:
                    stats['transform not checkable'] += 1
                elif ok:
                    stats['transform ok'] += 1
                else:
                    stats['transform failed'] += 1
                    print(f'  transform failed: {d["id"]} {d["name"]} -> {tr["to"]}')
        failed += stats['A/B failed'] + stats['EL failed'] + stats['transform failed']
        print(f'{path.name} ({time.time() - t0:.0f} s):')
        for k in sorted(stats):
            print(f'  {k}: {stats[k]}')
    print('FAILED' if failed else 'all checks passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
