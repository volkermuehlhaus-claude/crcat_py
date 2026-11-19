"""Evaluate catalog expressions numerically.

The catalog stores expressions as Python-syntax text (see tools/convert_mat.py). Here they
are compiled once into plain Python functions that work on mpmath numbers, so high-precision
evaluation needs no SymPy at run time.

Text conventions: ``Rational(p, q)`` and ``Float('...', 40)`` for non-integer numbers,
shared subexpressions as ``let`` definitions ``t1, t2, ...``, and ``root([c_n, ..., c_0], k)``
for MATLAB's root(p, z, k), the k-th root of a polynomial. MATLAB's numbering of the roots
is not documented, so every root is tried: an evaluation returns all candidate results.
"""

import itertools

import mpmath

__all__ = ['Compiled', 'compile_rows']


class _RootContext:
    """Supplies root values for one evaluation; records the polynomials it met."""

    def __init__(self, choice):
        self.choice = choice
        self.seen = {}

    def root(self, coeffs, k):
        coeffs = [mpmath.mpmathify(c) for c in coeffs]
        while len(coeffs) > 1 and coeffs[0] == 0:
            coeffs = coeffs[1:]
        if len(coeffs) < 2:
            raise ZeroDivisionError('root of a constant polynomial')
        key = tuple(coeffs)
        roots = _roots(key)
        self.seen.setdefault(key, len(roots))
        return roots[self.choice.get(key, 0)]


_ROOT_CACHE = {}


def _roots(key):
    r = _ROOT_CACHE.get(key)
    if r is None:
        r = mpmath.polyroots(list(key), maxsteps=200, extraprec=2 * mpmath.mp.prec)
        r = sorted((mpmath.mpmathify(x) for x in r),
                   key=lambda x: (float(mpmath.re(x)), float(mpmath.im(x))))
        if len(_ROOT_CACHE) > 10000:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[key] = r
    return r


def _rational(p, q):
    return mpmath.mpf(p) / q


def _float(s, digits=None):
    return mpmath.mpf(s)


class Compiled:
    """Expression rows compiled into a function of the given argument names.

    Calling the object with argument values returns a list of candidate results, each a
    list of rows (lists of values). Without ``root`` there is exactly one candidate.
    """

    def __init__(self, args, rows, lets=()):
        self.args = list(args)
        self.rows = rows
        self.has_root = any('root(' in t for row in rows for t in row) or \
            any('root(' in t for _, t in lets)
        # A row whose evaluation fails (division by zero at these values, ...) is left out;
        # a failing shared subexpression becomes NaN, so only rows that use it are lost.
        body = []
        for name, text in lets:
            body += ['    try:', f'        {name} = {text}', '    except _ERRORS:',
                     f'        {name} = _NAN']
        body.append('    _out = []')
        for row in rows:
            body += ['    try:', f'        _out.append([{", ".join(row)}])',
                     '    except _ERRORS:', '        pass']
        body.append('    return _out')
        src = f'def _f({", ".join(self.args)}):\n' + '\n'.join(body) + '\n'
        self._ns = {'Rational': _rational, 'Float': _float, 'root': None,
                    '_ERRORS': (ZeroDivisionError, ValueError, OverflowError),
                    '_NAN': mpmath.mpf('nan')}
        exec(compile(src, '<crcat expression>', 'exec'), self._ns)
        self._f = self._ns['_f']

    def __call__(self, *values):
        if not self.has_root:
            return [self._f(*values)]
        results = []
        pending = [{}]
        while pending:
            choice = pending.pop()
            ctx = _RootContext(choice)
            self._ns['root'] = ctx.root
            out = self._f(*values)
            new = [k for k in ctx.seen if k not in choice]
            if new:
                key = new[0]
                pending.extend({**choice, key: i} for i in range(ctx.seen[key]))
            else:
                results.append(out)
        return results


def compile_rows(args, entry):
    """Compile a catalog entry {'rows': ..., 'let': ...} for the argument names ``args``."""
    return Compiled(args, entry['rows'], entry.get('let', ()))


def compile_scalar(args, text):
    return Compiled(args, [[text]])


def first_value(compiled, *values):
    """Value of a scalar expression without root()."""
    return compiled(*values)[0][0][0]


def all_rows(compiled, *values):
    """All candidate rows (flattened over root choices)."""
    return list(itertools.chain.from_iterable(compiled(*values)))
