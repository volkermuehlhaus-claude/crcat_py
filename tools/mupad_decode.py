"""Decoder for MATLAB symbolic objects stored in CRCat's .mat catalogs.

MATLAB saves a ``sym`` object as the string ``symobj::fromString("<base64>", 2)``. The base64
text is zlib-compressed MuPAD binary serialization. This format is undocumented; the layout
below was worked out from the CRCat catalogs and is only as complete as those files need:

* The stream is a sequence of big-endian 32-bit words. Every node starts with the word
  ``0x000040TT`` (TT = node type) followed by a zero word.
* A blob holds several top-level statements. The value is in the last one,
  ``sysassign(symobjDummyIdentifier, hold(<value>))``.
* Repeated objects are written once and later referenced by number (type 0xc9). Numbers
  are 1-based per statement. Most nodes get their number after their children
  (post-order); builtins get it before (pre-order); procedures get none.

Values in the catalogs use only ``_plus``, ``_mult``, ``_power``, integers, rationals and
binary floats. Matrices are ``Dom::Matrix`` domain elements whose columns are sparse
polynomials in a dummy variable (the exponent is the row number). ``RootOf`` objects occur
only in the zeros/poles fields and are not converted.
"""

import base64
import re
import struct
import zlib

import sympy as sp
from sympy.printing.str import StrPrinter

# statement that holds the value: sysassign(symobjDummyIdentifier, hold(...))
VALUE_STATEMENT = (bytes.fromhex('0000401a 00000000 00000003 00004014 00000000 00000002'
                                 '00000009 00000009') + b'sysassign')

PRE_ORDER = {0x1f}          # builtin

# MATLAB's root(p, z, k), kept unevaluated
ROOT = sp.Function('root')
NOT_NUMBERED = {0x21}       # procedure


class DecodeError(Exception):
    pass


class Node:
    __slots__ = ('t', 'a', 'kids')

    def __init__(self, t, a=None, kids=()):
        self.t, self.a, self.kids = t, a, kids


def payload(sym_string):
    """Return the decompressed MuPAD stream from a ``symobj::fromString(...)`` string."""
    m = re.match(r'symobj::fromString\("([^"]+)"\s*,\s*\d+\)', sym_string, re.S)
    if not m:
        raise DecodeError('not a symobj::fromString string')
    b64 = re.sub(r'\s', '', m.group(1))
    return zlib.decompress(base64.b64decode(b64 + '=' * (-len(b64) % 4)))


def _limbs_to_int(limbs):
    """GMP-style limbs, least significant first."""
    return sum(limb << (32 * i) for i, limb in enumerate(limbs))


class _Parser:
    def __init__(self, b, pos, poly_numbered):
        self.b, self.p = b, pos
        self.table = [None]
        self.poly_numbered = poly_numbered

    def word(self):
        v = struct.unpack_from('>I', self.b, self.p)[0]
        self.p += 4
        return v

    def le_words(self, n):
        v = struct.unpack_from(f'<{n}I', self.b, self.p)
        self.p += 4 * n
        return v

    def kids(self, n):
        return tuple(self.node() for _ in range(n))

    def node(self):
        start = self.p
        h = self.word()
        if h >> 8 != 0x40:
            raise DecodeError(f'bad node header {h:08x} at byte {start}')
        t = h & 0xff
        if t == 0xc9:                                   # reference
            self.p += 12
            return Node('ref', struct.unpack_from('>I', self.b, self.p - 4)[0])
        numbered = t not in NOT_NUMBERED and (t != 0x22 or self.poly_numbered)
        slot = None
        if t in PRE_ORDER:
            slot = len(self.table)
            self.table.append(None)
        self.word()                                     # always zero
        if t in (0x14, 0x0a):                           # identifier, string
            self.word()
            n = self.word()
            self.word()
            s = self.b[self.p:self.p + n].decode('latin1')
            self.p += (n + 3) & ~3
            nd = Node('ident' if t == 0x14 else 'str', s)
        elif t == 0x03:                                 # integer
            self.word()
            n = self.word()
            v = _limbs_to_int(self.le_words(n))
            sign = self.word()
            if sign == 0xffffffff:
                v = -v
            elif sign not in (0, 1):
                raise DecodeError(f'integer sign {sign:x} at byte {start}')
            nd = Node('int', v)
        elif t == 0x04:                                 # rational
            self.word()
            n1, n2 = self.word(), self.word()
            num = _limbs_to_int(self.le_words(n1))
            den = _limbs_to_int(self.le_words(n2))
            sign = self.word()
            if sign == 0xffffffff:
                num = -num
            elif sign != 1:
                raise DecodeError(f'rational sign {sign:x} at byte {start}')
            nd = Node('rat', (num, den))
        elif t == 0x02:                                 # binary float: M * 2**e
            self.word()
            e = self.word()
            if e >= 0x80000000:
                e -= 1 << 32
            prec = self.word()
            n = self.word()
            limbs = self.le_words(n)                    # most significant first
            mant = sum(limb << (32 * (n - 1 - i)) for i, limb in enumerate(limbs))
            sign = self.word()
            if sign == 0xffffffff:
                mant = -mant
            elif sign not in (0, 1):
                raise DecodeError(f'float sign {sign:x} at byte {start}')
            nd = Node('float', (mant, e, prec))
        elif t in (0x1a, 0x17, 0x24):                   # expression, list, domain element
            n = self.word()
            nd = Node({0x1a: 'expr', 0x17: 'list', 0x24: 'ext'}[t], None, self.kids(n))
        elif t == 0x23:                                 # domain: key, constructor call
            self.word()
            nd = Node('domain', None, self.kids(self.word()))
        elif t == 0x1b:                                 # function environment
            n = self.word()
            self.word()
            nd = Node('funcenv', None, self.kids(n))
        elif t == 0x1f:                                 # builtin(code, ...)
            n = self.word()
            self.word()
            code = self.word()
            nd = Node('builtin', code, self.kids(n - 1))
        elif t == 0x0e:                                 # NIL
            self.word()
            nd = Node('nil')
        elif t == 0x19:                                 # table: header node + key/value pairs
            self.word()
            hdr = self.node()
            if hdr.t != 'tabhdr':
                raise DecodeError(f'table without header at byte {start}')
            nd = Node('table', None, (hdr,) + self.kids(2 * hdr.a))
        elif t == 0xca:                                 # table header: number of entries
            self.word()
            nd = Node('tabhdr', self.word())
        elif t == 0x21:                                 # procedure
            n = self.word()
            self.p += 12
            nd = Node('proc', None, self.kids(n - 1))
        elif t == 0x1c:                                 # local variable in a procedure
            self.p += 12
            nd = Node('var')
        elif t == 0x22:                                 # polynomial (matrix column)
            self.word()
            nt, nv = self.word(), self.word()
            self.p += 8
            exps = [self.word() for _ in range(nt * nv)]
            nd = Node('poly', tuple(exps), self.kids(max(nt, 1) + nv))
        else:
            raise DecodeError(f'unknown node type 0x{t:02x} at byte {start}')
        if slot is not None:
            self.table[slot] = nd
        elif numbered:
            self.table.append(nd)
        return nd


def parse_value(stream, poly_numbered=True):
    """Parse the value statement. Returns (value node, reference table)."""
    start = stream.rfind(VALUE_STATEMENT)
    if start < 0:
        raise DecodeError('value statement not found')
    pr = _Parser(stream, start, poly_numbered)
    top = pr.node()
    if pr.p != len(stream):
        raise DecodeError(f'parse ended at byte {pr.p} of {len(stream)}')
    hold = top.kids[2]
    return hold.kids[1], pr.table


class _Converter:
    def __init__(self, table, symbols):
        self.table = table
        self.symbols = symbols
        self.cache = {}

    def resolve(self, n):
        return self.table[n.a] if n.t == 'ref' else n

    def conv(self, n):
        n = self.resolve(n)
        key = id(n)
        if key in self.cache:
            return self.cache[key]
        v = self._conv(n)
        self.cache[key] = v
        return v

    def _conv(self, n):
        if n.t == 'ident':
            if n.a not in self.symbols:
                self.symbols[n.a] = sp.Symbol(n.a)
            return self.symbols[n.a]
        if n.t == 'int':
            return sp.Integer(n.a)
        if n.t == 'rat':
            return sp.Rational(*n.a)
        if n.t == 'float':
            mant, e, prec = n.a
            if mant == 0:
                return sp.Integer(0)
            while mant % 2 == 0:
                mant //= 2
                e += 1
            if e >= 0:
                return sp.Integer(mant * 2 ** e)
            if e >= -20:
                return sp.Rational(mant, 2 ** -e)
            return sp.Float(sp.Rational(mant, 2 ** -e), 40)
        if n.t == 'expr':
            op = self.resolve(n.kids[0])
            args = [self.conv(k) for k in n.kids[1:]]
            if op.t == 'ident' and op.a == '_plus':
                return sp.Add(*args)
            if op.t == 'ident' and op.a == '_mult':
                return sp.Mul(*args)
            if op.t == 'ident' and op.a == '_power' and len(args) == 2:
                return sp.Pow(*args)
            raise DecodeError(f'unsupported operator {getattr(op, "a", op.t)!r}')
        if n.t == 'ext':
            dom = self.resolve(n.kids[0])
            key = self.resolve(dom.kids[0])
            if key.t in ('str', 'ident') and key.a == 'RootOf':
                # MATLAB root(p, z, k): the k-th root of polynomial p in z
                return ROOT(*(self.conv(k) for k in n.kids[1:]))
            if key.t in ('str', 'ident'):
                raise DecodeError(f'unsupported domain element {key.a!r}')
            nrows = self.conv(n.kids[1])
            ncols = self.conv(n.kids[2])
            cols = self.resolve(n.kids[3])
            m = sp.zeros(int(nrows), int(ncols))
            for j, c in enumerate(cols.kids):
                poly = self.resolve(c)
                if not poly.a:                         # zero column
                    continue
                for coef, row in zip(poly.kids[:len(poly.a)], poly.a):
                    m[row - 1, j] = self.conv(coef)
            return m
        raise DecodeError(f'cannot convert node {n.t!r}')


class _Printer(_Converter):
    """Print values as infix text that ``sympy.sympify`` reads (no SymPy objects built).

    Returns (precedence, text) per node; 3 = atom, 2 = power, 1 = product, 0 = sum.
    Matrices become nested lists of strings.
    """

    MIN_SHARED = 25     # shared subexpressions longer than this get a name

    def __init__(self, table, value, share=True):
        super().__init__(table, None)
        self._root = _root_text
        self.share = share
        self.uses = {}
        self.lets = []
        self._count(value)

    def _count(self, n):
        """Count how often each node is used, following references."""
        stack = [n]
        while stack:
            n = self.resolve(stack.pop())
            k = id(n)
            self.uses[k] = self.uses.get(k, 0) + 1
            if self.uses[k] == 1 and n.t in ('expr', 'ext', 'list', 'poly'):
                stack.extend(n.kids)

    def conv(self, n):
        n = self.resolve(n)
        key = id(n)
        if key in self.cache:
            return self.cache[key]
        v = self._conv(n)
        if (self.share and n.t == 'expr' and self.uses.get(key, 0) > 1 and isinstance(v[1], str)
                and len(v[1]) > self.MIN_SHARED):
            name = f't{len(self.lets) + 1}'
            self.lets.append([name, v[1]])
            v = (3, name)
        self.cache[key] = v
        return v

    @staticmethod
    def _num(v):
        return (3, str(v)) if v >= 0 else (0, str(v))

    def _conv(self, n):
        if n.t == 'ident':
            return (3, n.a)
        if n.t == 'int':
            return self._num(n.a)
        if n.t == 'rat':
            return (3, 'Rational(%d, %d)' % n.a)
        if n.t == 'float':
            mant, e, prec = n.a
            if mant == 0:
                return (3, '0')
            while mant % 2 == 0:
                mant //= 2
                e += 1
            if e >= 0:
                return self._num(mant * 2 ** e)
            if e >= -20:
                return (3, f'Rational({mant}, {2 ** -e})')
            return (3, "Float('%s', 40)" % sp.Float(sp.Rational(mant, 2 ** -e), 40))
        if n.t == 'expr':
            op = self.resolve(n.kids[0])
            args = [self.conv(k) for k in n.kids[1:]]
            name = op.a if op.t == 'ident' else None
            if name == '_plus':
                text = args[0][1]
                for _, t in args[1:]:      # a negative term is a negated product or number
                    text += f' - {t[1:]}' if t.startswith('-') else f' + {t}'
                return (0, text)
            if name == '_mult':
                neg = (0, '-1') in args
                if neg:
                    args.remove((0, '-1'))
                text = '*'.join(t if p >= 1 and not t.startswith('-') else f'({t})'
                                for p, t in args)
                return (0, '-' + text) if neg else (1, text)
            if name == '_power' and len(args) == 2:
                (pb, b), (pe, e) = args
                b = b if pb == 3 else f'({b})'
                e = e if pe == 3 else f'({e})'
                return (2, f'{b}**{e}')
            raise DecodeError(f'unsupported operator {name or op.t!r}')
        if n.t == 'ext':
            dom = self.resolve(n.kids[0])
            key = self.resolve(dom.kids[0])
            if key.t in ('str', 'ident') and key.a == 'RootOf':
                # print the polynomial inline, so that z cannot hide in a shared name
                inline = _Printer(self.table, n, share=False)
                return (3, self._root(*(inline.conv(k)[1] for k in n.kids[1:])))
            if key.t in ('str', 'ident'):
                raise DecodeError(f'unsupported domain element {key.a!r}')
            nrows = int(self.conv(n.kids[1])[1])
            ncols = int(self.conv(n.kids[2])[1])
            m = [['0'] * ncols for _ in range(nrows)]
            for j, c in enumerate(self.resolve(n.kids[3]).kids):
                poly = self.resolve(c)
                for coef, row in zip(poly.kids[:len(poly.a)], poly.a):
                    m[row - 1][j] = self.conv(coef)[1]
            return (3, m)
        raise DecodeError(f'cannot convert node {n.t!r}')


class _CoefPrinter(StrPrinter):
    """SymPy printer whose output keeps exact numbers for high-precision evaluation."""

    def _print_Rational(self, expr):
        return f'Rational({expr.p}, {expr.q})'

    def _print_Float(self, expr):
        return "Float('%s', 40)" % sp.Float(expr, 40)


def _root_text(poly, var, k):
    """root(p, z, k) -> 'root([c_n, ..., c_0], k)' with p's coefficients in z."""
    names = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', poly)) - {'Rational', 'Float'}
    loc = {name: sp.Symbol(name) for name in names}
    loc.update(Rational=sp.Rational, Float=sp.Float)
    coeffs = sp.Poly(sp.sympify(poly, locals=loc), loc[var]).all_coeffs()
    pr = _CoefPrinter()
    return 'root([' + ', '.join(pr.doprint(c) for c in coeffs) + f'], {k})'


def to_text(sym_string):
    """Convert a MATLAB ``symobj::fromString`` string to infix text.

    Returns (value, lets): value is a string for scalars or a list of rows of strings for
    matrices; lets is a list of [name, text] definitions of shared subexpressions
    (names t1, t2, ...), in the order they must be evaluated. Numbers other than integers
    are written as ``Rational(p, q)`` or ``Float('...', 40)``; MATLAB's root(p, z, k) is
    written as ``root([c_n, ..., c_0], k)`` with the coefficients of p in z.
    """
    value, table = parse_value(payload(sym_string))
    pr = _Printer(table, value)
    text = pr.conv(value)[1]
    for name, let in pr.lets:
        if re.search(r'z', let):
            raise DecodeError(f'root variable z inside shared subexpression {name}')
    return text, pr.lets


def to_sympy(sym_string, symbols=None, poly_numbered=True):
    """Convert a MATLAB ``symobj::fromString`` string to a SymPy expression or Matrix."""
    value, table = parse_value(payload(sym_string), poly_numbered)
    return _Converter(table, symbols if symbols is not None else {}).conv(value)
