"""Direct least-squares fit of a complex rational polynomial (port of CRCat's CRFit.m).

Given admittance Y at frequencies f, find real coefficients of

    Yfit(s) = A(s) / B(s),  s = j*2*pi*f,

that minimize sum |B(s) Y - A(s)|^2 (Levy's linearized error). The normal equations are
built from frequency moments and solved directly in high-precision arithmetic (mpmath, like
MATLAB vpa in the original), which is what makes large polynomial orders usable.

Frequencies and element values may be in any consistent units, e.g. GHz, nH, nF, ohm.
"""

import mpmath
import numpy as np

__all__ = ['FitResult', 'crfit', 'evaluate', 'DEFAULT_DPS']

DEFAULT_DPS = 32        # MATLAB's default digits()

# return codes as in CRFit.m
OK = 1
BAD_INPUT = -1          # frequency and data vectors do not match
TOO_FEW_POINTS = -2     # not enough frequencies (or no coefficients) for the fit
SINGULAR = -3           # the normal equations are singular


class FitResult:
    """Fitted coefficients, highest power first (numpy polyval order), as mpmath numbers.

    ``a`` and ``b`` have the lengths of the flags passed to :func:`crfit`; coefficients that
    were not fitted are zero. One coefficient of ``b`` is normalized to 1. ``code`` is 1 if
    the fit worked, negative otherwise (then ``a`` and ``b`` are empty).
    """

    def __init__(self, a, b, code):
        self.a, self.b, self.code = a, b, code

    @property
    def ok(self):
        return self.code == OK

    def __repr__(self):
        return f'FitResult(code={self.code}, a={[mpmath.nstr(x, 6) for x in self.a]}, ' \
               f'b={[mpmath.nstr(x, 6) for x in self.b]})'


def _flags(flag):
    return [abs(float(x)) > 1e-4 for x in flag]


def crfit(y, freq, flag_a, flag_b, dps=DEFAULT_DPS):
    """Least-squares rational fit of admittance ``y`` at frequencies ``freq``.

    ``flag_a`` / ``flag_b`` mark (nonzero) the numerator / denominator coefficients to fit,
    highest power first; their lengths set the maximum number of terms. ``dps`` is the
    working precision in decimal digits.
    """
    y = np.asarray(y, dtype=complex).ravel()
    freq = np.asarray(freq).ravel()
    if y.shape != freq.shape or y.size == 0:
        return FitResult([], [], BAD_INPUT)
    fa0, fb0 = _flags(flag_a), _flags(flag_b)
    fa, fb = list(fa0), list(fb0)

    # Remove leading zeros (high-order terms that are zero).
    lead_a = fa.index(True) if True in fa else len(fa)
    lead_b = fb.index(True) if True in fb else len(fb)
    fa, fb = fa[lead_a:], fb[lead_b:]
    # Remove trailing zeros only if both DC terms are zero.
    trail = 0
    while fa and fb and not fa[-1] and not fb[-1]:
        fa, fb = fa[:-1], fb[:-1]
        trail += 1

    n, m = len(fa), len(fb)
    if y.size < n + m or n < 1 or m < 1:
        return FitResult([], [], TOO_FEW_POINTS)

    with mpmath.workdps(dps):
        full = _fill_matrix(y, freq, n, m)
        sol = _solve(full, fa + fb, n)
        if sol is None:
            return FitResult([], [], SINGULAR)
        a = [mpmath.mpf(0)] * lead_a + sol[:n] + [mpmath.mpf(0)] * trail
        b = [mpmath.mpf(0)] * lead_b + sol[n:] + [mpmath.mpf(0)] * trail
    return FitResult(a, b, OK)


def _cond(m):
    try:
        return mpmath.mnorm(m, 1) * mpmath.mnorm(mpmath.inverse(m), 1)
    except ZeroDivisionError:
        return mpmath.inf


def _scale(full):
    """Symmetric diagonal scaling 1/sqrt(diag): the normal matrix spans many decades when
    frequencies are in Hz, and mpmath's LU solver would call it singular. The scaled
    system has the same solution and a much smaller condition number."""
    return [1 / mpmath.sqrt(abs(full[i, i])) if full[i, i] != 0 else mpmath.mpf(1)
            for i in range(full.rows)]


def _sub(full, idx, d):
    k = len(idx)
    out = mpmath.matrix(k, k)
    for i, r in enumerate(idx):
        for j, c in enumerate(idx):
            out[i, j] = full[r, c] * d[r] * d[c]
    return out


def _solve(full, flags, n):
    """Normalize one denominator coefficient to 1 and solve for the rest (CRFit.m)."""
    flags = list(flags)
    size = len(flags)
    d = _scale(full)
    best = size - 3                       # 0-based: the w^2 coefficient (CRFit.m default)
    count_limit = size
    if count_limit < 100:
        cond_ok = mpmath.mpf('1e10')
        best_cond = mpmath.inf
        count = 0
        if count_limit > 10:
            count_limit = 3               # cond() gets slow for big matrices
        best = None
        for i in range(size - 1, n - 1, -1):          # start with the DC term of B
            if best_cond >= cond_ok and flags[i] and count < count_limit:
                count += 1
                flags[i] = False
                c = _cond(_sub(full, [k for k in range(size) if flags[k]], d))
                if c < best_cond:
                    best, best_cond = i, c
                flags[i] = True
    if best is None or not flags[best]:
        return None
    flags[best] = False
    idx = [k for k in range(size) if flags[k]]
    work = _sub(full, idx, d)
    rhs = mpmath.matrix([-full[r, best] * d[r] for r in idx])
    try:
        x = mpmath.lu_solve(work, rhs)
    except ZeroDivisionError:
        return None
    if any(mpmath.isnan(v) or mpmath.isinf(v) for v in x):
        return None
    sol = [mpmath.mpf(0)] * size
    for k, v in zip(idx, x):
        sol[k] = v * d[k]
    sol[best] = mpmath.mpf(1)
    return sol


def _fill_matrix(y, freq, n, m):
    """Normal-equation matrix of the linearized fit (FillMatrix in CRFit.m)."""
    w = [2 * mpmath.pi * mpmath.mpf(float(f)) for f in freq]
    g = [mpmath.mpf(float(v.real)) for v in y]
    bb = [mpmath.mpf(float(v.imag)) for v in y]
    w2 = [x * x for x in w]
    ymag2 = [gi * gi + bi * bi for gi, bi in zip(g, bb)]
    amax = max(n, m)

    wsum, gsum, ysum, bsum = [], [], [], []
    wp = [mpmath.mpf(1)] * len(w)                 # even powers 2*(i-1)
    for _ in range(amax):
        wsum.append(mpmath.fsum(wp))
        gsum.append(mpmath.fdot(g, wp))
        ysum.append(mpmath.fdot(ymag2, wp))
        wp = [a * b for a, b in zip(wp, w2)]
    wp = list(w)                                  # odd powers 2*i-1
    for _ in range(amax):
        bsum.append(mpmath.fdot(bb, wp))
        wp = [a * b for a, b in zip(wp, w2)]

    nm = n + m
    full = mpmath.matrix(nm, nm)

    def put(r1, c1, v):                           # 1-based indices as in CRFit.m
        full[r1 - 1, c1 - 1] = v

    for icol in range(1, n + 1):                  # upper left, n x n
        isrc = icol
        for irow in range(icol, n + 1, 2):
            v = wsum[isrc - 1]
            put(n - irow + 1, n - icol + 1, -v if (irow - icol) % 4 == 2 else v)
            isrc += 1
    for icol in range(1, m + 1):                  # lower right, m x m
        isrc = icol
        for irow in range(icol, m + 1, 2):
            v = ysum[isrc - 1]
            put(nm - irow + 1, nm - icol + 1, -v if (irow - icol) % 4 == 2 else v)
            isrc += 1
    for icol in range(1, amax + 1):               # upper right, G sums
        isrc = icol
        for irow in range(icol, amax + 1, 2):
            v = gsum[isrc - 1] if (irow - icol) % 4 == 2 else -gsum[isrc - 1]
            if irow <= n and icol <= m:
                put(n - irow + 1, nm - icol + 1, v)
            if icol <= n and irow <= m:
                put(n - icol + 1, nm - irow + 1, v)
            isrc += 1
    for icol in range(1, amax):                   # upper right, B sums
        isrc = icol
        for irow in range(icol + 1, amax + 1, 2):
            v = bsum[isrc - 1] if (irow - icol) % 4 == 3 else -bsum[isrc - 1]
            if irow <= n and icol <= m:
                put(n - irow + 1, nm - icol + 1, v)
            if icol <= n and irow <= m:
                put(n - icol + 1, nm - irow + 1, -v)
            isrc += 1
    for icol in range(nm - 1):                    # make it symmetric
        for irow in range(icol + 1, nm):
            full[irow, icol] = full[icol, irow]
    return full


def evaluate(a, b, freq, dps=DEFAULT_DPS):
    """Evaluate A(s)/B(s) at s = j*2*pi*f (port of CREval.m).

    Horner evaluation in ``dps`` digits, so large polynomials stay accurate; returns a
    complex numpy array.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    with mpmath.workdps(dps):
        out = np.empty(freq.size, dtype=complex)
        for i, f in enumerate(freq):
            s = mpmath.mpc(0, 2 * mpmath.pi * mpmath.mpf(float(f)))
            out[i] = complex(mpmath.polyval(list(a), s) / mpmath.polyval(list(b), s))
    return out
