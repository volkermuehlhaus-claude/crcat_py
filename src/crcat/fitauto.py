"""Automatic rational fit and interpolation (ports of CRFitAuto.m and CRInterp.m).

``fit_auto`` picks the number of polynomial terms and the working precision until the
rational fit reproduces the data to a given RMS error; ``interpolate`` uses it to evaluate
network data at other frequencies (also a limited extrapolation). The result is a rational
function of s = j*2*pi*f with real coefficients, so it is causal.
"""

import numpy as np

from .fit import TOO_FEW_POINTS, FitResult, crfit, evaluate

NOT_CONVERGED = -5     # the error limit was not reached

__all__ = ['fit_auto', 'interpolate']


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.abs(np.asarray(a) - np.asarray(b)) ** 2)))


def fit_auto(y, freq, max_error=1e-5, verbose=False):
    """Rational fit with automatic order and precision (CRFitAuto).

    Returns (FitResult, digits). Needs at least 10 frequencies. ``max_error`` (RMS, in
    the units of y) between 1e-10 and 1e-1. If the limit is not reached, the FitResult
    has a negative code (-2: not enough frequencies for the needed order, -5: no
    convergence in 10 tries).
    """
    y = np.asarray(y, dtype=complex).ravel()
    freq = np.asarray(freq, dtype=float).ravel()
    n = y.size
    if freq.size != n:
        raise ValueError(f'{n} data points but {freq.size} frequencies')
    if n < 10:
        raise ValueError(f'at least 10 frequencies needed for an automatic fit, {n} given')
    if not 1e-10 <= max_error <= 1e-1:
        raise ValueError('max_error must be between 1e-10 and 1e-1')

    err1 = err2 = 2 * max_error
    tries, try_limit = 0, 10
    terms_max = n // 2
    digits = terms = 0
    r1 = None
    while err1 > max_error and tries < try_limit:
        if tries == 0:
            # first guess from the sign changes of the slope of the (binned) imaginary part
            bin_size = 10 if n >= 100 else round(0.1 * n)
            size = n - n % bin_size
            summed = y.imag[:size].reshape(-1, bin_size).sum(axis=1)
            changes = int(np.sum(np.abs(np.diff((np.diff(summed) > 0).astype(int)))))
            digits = 32 + 5 * changes
            terms = round(0.3 * digits)
        elif err2 > max_error:
            digits = round(1.3 * digits)            # more precision
        else:
            terms = round(1.3 * terms)              # more terms
        if terms > terms_max:
            terms = terms_max
            tries = try_limit - 1                   # one more try
        if verbose:
            print(f'  {digits} digits, {2 * terms} polynomial terms: ', end='', flush=True)
        r1 = crfit(y, freq, [1] * terms, [1] * terms, digits)
        if r1.ok:
            fit1 = evaluate(r1.a, r1.b, freq, digits)
            err1 = _rmse(y, fit1)
            if verbose:
                print(f'fit 1 RMS error {err1:.3g}', end='', flush=True)
        else:
            terms = round(1.3 * terms)
            err2 = 2 * max_error
            if verbose:
                print(f'fit 1 failed (code {r1.code})', end='', flush=True)
        if r1.ok and err1 > max_error:
            # refit the fit: if the precision suffices, this gives the same values
            r2 = crfit(fit1, freq, [1] * terms, [1] * terms, digits)
            if r2.ok:
                err2 = _rmse(fit1, evaluate(r2.a, r2.b, freq, digits))
                if verbose:
                    print(f', fit 2 RMS error {err2:.3g}', end='')
            else:
                err2 = 2 * max_error
        if verbose:
            print()
        tries += 1
    if r1 is not None and r1.ok and err1 < max_error:
        return r1, digits
    if terms >= terms_max:
        code = TOO_FEW_POINTS
    else:
        code = r1.code if r1 is not None and not r1.ok else NOT_CONVERGED
    return FitResult([], [], code), digits


def interpolate(network, freq_new, max_error=1e-5, verbose=False):
    """Network data (scikit-rf Network) at new frequencies (Hz) via rational fits (CRInterp).

    Every S-parameter is fitted separately with :func:`fit_auto` (frequencies in GHz for
    the fit) and evaluated at ``freq_new``. Raises RuntimeError if a fit fails.
    """
    import skrf
    freq_new = np.asarray(freq_new, dtype=float).ravel()
    f = network.f / 1e9
    nports = network.nports
    s_new = np.empty((freq_new.size, nports, nports), dtype=complex)
    for i in range(nports):
        for j in range(nports):
            if verbose:
                print(f'S{i + 1}{j + 1}:')
            r, digits = fit_auto(network.s[:, i, j], f, max_error, verbose)
            if not r.ok:
                raise RuntimeError(f'automatic fit failed for S{i + 1}{j + 1} (code {r.code})')
            s_new[:, i, j] = evaluate(r.a, r.b, freq_new / 1e9, digits)
    return skrf.Network(frequency=skrf.Frequency.from_f(freq_new, unit='hz'), s=s_new,
                        z0=network.z0[0], name=network.name)
