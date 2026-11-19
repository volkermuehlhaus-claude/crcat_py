"""Chain components in series and shunt (port of CRSprout.m, experimental as in CRCat)."""

import numpy as np

__all__ = ['sprout']


def sprout(freq, components, original=None):
    """Admittance of components connected one after the other.

    The first component (with values) starts the circuit. After that, components at odd
    positions (1st, 3rd, ... counting from 1) are connected in series with everything so
    far, those at even positions in shunt. ``None`` or a component without values is a
    short in series and an open in shunt, so it changes nothing. Each component uses its
    first solution.

    With ``original``, return original minus the chain's admittance: what is left to model.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    y = np.zeros(freq.size, dtype=complex)
    started = False
    for k, comp in enumerate(components, start=1):
        if comp is None or not comp.values:
            continue
        yc = comp.eval(freq, solution=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            if not started:
                y, started = yc, True
            elif k % 2:
                y = 1 / (1 / y + 1 / yc)
            else:
                y = y + yc
    if original is not None:
        return np.asarray(original, dtype=complex).ravel() - y
    return y
