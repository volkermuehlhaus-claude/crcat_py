"""Data to fit, from Touchstone files (scikit-rf), with the selections of CRCat's SnP.Pull."""

import re

import numpy as np

__all__ = ['read', 'pull', 'admittance']


def read(path):
    """Read a Touchstone file (any .sNp, also Y/Z data) into a scikit-rf Network."""
    import skrf
    return skrf.Network(str(path))


def _parse(kind):
    """'y11', 'y1_12', '200y21' -> (letter, row, col); leading digits are ignored."""
    m = re.fullmatch(r'\d*([yYzZS])(\d*)(?:_(\d+))?', kind.strip())
    if not m:
        raise ValueError(f'unrecognized data selection {kind!r}')
    letter, a, b = m.groups()
    if b is not None:
        row, col = int(a), int(b)
    elif a == '':
        row, col = 1, 1
    elif len(a) == 2:
        row, col = int(a[0]), int(a[1])
    else:
        raise ValueError(f'{kind!r}: give two port numbers, with "_" between multi-digit ones')
    if row == 0 and col == 0:
        raise ValueError(f'{kind!r}: both port numbers cannot be zero')
    row, col = row or col, col or row
    return letter, row, col


def pull(network, kind='y11'):
    """Data to fit from a scikit-rf Network, selected as in CRCat (SnP.Pull).

    ``kind``:
      'ymm' (or 'ym0'): pi-model admittance from port m to ground (sum of column m of Y),
      'ymn': pi-model admittance between ports m and n (-Ymn),
      'zmm': tee-model impedance from port m to the center (Zmm minus the rest of column m),
      'zmn': tee-model impedance to ground (Zmn),
      'Ymn', 'Zmn', 'Smn': the network parameter itself.
    Multi-digit ports: 'y12_3'. Returns (data, description).

    CRCat's code (commit c805a25) returns -Ymn for 'Ymn'; that is an error there, as its
    author confirmed. The minus sign belongs only to 'ymn' with m != n.
    """
    letter, row, col = _parse(kind)
    n = network.nports
    if row > n or col > n:
        raise ValueError(f'{kind!r}: the data has only {n} ports')
    i, j = row - 1, col - 1
    if letter == 'y':
        y = network.y
        if i == j:
            return y[:, :, j].sum(axis=1), f'Pi admit port {row} to gnd'
        return -y[:, i, j], f'Pi admit port {col} to {row}'
    if letter == 'z':
        z = network.z
        if i == j:
            return 2 * z[:, i, j] - z[:, :, j].sum(axis=1), f'Tee imped port {row} to center'
        return z[:, i, j], f'Tee imped port {col} and {row} string to gnd'
    if letter == 'Y':
        return network.y[:, i, j], f'Y-Param port {col} to {row}'
    if letter == 'Z':
        return network.z[:, i, j], f'Z-Param port {col} to {row}'
    return network.s[:, i, j], f'S-Param port {col} to {row}'


def admittance(data, kind, z0=50.0):
    """Convert pulled data to the admittance that is fitted (as CRSeed does):
    impedance -> 1/Z, S-parameter -> (1 - S) / (1 + S) / z0."""
    letter = _parse(kind)[0]
    data = np.asarray(data, dtype=complex)
    if letter in 'zZ':
        return 1 / data
    if letter == 'S':
        return (1 - data) / (1 + data) / z0
    return data
