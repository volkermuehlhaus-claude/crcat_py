"""Find the catalog networks that best fit measured or simulated data (port of CRSeed.m /
CRSeedAddNoise.m).

Every network with synthesis equations is fitted to the data (rational fit, then the
closed-form synthesis), and the networks are ranked by RMS error of the admittance.
Fitting is done in Hz, so element values are in ohm, H and F.
"""

import pathlib
import re
import sys

import numpy as np

from . import catalog as _catalog
from . import data as _data
from .component import Component, use_ascii_units
from .fit import DEFAULT_DPS, crfit
from .sprout import sprout as _sprout

__all__ = ['seed', 'SeedResult', 'SeedEntry', 'Cancelled']


class SeedEntry:
    """One fitted network: ``component`` (solutions sorted by error), ``errors`` (RMS error
    of each solution), ``error`` (best), ``passive`` (Re Y >= 0 at all frequencies) and
    ``physical`` (all element values real and positive), both of the best solution."""

    def __init__(self, component, errors, passive=None):
        self.component = component
        self.errors = errors
        self.error = float(errors[0])
        self._passive = passive
        self.physical = component.is_physical(0)

    @property
    def passive(self):
        """Passivity of the best solution (computed when first asked for)."""
        if self._passive is None:
            self._passive = self.physical or self.component.is_passive(0)
        return self._passive

    @property
    def id(self):
        return self.component.network.id

    @property
    def name(self):
        return self.component.network.name

    def __repr__(self):
        p = '*' if self.passive else ' '
        return f'<CR{self.id}{p} {self.name} rms {self.error:.3g}: {self.component.format()}>'


class SeedResult:
    """Result of :func:`seed`: ``entries`` sorted by error, plus the data that was fitted."""

    def __init__(self, entries, skipped, freq, y, nfit, description, source):
        self.entries = sorted(entries, key=lambda e: e.error)
        self.skipped = skipped            # list of (Id, reason)
        self.freq, self.y, self.nfit = freq, y, nfit
        self.description, self.source = description, source

    def best(self, n=10, passive=False, physical=False):
        """The n best entries; only passive ones, or only ones with positive real
        element values (``physical``), if asked."""
        e = [x for x in self.entries
             if (x.passive or not passive) and (x.physical or not physical)]
        return e[:n]

    def report(self, n=100, n_passive=20):
        """Text summary like CRSeed's: the best fits, '*' marks passive models."""
        lines = [f'Fitting {self.description}' + (f', {self.source}' if self.source else ''),
                 f'{self.nfit} of {len(self.freq)} frequencies fitted, '
                 f'{len(self.entries)} networks fitted, {len(self.skipped)} skipped']
        for title, entries in ((f'Up to the {n} best fits (Id, RMS error; * = passive)',
                                self.best(n)),
                               (f'Up to the {n_passive} best passive fits',
                                self.best(n_passive, passive=True))):
            lines.append(title + ':')
            items = [f'({e.id}{"*" if e.passive else " "} {e.error:.3g})' for e in entries]
            for k in range(0, len(items), 10):
                lines.append('  ' + ' '.join(items[k:k + 10]))
        lines.append('Best passive models:')
        for e in self.best(10, passive=True):
            lines.append(f'  CR{e.id} {e.name}, rms {e.error:.3g}: {e.component.format()}')
        lines.append('Best models with positive element values:')
        for e in self.best(10, physical=True):
            lines.append(f'  CR{e.id} {e.name}, rms {e.error:.3g}: {e.component.format()}')
        return '\n'.join(lines)

    def plot(self, ids=None, path=None, z0=50.0):
        """Plot data and fitted models: reflection coefficient of the admittance (dB, against
        z0) and phase of the admittance. ``ids``: Ids to plot (default: the best model with
        positive element values and the best passive one)."""
        import matplotlib.pyplot as plt
        if ids is None:
            ids = [e.id for e in self.best(1, physical=True) + self.best(1, passive=True)]
            ids = list(dict.fromkeys(ids)) or [self.entries[0].id]
        y0 = 1 / z0
        f = self.freq
        fg = f / 1e9
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

        def draw(y, label, **kw):
            with np.errstate(divide='ignore'):
                ax1.plot(fg, 20 * np.log10(np.abs((y0 - y) / (y0 + y))), label=label, **kw)
            ax2.plot(fg, np.degrees(np.angle(y)), **kw)

        draw(self.y, 'data', color='k', lw=2.5, alpha=0.5)
        by_id = {e.id: e for e in self.entries}
        for i in ids:
            e = by_id[i]
            draw(e.component.eval(f, 0), f'CR{e.id} {e.name} (rms {e.error:.3g})')
        if self.nfit < len(f):
            for ax in (ax1, ax2):
                ax.axvline(fg[self.nfit - 1], color='gray', ls=':')
        ax1.set_ylabel('|reflection| (dB)')
        ax2.set_ylabel('phase of Y (deg)')
        ax2.set_xlabel('frequency (GHz)')
        ax1.set_title(self.description)
        ax1.legend(fontsize=8)
        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=120)
            plt.close(fig)
        return fig


def _load_data(data, fit, noise_floor_db, rng):
    """-> (freq in Hz, admittance to fit, description, source, z0)."""
    source = ''
    if isinstance(data, (str, pathlib.Path)):
        source = str(data)
        data = _data.read(data)
    if isinstance(data, tuple):
        freq, y = (np.asarray(x).ravel() for x in data)
        if _data._parse(fit)[0] not in 'yY':
            raise ValueError('for (freq, y) data use fit="y"')
        return freq.astype(float), y.astype(complex), 'admittance', source, 50.0
    net = data.copy()
    z0 = float(np.real(net.z0[0, 0])) or 50.0
    if noise_floor_db is not None:
        # SnP.AddNoise: Gaussian noise on S, standard deviation 20 dB below the floor
        sd = 10 ** ((noise_floor_db - 20) / 20)
        net.s = net.s + sd * (rng.standard_normal(net.s.shape) + 1j * rng.standard_normal(net.s.shape))
    pulled, desc = _data.pull(net, fit)
    return net.f.astype(float), _data.admittance(pulled, fit, z0), desc, source, z0


class Cancelled(Exception):
    """Raised when the ``cancel`` callback of a search returns True."""


def seed(data, catalog=None, fit='y11', ids=None, sprout=None, noise_floor_db=None,
         seed_rng=None, dps=DEFAULT_DPS, verbose=False, max_elements=None, progress=None,
         cancel=None):
    """Fit catalog networks to data and rank them (CRSeed).

    ``data``: Touchstone file name, scikit-rf Network, or (freq in Hz, admittance).
    ``fit``: what to fit, see :func:`crcat.data.pull` ('y11' pi-model shunt admittance of
    port 1, 'y21' series admittance, 'z11', 'Y11', 'S11', ...); a leading number fits only
    that many of the first frequencies (all are kept for plotting), e.g. '200y11'.
    ``ids``: Ids to try (default: all). ``sprout``: components already modeled; their
    chained admittance (see :func:`crcat.sprout.sprout`) is subtracted first.
    ``noise_floor_db``: add noise to the S-parameters first (CRSeedAddNoise), e.g. -40.
    ``max_elements``: skip networks with more elements. ``progress(fraction)`` is called
    while the search runs; if ``cancel()`` returns True the search stops with Cancelled.
    """
    cat = catalog or _catalog.load()
    rng = np.random.default_rng(seed_rng)
    freq, y, desc, source, z0 = _load_data(data, fit, noise_floor_db, rng)
    m = re.match(r'\d+', fit)
    nfit = min(int(m.group()), len(freq)) if m and int(m.group()) > 0 else len(freq)
    if sprout:
        y = _sprout(freq, sprout, y)
    ff, yf = freq[:nfit], y[:nfit]

    fits = {}                  # coefficient fit per signature
    entries, skipped = [], []
    nets = [cat[i] for i in ids] if ids is not None else list(cat)
    for k, net in enumerate(nets):
        if cancel is not None and cancel():
            raise Cancelled()
        if progress is not None and k % 10 == 0:
            progress(k / len(nets))
        if max_elements is not None and len(net.symVn) > max_elements:
            skipped.append((net.id, f'more than {max_elements} elements'))
            continue
        if not net.can_synthesize:
            skipped.append((net.id, 'no synthesis equations'))
            continue
        if net.flag_sig not in fits:
            fits[net.flag_sig] = crfit(yf, ff, net.flag_a, net.flag_b, dps)
        r = fits[net.flag_sig]
        if not r.ok:
            skipped.append((net.id, f'fit failed (code {r.code})'))
            continue
        comp = Component(net)
        try:
            errors = comp.fit(yf, ff, dps=dps, coefficients=r)
        except (ValueError, ZeroDivisionError) as ex:
            skipped.append((net.id, f'synthesis failed: {ex}'))
            continue
        if errors is None:
            skipped.append((net.id, 'no valid element values'))
            continue
        e = SeedEntry(comp, errors)
        entries.append(e)
        if verbose:
            print(f'CR{e.id} {e.name}: rms {e.error:.4g}{" passive" if e.passive else ""}: '
                  f'{comp.format()}')
    return SeedResult(entries, skipped, freq, y, nfit, desc, source)


def main(argv=None):
    """Command line: crcat-seed <touchstone file> [options]."""
    import argparse

    from . import __version__
    ap = argparse.ArgumentParser(
        prog='crcat-seed',
        description='Find the RLC networks (up to five elements) that best fit S/Y/Z data '
                    '(CRCat catalog search).')
    ap.add_argument('file', help='Touchstone file (.sNp, S, Y or Z data)')
    ap.add_argument('--fit', default='y11',
                    help="data to fit: y11 (pi shunt admittance of port 1, default), y21 "
                         "(pi series admittance), z11, Y11, Z11, S11 ...; a leading number "
                         "fits only that many of the first frequencies, e.g. 200y11")
    ap.add_argument('--ids', default=None,
                    help='comma-separated catalog Ids or names to try (default: all)')
    ap.add_argument('--top', type=int, default=30, help='number of best fits to list')
    ap.add_argument('--noise', type=float, default=None,
                    help='add noise to the S-parameters first, floor in dB (e.g. -40)')
    ap.add_argument('--plot', default=None,
                    help='save a plot of the best fit with positive element values and the '
                         'best passive fit (PNG)')
    ap.add_argument('--dps', type=int, default=DEFAULT_DPS, help='working precision, digits')
    ap.add_argument('-v', '--verbose', action='store_true', help='print every fitted network')
    ap.add_argument('--version', action='version', version=f'crcat {__version__}')
    args = ap.parse_args(argv)
    try:
        'Ωμ'.encode(sys.stdout.encoding or 'ascii')
    except (UnicodeEncodeError, LookupError):
        use_ascii_units()
    print(f'crcat-seed {__version__}: ' + ', '.join(f'{k}={v}' for k, v in vars(args).items()))
    cat = _catalog.load()
    ids = None
    if args.ids:
        ids = [cat.id(int(x) if x.strip().isdigit() else x.strip()) for x in args.ids.split(',')]
    r = seed(args.file, cat, fit=args.fit, ids=ids, noise_floor_db=args.noise, dps=args.dps,
             verbose=args.verbose)
    print(r.report(args.top, min(args.top, 20)))
    if args.plot:
        r.plot(path=args.plot)
        print(f'plot: {args.plot}')
