"""Plots of a model against the data (matplotlib is imported only when a plot is made; no
pyplot, so the GUI and the command line can both use them).

Views: 's' (magnitude and phase of the S-parameters), 'error' (|S model - S data| of each
S-parameter against frequency, with the error limit) and 'lrq' (2-ports: series L or C,
series R and Q, the quantities an inductor or capacitor is judged by).

Data are drawn as a thin line with open circles, the model as a solid line on top. Every
figure made here carries a :class:`Readout` (``fig.crcat_readout``): the values of all
curves at a frequency, for a marker.
"""


class Readout:
    """Values of the plotted curves at a frequency: ``text(index)``, ``index(f_ghz)``."""

    def __init__(self, f_ghz):
        self.f = np.asarray(f_ghz, dtype=float)
        self.rows = []                               # (label, function index -> text)

    def add(self, label, fmt):
        self.rows.append((label, fmt))

    def index(self, f_ghz):
        return int(np.argmin(np.abs(self.f - f_ghz)))

    def text(self, i):
        lines = [f'f = {self.f[i]:.6g} GHz']
        lines += [f'{label}: {fmt(i)}' for label, fmt in self.rows]
        return '\n'.join(lines)


def _markevery(n):
    return max(1, n // DATA_MARKERS)


def _pair(ax, x, d, m, color, label=None):
    """Data (thin line, open circles) and model (solid line on top) in one color."""
    ax.plot(x, d, color=color, lw=0.8, alpha=0.6, marker='o', markersize=4.5,
            markerfacecolor='none', markeredgewidth=0.9, markevery=_markevery(len(x)), zorder=2)
    ax.plot(x, m, color=color, lw=1.6, label=label, zorder=3)


def _style_legend(ax):
    """Two black entries that explain the line styles."""
    ax.plot([], [], color='k', lw=0.8, marker='o', markersize=4.5, markerfacecolor='none',
            label='data')
    ax.plot([], [], color='k', lw=1.6, label='model')


def _fmt(v, unit='', digits=4):
    return 'n/a' if not np.isfinite(v) else f'{v:.{digits}g}{unit}'


import numpy as np

__all__ = ['compare', 'VIEWS', 'views_for', 'MAX_TRACES', 'series_quantities', 'Readout']

DATA_MARKERS = 30       # open circles per data trace, about

VIEWS = {'s': 'S-parameters', 'error': 'Error', 'lrq': 'L (or C), R, Q',
         'size': 'Size against accuracy'}
MAX_TRACES = 8          # with more S-parameters, only the ones with the largest error


def views_for(model):
    """The views that make sense for a model ('lrq' needs two ports)."""
    return [v for v in VIEWS if v != 'lrq' or model.nports == 2]


def _data(model, network, all_data):
    """(f in Hz, S data, S model) in the fit band, or for all data frequencies."""
    keep = network.f > 0
    if not all_data:
        keep &= (network.f >= model.freq[0]) & (network.f <= model.freq[-1])
    f = network.f[keep]
    return f, network.s[keep], model.s_matrix(f)


def _traces(model, s_data, s_model):
    """S-parameters to draw: [(label, i, j)], reciprocal pairs once, coinciding ones merged,
    at most MAX_TRACES (those with the largest error). Returns (traces, number left out)."""
    n = model.nports
    items = []
    for i in range(n):
        for j in range(i + 1):
            err = float(np.sqrt(np.mean(np.abs(s_model[:, i, j] - s_data[:, i, j]) ** 2)))
            items.append([f'S{i + 1}{j + 1}', i, j, err])
    merged = []
    for it in items:                                 # same data and model: one trace
        for m in merged:
            if (np.allclose(s_data[:, it[1], it[2]], s_data[:, m[1], m[2]], atol=2e-3)
                    and np.allclose(s_model[:, it[1], it[2]], s_model[:, m[1], m[2]], atol=2e-3)):
                m[0] += ' = ' + it[0]
                break
        else:
            merged.append(it)
    left_out = max(len(merged) - MAX_TRACES, 0)
    if left_out:
        merged = sorted(merged, key=lambda m: -m[3])[:MAX_TRACES]
        merged.sort(key=lambda m: (m[1], m[2]))
    return [(m[0], m[1], m[2]) for m in merged], left_out


def _finish(fig, axes, model, f, title, all_data, legend_ax=None):
    for ax in axes:
        if all_data:
            ax.axvspan(model.freq[0] / 1e9, model.freq[-1] / 1e9, color='0.5', alpha=0.08)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('frequency (GHz)')
    axes[0].set_title(title + (', fit band shaded' if all_data else ''), fontsize=9)
    fig.tight_layout()
    if legend_ax is not None:                        # below the plots, never on the curves
        handles, labels = legend_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 5), fontsize=8,
                       frameon=False)
            rows = -(-len(labels) // 5)
            fig.subplots_adjust(bottom=0.15 + 0.05 * rows)


def _view_s(fig, model, network, all_data):
    f, s_data, s_model = _data(model, network, all_data)
    fg = f / 1e9
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    traces, left_out = _traces(model, s_data, s_model)
    readout = Readout(fg)
    for k, (label, i, j) in enumerate(traces):
        color = f'C{k % 10}'
        sd, sm = s_data[:, i, j], s_model[:, i, j]
        with np.errstate(divide='ignore'):
            _pair(ax1, fg, 20 * np.log10(np.abs(sd)), 20 * np.log10(np.abs(sm)), color, label)
        _pair(ax2, fg, np.degrees(np.angle(sd)), np.degrees(np.angle(sm)), color)

        def fmt(n, sd=sd, sm=sm):
            with np.errstate(divide='ignore'):
                d, m = 20 * np.log10(abs(sd[n])), 20 * np.log10(abs(sm[n]))
            return (f'data {d:7.2f} dB {np.degrees(np.angle(sd[n])):7.1f} deg, '
                    f'model {m:7.2f} dB {np.degrees(np.angle(sm[n])):7.1f} deg, '
                    f'|difference| {20 * np.log10(max(abs(sm[n] - sd[n]), 1e-300)):6.1f} dB')
        readout.add(label, fmt)
    fig.crcat_readout = readout
    _style_legend(ax1)
    ax1.set_ylabel('magnitude (dB)')
    ax2.set_ylabel('phase (deg)')
    title = f'circles: data, lines: model ({model.n_elements()} elements, {model.error_db():.1f} dB)'
    if left_out:
        title += f'; the {len(traces)} S-parameters with the largest error'
    _finish(fig, (ax1, ax2), model, f, title, all_data, ax1)


def _view_error(fig, model, network, all_data):
    f, s_data, s_model = _data(model, network, all_data)
    fg = f / 1e9
    ax = fig.subplots()
    traces, left_out = _traces(model, s_data, s_model)
    readout = Readout(fg)
    for k, (label, i, j) in enumerate(traces):
        with np.errstate(divide='ignore'):
            e = 20 * np.log10(np.abs(s_model[:, i, j] - s_data[:, i, j]))
        ax.plot(fg, e, color=f'C{k % 10}', lw=1.4, label=label)
        readout.add(label, lambda n, e=e: f'|S model - S data| {e[n]:6.1f} dB')
    fig.crcat_readout = readout
    ax.axhline(model.options.target_db, color='k', ls='--', lw=1,
               label=f'limit {model.options.target_db:g} dB')
    lo = min(model.options.target_db - 40, -80)
    ax.set_ylim(lo, max(0, ax.get_ylim()[1]))
    ax.set_ylabel('|S model - S data| (dB)')
    title = f'error of each S-parameter (RMS of all in the fit band: {model.error_db():.1f} dB)'
    if left_out:
        title += f'; the {len(traces)} largest'
    _finish(fig, (ax,), model, f, title, all_data, ax)


def series_quantities(s, z0):
    """Series impedance and port-1 input impedance of 2-port S-parameters: the usual
    inductor / capacitor figures. Returns (Zseries = -1/Y21, Zin = 1/Y11)."""
    eye = np.eye(2)
    y = np.linalg.solve(z0 * (eye + s), eye - s)     # Y = (1/z0) (I + S)^-1 (I - S)
    with np.errstate(all='ignore'):
        return -1 / y[:, 1, 0], 1 / y[:, 0, 0]


def _limits(d):
    """Axis limits for data values d: the bulk of the data (2nd to 98th percentile, so a few
    points at the band edge do not set the scale), with a margin of a quarter of the span
    and at least 15 % of the largest value, so a flat curve does not lie on the frame.
    Starts at zero for positive data."""
    lo, hi = np.nanpercentile(d, [2, 98])
    pad = max(0.25 * (hi - lo), 0.15 * max(abs(lo), abs(hi)), 1e-30)
    return (min(lo - pad, 0.0) if lo >= 0 else lo - pad), hi + pad


def _view_lrq(fig, model, network, all_data):
    f, s_data, s_model = _data(model, network, all_data)
    fg = f / 1e9
    w = 2 * np.pi * f
    (zs_d, zin_d), (zs_m, zin_m) = (series_quantities(s, model.z0) for s in (s_data, s_model))
    in_band = (f >= model.freq[0]) & (f <= model.freq[-1])
    capacitive = np.median(zs_d.imag[in_band]) < 0
    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
    with np.errstate(all='ignore'):
        if capacitive:
            rows = [(ax1, -1e15 / (w * zs_d.imag), -1e15 / (w * zs_m.imag), 'series C (fF)')]
        else:
            rows = [(ax1, 1e12 * zs_d.imag / w, 1e12 * zs_m.imag / w, 'series L (pH)')]
        rows += [(ax2, zs_d.real, zs_m.real, 'series R (ohm)'),
                 (ax3, np.abs(zin_d.imag) / zin_d.real, np.abs(zin_m.imag) / zin_m.real,
                  'Q at port 1')]
    readout = Readout(fg)
    for ax, d, m, label in rows:
        _pair(ax, fg, d, m, 'C0')
        ax.set_ylabel(label)
        name, _, unit = label.partition(' (')
        unit = ' ' + unit.rstrip(')') if unit else ''
        readout.add(name, lambda n, d=d, m=m, unit=unit:
                    f'data {_fmt(d[n], unit)}, model {_fmt(m[n], unit)}')
        ok = np.isfinite(d) & in_band                # scale to the data in the fit band
        if ok.any():
            pos = d[ok][d[ok] > 0]
            if label.startswith('Q') and pos.size and pos.max() > 50 * pos.min():
                ax.set_yscale('log')                 # Q over decades (a capacitor from low f)
                ax.set_ylim(pos.min() / 2, pos.max() * 2)
            else:
                ax.set_ylim(*_limits(d[ok]))
    fig.crcat_readout = readout
    _style_legend(ax1)
    _finish(fig, (ax1, ax2, ax3), model, f,
            'from -1/Y21 (series) and 1/Y11 (Q); circles: data, lines: model',
            all_data, ax1)


def _view_size(fig, model, network, all_data):
    """Error of the whole model against its number of elements, for the models that can be
    put together from the candidates tried (Model.tradeoff); the one in use is marked."""
    points = model.tradeoff()
    ax = fig.subplots()
    x = [p.n_elements for p in points]
    y = [p.error_db for p in points]
    ax.plot(x, y, color='C0', lw=1.2, marker='o', markersize=6)
    ax.plot([model.n_elements()], [model.error_db()], color='C3', marker='o', markersize=11,
            markerfacecolor='none', markeredgewidth=2, ls='none', label='model in use')
    ax.axhline(model.options.target_db, color='k', ls='--', lw=1,
               label=f'limit {model.options.target_db:g} dB')
    ax.set_xlabel('elements in the model')
    ax.set_ylabel('S-parameter error of the model (dB)')
    ax.set_title('size against accuracy: click a point to use that model', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.crcat_tradeoff = points
    fig.crcat_readout = Readout(np.array([]))


def compare(model, network, fig=None, view='s', all_data=False):
    """Draw data against model into ``fig`` (a matplotlib Figure; a new one if None).

    ``view``: 's', 'error', 'lrq' or 'size' (see the module text; 'size' is the error
    of the model against its number of elements, with the points in ``fig.crcat_tradeoff``).
    ``all_data``: all frequencies of the data instead of the fit band only; the fit band is
    then shaded.
    """
    if view not in views_for(model):
        raise ValueError(f'view {view!r} is not available: use one of {views_for(model)}')
    if fig is None:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 6.5))
    fig.clear()
    fig.crcat_tradeoff = None
    {'s': _view_s, 'error': _view_error, 'lrq': _view_lrq,
     'size': _view_size}[view](fig, model, network, all_data)
    return fig
