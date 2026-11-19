"""Plots (matplotlib is imported only when a plot is made; no pyplot, so the GUI and the
command line can both use them)."""

import numpy as np

__all__ = ['compare']


def compare(model, network, fig=None):
    """S-parameters of the data (all frequencies) and of the model: magnitude and phase.

    Draws into ``fig`` (a matplotlib Figure) or makes a new one; returns the figure.
    Reciprocal pairs are drawn once (S21, not S12).
    """
    if fig is None:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(9, 6))
    fig.clear()
    f = network.f[network.f > 0]
    fg = f / 1e9
    s_data = network.s[network.f > 0]
    s_model = model.s_matrix(f)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    k = 0
    for i in range(model.nports):
        for j in range(i + 1):
            color = f'C{k % 10}'
            k += 1
            with np.errstate(divide='ignore'):
                ax1.plot(fg, 20 * np.log10(np.abs(s_data[:, i, j])), color=color, lw=3,
                         alpha=0.35)
                ax1.plot(fg, 20 * np.log10(np.abs(s_model[:, i, j])), color=color, lw=1.2,
                         label=f'S{i + 1}{j + 1}')
            ax2.plot(fg, np.degrees(np.angle(s_data[:, i, j])), color=color, lw=3, alpha=0.35)
            ax2.plot(fg, np.degrees(np.angle(s_model[:, i, j])), color=color, lw=1.2)
    for ax in (ax1, ax2):
        ax.axvspan(model.freq[0] / 1e9, model.freq[-1] / 1e9, color='0.5', alpha=0.08)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel('magnitude (dB)')
    ax2.set_ylabel('phase (deg)')
    ax2.set_xlabel('frequency (GHz)')
    ax1.set_title(f'wide: data, thin: model ({model.n_elements()} elements, '
                  f'{model.error_db():.1f} dB), fit band shaded', fontsize=10)
    ax1.legend(fontsize=8, ncol=min(k, 6))
    fig.tight_layout()
    return fig
