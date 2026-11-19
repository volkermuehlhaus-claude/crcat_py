"""Schematics of a :class:`crcat.model.Model`: one picture per result row.

Built on :mod:`crcat.draw` (CRCat's layouts). A row is a branch (a catalog network between
two nodes, or "open" / "short") or a pair of coupled coils. No GUI code here; matplotlib is
imported only when something is drawn.
"""

import math

import numpy as np

from . import draw
from .component import format_value
from .model import Branch

__all__ = ['draw_row', 'figure_row', 'figure_model', 'row_title']


def _terminal(ax, at, node, side, fontsize):
    """Terminal of a row: ground symbol for node '0', else a labelled circle."""
    if node == '0':
        if side > 0:
            draw.draw_ground(ax, at)
        else:                                        # ground on the left: mirrored
            for k, half in enumerate((0.3, 0.2, 0.1)):
                x = at[0] - 0.11 * (k + 1)
                ax.plot([x, x], [at[1] - half, at[1] + half], color='k', lw=1.2)
            ax.plot([at[0] - 0.11, at[0]], [at[1], at[1]], color='k', lw=1.2)
        return
    r = 0.1
    th = np.linspace(0, 2 * math.pi, 41)
    cx = at[0] + side * r
    ax.plot(cx + r * np.cos(th), at[1] + r * np.sin(th), color='k', lw=1.2)
    ax.text(cx, at[1] + 1.8 * r, node, fontsize=fontsize, ha='center', va='bottom')


def _frame(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin - 0.7, xmax + 0.8)
    ax.set_ylim(ymin - 1.1, ymax + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')


def row_title(row):
    if isinstance(row, Branch):
        return f'{row.name}: {row.chosen.label}'
    return f'{row.name}: coupled coils'


def _draw_branch(ax, b, fontsize):
    c = b.chosen
    if c.is_trivial:
        _terminal(ax, (0.0, 0.0), b.node1, -1, fontsize)
        _terminal(ax, (1.5, 0.0), b.node2, 1, fontsize)
        if c.trivial == 'short':
            draw.draw_element(ax, 'W', (0.0, 0.0), (1.5, 0.0))
        ax.text(0.75, -0.25, c.trivial, fontsize=fontsize, ha='center', va='top')
        _frame(ax, 0, 1.5, 0, 0)
        return
    comp = c.component
    if not draw.has_layout(comp):
        ax.text(0.5, 0.5, c.values_text(), fontsize=fontsize, ha='center', va='center',
                transform=ax.transAxes, wrap=True)
        ax.axis('off')
        return
    end = draw.draw_network(ax, comp, fontsize=fontsize)
    _terminal(ax, (0.0, 0.0), b.node1, -1, fontsize)
    _terminal(ax, end, b.node2, 1, fontsize)
    seg = draw.layout(comp)
    xs = [v for _, p, q in seg for v in (p[0], q[0])]
    ys = [v for _, p, q in seg for v in (p[1], q[1])]
    _frame(ax, min(xs), max(xs), min(ys), max(ys))


def _draw_coupled(ax, g, fontsize):
    """Two coils, one above the other, joined at the right; coupling factor between."""
    i, j, _ = g.ports
    ends, extents, y0 = [], [], 0.0
    for port, comp in zip((i, j), g.coils):
        seg = draw.layout(comp)
        ys = [v for _, p, q in seg for v in (p[1], q[1])]
        xs = [v for _, p, q in seg for v in (p[0], q[0])]
        y0 -= max(ys)                                # top of this coil below the last one
        end = draw.draw_network(ax, comp, origin=(0.0, y0), fontsize=fontsize)
        _terminal(ax, (0.0, y0), f'p{port}', -1, fontsize)
        ends.append(end)
        extents.append((max(xs), y0 + min(ys), y0 + max(ys)))
        y0 += min(ys) - 2.0                          # room for labels and the k text
    xr = max(e[0] for e in ends) + 0.4
    for e in ends:                                   # join the coil ends
        draw.draw_element(ax, 'W', e, (xr, e[1]))
    draw.draw_element(ax, 'W', (xr, ends[0][1]), (xr, ends[1][1]))
    ym = 0.5 * (ends[0][1] + ends[1][1])
    text = f'k = {g.k:.3g}'
    if any(sum(sym[0] == 'L' for sym in comp.network.symVn) > 1 for comp in g.coils):
        names = [comp.network.symVn[n] for comp, n in zip(g.coils, g.index)]
        text += f'  ({names[0]} above, {names[1]} below)'    # which inductors are coupled
    ax.text(0.0, 0.5 * (extents[0][1] + extents[1][2]) - 0.35, text, fontsize=fontsize,
            ha='left', va='center')
    x = xr
    if g.rc > 0:
        draw.draw_element(ax, 'R', (xr, ym), (xr + 1.5, ym), 'Rc\n' + format_value(g.rc, 'R'),
                          fontsize=fontsize)
        x = xr + 1.5
    else:
        draw.draw_element(ax, 'W', (xr, ym), (xr + 0.3, ym))
        x = xr + 0.3
    _terminal(ax, (x, ym), g.node_k, 1, fontsize)
    _frame(ax, 0, x, min(e[1] for e in extents), max(e[2] for e in extents))


def draw_row(ax, row, fontsize=8, title=True):
    """Draw one result row (a Branch or a Coupled) on the axes."""
    ax.clear()
    if isinstance(row, Branch):
        _draw_branch(ax, row, fontsize)
    else:
        _draw_coupled(ax, row, fontsize)
    if title:
        ax.set_title(row_title(row), fontsize=fontsize + 1)


def figure_row(row, fig=None):
    if fig is None:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 2.6))
    fig.clear()
    draw_row(fig.subplots(), row)
    fig.tight_layout()
    return fig


def figure_model(model, fig=None, columns=3):
    """All rows of a model in a grid, each with its nodes labelled."""
    rows = model.rows()
    cols = min(columns, len(rows))
    nrows = math.ceil(len(rows) / cols)
    if fig is None:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(4.6 * cols, 2.8 * nrows))
    fig.clear()
    axes = np.atleast_1d(fig.subplots(nrows, cols)).ravel()
    for ax in axes[len(rows):]:
        ax.axis('off')
    for ax, row in zip(axes, rows):
        draw_row(ax, row)
    fig.suptitle(f'{model.source or "model"}: {model.form} form, {model.n_elements()} '
                 f'elements, error {model.error_db():.1f} dB', fontsize=10)
    fig.tight_layout()
    return fig
