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

__all__ = ['draw_row', 'figure_row', 'figure_rows', 'figure_model', 'figure_full', 'row_title',
           'draw_full',
           'can_draw_full', 'is_simple']

FULL_MAX_ELEMENTS = 14   # up to here the whole model is the default picture in the window
HIGHLIGHT = 'tab:blue'   # the selected row, in the picture of the whole model
STRETCH = 1.7            # parallel parts of a network are drawn this much further apart
LABEL_ROOM = 1.5         # room right of an upright element for its text
DROP = 0.5               # wire from the top line down to a branch to ground
LINE_LENGTH = 1.6        # length of the symbol of a line at a port
COUPLING_ROOM = 2.0      # room between two coupled coils for the coupling factor


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


# --- the whole model in one picture -----------------------------------------------------------

def can_draw_full(model):
    """True if :func:`draw_full` can draw the model: one or two ports, and a layout for
    every network in it."""
    if model.nports > 2:
        return False
    comps = [b.chosen.component for b in model.free_branches() if not b.chosen.is_trivial]
    comps += [c for g in model.couplings for c in g.coils]
    return all(draw.has_layout(c) for c in comps)


def is_simple(model):
    """True if the picture of the whole model is the better default: it can be drawn and
    the model is small."""
    return can_draw_full(model) and model.n_elements() <= FULL_MAX_ELEMENTS


class _Box:
    """Bounding box of what was drawn."""

    def __init__(self):
        self.x, self.y = [], []

    def add(self, *points):
        for x, y in points:
            self.x.append(x)
            self.y.append(y)


def _length(comp):
    return draw._extent(comp)[1]


def _half(comp):
    """Half the width of a network across its direction, as drawn."""
    _, _, ymin, ymax = draw._extent(comp)
    return STRETCH * max(abs(ymin), abs(ymax))


def _place(ax, comp, start, along, color, lw, fontsize, box):
    """Draw a catalog network from ``start`` in direction ``along`` ('right', 'left' or
    'down'); returns the point of its other terminal."""
    sx, sy = start

    def at(x, y):
        if along == 'right':
            return sx + x, sy + STRETCH * y
        if along == 'left':
            return sx - x, sy + STRETCH * y
        return sx + STRETCH * y, sy - x
    labels = draw._labels(comp, 0, True)
    for label, p, q in draw.layout(comp):
        a, b = at(*p), at(*q)
        if (b[0], b[1]) < (a[0], a[1]):          # left to right, or upwards: text below / right
            a, b = b, a
        box.add(a, b)
        if label == 'W':
            draw.draw_element(ax, 'W', a, b, color=color, lw=lw)
        else:
            draw.draw_element(ax, label[0], a, b, labels[label], color, lw, fontsize)
    return at(_length(comp), 0.0)


def _wire(ax, a, b, box, color='k', lw=1.2):
    draw.draw_element(ax, 'W', a, b, color=color, lw=lw)
    box.add(a, b)


def _ground_below(ax, at, box, color='k', lw=1.2):
    x, y = at
    ax.plot([x, x], [y, y - 0.15], color=color, lw=lw)
    for k, half in enumerate((0.3, 0.2, 0.1)):
        yy = y - 0.15 - 0.11 * k
        ax.plot([x - half, x + half], [yy, yy], color=color, lw=lw)
    box.add((x - 0.3, y - 0.4), (x + 0.3, y))


def _port(ax, at, name, side, fontsize, box):
    _terminal(ax, at, name, side, fontsize)
    box.add((at[0] + side * 0.3, at[1] + 0.45), at)


def _line_symbol(ax, a, b, text, fontsize, box):
    """A piece of line between a and b (on one level): a long box with its values."""
    (x1, y), (x2, _) = a, b
    lo, hi = min(x1, x2) + 0.2, max(x1, x2) - 0.2
    ax.plot([min(x1, x2), lo], [y, y], color='k', lw=1.2)
    ax.plot([hi, max(x1, x2)], [y, y], color='k', lw=1.2)
    ax.plot([lo, hi, hi, lo, lo], [y - 0.14, y - 0.14, y + 0.14, y + 0.14, y - 0.14],
            color='k', lw=1.2)
    ax.text(0.5 * (lo + hi), y - 0.25, text, fontsize=fontsize, ha='center', va='top')
    box.add((min(x1, x2), y - 0.9), (max(x1, x2), y + 0.14))


def _branch_down(ax, comp, node, color, lw, fontsize, box):
    """A network hanging from the point ``node`` on the top line; returns its lower end."""
    top = (node[0], node[1] - DROP)
    _wire(ax, node, top, box, color, lw)
    end = _place(ax, comp, top, 'down', color, lw, fontsize, box)
    box.add((node[0] + _half(comp) + LABEL_ROOM, end[1]))
    return end


def draw_full(ax, model, highlight=None, fontsize=8):
    """Draw the whole model as one circuit (1- and 2-port models, see :func:`can_draw_full`).

    ``highlight``: a result row (a Branch or a Coupled of the model) drawn in colour.
    Pi form: the branch between the ports along the top, the branches to ground hanging
    down. Tee form: two legs along the top, the centre leg down. Lines at the ports are
    boxes between a port and the lumped part; coupled coils hang from the two ports.
    """
    ax.clear()
    box = _Box()

    def style(row):
        return (HIGHLIGHT, 1.9) if row is highlight and row is not None else ('k', 1.2)

    def comp_of(b):
        if b is None or b.group is not None or b.chosen.is_trivial:
            return None
        return b.chosen.component

    def half(comp):
        return _half(comp) if comp is not None else 0.0

    def room_right(comp):                        # right of a hanging network: its texts
        return half(comp) + (LABEL_ROOM if comp is not None else 0.0) + 0.3

    by_name = {b.name: b for b in model.branches}
    group = model.couplings[0] if model.couplings else None
    n = model.nports
    ports = [f'p{i + 1}' for i in range(n)]
    delays = [float(d) for d in model.port_delay]

    # --- what lies along the top line and what hangs from it ---------------------------------
    if model.form == 'tee':
        legs = (by_name['p1 - c'], by_name['p2 - c'])
        hanging = {'c': (by_name['c - gnd'], comp_of(by_name['c - gnd']))}
    else:
        series = by_name.get('p1 - p2') if n == 2 else None
        hanging = {}
        for k, port in enumerate(ports):
            if group is not None:
                hanging[port] = (group, group.coils[k])
            else:
                b = by_name.get(f'{port} - gnd')
                hanging[port] = (b, comp_of(b))

    # --- positions of the nodes along the top line --------------------------------------------
    nodes = {'p1': 0.0}
    if model.form == 'tee':
        left, right = comp_of(legs[0]), comp_of(legs[1])
        c_comp = hanging['c'][1]
        nodes['c'] = (_length(left) if left is not None else 1.0) + half(c_comp) + 0.3
        nodes['p2'] = (nodes['c'] + room_right(c_comp)
                       + (_length(right) if right is not None else 1.0))
    elif n == 2:
        mid = comp_of(series)
        nodes['p2'] = (room_right(hanging['p1'][1]) + (_length(mid) if mid is not None else 1.0)
                       + half(hanging['p2'][1]) + 0.3)
        if group is not None:                    # room for "k = ..." between the coils
            nodes['p2'] = max(nodes['p2'], room_right(hanging['p1'][1]) + COUPLING_ROOM
                              + half(hanging['p2'][1]))

    # --- ports, with the lines at the ports ---------------------------------------------------
    for k, port in enumerate(ports):
        side = -1 if k == 0 else 1
        xn = nodes[port]
        own = hanging.get(port, (None, None))[1]
        reach = half(own) + (LABEL_ROOM if side > 0 and own is not None else 0.0) + 0.5
        if n == 1:
            reach = 0.5
        x_line = xn + side * reach
        x_port = x_line
        if delays[k] > 0:
            x_port = x_line + side * LINE_LENGTH
            text = (f'{format_value(model.port_z0[k], "R")}\n'
                    f'{model.port_delay[k] * 1e12:.4g} ps')
            _line_symbol(ax, (x_line, 0.0), (x_port, 0.0), text, fontsize, box)
        _wire(ax, (xn, 0.0), (x_line, 0.0), box)
        _port(ax, (x_port, 0.0), port, side, fontsize, box)

    # --- the top line ---------------------------------------------------------------------------
    if n == 1:
        row, comp = hanging['p1']
        color, lw = style(row)
        if comp is not None:
            _wire(ax, (0.0, 0.0), (0.5, 0.0), box, color, lw)
            end = _place(ax, comp, (0.5, 0.0), 'right', color, lw, fontsize, box)
            corner = (end[0] + 0.4, 0.0)
            _wire(ax, end, corner, box, color, lw)
            _wire(ax, corner, (corner[0], -0.4), box, color, lw)
            _ground_below(ax, (corner[0], -0.4), box, color, lw)
            box.add((0.0, -0.9 - half(comp)), (0.0, half(comp)))
        else:
            ax.text(0.9, -0.1, 'open', fontsize=fontsize, ha='left', va='top')
            box.add((2.0, -0.6))
        hanging = {}
    elif model.form == 'tee':
        for b, along, port in ((legs[0], 'right', 'p1'), (legs[1], 'left', 'p2')):
            comp = comp_of(b)
            color, lw = style(b)
            xa, xc = nodes[port], nodes['c']
            if comp is not None:
                end = _place(ax, comp, (xa, 0.0), along, color, lw, fontsize, box)
                _wire(ax, end, (xc, 0.0), box, color, lw)
            elif b.chosen.trivial == 'short':
                _wire(ax, (xa, 0.0), (xc, 0.0), box, color, lw)
            else:                                # an open leg: this port is not connected
                ax.text(0.5 * (xa + xc), -0.1, 'open', fontsize=fontsize, ha='center', va='top',
                        color=color)
    elif series is not None:
        comp = comp_of(series)
        color, lw = style(series)
        start = room_right(hanging['p1'][1])
        if comp is not None:
            _wire(ax, (nodes['p1'], 0.0), (start, 0.0), box, color, lw)
            end = _place(ax, comp, (start, 0.0), 'right', color, lw, fontsize, box)
            _wire(ax, end, (nodes['p2'], 0.0), box, color, lw)
        elif series.chosen.trivial == 'short':
            _wire(ax, (nodes['p1'], 0.0), (nodes['p2'], 0.0), box, color, lw)

    # --- what hangs down ----------------------------------------------------------------------
    ends = {}
    for node, (row, comp) in hanging.items():
        if comp is not None:
            color, lw = style(row)
            ends[node] = _branch_down(ax, comp, (nodes[node], 0.0), color, lw, fontsize, box)
    if group is not None and len(ends) == 2:
        color, lw = style(group)
        (xa, ya), (xb, yb) = ends['p1'], ends['p2']
        xk = 0.5 * (room_right(hanging['p1'][1]) + nodes['p2'] - half(hanging['p2'][1]))
        ax.text(xk, 0.5 * (min(ya, yb) - DROP), f'k = {group.k:.3g}', fontsize=fontsize,
                ha='center', va='center', color=color)
        if group.rc > 0:                         # joined, then the common resistor to ground
            yj = min(ya, yb) - 0.4
            xm = 0.5 * (xa + xb)
            _wire(ax, (xa, ya), (xa, yj), box, color, lw)
            _wire(ax, (xb, yb), (xb, yj), box, color, lw)
            _wire(ax, (xa, yj), (xb, yj), box, color, lw)
            draw.draw_element(ax, 'R', (xm, yj - 1.5), (xm, yj),
                              'Rc\n' + format_value(group.rc, 'R'), color, lw, fontsize)
            box.add((xm, yj - 1.5), (xm + LABEL_ROOM, yj))
            _ground_below(ax, (xm, yj - 1.5), box, color, lw)
        else:
            _ground_below(ax, (xa, ya), box, color, lw)
            _ground_below(ax, (xb, yb), box, color, lw)
    else:
        for node, end in ends.items():
            color, lw = style(hanging[node][0])
            _ground_below(ax, end, box, color, lw)

    # --- nodes without a DC path: what the netlist adds ----------------------------------------
    floating = model.floating_nodes()
    ohm = model.options.float_ohm('spice')
    x0, x1 = min(box.x), max(box.x)
    y0, y1 = min(box.y), max(box.y)
    if floating and ohm:
        unit = format_value(1.0, 'R')[1:]        # the ohm sign in use
        value = f'{ohm / 1e9:g} G{unit}' if ohm >= 1e9 else format_value(ohm, 'R')
        ax.text(x0, y0 - 0.5, f'+ {value} from node ' + ', '.join(floating)
                + ' to ground (operating point only)', fontsize=fontsize, color='0.45',
                ha='left', va='top')
        y0 -= 0.9
    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(y0 - 0.5, y1 + 0.7)
    ax.set_aspect('equal')
    ax.axis('off')
    # texts keep their size while the picture is scaled: make them fit a large circuit
    fig = ax.figure
    pos = ax.get_position()
    width = fig.get_size_inches()[0] * 72 * pos.width
    height = fig.get_size_inches()[1] * 72 * pos.height
    per_unit = min(width / (x1 - x0 + 1.0), height / (y1 - y0 + 1.2))
    size = min(fontsize + 1.0, max(4.5, 0.2 * per_unit))
    for t in ax.texts:
        t.set_fontsize(size)


def figure_full(model, fig=None, highlight=None):
    """The whole model in one figure (see :func:`draw_full`)."""
    if fig is None:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(9.0, 5.0))
    fig.clear()
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    draw_full(ax, model, highlight)
    fig.suptitle(f'{model.source or "model"}: {model.form} form, {model.n_elements()} '
                 f'elements, error {model.error_db():.1f} dB', fontsize=10)
    return fig


def figure_model(model, fig=None, columns=3):
    """The schematic of a model: the whole circuit in one picture if it can be drawn
    (:func:`can_draw_full`), else all rows in a grid, each with its nodes labelled."""
    if can_draw_full(model):
        return figure_full(model, fig)
    return figure_rows(model, fig, columns)


def figure_rows(model, fig=None, columns=3):
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
