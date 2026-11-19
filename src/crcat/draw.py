"""Schematics of catalog networks (port of CRCat's Draw, DrawArray and DrawRLC.m).

The layout of every topology comes from CRCat (``data/draw_layout.json``, the coordinate
tables of CRCat.Draw): each network is drawn left to right from its port terminal (node 1)
to its ground terminal (node 0). One length unit is the body of a resistor; an element with
its two leads is 1.5 units long.

matplotlib is imported only when something is drawn, and no pyplot is used.
"""

import functools
import json
import math
import pathlib

import numpy as np

from .component import format_value

__all__ = ['draw_network', 'draw_element', 'draw_port', 'draw_ground', 'figure',
           'figure_array', 'has_layout', 'layout']

LEAD = 0.25
_UNIT = 1.0


@functools.lru_cache(maxsize=None)
def _layouts():
    path = pathlib.Path(__file__).parent / 'data' / 'draw_layout.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _network(obj):
    return getattr(obj, 'network', obj)


def layout(network):
    """Segments of a network's schematic: list of (label, (x1, y1), (x2, y2)); label is
    an element name or 'W' for a wire. KeyError if CRCat has no layout for the topology."""
    net = _network(network)
    group = net.name.split('-')[0]
    lay = _layouts()[group]
    nseg = len(lay['x']) // 2
    names = net.symVn if len(net.symVn) + lay['wires'] == nseg else net.symV
    if len(names) + lay['wires'] != nseg:
        raise KeyError(f'the layout of {group} does not fit network {net.name}')
    labels = list(names) + ['W'] * lay['wires']
    return [(labels[k], (lay['x'][2 * k], lay['y'][2 * k]),
             (lay['x'][2 * k + 1], lay['y'][2 * k + 1])) for k in range(nseg)]


def has_layout(network):
    try:
        layout(network)
    except KeyError:
        return False
    return True


class _Frame:
    """Places local coordinates (along the element, across it) on the axes."""

    def __init__(self, start, end, scale=1.0):
        self.x0, self.y0 = start
        self.angle = math.atan2(end[1] - start[1], end[0] - start[0])
        self.scale = scale

    def xy(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        c, s = math.cos(self.angle), math.sin(self.angle)
        return (self.x0 + self.scale * (c * x - s * y), self.y0 + self.scale * (s * x + c * y))


def _symbol(kind, lead):
    """Local outline of an element: body of length 1 between two leads (DrawRLC.m)."""
    total = 1.0 + 2 * lead
    if kind == 'R':
        x = lead + np.array([-lead, 0, 0.1 / 1.2, 0.3 / 1.2, 0.5 / 1.2, 0.7 / 1.2, 0.9 / 1.2,
                             1.1 / 1.2, 1.0, 1.0 + lead])
        y = np.array([0, 0, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0, 0]) / 1.2
        return x, y, -0.25
    if kind == 'L':
        th = np.linspace(math.pi, 0, 21)
        xs, ys = [0.0], [0.0]
        for k in range(4):                           # four loops of radius 0.125
            xs += list(lead + 0.25 * k + 0.125 + 0.125 * np.cos(th))
            ys += list(0.125 * np.sin(th) + np.where((th < math.pi) & (th > 0), 0.05, 0.0))
        xs.append(total)
        ys.append(0.0)
        return np.array(xs), np.array(ys), -0.12
    if kind == 'C':
        half, a, b = 0.25, lead + 0.42, lead + 0.58  # plates around the middle
        nan = float('nan')
        x = np.array([0, a, nan, a, a, nan, b, b, nan, b, total])
        y = np.array([0, 0, nan, half, -half, nan, half, -half, nan, 0, 0])
        return x, y, -0.33
    raise ValueError(f'unknown element kind {kind!r}')


def draw_element(ax, kind, start, end, label='', color='k', lw=1.2, fontsize=8):
    """Draw R, L, C or a wire ('W') from start to end. The symbol keeps its size; the
    leads take up the rest of the distance (a distance below 1 shrinks the symbol)."""
    if kind == 'W':
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw,
                solid_capstyle='round')
        return
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    scale = min(1.0, length / (1.0 + 2 * LEAD))
    lead = (length / scale - 1.0) / 2
    fr = _Frame(start, end, scale)
    x, y, text_y = _symbol(kind, lead)
    ax.plot(*fr.xy(x, y), color=color, lw=lw, solid_capstyle='round')
    if label:
        tx, ty = fr.xy(0.5 * (1.0 + 2 * lead), text_y)
        flat = abs(math.sin(fr.angle)) < 0.5         # text under flat, beside steep elements
        ax.text(float(tx), float(ty), label, color=color, fontsize=fontsize,
                ha='center' if flat else 'left', va='top' if flat else 'center')


def draw_port(ax, at, label='', color='k', lw=1.2, fontsize=8, radius=0.1):
    """Small circle left of the point ``at``, with a label."""
    th = np.linspace(0, 2 * math.pi, 41)
    ax.plot(at[0] - radius + radius * np.cos(th), at[1] + radius * np.sin(th), color=color, lw=lw)
    if label:
        ax.text(at[0] - radius, at[1] + 1.6 * radius, label, color=color, fontsize=fontsize,
                ha='center', va='bottom')


def draw_ground(ax, at, color='k', lw=1.2):
    """Ground symbol right of the point ``at``."""
    for k, half in enumerate((0.3, 0.2, 0.1)):
        x = at[0] + 0.11 * (k + 1)
        ax.plot([x, x], [at[1] - half, at[1] + half], color=color, lw=lw)
    ax.plot([at[0], at[0] + 0.11], [at[1], at[1]], color=color, lw=lw)


def _labels(obj, solution, values):
    net = _network(obj)
    vals = getattr(obj, 'values', None)
    out = {}
    for k, sym in enumerate(net.symVn):
        text = sym
        if values and vals:
            text += '\n' + format_value(vals[solution][k], sym)
        out[sym] = text
    return out


def draw_network(ax, obj, origin=(0.0, 0.0), solution=0, values=True, color='k', lw=1.2,
                 fontsize=8):
    """Draw a catalog network (or a Component, then with element values) on ``ax``.

    The port terminal is at ``origin``; returns the position of the ground terminal.
    """
    labels = _labels(obj, solution, values)
    end = origin
    for label, (x1, y1), (x2, y2) in layout(obj):
        p, q = (origin[0] + x1, origin[1] + y1), (origin[0] + x2, origin[1] + y2)
        if label == 'W':
            draw_element(ax, 'W', p, q, color=color, lw=lw)
        else:
            draw_element(ax, label[0], p, q, labels[label], color, lw, fontsize)
        end = q
    return end


def _extent(obj):
    xs = [v for _, p, q in layout(obj) for v in (p[0], q[0])]
    ys = [v for _, p, q in layout(obj) for v in (p[1], q[1])]
    return min(xs), max(xs), min(ys), max(ys)


def _setup(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin - 0.5, xmax + 0.7)
    ax.set_ylim(ymin - 1.0, ymax + 0.7)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_one(ax, obj, title=None, solution=0, values=True, fontsize=8):
    """A network with port, ground and title, filling the axes."""
    net = _network(obj)
    end = draw_network(ax, obj, solution=solution, values=values, fontsize=fontsize)
    draw_port(ax, (0.0, 0.0), '1', fontsize=fontsize)
    draw_ground(ax, end)
    _setup(ax, *_extent(obj))
    ax.set_title(f'{net.name}, CR{net.id}' if title is None else title, fontsize=fontsize + 1)


def figure(obj, fig=None, solution=0, values=True):
    """Figure with the schematic of one network or Component (CRCat.Draw)."""
    if fig is None:
        from matplotlib.figure import Figure
        xmin, xmax, ymin, ymax = _extent(obj)
        fig = Figure(figsize=(1.1 * (xmax - xmin + 1.6), 1.1 * (ymax - ymin + 2.0)))
    fig.clear()
    draw_one(fig.subplots(), obj, solution=solution, values=values)
    return fig


def figure_array(objs, fig=None, values=True, columns=5):
    """Figure with up to 10 networks or Components in a grid (CRCat.DrawArray)."""
    objs = list(objs)[:10]
    if not objs:
        raise ValueError('nothing to draw')
    cols = min(columns, len(objs))
    rows = math.ceil(len(objs) / cols)
    if fig is None:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(3.2 * cols, 2.6 * rows))
    fig.clear()
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes[len(objs):]:
        ax.axis('off')
    for ax, obj in zip(axes, objs):
        draw_one(ax, obj, values=values)
    fig.tight_layout()
    return fig


def save(fig, path, dpi=120):
    """Save a figure made here to a file (PNG, SVG, PDF by extension)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi)
