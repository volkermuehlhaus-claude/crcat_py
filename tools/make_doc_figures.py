"""Make the pictures of the README (doc/*.png): concept diagrams and example schematics.

    python tools/make_doc_figures.py [name ...]

Without names all pictures are made. The examples with real data need the sibling
repositories that the README names (snp2le, lumpedmodel); they are skipped if a file is
missing.
"""

import pathlib
import sys

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import crcat
from crcat import model, plots, schematic

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / 'doc'
sys.path.insert(0, str(ROOT / 'tests'))

IND = ROOT.parent.parent / 'github' / 'snp2le' / 'snp2le' / 'examples' / 'ind_500pH_ihp-sg13cmos5l.s2p'
LINE = ROOT.parent / 'lumpedmodel' / 'rlgc_from_s2p' / 'line_880um.s2p'
MIM = ROOT.parent / 'lumpedmodel' / 'mim_from_s2p' / 'mim_170fF.s2p'

BLUE, SOFT, GREY, WARM = '#1f77b4', '#e8f1fa', '#f3f3f3', '#fdf1e0'


def new(width, height):
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def save(fig, name):
    fig.savefig(DOC / name, dpi=110, facecolor='white')
    print('wrote', DOC / name)


def box(ax, x, y, w, h, title, text='', face=SOFT, edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                facecolor=face, edgecolor=edge, linewidth=1.3))
    if text:
        ax.text(x + w / 2, y + h - 0.2, title, ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(x + w / 2, y + h - 0.62, text, ha='center', va='top', fontsize=8.5, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=10, fontweight='bold')


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, color='0.3', lw=1.3))


# --- concept: from the file to the netlist ----------------------------------------------------

def concept_flow():
    fig = new(11.0, 4.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 8.0)
    ax.axis('off')
    y, h, w, gap = 5.0, 2.25, 3.6, 0.7
    steps = [('Touchstone file', '.s1p  .s2p  .sNp\nEM simulation\nor measurement', GREY, '0.4'),
             ('Data check', 'kind of part, length\npi or tee form\nDC region', SOFT, BLUE),
             ('Branches', 'port to ground,\nport to port:\none admittance each', SOFT, BLUE),
             ('Catalog search', '623 RLC networks\nrational fit, closed-\nform element values', SOFT, BLUE),
             ('Judge and choose', 'S-parameters,\nreactance, loss;\nrefine the values', SOFT, BLUE)]
    xs = [0.5 + k * (w + gap) for k in range(5)]
    for k, (title, text, face, edge) in enumerate(steps):
        box(ax, xs[k], y, w, h, title, text, face, edge)
        if k:
            arrow(ax, (xs[k] - gap + 0.05, y + h / 2), (xs[k] - 0.05, y + h / 2))
    ax.plot([xs[2], xs[4] + w], [y + h + 0.22] * 2, color='0.6', lw=1.0, linestyle='--')
    ax.text(xs[3] + w / 2, y + h + 0.32, 'for every branch', ha='center', va='bottom', fontsize=8.5,
            color='0.35')
    # the model, under the last step
    ym, hm = 2.35, 1.65
    box(ax, xs[4], ym, w, hm, 'Model', 'a few R, L, C\nper branch', WARM, '#c77f1a')
    arrow(ax, (xs[4] + w / 2, y - 0.05), (xs[4] + w / 2, ym + hm + 0.05))
    # what comes out of it, under the first four steps
    outs = [('Netlist', 'SPICE and Spectre\nsubcircuit'), ('Testbench', 'S-parameter run\nof the netlist'),
            ('Verdict', 'good / fair / poor,\nL and Q against the data'),
            ('Plots, schematic', 'data against model,\nthe circuit as a picture')]
    for k, (title, text) in enumerate(outs):
        box(ax, xs[k], ym, w, hm, title, text, GREY, '0.45')
    yb = 1.55
    ax.plot([xs[4] + w / 2, xs[4] + w / 2], [ym - 0.05, yb], color='0.3', lw=1.3)
    ax.plot([xs[0] + w / 2, xs[4] + w / 2], [yb, yb], color='0.3', lw=1.3)
    for k in range(4):
        arrow(ax, (xs[k] + w / 2, yb), (xs[k] + w / 2, ym - 0.05))
    ax.text(11.0, 0.55, 'The search and the choice run once per branch; the model is the sum of the '
            'chosen networks.', ha='center', fontsize=9, color='0.25')
    save(fig, 'concept_flow.png')


# --- concept: the forms of a model ------------------------------------------------------------

def _block(ax, x, y, w, h, text, face=SOFT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.06',
                                facecolor=face, edgecolor=BLUE, linewidth=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9)


def _wire(ax, *points):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color='k', lw=1.2, solid_capstyle='round')


def _gnd(ax, x, y):
    _wire(ax, (x, y), (x, y - 0.12))
    for k, half in enumerate((0.22, 0.14, 0.07)):
        ax.plot([x - half, x + half], [y - 0.12 - 0.08 * k] * 2, color='k', lw=1.2)


def _terminal(ax, x, y, name, side):
    ax.plot([x], [y], marker='o', ms=6, mfc='white', mec='k', mew=1.2)
    ax.text(x + side * 0.12, y + 0.17, name, ha='center', va='bottom', fontsize=9)


def _shunt(ax, x, y, text):
    _wire(ax, (x, y), (x, y - 0.65))
    _block(ax, x - 0.75, y - 1.3, 1.5, 0.65, text)
    _wire(ax, (x, y - 1.3), (x, y - 1.5))
    _gnd(ax, x, y - 1.5)


def concept_forms():
    fig = new(11.0, 3.9)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 7.8)
    ax.set_aspect('equal')
    ax.axis('off')
    for x, title, note in ((0.3, 'pi form, 2 ports', 'three branches'),
                           (7.4, 'pi form, 3 ports', 'six branches; n ports: n (n + 1) / 2'),
                           (15.7, 'tee form, 2 ports', 'three legs, one inner node')):
        ax.text(x, 7.4, title, fontsize=11, fontweight='bold', va='top')
        ax.text(x, 6.85, note, fontsize=8.5, color='0.35', va='top')
    y = 4.0
    # pi, 2 ports
    x0 = 0.6
    _terminal(ax, x0, y, 'p1', 0)
    _terminal(ax, x0 + 5.6, y, 'p2', 0)
    _wire(ax, (x0, y), (x0 + 2.0, y))
    _block(ax, x0 + 2.0, y - 0.33, 1.6, 0.66, 'p1 - p2')
    _wire(ax, (x0 + 3.6, y), (x0 + 5.6, y))
    _shunt(ax, x0 + 1.0, y, 'p1 - gnd')
    _shunt(ax, x0 + 4.6, y, 'p2 - gnd')
    # pi, 3 ports: p1 and p3 at the ends, p2 from above; the branch p1 - p3 goes over the top
    x0 = 7.9
    xs = (x0 + 1.0, x0 + 3.6, x0 + 6.2)
    _terminal(ax, x0, y, 'p1', 0)
    _terminal(ax, x0 + 7.2, y, 'p3', 0)
    _wire(ax, (x0, y), (xs[0], y))
    _wire(ax, (xs[2], y), (x0 + 7.2, y))
    yt, yp = y + 0.95, y + 1.7
    _terminal(ax, xs[1], yp, 'p2', 0)
    _wire(ax, (xs[1], yp), (xs[1], y))
    for a, b, text in ((0, 1, 'p1 - p2'), (1, 2, 'p2 - p3')):
        xm = (xs[a] + xs[b]) / 2
        _wire(ax, (xs[a], y), (xm - 0.7, y))
        _block(ax, xm - 0.7, y - 0.33, 1.4, 0.66, text)
        _wire(ax, (xm + 0.7, y), (xs[b], y))
    xm = (xs[0] + xs[1]) / 2                       # box of p1 - p3 left of the stem of p2
    _wire(ax, (xs[0], y), (xs[0], yt), (xm - 0.7, yt))
    _block(ax, xm - 0.7, yt - 0.33, 1.4, 0.66, 'p1 - p3')
    _wire(ax, (xm + 0.7, yt), (xs[1] - 0.14, yt))            # a gap: no connection to p2
    _wire(ax, (xs[1] + 0.14, yt), (xs[2], yt), (xs[2], y))
    for k, x in enumerate(xs):
        _shunt(ax, x, y, f'p{k + 1} - gnd')
        ax.plot([x], [y], marker='o', ms=3.5, color='k')
    # tee
    x0 = 16.2
    _terminal(ax, x0, y, 'p1', 0)
    _terminal(ax, x0 + 5.2, y, 'p2', 0)
    _wire(ax, (x0, y), (x0 + 0.45, y))
    _block(ax, x0 + 0.45, y - 0.33, 1.45, 0.66, 'p1 - c')
    _wire(ax, (x0 + 1.9, y), (x0 + 3.3, y))
    _block(ax, x0 + 3.3, y - 0.33, 1.45, 0.66, 'p2 - c')
    _wire(ax, (x0 + 4.75, y), (x0 + 5.2, y))
    ax.plot([x0 + 2.6], [y], marker='o', ms=3.5, color='k')
    ax.text(x0 + 2.6, y + 0.18, 'c', ha='center', va='bottom', fontsize=9)
    _shunt(ax, x0 + 2.6, y, 'c - gnd')
    ax.text(11.0, 0.75, 'Each box is one network of the catalog: up to five R, L, C, or nothing at all '
            '(an open branch, a shorted leg).', ha='center', fontsize=9, color='0.25')
    save(fig, 'concept_forms.png')


# --- example schematics -----------------------------------------------------------------------

def _full(m, name, title, size=(8.6, 4.2), highlight=None):
    fig = new(*size)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
    schematic.draw_full(ax, m, highlight)
    fig.suptitle(title, fontsize=10)
    save(fig, name)


def schematic_examples():
    from test_dc import TEE_CAPS
    from test_model import make_tee, make_transformer
    f = np.linspace(0.5e9, 40e9, 80)
    m = model.extract(make_tee(*TEE_CAPS, f=f), model.Options(form='tee', pick='best'))
    _full(m, 'schematic_tee.png', 'Tee form: two legs along the top, the centre leg down', (8.0, 3.8))
    m = model.extract(make_transformer(0.3e-9, 0.3e-9, -0.4, 1.0, 1.0, rc=0.5),
                      model.Options(form='pi', pick='best'))
    _full(m, 'schematic_coupled.png', 'Coupled coils with a common resistor', (7.0, 4.2),
          m.couplings[0])
    if LINE.exists():
        m = model.extract(crcat.data.read(str(LINE)))
        _full(m, 'schematic_line.png', f'880 um line, 0.275 to 110 GHz: lines at the ports and '
              f'{m.n_elements() - 2} lumped elements, {m.error_db():.1f} dB', (9.0, 3.6))


def dc_before_after():
    if not IND.exists():
        return print('skipped (no data):', IND)
    net = crcat.data.read(str(IND))
    fig = new(11.0, 4.0)
    for k, (dc, title) in enumerate((('off', 'Without the rule: a resistor from each port to ground'),
                                     ('data', 'With the rule (default): no DC path to ground'))):
        m = model.extract(net, model.Options(dc=dc))
        ax = fig.add_axes([0.01 + 0.5 * k, 0.02, 0.48, 0.8])
        leak = [b for b in m.free_branches() if b.node2 == '0']
        schematic.draw_full(ax, m, leak[0] if leak else None)
        fig.text(0.25 + 0.5 * k, 0.93, title, ha='center', fontsize=10, fontweight='bold')
        fig.text(0.25 + 0.5 * k, 0.865, m.dc_text() + f'; {m.error_db():.1f} dB', ha='center',
                 fontsize=9, color='0.3')
    save(fig, 'dc_before_after.png')


def target_frequency():
    if not IND.exists():
        return print('skipped (no data):', IND)
    net = crcat.data.read(str(IND))
    for name, ftarget in (('target_without.png', None), ('target_with.png', 20e9)):
        m = model.extract(net, model.Options(ftarget=ftarget))
        fig = new(5.4, 5.0)
        plots.compare(m, net, fig, view='lrq')
        for ax in fig.axes:
            ax.axvline(20, color='0.4', lw=0.9, ls=':')
        save(fig, name)


FIGURES = {'concept_flow': concept_flow, 'concept_forms': concept_forms,
           'schematic_examples': schematic_examples, 'dc_before_after': dc_before_after,
           'target_frequency': target_frequency}

if __name__ == '__main__':
    for key in (sys.argv[1:] or FIGURES):
        FIGURES[key]()
