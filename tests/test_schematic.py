"""Schematics of model rows: branches, open / short, coupled coils."""

import pytest

pytest.importorskip('matplotlib')

import numpy as np  # noqa: E402
import skrf  # noqa: E402

from crcat import model, schematic  # noqa: E402
from crcat.draw import save  # noqa: E402
from test_model import TEE, TWO_PORT, make_network, make_tee, make_transformer  # noqa: E402


def texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_branch_rows(tmp_path):
    m = model.extract(make_network(2, TWO_PORT), model.Options(pick='best'))
    fig = schematic.figure_row(m.rows()[2])
    assert fig.axes[0].get_title() == 'p1 - p2: CR7 SerRL'
    assert {'p1', 'p2', 'R\n2\u03a9', 'L\n400pH'} <= set(texts(fig))
    fig = schematic.figure_rows(m)
    assert [ax.get_title() for ax in fig.axes] == ['p1 - gnd: CR3 C', 'p2 - gnd: CR8 SerRC',
                                                   'p1 - p2: CR7 SerRL']
    assert 'gnd' not in texts(fig)                         # ground is a symbol, not a label
    save(fig, tmp_path / 'model.png')
    assert (tmp_path / 'model.png').stat().st_size > 3000


def test_open_and_short_rows():
    m = model.extract(make_tee(None, TEE[1], TEE[2]),
                      model.Options(form='tee', pick='best', reactance_percent=0,
                                    loss_percent=0))
    assert m.branches[0].chosen.label == 'short'
    fig = schematic.figure_rows(m)
    assert 'short' in texts(fig) and 'c' in texts(fig)
    m = model.extract(make_network(2, {(1, 2): TWO_PORT[(1, 2)]}), model.Options(pick='best'))
    assert [b.chosen.label for b in m.branches[:2]] == ['open', 'open']
    assert texts(schematic.figure_rows(m)).count('open') == 2
    full = texts(schematic.figure_model(m))          # the whole model: open branches left out
    assert 'open' not in full and {'p1', 'p2'} <= set(full) and len(full) == 2 + 2


def test_coupled_row():
    m = model.extract(make_transformer(0.3e-9, 0.3e-9, -0.4, 1.0, 1.0, rc=0.5),
                      model.Options(form='pi', pick='best'))
    g = m.couplings[0]
    fig = schematic.figure_row(g)
    t = texts(fig)
    assert fig.axes[0].get_title() == 'p1, p2 - gnd: coupled coils'
    assert any(x.startswith('k = -0.4') for x in t) and 'Rc\n500m\u03a9' in t
    assert {'p1', 'p2'} <= set(t)


# --- the whole model in one picture -------------------------------------------------------------

def _axes():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 4.5))
    FigureCanvasAgg(fig)
    return fig, fig.add_axes([0.02, 0.02, 0.96, 0.96])


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def _inside(ax):
    """True if every line that was drawn lies within the axis limits."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for ln in ax.lines:
        x, y = (np.asarray(v, dtype=float) for v in ln.get_data())
        ok = np.isfinite(x) & np.isfinite(y)
        if np.any(x[ok] < x0) or np.any(x[ok] > x1) or np.any(y[ok] < y0) or np.any(y[ok] > y1):
            return False
    return True


def test_full_schematic_of_a_pi_model():
    m = model.extract(make_network(2, TWO_PORT), model.Options(pick='best'))
    assert schematic.can_draw_full(m) and schematic.is_simple(m)
    fig, ax = _axes()
    schematic.draw_full(ax, m)
    texts = _texts(ax)
    # every element once, with its value; the two ports
    assert sorted(t.split('\n')[0] for t in texts if '\n' in t) == ['C', 'C', 'L', 'R', 'R']
    assert {'p1', 'p2'} <= set(texts) and len(texts) == m.n_elements() + 2
    assert _inside(ax) and not ax.axison
    assert all(ln.get_color() == 'k' for ln in ax.lines)
    # one row in colour: its lines and texts, and nothing else
    for row in m.rows():
        schematic.draw_full(ax, m, highlight=row)
        marked = [t.get_text() for t in ax.texts if t.get_color() == schematic.HIGHLIGHT]
        assert len(marked) == row.chosen.n_elements
        assert any(ln.get_color() == schematic.HIGHLIGHT for ln in ax.lines)
        assert any(ln.get_color() == 'k' for ln in ax.lines)
    # the saved picture of such a model is this one circuit
    fig = schematic.figure_model(m)
    assert len(fig.axes) == 1 and 'pi form, 5 elements' in fig._suptitle.get_text()


def test_full_schematic_of_other_forms():
    f = np.linspace(0.5e9, 40e9, 80)
    fig, ax = _axes()
    # one port
    one = model.extract(make_network(1, {(1,): ('SerRLC', [1.0, 50e-12, 170e-15])}, f=f),
                        model.Options(pick='best'))
    schematic.draw_full(ax, one)
    assert schematic.can_draw_full(one) and 'p1' in _texts(ax) and 'p2' not in _texts(ax)
    assert len(_texts(ax)) == 4 and _inside(ax)
    # tee form with capacitive legs: the centre node has no DC path, the picture says so
    tee = model.extract(make_tee(('SerRC', [1.0, 300e-15]), ('SerRC', [1.5, 200e-15]),
                                 ('SerRC', [8.0, 30e-15]), f=f),
                        model.Options(form='tee', pick='best'))
    schematic.draw_full(ax, tee, highlight=tee.branches[2])
    texts = _texts(ax)
    note = [t for t in texts if 'operating point only' in t]
    assert len(note) == 1 and 'from node c to ground' in note[0] and '1 G' in note[0]
    assert len(texts) == 6 + 2 + 1 and _inside(ax)
    tee.options.float_resistor = 0               # no resistor in the netlist: no note
    schematic.draw_full(ax, tee)
    assert not any('operating point' in t for t in _texts(ax))
    # a tee with a shorted leg
    short = model.extract(make_tee(None, ('SerRL', [1.5, 0.15e-9]), ('SerRC', [8.0, 300e-15])),
                          model.Options(form='tee', pick='best'))
    schematic.draw_full(ax, short)
    assert short.branches[0].chosen.trivial == 'short' and len(_texts(ax)) == 4 + 2 and _inside(ax)
    # lines at the ports
    core = make_network(2, TWO_PORT, f=np.linspace(0.5e9, 100e9, 120))
    net = skrf.Network(frequency=core.frequency, z0=50,
                       s=model.shift_ports(core.s, core.f, 1.5e-12))
    line = model.extract(net, model.Options(delay=1.5e-12, pick='best'))
    schematic.draw_full(ax, line)
    assert sum('1.5 ps' in t for t in _texts(ax)) == 2 and _inside(ax)
    # coupled coils
    coupled = model.extract(make_transformer(0.3e-9, 0.3e-9, -0.4, 1.0, 1.0, rc=0.5),
                            model.Options(form='pi', pick='best'))
    schematic.draw_full(ax, coupled, highlight=coupled.couplings[0])
    texts = _texts(ax)
    assert 'k = -0.4' in texts and any(t.startswith('Rc\n') for t in texts) and _inside(ax)
    assert sum(t.get_color() == schematic.HIGHLIGHT for t in ax.texts) == 6   # R L R L, Rc, k


def test_full_schematic_limits():
    # three ports: the rows in a grid, as before
    three = make_network(3, {(1,): ('C', [20e-15]), (2,): ('C', [30e-15]), (3,): ('C', [10e-15]),
                             (1, 2): ('SerRL', [1.0, 0.3e-9]), (2, 3): ('SerRL', [2.0, 0.5e-9])})
    m3 = model.extract(three, model.Options(pick='best', coupling=False))
    assert not schematic.can_draw_full(m3) and not schematic.is_simple(m3)
    assert len(schematic.figure_model(m3).axes) > 1
    # a large 2-port can be drawn, but is not the default picture
    parts = {(1,): ('VII-4', [3000.0, 0.7e-15, 20.0, 0.8e-15]),
             (2,): ('VII-4', [2500.0, 0.9e-15, 25.0, 0.6e-15]),
             (1, 2): ('XII-4', [0.25, 40e-12, 90.0, 280e-12]),
             }
    big = model.extract(make_network(2, parts), model.Options(pick='best', dc='off'))
    if big.n_elements() > schematic.FULL_MAX_ELEMENTS:
        assert schematic.can_draw_full(big) and not schematic.is_simple(big)
    fig, ax = _axes()
    schematic.draw_full(ax, big)
    assert _inside(ax) and len(_texts(ax)) == big.n_elements() + 2
    assert all(t.get_fontsize() <= 9 for t in ax.texts)
