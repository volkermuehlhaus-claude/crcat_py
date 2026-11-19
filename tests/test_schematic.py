"""Schematics of model rows: branches, open / short, coupled coils."""

import pytest

pytest.importorskip('matplotlib')

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
    fig = schematic.figure_model(m)
    assert [ax.get_title() for ax in fig.axes] == ['p1 - gnd: CR3 C', 'p2 - gnd: CR8 SerRC',
                                                   'p1 - p2: CR7 SerRL']
    assert 'gnd' not in texts(fig)                         # ground is a symbol, not a label
    save(fig, tmp_path / 'model.png')
    assert (tmp_path / 'model.png').stat().st_size > 3000


def test_open_and_short_rows():
    m = model.extract(make_tee(None, TEE[1], TEE[2]), model.Options(form='tee', pick='best'))
    assert m.branches[0].chosen.label == 'short'
    fig = schematic.figure_model(m)
    assert 'short' in texts(fig) and 'c' in texts(fig)
    m = model.extract(make_network(2, {(1, 2): TWO_PORT[(1, 2)]}), model.Options(pick='best'))
    assert [b.chosen.label for b in m.branches[:2]] == ['open', 'open']
    assert texts(schematic.figure_model(m)).count('open') == 2


def test_coupled_row():
    m = model.extract(make_transformer(0.3e-9, 0.3e-9, -0.4, 1.0, 1.0, rc=0.5),
                      model.Options(form='pi', pick='best'))
    g = m.couplings[0]
    fig = schematic.figure_row(g)
    t = texts(fig)
    assert fig.axes[0].get_title() == 'p1, p2 - gnd: coupled coils'
    assert any(x.startswith('k = -0.4') for x in t) and 'Rc\n500m\u03a9' in t
    assert {'p1', 'p2'} <= set(t)
