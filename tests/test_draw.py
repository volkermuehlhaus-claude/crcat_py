"""Schematics (port of CRCat's Draw / DrawArray)."""

import pytest

pytest.importorskip('matplotlib')

import crcat  # noqa: E402
from crcat import draw  # noqa: E402
from crcat.component import Component  # noqa: E402


def test_every_base_network_has_a_layout():
    cat = crcat.load('base')
    for net in cat:
        segments = draw.layout(net)
        names = [label for label, _, _ in segments if label != 'W']
        assert names == net.symVn, net.name


def test_layout_matches_crcat_table():
    # CRCat.m, case 'SerRLC': aX = [0 1.5 1.5 3 3 4.5], aY = 0
    seg = draw.layout(crcat.load('base')['SerRLC'])
    assert seg == [('R', (0.0, 0.0), (1.5, 0.0)), ('L', (1.5, 0.0), (3.0, 0.0)),
                   ('C', (3.0, 0.0), (4.5, 0.0))]
    # a bridge (topology XXXIII) has two wires and five elements
    seg = draw.layout(crcat.load('base')['XXXIII-40'])
    assert [s[0] for s in seg].count('W') == 2 and len(seg) == 7


def test_figures(tmp_path):
    cat = crcat.load()
    c = Component('XXXII-39', cat, values=[0.5e-9, 0.4e-9, 10, 0.2e-9, 3e-12])
    fig = draw.figure(c)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert 'La\n500pH' in texts and 'Rc\n10\u03a9' in texts and '1' in texts
    assert fig.axes[0].get_title() == 'XXXII-39, CR500'
    draw.save(fig, tmp_path / 'one.png')
    assert (tmp_path / 'one.png').stat().st_size > 2000
    # a bare network: names only
    texts = [t.get_text() for t in draw.figure(cat[500]).axes[0].texts]
    assert 'La' in texts and not any('H' in t for t in texts)
    fig = draw.figure_array([cat[i] for i in range(1, 15)])
    assert sum(bool(ax.get_title()) for ax in fig.axes) == 10          # at most ten
    draw.save(fig, tmp_path / 'array.svg')
    with pytest.raises(ValueError):
        draw.figure_array([])


def test_no_layout_for_unknown_topology():
    cat = crcat.load()
    assert not draw.has_layout(cat['FrontHalfModel'])
    with pytest.raises(KeyError):
        draw.figure(cat['FrontHalfModel'])
