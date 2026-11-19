"""Fit, components, catalog relations, data selection, sprout, automatic fit.

Expected values come from CRCat's README and its MATLAB tests (CRCompTest.m,
CRSproutTest.m, CRFitAutoTest.m).
"""

import pathlib

import mpmath
import numpy as np
import pytest

from crcat import catalog, data
from crcat.component import Component, _sign_changes, format_value
from crcat.fit import BAD_INPUT, TOO_FEW_POINTS, crfit, evaluate
from crcat.sprout import sprout

DATA = pathlib.Path(__file__).parent / 'data'


@pytest.fixture(scope='module')
def cat():
    return catalog.load()


def close(a, b, rel=1e-6):
    return all(abs(complex(x) - complex(y)) <= rel * max(abs(complex(y)), 1e-30)
               for x, y in zip(a, b))


# --- catalog ----------------------------------------------------------------------------

def test_catalog_lookup(cat):
    assert len(cat) == 631 and len(catalog.load('base')) == 623
    assert cat.id('XXXII-39') == 500 == cat.id('CR500') == cat.id(500)
    assert cat[624].name == 'XXXII-39_original'
    assert sum(n.can_synthesize for n in catalog.load('base')) == 491


def test_relations(cat):
    # README: 500 -> 239 Equiv, 413 Equiv (624.. are cross-catalog, not in this port)
    assert cat.relations(500) == [(239, 'Equiv'), (413, 'Equiv'), (500, 'Self')]
    assert cat.relationship(500, 243) == 'NoXfrm'


def test_transform(cat):
    # README: CR500 with test values -> XXXI-33 (CR413) and back
    c = Component(500, cat)
    t = c.transform(413)
    assert t.name == 'XXXI-33'
    # README values, rounded to 4 decimals
    assert np.allclose(t.values[0], [17.3, 9.5631, 53.5379, 6.4071, 0.0414], rtol=0, atol=5e-5)
    assert close(t.transform(500).values[0], c.values[0], 1e-9)
    f = np.linspace(0.1, 3, 7)
    assert np.allclose(t.eval(f), c.eval(f), rtol=1e-10)


# --- rational fit -----------------------------------------------------------------------

def test_crfit_return_codes():
    assert crfit([1, 2], [1], [1], [1]).code == BAD_INPUT
    assert crfit([1, 2], [1, 2], [1, 1], [1, 1]).code == TOO_FEW_POINTS


def test_crfit_exact_rational():
    # Y = (2 + 3s) / (1 + 0.5 s + 0.25 s^2) is recovered exactly (b normalized to 1)
    f = np.linspace(0.05, 2, 25)
    s = 2j * np.pi * f
    y = (2 + 3 * s) / (1 + 0.5 * s + 0.25 * s ** 2)
    r = crfit(y, f, [1, 1], [1, 1, 1])
    assert r.ok
    k = r.b[-1]                                       # data are double precision
    assert close([x / k for x in r.a], [3, 2], 1e-13)
    assert close([x / k for x in r.b], [0.25, 0.5, 1], 1e-13)
    assert np.allclose(evaluate(r.a, r.b, f), y, rtol=1e-13)


# --- components (CRCompTest.m) ----------------------------------------------------------

def test_component_fit_recovers_values():
    custom = catalog.load('custom')
    base = Component(1, custom)                       # XXXII-39_original, test values
    f = np.arange(1, 21)
    y = base.eval(f, 0)
    c = Component(1, custom, values=[1, 1, 1, 1, 1])
    err = c.fit(y, f)
    assert err[0] < 1e-7                               # |y| ~ 0.01..0.04, double precision data
    assert close(c.values[0], base.values[0], 1e-6)
    # README: the version with Cdd fixed recovers the others given Cdd = 0.092
    fixed = Component(2, custom)
    assert fixed.network.fixed == ['Cdd']
    err = fixed.fit(y, f, fixed=[0.092])
    assert err[0] < 1e-7                               # README: 2.39e-9
    assert close(fixed.values[0], [4.3, 13, 5.4, 27, 0.092], 1e-6)


def test_component_fit_sonnet_data(cat):
    # README: Sonnet netlist data of XXXII-39 -> La 0.5 nH, Lb 0.4 nH, Rc 10, Ld 0.2 nH, Cd 3 pF
    net = data.read(DATA / 'XXXII-39.y1p')
    y, desc = data.pull(net, 'y11')
    assert desc == 'Pi admit port 1 to gnd'
    c = Component(500, cat)
    err = c.fit(y, net.f)
    assert err[0] < 1e-11                              # README: 3.128e-13
    assert close(c.values[0], [0.5e-9, 0.4e-9, 10, 0.2e-9, 3e-12], 1e-7)
    assert c.is_passive()
    assert c.format() == 'La=500pH Lb=400pH Rc=10Ω Ld=200pH Cd=3pF'


def test_component_fit_root_network(cat):
    # a network whose synthesis equations contain root(p, z, k)
    n = cat['XV-2']
    assert n._synthesis.has_root
    c = Component(n)
    f = np.linspace(0.05, 3, 30)
    y = c.eval(f, 0)
    fit = Component(n, values=[1] * len(n.symVn))
    err = fit.fit(y, f)
    assert err[0] < 1e-8
    assert close(fit.values[0], c.values[0], 1e-6)


# --- passivity --------------------------------------------------------------------------

def test_sign_changes():
    assert _sign_changes([1, 0, -1]) == [1.0]          # w^2 - 1 crosses at 1
    assert _sign_changes([1, -2, 1]) == []             # (w-1)^2 only touches zero
    assert _sign_changes([-1, 0, -1]) == [0.0]         # negative everywhere
    assert _sign_changes([1, -3, 2, 0, 0]) == [1.0, 2.0]
    assert _sign_changes([0, 0]) == []                 # lossless


def test_passive(cat):
    assert Component(500, cat, values=[0.5e-9, 0.4e-9, 10, 0.2e-9, 3e-12]).is_passive()
    # XXXII-44 (CR505) with the rounded values of CRCat's README: Re Y < 0 far above 1e30 rad/s
    c = Component(505, cat, values=[500e-12, 400e-12, 2.7e-12, 13.99, 1.588e-12])
    assert c.gain_frequencies()[0] > 1e30
    assert not Component('R', cat, values=[-5]).is_passive()


def test_format_value():
    assert format_value(3e-12, 'Cd') == '3pF'
    assert format_value(1.5e3, 'R') == '1.5kΩ'
    assert format_value(0, 'L') == '0H'


# --- data selection ---------------------------------------------------------------------

def test_pull_two_port():
    net = data.read(DATA / 'hairpin_InBand_Nf251.s2p')
    y, z = net.y, net.z
    assert np.allclose(data.pull(net, 'y11')[0], y[:, 0, 0] + y[:, 1, 0])
    assert np.allclose(data.pull(net, 'y10')[0], y[:, 0, 0] + y[:, 1, 0])
    assert np.allclose(data.pull(net, 'y21')[0], -y[:, 1, 0])
    assert np.allclose(data.pull(net, 'Y21')[0], y[:, 1, 0])
    assert np.allclose(data.pull(net, 'z22')[0], z[:, 1, 1] - z[:, 0, 1])
    assert np.allclose(data.pull(net, 'z12')[0], z[:, 0, 1])
    assert np.allclose(data.pull(net, '200S21')[0], net.s[:, 1, 0])
    s = data.pull(net, 'S11')[0]
    assert np.allclose(data.admittance(s, 'S11'), (1 - s) / (1 + s) / 50)
    assert np.allclose(data.admittance(z[:, 0, 0], 'Z11'), 1 / z[:, 0, 0])
    with pytest.raises(ValueError):
        data.pull(net, 'y31')


# --- sprout (CRSproutTest.m) ------------------------------------------------------------

def test_sprout(cat):
    f = np.arange(1, 11)
    vals = [1.0, 2.0, 0.3, 4.0, 0.05]                  # La Rb Lb Lc Cc of XXIII-9
    target = Component('XXIII-9', cat, values=vals).eval(f, 0)
    chain = [Component('L', cat, values=[vals[2]]), Component('R', cat, values=[vals[1]]),
             Component('ParLC', cat, values=vals[3:5]), Component('L', cat, values=[vals[0]])]
    assert np.allclose(sprout(f, chain), target, rtol=1e-12)
    assert np.allclose(sprout(f, chain, target), 0, atol=1e-12)
    # empty components change nothing (test 4: series short / shunt open)
    gappy = [chain[0], None, None, chain[1], chain[2], None, None, chain[3]]
    assert np.allclose(sprout(f, gappy), target, rtol=1e-12)
    assert np.allclose(sprout(f, []), 0)
