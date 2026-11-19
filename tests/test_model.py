"""From network data to a netlist: model extraction, netlist writers, crcat-netlist."""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import skrf

import crcat
from crcat import model, netlist
from crcat.cli_netlist import main
from crcat.component import Component

DATA = pathlib.Path(__file__).parent / 'data'
F = np.linspace(0.5e9, 40e9, 80)


def make_network(nports, parts, f=F, z0=50.0):
    """Network from {ports: (catalog name, SI values)}: the pi model evaluated exactly."""
    cat = crcat.load('base')
    y = np.zeros((f.size, nports, nports), dtype=complex)
    for ports, (name, values) in parts.items():
        model._stamp(y, ports, Component(name, cat, values=values).eval(f, 0))
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit='hz'), s=model._y2s(y, z0), z0=z0)


def chosen(m):
    return {b.ports: b.chosen for b in m.branches}


EXACT = model.Options(pick='best')     # exact data: the generating networks come back

TWO_PORT = {(1,): ('C', [20e-15]), (2,): ('SerRC', [150.0, 35e-15]),
            (1, 2): ('SerRL', [2.0, 0.4e-9])}


def test_round_trip_two_port():
    m = model.extract(make_network(2, TWO_PORT), EXACT)
    c = chosen(m)
    assert m.error_db() < -150
    assert [c[p].n_elements for p in ((1,), (2,), (1, 2))] == [1, 2, 2]
    assert c[(1,)].label == 'CR3 C'
    assert np.allclose(c[(1,)].component.values[0], [20e-15], rtol=1e-8)
    assert np.allclose(sorted(c[(1, 2)].component.values[0]), sorted([2.0, 0.4e-9]), rtol=1e-7)
    assert all(v > 0 for *_, v in m.elements())
    assert not any(b.warning() for b in m.branches)
    assert 'CR3 C' in m.summary()


def test_round_trip_one_port_and_three_port():
    m = model.extract(make_network(1, {(1,): ('SerRLC', [3.0, 0.2e-9, 100e-15])}), EXACT)
    assert m.error_db() < -150 and m.n_elements() == 3
    # three ports: two branches are missing and must come out open
    parts = {(1,): ('C', [10e-15]), (3,): ('C', [15e-15]), (1, 3): ('SerRL', [1.5, 0.3e-9]),
             (2, 3): ('SerRL', [1.0, 0.25e-9])}
    m = model.extract(make_network(3, parts), EXACT)
    c = chosen(m)
    assert m.error_db() < -150
    assert c[(2,)].is_open and c[(1, 2)].is_open
    assert {p for p, x in c.items() if not x.is_open} == set(parts)


def test_simplest_and_best():
    # a small series resistance in the capacitor branch: negligible at a loose limit
    parts = dict(TWO_PORT)
    parts[(1,)] = ('SerRC', [0.5, 20e-15])
    net = make_network(2, parts)
    simple = chosen(model.extract(net, model.Options(target_db=-40)))
    best = chosen(model.extract(net, model.Options(pick='best')))
    assert simple[(1,)].label == 'CR3 C' and best[(1,)].n_elements == 2
    assert best[(1,)].error < simple[(1,)].error <= 0.01


@pytest.mark.slow
def test_require_modes():
    # a negative inductance between the ports (magnetic coupling in a pi model)
    parts = {(1,): ('C', [20e-15]), (2,): ('C', [20e-15]), (1, 2): ('L', [-0.3e-9])}
    net = make_network(2, parts)
    m = model.extract(net)                                   # positive: cannot fit
    b = m.branches[2]
    assert b.ports == (1, 2) and b.target_missed and 'limit not reached' in b.warning()
    assert all(v > 0 for *_, v in m.elements())
    m = model.extract(net, model.Options(require='positive-r', pick='best'))
    assert m.error_db() < -150
    assert np.allclose(chosen(m)[(1, 2)].component.values[0], [-0.3e-9], rtol=1e-8)
    # negative resistance: allowed only with 'any'
    net = make_network(1, {(1,): ('SerRL', [-2.0, 0.4e-9])})
    assert model.extract(net, model.Options(require='positive-r')).error_db() > -40
    assert model.extract(net, model.Options(require='passive')).error_db() > -40
    assert model.extract(net, model.Options(require='any', pick='best')).error_db() < -150


@pytest.mark.slow
def test_full_search_gives_the_same_choice():
    # data that no small network fits exactly: five-element branch, up to three allowed
    net = skrf.Network(str(DATA / 'XXXII-39.y1p'))
    for kw in ({'max_elements': 3}, {'max_elements': 4, 'pick': 'best'}, {'target_db': -25}):
        fast = model.extract(net, model.Options(**kw))
        full = model.extract(net, model.Options(full_search=True, **kw))
        assert fast.branches[0].chosen.label == full.branches[0].chosen.label, kw
        assert np.isclose(fast.error(), full.error()), kw
        assert len(full.branches[0].candidates) >= len(fast.branches[0].candidates)


def test_choose_alternative():
    m = model.extract(make_network(2, TWO_PORT), model.Options(full_search=True, pick='best'))
    b = m.branches[2]
    assert len(b.candidates) > 3 and b.candidates[0] is b.chosen
    m.choose(2, len(b.candidates) - 1)                       # the worst candidate
    assert b.chosen is b.candidates[-1] and m.error_db() > -100
    m.choose(2, 0)
    assert m.error_db() < -150


def test_options_and_band():
    net = make_network(2, TWO_PORT)
    m = model.extract(net, model.Options(fmin=5e9, fmax=20e9))
    assert m.freq[0] >= 5e9 and m.freq[-1] <= 20e9
    for bad in ({'require': 'x'}, {'pick': 'x'}, {'tolerance': 0.5}, {'max_elements': 6},
                {'target_db': 3}):
        with pytest.raises(ValueError):
            model.extract(net, model.Options(**bad))
    with pytest.raises(ValueError):
        model.extract(net, model.Options(fmin=39e9))            # too few points


def test_progress_and_cancel():
    net = make_network(1, {(1,): ('SerRL', [2.0, 0.4e-9])})
    seen = []
    model.extract(net, progress=lambda frac, text: seen.append((frac, text)))
    assert seen[-1] == (1.0, 'done') and all(0 <= f <= 1 for f, _ in seen)
    assert [f for f, _ in seen] == sorted(f for f, _ in seen)
    with pytest.raises(model.Cancelled):
        model.extract(net, cancel=lambda: len(seen) > 0)


# --- netlists ---------------------------------------------------------------------------

def spice_s_parameters(text, f, z0=50.0):
    """S-parameters of a SPICE subcircuit of R, L, C by nodal analysis (independent of
    the model code)."""
    lines = [ln.split() for ln in text.splitlines() if ln and ln[0] not in '*.']
    head = next(ln.split() for ln in text.splitlines() if ln.upper().startswith('.SUBCKT'))
    ports = head[2:]
    nodes = ports + sorted({n for _, a, b, _ in lines for n in (a, b)} - set(ports) - {'0'})
    idx = {n: k for k, n in enumerate(nodes)}
    s = 2j * np.pi * f
    y = np.zeros((f.size, len(nodes), len(nodes)), dtype=complex)
    for name, a, b, value in lines:
        v = float(value)
        g = {'R': 1 / v + 0 * s, 'L': 1 / (s * v), 'C': s * v}[name[0].upper()]
        for p, q in ((a, b), (b, a)):
            if p != '0':
                y[:, idx[p], idx[p]] += g
                if q != '0':
                    y[:, idx[p], idx[q]] -= g
    n = len(ports)
    yp = y[:, :n, :n]
    if len(nodes) > n:                               # eliminate the internal nodes
        yp = yp - y[:, :n, n:] @ np.linalg.inv(y[:, n:, n:]) @ y[:, n:, :n]
    eye = np.eye(n)
    return (eye - z0 * yp) @ np.linalg.inv(eye + z0 * yp)


def test_spice_netlist_gives_the_model_response():
    parts = dict(TWO_PORT)
    parts[(1, 2)] = ('SerRLC', [1.2, 0.3e-9, 200e-15])
    m = model.extract(make_network(2, parts), EXACT)
    text = netlist.to_spice(m, 'dut')
    assert text.count('.SUBCKT dut p1 p2') == 1 and text.rstrip().endswith('.ENDS dut')
    assert np.allclose(spice_s_parameters(text, F), m.s_matrix(F), atol=1e-9)
    assert len([ln for ln in text.splitlines() if ln[0] in 'RLC']) == m.n_elements()


def test_spectre_netlist():
    m = model.extract(make_network(2, TWO_PORT), EXACT)
    text = netlist.to_spectre(m, 'dut')
    assert 'subckt dut (p1 p2)' in text and 'ends dut' in text
    assert text.count(' resistor r=') == 2 and text.count(' inductor l=') == 1
    assert text.count(' capacitor c=') == 2
    with pytest.raises(ValueError):
        netlist.render(m, 'verilog')
    assert netlist.subckt_name('170fF mim-v2') == 'm_170fF_mim_v2'


def test_negative_values_are_flagged():
    net = make_network(1, {(1,): ('L', [-0.3e-9])})
    m = model.extract(net, model.Options(require='positive-r'))
    assert 'WARNING: negative element values: L_p1' in netlist.to_spice(m)


# --- command line -----------------------------------------------------------------------

def test_cli(tmp_path, capsys):
    src = tmp_path / 'dut.s2p'
    make_network(2, TWO_PORT).write_touchstone(str(src.with_suffix('')))
    main([str(src), '--format', 'both', '--fmax', '30', '--pick', 'best', '-o', str(tmp_path / 'out')])
    out = capsys.readouterr().out
    assert 'crcat-netlist' in out and 'fmax=30.0' in out and 'model: 5 elements' in out
    files = {p.name for p in (tmp_path / 'out').iterdir()}
    assert files == {'dut_crcat.sp', 'dut_crcat.scs', 'dut_crcat.s2p', 'dut_crcat.json',
                     'dut_crcat.png'}
    d = json.loads((tmp_path / 'out' / 'dut_crcat.json').read_text())
    assert d['ports'] == 2 and len(d['branches']) == 3 and d['error_db'] < -100
    back = skrf.Network(str(tmp_path / 'out' / 'dut_crcat.s2p'))
    ref = skrf.Network(str(src))
    assert np.allclose(back.s, ref.s, atol=1e-6)
    assert '.SUBCKT dut_model p1 p2' in (tmp_path / 'out' / 'dut_crcat.sp').read_text()


def test_core_does_not_load_the_gui():
    code = ('import sys, crcat, crcat.model, crcat.netlist; '
            'print(any(m.startswith("PySide6") for m in sys.modules), '
            '"matplotlib.pyplot" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr
