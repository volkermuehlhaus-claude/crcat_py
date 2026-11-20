"""DC safeguard: no leakage path where the data have none; resistor at floating nodes."""

import math
import os

import numpy as np
import pytest

import crcat
from crcat import dc, model, netlist, testbench
from crcat.cli_netlist import main
from crcat.component import Component
from test_easy import find_ngspice
from test_model import make_network, make_tee

F = np.linspace(0.5e9, 40e9, 80)


def test_which_networks_conduct():
    cat = {n.name: n for n in crcat.load('base')}
    open_at_dc = ['C', 'SerRC', 'SerLC', 'SerRLC']
    path = ['R', 'L', 'ParRL', 'ParRC', 'ParLC', 'SerRL', 'ParRLC']
    assert [dc.conducts(cat[n]) for n in open_at_dc] == [False] * len(open_at_dc)
    assert [dc.conducts(cat[n]) for n in path] == [True] * len(path)
    # the rule agrees with the admittance at a very low frequency, for every network
    for net in crcat.load('base'):
        if not net.can_synthesize or net.fixed:
            continue
        values = [1e-9 if s[0] == 'L' else 1e-12 if s[0] == 'C' else 10.0 for s in net.symVn]
        y = Component(net, values=values).eval(np.array([1e-3]), 0)[0]
        assert (abs(y) > 1e-6) == dc.conducts(net), net.name


def test_what_the_data_say_about_dc():
    cat = crcat.load('base')

    def y(name, values, f=F):
        return Component(name, cat, values=values).eval(f, 0)
    assert dc.dc_class(F, y('C', [20e-15])) == 'open'
    assert dc.dc_class(F, y('SerRC', [150.0, 35e-15])) == 'open'       # loss, but no conduction
    assert dc.dc_class(F, y('SerRLC', [1.0, 10e-12, 170e-15])) == 'open'
    assert dc.dc_class(F, y('ParRC', [5000.0, 14e-15])) == 'path'      # it conducts
    assert dc.dc_class(F, y('SerRL', [2.0, 0.4e-9])) == 'path'
    assert dc.dc_class(F, y('R', [50.0])) == 'path'
    assert dc.dc_class(F, 0 * F) == 'open'
    high = np.linspace(60e9, 100e9, 40)                                # too high to tell
    assert dc.dc_class(high, y('C', [20e-15], high)) is None


PARTS = {(1,): ('SerRC', [150.0, 35e-15]), (2,): ('ParRC', [3000.0, 20e-15]),
         (1, 2): ('SerRL', [2.0, 0.4e-9])}


def test_safeguard_follows_the_data():
    net = make_network(2, PARTS, f=F)
    m = model.extract(net, model.Options(pick='best'))                 # dc 'data' is the default
    by = {b.name: b for b in m.branches}
    assert [by[n].dc for n in ('p1 - gnd', 'p2 - gnd', 'p1 - p2')] == ['open', 'path', 'path']
    # the open branch has no candidate with a DC path, the others may have
    assert not any(c.conducts for c in by['p1 - gnd'].candidates)
    assert by['p2 - gnd'].chosen.conducts and by['p1 - p2'].chosen.conducts
    assert m.error_db() < -100 and not m.warnings()
    r = m.dc_resistances()
    assert np.isclose(r[('p1', 'p2')], 2.0) and np.isclose(r[('p2', '0')], 3000.0)
    assert np.isclose(r[('p1', '0')], 3002.0)                          # through the other port
    assert m.dc_text().startswith('DC paths: p1 - p2 2')
    assert m.dc_text() in m.summary() and len(m.to_dict()['dc_ohm']) == 3
    # without the rule, networks with a DC path are among the candidates of that branch
    off = model.extract(net, model.Options(pick='best', dc='off'))
    assert any(c.conducts for c in off.branches[0].candidates)
    # no path to ground at all, whatever the data say: the conducting branch is replaced
    strict = model.extract(net, model.Options(dc='no-ground'))
    r = strict.dc_resistances()
    assert math.isinf(r[('p1', '0')]) and math.isinf(r[('p2', '0')]) and np.isclose(r[('p1', 'p2')], 2, rtol=0.2)
    assert strict.dc_text().endswith('none to ground')
    assert not any(c.conducts for b in strict.branches if b.node2 == '0' for c in b.candidates)
    with pytest.raises(ValueError):
        model.extract(net, model.Options(dc='maybe'))


def test_a_leak_is_reported_when_the_rule_is_off():
    net = make_network(2, PARTS, f=F)
    m = model.extract(net, model.Options(dc='off'))
    b = m.branches[0]
    leak = next(c for c in b.candidates if c.conducts)
    m.choose(0, leak)
    assert any('leakage path at DC' in w for w in m.warnings())


TEE_CAPS = (('SerRC', [1.0, 300e-15]), ('SerRC', [1.5, 200e-15]), ('SerRC', [8.0, 30e-15]))


def test_floating_nodes_get_a_resistor():
    f = np.linspace(0.5e9, 40e9, 80)
    m = model.extract(make_tee(*TEE_CAPS, f=f), model.Options(form='tee', pick='best'))
    assert m.error_db() < -100 and [b.dc for b in m.branches] == ['open'] * 3
    assert m.floating_nodes() == ['c']                                 # one node per island
    assert m.dc_text() == 'DC: no path between the ports or to ground'
    spice = netlist.to_spice(m, 'x')
    assert 'Rdc_c c 0 1e+09' in spice and 'no DC path at these nodes' in spice
    assert 'Rdc_' not in netlist.to_spectre(m, 'x')                    # Spectre adds its own gmin
    assert m.n_elements() == 6                                         # not an element of the model
    m.options.float_resistor = 0
    assert 'Rdc_' not in netlist.to_spice(m, 'x')
    m.options.float_resistor = 5e8
    assert 'Rdc_c c 0 5e+08' in netlist.to_spice(m, 'x')
    assert 'Rdc_c (c 0) resistor r=5e+08' in netlist.to_spectre(m, 'x')
    with pytest.raises(ValueError):
        model.Options(float_resistor=-1).check()
    # a model whose nodes all have a DC path gets none
    plain = model.extract(make_network(2, PARTS, f=F), model.Options(pick='best'))
    assert plain.floating_nodes() == [] and 'Rdc_' not in netlist.to_spice(plain, 'x')


def test_floating_node_resistor_in_ngspice(tmp_path):
    """With the resistor ngspice finds the operating point without the 'singular matrix'
    fallbacks, the S-parameters are those of the model, and no DC current flows."""
    exe = find_ngspice()
    if exe is None:
        pytest.skip('ngspice not found')
    import subprocess
    f = np.linspace(0.5e9, 40e9, 80)
    m = model.extract(make_tee(*TEE_CAPS, f=f), model.Options(form='tee', pick='best'))
    for ohm, clean in ((1e9, True), (0, False)):
        m.options.float_resistor = ohm
        (tmp_path / 'dut.sp').write_text(netlist.to_spice(m, 'dut'), encoding='utf-8')
        (tmp_path / 'tb.sp').write_text(testbench.ngspice(m, 'dut.sp', 'dut'), encoding='utf-8')
        freq, s = testbench.run_ngspice(tmp_path / 'tb.sp', exe)
        assert np.abs(s - m.s_matrix(m.freq)).max() < 1e-5
        (tmp_path / 'op.sp').write_text(
            '* operating point\n.include "dut.sp"\nX1 p1 p2 dut\nV1 p1 0 dc 1\nV2 p2 0 dc 0\n'
            '.control\nop\nprint i(V1) i(V2)\n.endc\n.end\n', encoding='utf-8')
        r = subprocess.run([exe, '-b', 'op.sp'], cwd=tmp_path, capture_output=True, text=True,
                           timeout=60)
        out = r.stdout + r.stderr
        assert ('singular matrix' not in out) == clean
        if clean:
            currents = [float(ln.split('=')[1]) for ln in out.splitlines() if ln.startswith('i(v')]
            assert len(currents) == 2 and max(abs(i) for i in currents) < 1e-12


def test_cli_dc_options(tmp_path, capsys):
    src = tmp_path / 'dut.s2p'
    make_network(2, PARTS, f=F).write_touchstone(str(src.with_suffix('')))
    main([str(src), '--dc', 'no-ground', '--float-resistor', '2e9', '--noplot'])
    out = capsys.readouterr().out
    assert 'dc=no-ground' in out and 'DC paths no-ground' in out and 'none to ground' in out
    assert os.path.exists(tmp_path / 'dut_crcat.sp')
