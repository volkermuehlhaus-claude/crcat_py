"""Data check, verdict with suggestions, size against accuracy, re-selection, testbench."""

import copy
import os
import pathlib
import shutil

import numpy as np
import pytest
import skrf

from crcat import advice, check, model, netlist, testbench
from test_model import TEE, TWO_PORT, make_network, make_tee, make_transformer

DATA = pathlib.Path(__file__).parent / 'data'


def line(z0, delay, f, loss=0.0):
    """Matched-reference S-parameters of a line (ABCD), with a little series loss."""
    theta = 2 * np.pi * f * delay * (1 - 1j * loss)
    a, b, c = np.cos(theta), 1j * z0 * np.sin(theta), 1j * np.sin(theta) / z0
    den = 2 * a + b / 50 + c * 50
    s = np.empty((f.size, 2, 2), dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = (b / 50 - c * 50) / den
    s[:, 0, 1] = s[:, 1, 0] = 2 / den
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit='hz'), s=s, z0=50)


def _line_y(z0, delay, f, loss=0.01):
    theta = 2 * np.pi * f * delay * (1 - 1j * loss)
    return -1j / (z0 * np.tan(theta)), 1j / (z0 * np.sin(theta))


def _network(y, f):
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit='hz'), s=model._y2s(y, 50.0), z0=50)


def two_lines(f, a=(50.0, 6e-12), b=(70.0, 4e-12), coupling=0.0):
    """4-port: line a from port 1 to 3, line b from 2 to 4, a capacitor between the lines
    at each end."""
    y = np.zeros((f.size, 4, 4), dtype=complex)
    for (i, j), (z0, delay) in (((0, 2), a), ((1, 3), b)):
        d, o = _line_y(z0, delay, f)
        y[:, i, i] += d
        y[:, j, j] += d
        y[:, i, j] += o
        y[:, j, i] += o
    jwc = 2j * np.pi * f * coupling
    for i, j in ((0, 1), (2, 3)):
        y[:, i, i] += jwc
        y[:, j, j] += jwc
        y[:, i, j] -= jwc
        y[:, j, i] -= jwc
    return _network(y, f)


def star(f, arms, c_centre=0.0):
    """n-port: a line (z0, delay) from each port to one common point with a capacitor."""
    n = len(arms)
    y = np.zeros((f.size, n + 1, n + 1), dtype=complex)
    for i, (z0, delay) in enumerate(arms):
        d, o = _line_y(z0, delay, f)
        y[:, i, i] += d
        y[:, n, n] += d
        y[:, i, n] += o
        y[:, n, i] += o
    y[:, n, n] += 2j * np.pi * f * c_centre
    reduced = y[:, :n, :n] - (np.einsum('fi,fj->fij', y[:, :n, n], y[:, n, :n])
                              / y[:, n, n][:, None, None])
    return _network(reduced, f)


# --- data check -----------------------------------------------------------------------------

def test_check_kinds_and_verdicts():
    f = np.linspace(0.5e9, 40e9, 80)
    ind = make_network(2, {(1,): ('C', [20e-15]), (2,): ('C', [20e-15]),
                           (1, 2): ('SerRL', [2.0, 0.4e-9])}, f=f)
    c = check.check(ind)
    assert c.kind == 'inductor' and c.verdict == 'suited' and c.form == 'pi' and c.fmax is None
    assert abs(c.line['L'] - 0.4e-9) < 0.02e-9 and abs(c.line['C'] - 40e-15) < 2e-15
    assert c.srf is None and c.nports == 2 and c.points == 80 and c.passive
    assert c.lines()[0].endswith('inductor') and c.lines()[-1].startswith('suited')
    assert c.brief() == ['inductor, 2 port(s), 0.5 to 40 GHz, 80 points', '3 branches, pi form']
    assert sum(len(x) for x in c.brief()) < sum(len(x) for x in c.lines())
    cap = make_network(2, {(1, 2): ('SerRC', [1.0, 100e-15])}, f=f)
    assert check.check(cap).kind == 'capacitor'
    one = skrf.Network(str(DATA / 'XXXII-39.y1p'))
    assert check.check(one).kind == 'one-port, inductive'
    with pytest.raises(ValueError):
        check.check(ind, fmin=39.5e9)


def test_check_finds_a_line_and_the_frequency_to_stop_at():
    f = np.linspace(0.5e9, 100e9, 200)
    c = check.check(line(50.0, 6e-12, f, loss=0.01))       # 216 deg at 100 GHz
    assert c.kind == 'line' and c.verdict == 'limited'
    assert abs(c.line['z0'] - 50) < 2 and abs(c.line['delay'] - 6e-12) < 0.2e-12
    assert 200 < c.line['degrees'] < 230
    assert abs(c.fmax - 100e9 * 45 / c.line['degrees']) < 1e6      # 45 deg
    assert 'distributed' in c.reasons[0]
    assert ' deg long at 100 GHz: a line at each port is added (' in c.brief()[1]
    assert len(c.brief()) == 2 and len(c.brief()[1]) < 0.5 * len(c.reasons[0])
    short = check.check(line(50.0, 6e-12, f, loss=0.01), fmax=30e9)   # 65 deg: not yet distributed
    assert short.verdict == 'suited' and short.fmax is None


def test_check_tee_form_and_coupling():
    c = check.check(make_tee(*TEE), fmax=10e9)             # 37 deg as a line
    assert c.form == 'tee' and c.verdict == 'suited' and len(c.unphysical('pi')) >= 1
    assert check.check(make_network(2, TWO_PORT)).form == 'pi'
    c = check.check(make_transformer(0.3e-9, 0.5e-9, -0.6, 1.0, 2.0))
    assert c.verdict == 'suited' and len(c.couplings) == 1
    assert c.couplings[0][0] == (1, 2, 0) and abs(c.couplings[0][1] + 0.6) < 0.01
    assert 'coupled coils will be tried' in ' '.join(c.reasons)


def test_check_not_suited():
    # many branches with a negative conductance: no pi of positive elements
    parts = {(1, 2): ('ParRC', [-300.0, 20e-15]), (1, 3): ('ParRC', [-400.0, 30e-15]),
             (2, 3): ('ParRC', [-500.0, 10e-15]), (1,): ('ParRC', [100.0, 10e-15]),
             (2,): ('ParRC', [100.0, 10e-15]), (3,): ('ParRC', [100.0, 10e-15])}
    c = check.check(make_network(3, parts))
    assert c.verdict == 'not suited' and c.kind == 'multi-port'
    assert '3 of 6 branches' in c.reasons[0]


# --- automatic line at the ports ---------------------------------------------------------------

def test_automatic_line_for_a_distributed_part():
    f = np.linspace(0.5e9, 100e9, 120)
    net = line(70.0, 6e-12, f, loss=0.01)                  # 216 deg at 100 GHz, not 50 ohm
    c = check.check(net)
    assert abs(c.port_line['z0'] - 70) < 2
    assert abs(c.port_line['delay'] - 6e-12 * (1 - 30 / 216) / 2) < 0.1e-12
    assert 'at each port is tried' in c.reasons[0]
    m = model.extract(net, model.Options(coupling=False))  # delay 'auto' is the default
    assert m.delay == c.port_line['delay'] and m.line_z0 == c.port_line['z0']
    assert m.error_db() < -30 and 'chosen automatically' in m.line_note
    assert m.line_note in m.summary()
    assert advice.assess(m, net, c).level in ('good', 'fair')
    assert f'Z0={m.line_z0:.6g} TD={m.delay:.6g}' in netlist.to_spice(m, 'x')
    # a part that is short in the band gets no line, and neither does 'off'
    assert check.check(net, fmax=30e9).port_line is None
    short = model.extract(net, model.Options(coupling=False, fmax=30e9))
    assert short.delay == 0 and short.line_note == ''


def test_automatic_line_is_dropped_when_it_does_not_help(monkeypatch):
    # the check proposes a line of a wrong impedance: the model without it is kept
    f = np.linspace(0.5e9, 100e9, 120)
    net = line(50.0, 6e-12, f, loss=0.01)
    real = check.check

    def wrong(*args, **kwargs):
        c = real(*args, **kwargs)
        c.port_lines = [{'z0': 8.0, 'delay': 0.2e-12}] * 2
        return c
    monkeypatch.setattr(check, 'check', wrong)
    m = model.extract(net, model.Options(coupling=False, max_elements=2, refine=False))
    plain = model.extract(net, model.Options(coupling=False, max_elements=2, refine=False,
                                             delay='off'))
    if m.delay:                                  # kept only if it is the better one
        assert m.error() < plain.error() and 'without it' in m.line_note
    else:
        assert np.isclose(m.error(), plain.error()) and m.line_note.startswith('no line')


# --- lines at the ports of a multi-port ---------------------------------------------------------

F100 = np.linspace(0.5e9, 100e9, 120)
STAR = [(60.0, 3e-12), (40.0, 3e-12), (50.0, 3e-12)]         # 216 deg between two ports


def test_shift_ports_with_one_line_per_port():
    net = star(F100, STAR)
    zl, td = np.array([60.0, 40.0, 50.0]), np.array([1e-12, 2e-12, 0.0])
    other = net.copy()
    other.renormalize(zl, s_def='power')
    assert np.allclose(model._renormalize(net.s, np.full(3, 50.0), zl), other.s)
    longer = model.shift_ports(net.s, F100, td, zl, 50.0)
    assert not np.allclose(longer, net.s)
    assert np.allclose(model.shift_ports(longer, F100, -td, zl, 50.0), net.s)
    # one value for all ports, reference impedance: the plain phase shift
    assert np.allclose(model.shift_ports(net.s, F100, 1e-12),
                       net.s * np.exp(-4j * np.pi * F100 * 1e-12)[:, None, None])


def test_check_finds_the_lines_of_a_multi_port():
    c = check.check(star(F100, STAR))
    (g,) = c.line_groups
    assert g['ports'] == [1, 2, 3] and 210 < g['degrees'] < 222 and g['fit_db'] < -40
    assert np.allclose(g['z0'], [60, 40, 50], rtol=0.02) and np.allclose(g['delay'], 3e-12, rtol=0.02)
    assert np.allclose([v['delay'] for v in c.port_lines], 3e-12 * (1 - 30 / g['degrees']), rtol=0.02)
    assert c.verdict == 'limited' and 'a line at each of these ports is tried' in c.reasons[0]
    assert any(ln.startswith('as lines to one point: p1 60') for ln in c.lines())
    # arms of different length, a fourth arm, a capacitor where they meet
    arms = [(50.0, 2e-12), (60.0, 3e-12), (40.0, 3e-12), (50.0, 4e-12)]
    (g,) = check.check(star(F100, arms, c_centre=20e-15)).line_groups
    assert np.allclose(g['z0'], [50, 60, 40, 50], rtol=0.03)
    assert np.allclose(g['delay'], [2e-12, 3e-12, 3e-12, 4e-12], rtol=0.03)
    # two separate lines in a 4-port: two groups, each port the line of its group
    c = check.check(two_lines(F100))
    assert [g['ports'] for g in c.line_groups] == [[1, 3], [2, 4]]
    assert np.allclose([v['z0'] for v in c.port_lines], [50, 70, 50, 70], rtol=0.02)
    # nothing for a short part, and nothing when the data start where it is long already
    assert check.check(star(F100, STAR), fmax=30e9).port_lines is None
    assert check.check(star(np.linspace(60e9, 100e9, 80), STAR)).port_lines is None
    lumped = make_network(3, {(1, 2): ('SerRL', [1.0, 0.3e-9]), (2, 3): ('SerRL', [1.0, 0.2e-9]),
                              (1,): ('C', [20e-15]), (3,): ('C', [30e-15])})
    assert check.check(lumped).port_lines is None


def test_automatic_lines_for_a_star():
    net = star(F100, STAR)
    m = model.extract(net, model.Options(coupling=False, max_elements=3))
    assert m.error_db() < -35 and not m.same_lines()
    assert np.allclose(m.port_z0, [60, 40, 50], rtol=0.02) and np.all(m.port_delay > 2e-12)
    assert m.summary().count('lines at the ports') == 2 and 'chosen automatically' in m.line_note
    assert [v['port'] for v in m.to_dict()['port_lines']] == [1, 2, 3]
    text = netlist.to_spice(m, 'x')
    assert f'T_p2 p2 0 p2_c 0 Z0={m.port_z0[1]:.6g} TD={m.port_delay[1]:.6g}' in text
    assert f'tline z0={m.port_z0[0]:.6g} td={m.port_delay[0]:.6g}' in netlist.to_spectre(m, 'x')
    plain = model.extract(net, model.Options(coupling=False, max_elements=3, delay='off'))
    assert plain.delay == 0 and plain.error_db() > -15


def test_automatic_lines_for_two_lines_in_a_four_port():
    m = model.extract(two_lines(F100), model.Options(coupling=False, max_elements=3))
    assert m.error_db() < -35
    assert np.allclose(m.port_z0, [50, 70, 50, 70], rtol=0.02)
    assert np.allclose(m.port_delay[[0, 1]], m.port_delay[[2, 3]], rtol=0.01)


# --- verdict and suggestions ----------------------------------------------------------------

def test_assess_good():
    net = make_network(2, {(1,): ('C', [20e-15]), (2,): ('C', [20e-15]),
                           (1, 2): ('SerRL', [2.0, 0.4e-9])})
    m = model.extract(net)
    a = advice.assess(m, net, check.check(net))
    assert a.level == 'good' and not a.suggestions
    assert 'L within' in a.text and 'Q within' in a.text and 'elements' in a.text
    assert a.quantities['L'] < 0.01 and a.quantities['Q'] < 0.01
    assert a.lines()[0].startswith('good: 0.5 to 40 GHz')


def test_assess_suggests_a_band_for_a_line():
    f = np.linspace(0.5e9, 100e9, 120)
    net = line(50.0, 6e-12, f, loss=0.01)
    looked = check.check(net)
    m = model.extract(net, model.Options(coupling=False, delay='off'))
    a = advice.assess(m, net, looked)
    assert a.level == 'poor' and a.suggestions
    s = a.suggestions[0]
    assert s.label.startswith('Fit to ') and 10e9 < s.changes['fmax'] < 35e9
    # taking the suggestion gives a model that is not poor any more
    opt = model.Options(coupling=False, delay='off', **s.changes)
    again = model.extract(net, opt)
    assert advice.assess(again, net).level in ('good', 'fair')
    assert again.error() < 0.2 * m.error()


def test_assess_suggests_negative_reactances_for_coupling_without_coils():
    net = make_transformer(0.3e-9, 0.5e-9, -0.6, 1.0, 2.0)
    m = model.extract(net, model.Options(form='pi', coupling=False))
    a = advice.assess(m, net)
    assert a.level != 'good'
    assert {'require': 'positive-r'} in [s.changes for s in a.suggestions]


# --- size against accuracy, re-selection ------------------------------------------------------

def skin_inductor():
    f = np.linspace(0.5e9, 40e9, 60)
    return make_network(2, {(1,): ('C', [20e-15]), (2,): ('C', [25e-15]),
                            (1, 2): ('X-1', [0.5, 0.4e-9, 2.0, 0.1e-9])}, f=f)


def test_tradeoff_and_apply():
    net = skin_inductor()
    m = model.extract(net)
    before = (m.n_elements(), m.error())
    points = m.tradeoff()
    assert (m.n_elements(), m.error()) == before             # the model itself is untouched
    assert len(points) >= 3
    assert [p.n_elements for p in points] == sorted({p.n_elements for p in points})
    assert all(b.error < a.error for a, b in zip(points, points[1:]))
    assert points[0].n_elements == 0 and points[-1].error_db < -40
    small = points[1]
    m.apply(small)
    assert m.n_elements() == small.n_elements and np.isclose(m.error(), small.error)
    assert len(netlist.to_spice(m, 'x').splitlines()) > 3


def test_reselect_without_a_new_search():
    net = skin_inductor()
    m = model.extract(net, model.Options(target_db=-30))
    n0 = m.n_elements()
    strict = copy.copy(m.options)
    strict.target_db, strict.loss_percent = -60.0, 5.0
    m.reselect(strict)
    assert m.n_elements() > n0 and m.options.target_db == -60
    assert all(b.chosen is b.candidates[0] for b in m.free_branches())
    loose = copy.copy(m.options)
    loose.target_db, loose.loss_percent, loose.reactance_percent = -15.0, 0.0, 0.0
    m.reselect(loose)
    assert m.n_elements() < n0
    few = copy.copy(m.options)
    few.max_elements = 1
    m.reselect(few)
    assert all(b.chosen.n_elements <= 1 for b in m.free_branches())


# --- testbench --------------------------------------------------------------------------------

def find_ngspice():
    for name in ('ngspice_con', 'ngspice'):
        exe = shutil.which(name)
        if exe:
            return exe
    for folder in (r'C:\Program Files\Qucs-S\bin', r'C:\Program Files\ngspice\bin',
                   os.environ.get('NGSPICE_HOME', '')):
        exe = pathlib.Path(folder) / 'ngspice_con.exe'
        if folder and exe.exists():
            return str(exe)
    return None


def test_testbench_text():
    m = model.extract(make_network(2, TWO_PORT), model.Options(pick='best'))
    text = testbench.ngspice(m, 'dut.sp', 'dut')
    assert '.include "dut.sp"' in text and 'X1 p1 p2 dut' in text
    assert 'V2 p2 0 dc 0 ac 1 portnum 2 z0 50' in text and '.sp lin 80 500000000 4e+10' in text
    text = testbench.spectre(m, 'dut.scs', 'dut')
    assert 'PORT2 (p2 0) port r=50 num=2' in text and 'sp1 sp ports=[PORT1 PORT2]' in text


@pytest.mark.parametrize('case', ['pi', 'coupled', 'line', 'auto line', 'star'])
def test_netlist_in_ngspice(case, tmp_path):
    """The written netlist, simulated by ngspice, gives the S-parameters of the model."""
    exe = find_ngspice()
    if exe is None:
        pytest.skip('ngspice not found')
    if case == 'pi':
        m = model.extract(make_network(2, TWO_PORT), model.Options(pick='best'))
    elif case == 'coupled':
        m = model.extract(make_transformer(0.3e-9, 0.3e-9, -0.4, 1.0, 1.0, rc=0.5),
                          model.Options(form='pi', pick='best'))
        assert m.couplings
    elif case == 'star':
        m = model.extract(star(F100, STAR), model.Options(coupling=False, max_elements=3))
        assert not m.same_lines()
    elif case == 'auto line':
        m = model.extract(line(70.0, 6e-12, np.linspace(0.5e9, 100e9, 120), loss=0.01),
                          model.Options(coupling=False))
        assert m.delay and m.line_z0 != 50
    else:
        f = np.linspace(0.5e9, 100e9, 120)
        core = make_network(2, TWO_PORT, f=f)
        net = skrf.Network(frequency=core.frequency, s=model.shift_ports(core.s, f, 1.5e-12), z0=50)
        m = model.extract(net, model.Options(delay=1.5e-12, pick='best'))
    (tmp_path / 'dut.sp').write_text(netlist.to_spice(m, 'dut'), encoding='utf-8')
    (tmp_path / 'tb.sp').write_text(testbench.ngspice(m, 'dut.sp', 'dut'), encoding='utf-8')
    f, s = testbench.run_ngspice(tmp_path / 'tb.sp', exe)
    assert np.allclose(f, m.freq, rtol=1e-6)
    assert np.abs(s - m.s_matrix(m.freq)).max() < 1e-5      # the netlist has six digits
