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
    only_pi = dict(form='pi', coupling=False)
    m = model.extract(net, model.Options(**only_pi))         # positive: cannot fit
    b = m.branches[2]
    assert b.ports == (1, 2) and b.target_missed and 'limit not reached' in b.warning()
    assert all(v > 0 for *_, v in m.elements())
    m = model.extract(net, model.Options(require='positive-r', pick='best', **only_pi))
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


# --- the two error measures ------------------------------------------------------------

def low_impedance_inductor():
    """A small inductor with skin effect: nearly a through in 50 ohm (|S11| < 0.03)."""
    return make_network(2, {(1, 2): ('X-1', [0.4, 60e-12, 1.0, 15e-12])},
                        f=np.linspace(0.5e9, 60e9, 80))


def test_port_loss_error_protects_the_resistance():
    # 0.3 ohm in series with 60 pH, logarithmic frequencies: nearly a through in 50 ohm
    f = np.geomspace(0.05e9, 20e9, 60)
    net = make_network(2, {(1, 2): ('SerRL', [0.3, 60e-12])}, f=f)
    m = model.extract(net, model.Options(form='pi', coupling=False))
    b = m.branches[2]
    c = b.chosen
    assert c.label.startswith('CR7 SerRL') and c.loss < 1e-6 and c.score <= 1
    assert np.allclose(c.component.values[0], [0.3, 60e-12], rtol=1e-5)
    assert 'loss' in m.summary() and 'X' in m.summary()
    # the same branch without its resistance: fine by the S-parameters, the loss of the
    # ports entirely wrong (it is all in this branch)
    s = model._BranchSearch.__new__(model._BranchSearch)
    y_l = 1 / (2j * np.pi * f * 60e-12)
    sd = model._Data(f, (net.y + net.y.transpose(0, 2, 1)) / 2, net.s, 50.0, None, m.options,
                     None)

    def diff(yc):
        y = sd.y.copy()
        model._stamp(y, (1, 2), yc - b.y)
        return sd.y_diff(y)
    search = model._BranchSearch(b, diff, sd, crcat.load('base'), m.options, lambda *a: None)
    score, s_error, ex, el = search.measure(y_l)
    assert s_error < 0.01 and el > 0.99                     # passes -40 dB, loss 100 % off
    assert score > 3
    with pytest.raises(ValueError):
        model.extract(net, model.Options(loss_percent=-1))


def test_netlist_admittance_matches_the_catalog():
    cat = crcat.load('base')
    s = 2j * np.pi * F
    for name in ('SerRLC', 'XXXII-39', 'XXXIII-40', 'X-1'):
        comp = Component(name, cat, values=[v * 1e-9 if k else v for k, v in
                                            enumerate([3.0] + [0.2] * 4)][:len(cat[name].symVn)])
        y = model.netlist_admittance(comp.network.connections, comp.values[0], s)
        assert np.allclose(y, comp.eval(F, 0), rtol=1e-9), name


def test_refinement_improves_and_keeps_signs():
    # a branch whose resistance rises with frequency (skin effect): a series R-L from the
    # closed-form fit gets the reactance, the refinement moves its R towards the data
    f = np.linspace(0.5e9, 40e9, 60)
    net = make_network(2, {(1, 2): ('X-1', [0.5, 0.4e-9, 2.0, 0.1e-9])}, f=f)
    opt = dict(form='pi', coupling=False, max_elements=2)
    plain = model.extract(net, model.Options(refine=False, **opt)).branches[2].chosen
    refined = model.extract(net, model.Options(**opt)).branches[2].chosen
    assert refined.refined and refined.score < plain.score
    assert refined.loss < plain.loss
    assert all(v > 0 for v in refined.component.values[0])


def test_negligible_branch_stays_open_with_the_port_limits():
    # a tiny capacitance to ground: no network is wanted for it, whatever its own error
    parts = {(1,): ('C', [0.02e-15]), (1, 2): ('SerRL', [2.0, 0.4e-9])}
    m = model.extract(make_network(2, parts))
    assert chosen(m)[(1,)].label == 'open'


# --- plots ------------------------------------------------------------------------------

def test_plot_views():
    from crcat import plots
    net = low_impedance_inductor()
    m = model.extract(net, model.Options(fmax=40e9))
    assert plots.views_for(m) == ['s', 'error', 'lrq']
    fig = plots.compare(m, net)                                    # fit band only
    assert fig.axes[0].get_xlim()[1] < 45 and 'shaded' not in fig.axes[0].get_title()
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == ['S11 = S22', 'S21']                          # coinciding traces merged
    fig = plots.compare(m, net, view='s', all_data=True)
    assert fig.axes[0].get_xlim()[1] > 55 and 'fit band shaded' in fig.axes[0].get_title()
    fig = plots.compare(m, net, view='error')
    assert any('limit -40 dB' == t.get_text() for t in fig.legends[0].get_texts())
    fig = plots.compare(m, net, view='lrq')
    assert [ax.get_ylabel() for ax in fig.axes] == ['series L (pH)', 'series R (ohm)', 'Q at port 1']
    # a flat curve keeps a margin to the frame, and one outlier does not set the scale
    lo, hi = plots._limits(np.full(50, 73.6))
    assert lo == 0 and 80 < hi < 95
    lo, hi = plots._limits(np.r_[1500.0, np.full(99, 5.0)])
    assert hi < 20
    lo, hi = plots._limits(np.linspace(-3, -1, 50))
    assert lo < -3 and -1 < hi < 0
    zs, zin = plots.series_quantities(net.s, 50.0)
    assert np.allclose(zs.imag[0] / (2 * np.pi * net.f[0]), 75e-12, rtol=0.02)
    # a capacitor shows C, a 3-port has no L/R/Q view, many ports show the worst traces only
    cap = make_network(2, {(1, 2): ('SerRC', [1.0, 100e-15])})
    fig = plots.compare(model.extract(cap), cap, view='lrq')
    assert fig.axes[0].get_ylabel() == 'series C (fF)'
    three = make_network(3, {(1,): ('C', [10e-15]), (1, 3): ('SerRL', [1.5, 0.3e-9]),
                             (2, 3): ('SerRL', [1.0, 0.25e-9])})
    m3 = model.extract(three)
    assert plots.views_for(m3) == ['s', 'error']
    with pytest.raises(ValueError):
        plots.compare(m3, three, view='lrq')
    plots.MAX_TRACES, keep = 3, plots.MAX_TRACES
    try:
        fig = plots.compare(m3, three)
        assert len(fig.legends[0].get_texts()) == 3 and 'largest error' in fig.axes[0].get_title()
    finally:
        plots.MAX_TRACES = keep


# --- tee form and coupled coils ---------------------------------------------------------

def make_tee(leg1, leg2, center, f=F, z0=50.0):
    """2-port tee from (catalog name, values) per leg; None = short (legs) or open."""
    cat = crcat.load('base')

    def y(part):
        return Component(part[0], cat, values=part[1]).eval(f, 0)
    za = 0 * f if leg1 is None else 1 / y(leg1)
    zb = 0 * f if leg2 is None else 1 / y(leg2)
    yc = 0 * f if center is None else y(center)
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit='hz'),
                        s=model._tee_s(za, zb, yc, z0), z0=z0)


TEE = (('SerRL', [1.0, 0.2e-9]), ('SerRL', [1.5, 0.15e-9]), ('SerRC', [8.0, 300e-15]))


def test_tee_form_round_trip():
    net = make_tee(*TEE)
    m = model.extract(net, model.Options(form='tee', pick='best'))
    assert m.form == 'tee' and m.error_db() < -150 and m.n_elements() == 6
    assert [b.name for b in m.branches] == ['p1 - c', 'p2 - c', 'c - gnd']
    text = netlist.to_spice(m, 'dut')
    assert np.allclose(spice_s_parameters(text, F), m.s_matrix(F), atol=1e-9)
    assert ' c ' in text or text.count(' c\n')
    with pytest.raises(ValueError):
        model.extract(make_network(1, {(1,): ('C', [1e-14])}), model.Options(form='tee'))


@pytest.mark.slow
def test_auto_takes_the_tee_form_when_pi_fails():
    net = make_tee(*TEE)
    pi = model.extract(net, model.Options(form='pi', coupling=False))
    assert pi.warnings() and pi.error_db() > -40          # a tee is no pi of small branches
    auto = model.extract(net)
    assert auto.form == 'tee' and not auto.warnings() and auto.error() < pi.error()
    # data that are a pi stay a pi
    assert model.extract(make_network(2, TWO_PORT)).form == 'pi'


def test_tee_with_shorted_legs():
    # only the center leg: both series legs are shorts; one merges the center node into
    # p1, the other becomes a resistor of 1e-6 ohm between the ports. Port 1 is then
    # shorted to port 2 and its admittance is infinite: S-parameter error only.
    m = model.extract(make_tee(None, None, TEE[2]),
                      model.Options(form='tee', pick='best', reactance_percent=0,
                                    loss_percent=0))
    assert [b.chosen.label for b in m.branches[:2]] == ['short', 'short']
    assert m.error_db() < -150
    text = netlist.to_spice(m, 'dut')
    assert 'Rshort_p2c p2 p1 1e-06' in text
    assert not any(' c ' in ln + ' ' for ln in text.splitlines() if ln[0] in 'RLC')
    assert np.allclose(spice_s_parameters(text, F), m.s_matrix(F), atol=1e-6)


def make_transformer(l1, l2, k, r1, r2, f=F, z0=50.0, rc=0.0):
    w = 2 * np.pi * f
    z = np.zeros((f.size, 2, 2), dtype=complex)
    z[:, 0, 0] = r1 + rc + 1j * w * l1
    z[:, 1, 1] = r2 + rc + 1j * w * l2
    z[:, 0, 1] = z[:, 1, 0] = rc + 1j * w * k * np.sqrt(l1 * l2)
    y = np.linalg.inv(z)
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit='hz'), s=model._y2s(y, z0), z0=z0)


def test_coupled_coils():
    # opposite winding sense: the pi form needs a negative inductance between the ports
    net = make_transformer(0.3e-9, 0.5e-9, -0.6, 1.0, 2.0)
    plain = model.extract(net, model.Options(form='pi', coupling=False))
    assert plain.warnings()                                # needs a negative inductance
    m = model.extract(net, model.Options(form='pi', pick='best'))
    assert len(m.couplings) == 1 and not m.warnings()
    g = m.couplings[0]
    assert g.ports == (1, 2, 0) and abs(g.k + 0.6) < 1e-3 and g.rc == 0
    assert [c.network.name for c in g.coils] == ['SerRL', 'SerRL']
    assert np.allclose(sorted(g.coils[1].values[0]), [0.5e-9, 2.0], rtol=1e-3)
    assert m.error_db() < -60 and all(v > 0 for _, kind, *_, v in m.elements() if kind != 'K')
    assert [r.name for r in m.rows()] == ['p1 - p2', 'p1, p2 - gnd']
    assert m.rows()[0].chosen.label == 'open'
    text = netlist.to_spice(m, 'dut')
    assert 'K_k12 L_k12a L_k12b -0.6' in text
    assert np.allclose(spice_s_parameters(text, F), m.s_matrix(F), atol=1e-9)
    assert 'mutual_inductor coupling=-0.6' in netlist.to_spectre(m, 'dut')
    d = m.to_dict()
    assert len(d['couplings']) == 1 and len(d['branches']) == 1


@pytest.mark.slow
def test_coupled_coils_with_common_resistor_and_negative_k():
    net = make_transformer(0.3e-9, 0.3e-9, -0.4, 1.0, 1.0, rc=0.5)
    m = model.extract(net, model.Options(form='pi', pick='best'))
    g = m.couplings[0]
    assert abs(g.k + 0.4) < 1e-3 and abs(g.rc - 0.5) < 1e-3 and m.error_db() < -60
    text = netlist.to_spice(m, 'dut')
    assert 'Rc_k12 k12_m 0 0.5' in text
    assert np.allclose(spice_s_parameters(text, F), m.s_matrix(F), atol=1e-9)


# --- netlists ---------------------------------------------------------------------------

def spice_s_parameters(text, f, z0=50.0):
    """S-parameters of a SPICE subcircuit of R, L, C by nodal analysis (independent of
    the model code)."""
    lines = [ln.split() for ln in text.splitlines() if ln and ln[0] not in '*.']
    couplings = [ln for ln in lines if ln[0][0] == 'K']
    lines = [ln for ln in lines if ln[0][0] != 'K']
    coupled = {name for _, a, b, _ in couplings for name in (a, b)}
    head = next(ln.split() for ln in text.splitlines() if ln.upper().startswith('.SUBCKT'))
    ports = head[2:]
    nodes = ports + sorted({n for _, a, b, _ in lines for n in (a, b)} - set(ports) - {'0'})
    idx = {n: k for k, n in enumerate(nodes)}
    s = 2j * np.pi * f
    y = np.zeros((f.size, len(nodes), len(nodes)), dtype=complex)
    def stamp(a, b, c, e, g):                        # current a->b driven by voltage c->e
        for p, sp in ((a, 1), (b, -1)):
            for q, sq in ((c, 1), (e, -1)):
                if p != '0' and q != '0':
                    y[:, idx[p], idx[q]] += sp * sq * g

    by_name = {ln[0]: ln for ln in lines}
    for _, la, lb, k in couplings:
        (_, a1, b1, v1), (_, a2, b2, v2) = by_name[la], by_name[lb]
        l1, l2 = float(v1), float(v2)
        m = float(k) * np.sqrt(l1 * l2)
        det = s * (l1 * l2 - m * m)                  # inverse of s * [[l1, m], [m, l2]]
        stamp(a1, b1, a1, b1, l2 / det)
        stamp(a2, b2, a2, b2, l1 / det)
        stamp(a1, b1, a2, b2, -m / det)
        stamp(a2, b2, a1, b1, -m / det)
    for name, a, b, value in lines:
        if name in coupled:
            continue
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
    assert 'crcat-netlist' in out and 'fmax=30.0' in out
    assert 'model (pi form): 5 elements' in out
    files = {p.name for p in (tmp_path / 'out').iterdir()}
    assert files == {'dut_crcat.sp', 'dut_crcat.scs', 'dut_crcat.s2p', 'dut_crcat.json',
                     'dut_crcat.png', 'dut_crcat_error.png', 'dut_crcat_lrq.png',
                     'dut_crcat_schematic.png'}
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
