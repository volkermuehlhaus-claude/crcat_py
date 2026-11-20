"""S-parameter testbenches for a written netlist, to confirm a model in the simulator.

``ngspice`` (version 37 or later, which has the .sp analysis) and ``spectre``. The ngspice
bench was run with ngspice 46; the Spectre one is written from the documentation and has
not been run.
"""

import pathlib
import subprocess

import numpy as np

__all__ = ['ngspice', 'spectre', 'run_ngspice']


def _ports(model):
    return [f'p{i + 1}' for i in range(model.nports)]


def _points(model):
    return int(model.freq.size), float(model.freq[0]), float(model.freq[-1])


def ngspice(model, netlist_file, name, output='tb_out.txt'):
    """ngspice input: the subcircuit between ports of the reference impedance, .sp
    analysis over the fit band, all S-parameters written to ``output`` (wrdata)."""
    n, f0, f1 = _points(model)
    ports = _ports(model)
    cols = ' '.join(f'real(S_{i}_{j}) imag(S_{i}_{j})'
                    for i in range(1, model.nports + 1) for j in range(1, model.nports + 1))
    lines = [f'* S-parameter testbench for {name} (crcat)', f'.include "{netlist_file}"',
             f'X1 {" ".join(ports)} {name}']
    lines += [f'V{k} {p} 0 dc 0 ac 1 portnum {k} z0 {model.z0:g}' for k, p in enumerate(ports, 1)]
    lines += [f'.sp lin {n} {f0:.9g} {f1:.9g}', '.control', 'run', f'wrdata {output} {cols}',
              '.endc', '.end']
    return '\n'.join(lines) + '\n'


def spectre(model, netlist_file, name, output=None):
    """Spectre input: ports, sp analysis over the fit band, Touchstone output."""
    n, f0, f1 = _points(model)
    ports = _ports(model)
    output = output or f'{name}_tb.s{model.nports}p'
    lines = [f'// S-parameter testbench for {name} (crcat)', 'simulator lang=spectre',
             f'include "{netlist_file}"', f'X1 ({" ".join(ports)}) {name}']
    lines += [f'PORT{k} ({p} 0) port r={model.z0:g} num={k}' for k, p in enumerate(ports, 1)]
    lines.append(f'sp1 sp ports=[{" ".join(f"PORT{k}" for k in range(1, model.nports + 1))}] '
                 f'start={f0:.9g} stop={f1:.9g} lin={n} file="{output}" datafmt=touchstone')
    return '\n'.join(lines) + '\n'


def run_ngspice(bench_file, executable='ngspice', timeout=120):
    """Run an ngspice bench made by :func:`ngspice` and read its output.

    Returns (frequencies in Hz, S of shape (frequency, n, n)). The output file is the
    one named in the bench (in the folder of the bench)."""
    bench_file = pathlib.Path(bench_file)
    text = bench_file.read_text(encoding='utf-8')
    out = next(ln.split()[1] for ln in text.splitlines() if ln.startswith('wrdata'))
    n = sum(ln.startswith('V') and 'portnum' in ln for ln in text.splitlines())
    r = subprocess.run([executable, '-b', bench_file.name], cwd=bench_file.parent,
                       capture_output=True, text=True, timeout=timeout)
    path = bench_file.parent / out
    if not path.exists():
        raise RuntimeError(f'ngspice wrote no output: {r.stdout[-500:]} {r.stderr[-500:]}')
    d = np.loadtxt(path)
    values = d[:, 1::2]                              # wrdata repeats the frequency per column
    s = (values[:, 0::2] + 1j * values[:, 1::2]).reshape(-1, n, n)
    return d[:, 0], s
