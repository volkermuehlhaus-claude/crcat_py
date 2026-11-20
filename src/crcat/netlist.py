"""Netlist text for a :class:`crcat.model.Model`: SPICE and Spectre subcircuits.

Coupled coils are written as a K element (SPICE) or mutual_inductor (Spectre), the lines
at the ports (Options.delay) as T elements (SPICE) or tline (Spectre).

Ports are p1..pN, ground is node 0 (the global ground of both simulators). Values are in
ohm, H and F.
"""

from . import __version__

__all__ = ['to_spice', 'to_spectre', 'FORMATS', 'EXTENSION', 'render', 'subckt_name']

FORMATS = ('spice', 'spectre')
EXTENSION = {'spice': '.sp', 'spectre': '.scs'}
_SPECTRE = {'R': ('resistor', 'r'), 'L': ('inductor', 'l'), 'C': ('capacitor', 'c')}


def subckt_name(text):
    """A valid subcircuit name from any text (file name)."""
    name = ''.join(c if c.isalnum() or c == '_' else '_' for c in text)
    return name if name[:1].isalpha() else 'm_' + name


def _header(model, comment):
    o = model.options
    lines = [f'Lumped model, one RLC network of the CRCat catalog per branch (crcat {__version__})']
    if model.source:
        lines.append(f'Data: {model.source}')
    lines.append(f'Fit band {model.freq[0] / 1e9:.6g} to {model.freq[-1] / 1e9:.6g} GHz, '
                 f'{model.form} form, elements {o.require}, choice {o.pick}')
    lines.append(f'S-parameter error of the model: {model.error_db():.1f} dB rms, '
                 f'reference {model.z0:g} ohm')
    negative = [n for n, kind, *_, v in model.elements() if kind in 'RLC' and v <= 0]
    if negative:
        lines.append('WARNING: negative element values: ' + ' '.join(negative))
    lines += ['WARNING: ' + w for w in model.warnings()]
    return [f'{comment} {line}' for line in lines]


def _ports(model):
    return [f'p{i + 1}' for i in range(model.nports)]


def _line_z0(model, port):
    """Impedance of the line at a port ('p3')."""
    return float(model.port_z0[int(port[1:]) - 1])


def to_spice(model, name='crcat_model'):
    """SPICE subcircuit (ngspice, Xyce, HSPICE)."""
    lines = _header(model, '*')
    lines.append(f'.SUBCKT {name} {" ".join(_ports(model))}')
    for comment, part in model.sections():
        lines.append('* ' + comment)
        for ename, kind, n1, n2, value in part:      # for K: the two inductor names
            if kind == 'T':                          # line between a port and the lumped part
                lines.append(f'{ename} {n1} 0 {n2} 0 Z0={_line_z0(model, n1):.6g} TD={value:.6g}')
                continue
            lines.append(f'{ename} {n1} {n2} {value:.6g}')
    lines.append(f'.ENDS {name}')
    return '\n'.join(lines) + '\n'


def to_spectre(model, name='crcat_model'):
    """Spectre subcircuit."""
    lines = _header(model, '//')
    lines += ['simulator lang=spectre', f'subckt {name} ({" ".join(_ports(model))})']
    for comment, part in model.sections():
        lines.append('// ' + comment)
        for ename, kind, n1, n2, value in part:
            if kind == 'K':
                lines.append(f'    {ename} mutual_inductor coupling={value:.6g} '
                             f'ind1={n1} ind2={n2}')
                continue
            if kind == 'T':
                lines.append(f'    {ename} ({n1} 0 {n2} 0) tline z0={_line_z0(model, n1):.6g} '
                             f'td={value:.6g}')
                continue
            prim, par = _SPECTRE[kind]
            lines.append(f'    {ename} ({n1} {n2}) {prim} {par}={value:.6g}')
    lines.append(f'ends {name}')
    return '\n'.join(lines) + '\n'


def render(model, fmt, name='crcat_model'):
    if fmt == 'spice':
        return to_spice(model, name)
    if fmt == 'spectre':
        return to_spectre(model, name)
    raise ValueError(f'unknown netlist format {fmt!r}: use one of {FORMATS}')
