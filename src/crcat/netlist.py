"""Netlist text for a :class:`crcat.model.Model`: SPICE and Spectre subcircuits.

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
                 f'elements {o.require}, choice {o.pick}')
    lines.append(f'S-parameter error of the model: {model.error_db():.1f} dB rms, '
                 f'reference {model.z0:g} ohm')
    negative = [n for n, *_, v in model.elements() if v <= 0]
    if negative:
        lines.append('WARNING: negative element values: ' + ' '.join(negative))
    lines += ['WARNING: ' + b.warning() for b in model.branches if b.warning()]
    return [f'{comment} {line}' for line in lines]


def _ports(model):
    return [f'p{i + 1}' for i in range(model.nports)]


def _branch_comment(b):
    return f'branch {b.name}: {b.chosen.label}, error {b.chosen.error_db:.1f} dB'


def to_spice(model, name='crcat_model'):
    """SPICE subcircuit (ngspice, Xyce, HSPICE)."""
    lines = _header(model, '*')
    lines.append(f'.SUBCKT {name} {" ".join(_ports(model))}')
    for b in model.branches:
        lines.append('* ' + _branch_comment(b))
        for ename, kind, n1, n2, value in b.elements():
            lines.append(f'{ename} {n1} {n2} {value:.6g}')
    lines.append(f'.ENDS {name}')
    return '\n'.join(lines) + '\n'


def to_spectre(model, name='crcat_model'):
    """Spectre subcircuit."""
    lines = _header(model, '//')
    lines += ['simulator lang=spectre', f'subckt {name} ({" ".join(_ports(model))})']
    for b in model.branches:
        lines.append('// ' + _branch_comment(b))
        for ename, kind, n1, n2, value in b.elements():
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
