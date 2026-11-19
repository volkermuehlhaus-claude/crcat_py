"""Command line: crcat-netlist <touchstone file> -> lumped-element netlist."""

import argparse
import json
import pathlib
import sys

from . import __version__, data, model, netlist
from .component import use_ascii_units


def write_outputs(m, network, outdir, base, formats, name, plot=True):
    """Write netlist(s), model response, JSON, plot and schematic; return the files."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for fmt in formats:
        p = outdir / f'{base}_crcat{netlist.EXTENSION[fmt]}'
        p.write_text(netlist.render(m, fmt, name), encoding='utf-8')
        files.append(p)
    p = outdir / f'{base}_crcat.s{m.nports}p'
    m.to_network(network.f[network.f > 0]).write_touchstone(str(p.with_suffix('')), form='ri')
    files.append(p)
    p = outdir / f'{base}_crcat.json'
    p.write_text(json.dumps(m.to_dict(), indent=1), encoding='utf-8')
    files.append(p)
    if plot:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from .plots import compare
        fig = compare(m, network)
        FigureCanvasAgg(fig)
        p = outdir / f'{base}_crcat.png'
        fig.savefig(p, dpi=120)
        files.append(p)
        from .schematic import figure_model
        fig = figure_model(m)
        FigureCanvasAgg(fig)
        p = outdir / f'{base}_crcat_schematic.png'
        fig.savefig(p, dpi=120)
        files.append(p)
    return files


def main(argv=None):
    d = model.Options()
    ap = argparse.ArgumentParser(
        prog='crcat-netlist',
        description='Lumped-element netlist from S-parameter data: a pi network with one '
                    'RLC network (up to five elements, CRCat catalog) per branch.')
    ap.add_argument('file', help='Touchstone file (.sNp)')
    ap.add_argument('--fmin', type=float, default=None, help='lower end of the fit band, GHz')
    ap.add_argument('--fmax', type=float, default=None, help='upper end of the fit band, GHz')
    ap.add_argument('--require', choices=model.REQUIRE, default=d.require,
                    help='element values: positive = all positive (default); positive-r = '
                         'R positive, L and C may be negative; passive = any sign if the '
                         'branch stays passive; any = no restriction')
    ap.add_argument('--pick', choices=model.PICK, default=d.pick,
                    help='simplest: fewest elements within the error limit (default); '
                         'best: lowest error')
    ap.add_argument('--target-db', type=float, default=d.target_db,
                    help='error limit per branch, RMS S-parameter difference in dB '
                         f'(default {d.target_db:g})')
    ap.add_argument('--tolerance', type=float, default=d.tolerance,
                    help='if the limit cannot be reached: accept up to this factor times '
                         f'the lowest error (default {d.tolerance:g})')
    ap.add_argument('--max-elements', type=int, default=d.max_elements,
                    help=f'most elements per branch, 1..5 (default {d.max_elements})')
    ap.add_argument('--form', choices=model.FORM, default=d.form,
                    help='pi; tee (2-ports); auto (default): pi, and for a 2-port the tee '
                         'form too if a pi branch misses the limit')
    ap.add_argument('--no-coupling', action='store_true',
                    help='do not replace branches by coupled coils (K element)')
    ap.add_argument('--format', choices=netlist.FORMATS + ('both',), default='spice',
                    help='netlist format (default spice)')
    ap.add_argument('--name', default=None, help='subcircuit name (default: from the file)')
    ap.add_argument('-o', '--outdir', default=None,
                    help='output folder (default: the folder of the input file)')
    ap.add_argument('--full-search', action='store_true',
                    help='synthesize every catalog network (slower, same result)')
    ap.add_argument('--noplot', action='store_true', help='do not write the comparison plot')
    ap.add_argument('--version', action='version', version=f'crcat {__version__}')
    args = ap.parse_args(argv)
    try:
        'Ωμ'.encode(sys.stdout.encoding or 'ascii')
    except (UnicodeEncodeError, LookupError):
        use_ascii_units()
    print(f'crcat-netlist {__version__}: ' + ', '.join(f'{k}={v}' for k, v in vars(args).items()))

    path = pathlib.Path(args.file)
    network = data.read(path)
    try:
        opt = model.Options(
            fmin=None if args.fmin is None else args.fmin * 1e9,
            fmax=None if args.fmax is None else args.fmax * 1e9,
            require=args.require, pick=args.pick, target_db=args.target_db,
            tolerance=args.tolerance, max_elements=args.max_elements,
            form=args.form, coupling=not args.no_coupling, full_search=args.full_search)
        m = model.extract(network, opt)
    except ValueError as ex:
        sys.exit(f'error: {ex}')
    m.source = path.name
    print('\n'.join(model.describe(m)))
    formats = netlist.FORMATS if args.format == 'both' else (args.format,)
    name = args.name or netlist.subckt_name(path.stem + '_model')
    files = write_outputs(m, network, args.outdir or path.parent, path.stem, formats, name,
                          plot=not args.noplot)
    for p in files:
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
