"""Command line: crcat-netlist <touchstone file> -> lumped-element netlist."""

import argparse
import json
import pathlib
import sys

from . import __version__, advice, check, data, model, netlist, testbench
from .component import use_ascii_units


def write_outputs(m, network, outdir, base, formats, name, plot=True, all_data=False):
    """Write netlist(s) with testbenches, model response, JSON, plots and schematic;
    return the files."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for fmt in formats:
        p = outdir / f'{base}_crcat{netlist.EXTENSION[fmt]}'
        p.write_text(netlist.render(m, fmt, name), encoding='utf-8')
        files.append(p)
        bench = outdir / f'{base}_crcat_tb{netlist.EXTENSION[fmt]}'      # S-parameter testbench
        make = testbench.ngspice if fmt == 'spice' else testbench.spectre
        bench.write_text(make(m, p.name, name), encoding='utf-8')
        files.append(bench)
    p = outdir / f'{base}_crcat.s{m.nports}p'
    m.to_network(network.f[network.f > 0]).write_touchstone(str(p.with_suffix('')), form='ri')
    files.append(p)
    p = outdir / f'{base}_crcat.json'
    p.write_text(json.dumps(m.to_dict(), indent=1), encoding='utf-8')
    files.append(p)
    if plot:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from .plots import compare
        from .plots import views_for
        for view in [v for v in views_for(m) if v != 'size']:
            fig = compare(m, network, view=view, all_data=all_data)
            FigureCanvasAgg(fig)
            p = outdir / (f'{base}_crcat.png' if view == 's' else f'{base}_crcat_{view}.png')
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
    ap.add_argument('--ftarget', type=float, default=None,
                    help='frequency of use, GHz: the error limits are three times tighter '
                         'around it, and the result is also reported there')
    ap.add_argument('--quality', choices=tuple(model.QUALITY), default='standard',
                    help='error limits and choice in one word: small (fewer elements, more '
                         'error), standard (default), accurate (lowest error); --pick and '
                         'the three limit options override it')
    ap.add_argument('--keep-low', action='store_true',
                    help='without --fmin: also fit the DC region at the low end, where the '
                         'S-parameters do not change with frequency (left out by default)')
    ap.add_argument('--require', choices=model.REQUIRE, default=d.require,
                    help='element values: positive = all positive (default); positive-r = '
                         'R positive, L and C may be negative; passive = any sign if the '
                         'branch stays passive; any = no restriction')
    ap.add_argument('--pick', choices=model.PICK, default=None,
                    help='simplest: fewest elements within the error limit (default); '
                         'best: lowest error')
    ap.add_argument('--target-db', type=float, default=None,
                    help='S-parameter error limit per branch, RMS S-parameter difference in dB '
                         f'(default {d.target_db:g})')
    ap.add_argument('--reactance-percent', type=float, default=None,
                    help='reactance error limit of a branch: RMS of Im dYii / |Yii| of the '
                         f'port admittances, percent; 0 = off (default {d.reactance_percent:g})')
    ap.add_argument('--loss-percent', type=float, default=None,
                    help='loss error limit of a branch: RMS of Re dYii / Re Yii of the port '
                         f'admittances, percent; 0 = off (default {d.loss_percent:g})')
    ap.add_argument('--no-refine', action='store_true',
                    help='keep the element values of the closed-form synthesis')
    ap.add_argument('--tolerance', type=float, default=d.tolerance,
                    help='if the limits cannot be reached: accept up to this factor times '
                         f'the best candidate (default {d.tolerance:g})')
    ap.add_argument('--max-elements', type=int, default=d.max_elements,
                    help=f'most elements per branch, 1..5 (default {d.max_elements})')
    ap.add_argument('--form', choices=model.FORM, default=d.form,
                    help='pi; tee (2-ports); auto (default): pi, and for a 2-port the tee '
                         'form too if a pi branch misses the limit')
    ap.add_argument('--delay', default='auto',
                    help="line at each port, removed before the fit and put back in the netlist "
                         "as a line element: auto (default: only for a 2-port that is "
                         "distributed in the fit band), off, or the delay in ps")
    ap.add_argument('--line-z0', type=float, default=None,
                    help='impedance in ohm of the line of a given --delay (default: the '
                         'reference impedance of the data)')
    ap.add_argument('--sequential', action='store_true',
                    help='pi form: fit the branches to ground to the port admittance minus the '
                         'models (not the data) of the branches between ports')
    ap.add_argument('--dc', choices=model.DC, default=d.dc,
                    help='safeguard against leakage at DC. data (default): a branch that is '
                         'open at DC in the data gets no network with a DC path; no-ground: '
                         'also never a DC path from a port to ground; off: no such rule')
    ap.add_argument('--float-resistor', default='auto',
                    help='resistor in ohm to ground at internal nodes without a DC path, for '
                         'the operating point of the simulator: auto (default: 1e9 in SPICE '
                         'netlists, none in Spectre netlists), 0 for none, or a value')
    ap.add_argument('--no-coupling', action='store_true',
                    help='do not replace branches by coupled coils (K element)')
    ap.add_argument('--format', choices=netlist.FORMATS + ('both',), default='spice',
                    help='netlist format (default spice)')
    ap.add_argument('--name', default=None, help='subcircuit name (default: from the file)')
    ap.add_argument('-o', '--outdir', default=None,
                    help='output folder (default: the folder of the input file)')
    ap.add_argument('--full-search', action='store_true',
                    help='synthesize every catalog network (slower, same result)')
    ap.add_argument('--check-only', action='store_true',
                    help='only look at the data (kind of part, size, suited or not): no fit')
    ap.add_argument('--noplot', action='store_true', help='do not write the plots')
    ap.add_argument('--plot-all-data', action='store_true',
                    help='plots over all frequencies of the data, not only the fit band')
    ap.add_argument('--version', action='version', version=f'crcat {__version__}')
    args = ap.parse_args(argv)
    for key, value in model.QUALITY[args.quality].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    try:
        'Ωμ'.encode(sys.stdout.encoding or 'ascii')
    except (UnicodeEncodeError, LookupError):
        use_ascii_units()
    print(f'crcat-netlist {__version__}: ' + ', '.join(f'{k}={v}' for k, v in vars(args).items()))

    path = pathlib.Path(args.file)
    network = data.read(path)
    try:
        looked = check.check(network, None if args.fmin is None else args.fmin * 1e9,
                             None if args.fmax is None else args.fmax * 1e9, args.target_db,
                             not args.keep_low)
    except ValueError as ex:
        sys.exit(f'error: {ex}')
    print('data check: ' + '\n            '.join(looked.lines()))
    if args.check_only:
        return
    try:
        opt = model.Options(
            fmin=None if args.fmin is None else args.fmin * 1e9,
            fmax=None if args.fmax is None else args.fmax * 1e9,
            ftarget=None if args.ftarget is None else args.ftarget * 1e9,
            require=args.require, pick=args.pick, target_db=args.target_db,
            reactance_percent=args.reactance_percent, loss_percent=args.loss_percent,
            refine=not args.no_refine,
            tolerance=args.tolerance, max_elements=args.max_elements,
            form=args.form, coupling=not args.no_coupling, full_search=args.full_search,
            sequential=args.sequential,
            delay=args.delay if args.delay in ('off', 'auto') else float(args.delay) * 1e-12,
            line_z0=args.line_z0, dc=args.dc, trim_low=not args.keep_low,
            float_resistor='auto' if args.float_resistor == 'auto' else float(args.float_resistor))
        m = model.extract(network, opt)
    except ValueError as ex:
        sys.exit(f'error: {ex}')
    m.source = path.name
    print('\n'.join(model.describe(m)))
    print('\n'.join(advice.assess(m, network, looked).lines()))
    formats = netlist.FORMATS if args.format == 'both' else (args.format,)
    name = args.name or netlist.subckt_name(path.stem + '_model')
    files = write_outputs(m, network, args.outdir or path.parent, path.stem, formats, name,
                          plot=not args.noplot, all_data=args.plot_all_data)
    for p in files:
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
