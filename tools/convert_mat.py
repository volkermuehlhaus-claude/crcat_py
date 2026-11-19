"""Convert CRCat's MATLAB catalogs (.mat) to the JSON files used by the crcat package.

Usage:
    python tools/convert_mat.py <CRCat checkout> [--out src/crcat/data]

Reads ``@CRCat/Catalogs/ABaseCatalog.mat`` and ``ACustomCatalog.mat`` from a checkout of
https://github.com/Sonnet-Software/CRCat and writes ``catalog_base.json`` and
``catalog_custom.json``. Needs the ``mat-io`` package (``pip install mat-io``), which is a
development dependency only.

Symbolic fields are written as SymPy strings (``sympy.sympify`` reads them back). Dropped:
``symZeros`` / ``symPoles`` (MuPAD RootOf objects; computed numerically from A and B instead).
"""

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mupad_decode import to_text  # noqa: E402

__version__ = '0.1'


def _is_sym(v):
    return type(v).__name__ == 'MatlabOpaque' and getattr(v, 'classname', '') == 'sym'


def _sym(v):
    """sym object -> (rows of expression strings, lets); scalars become [[x]]."""
    x, lets = to_text(str(v.properties['s'].ravel()[0]))
    return (x if isinstance(x, list) else [[x]]), lets


def _scalar(v):
    x, lets = _sym(v)
    if len(x) != 1 or len(x[0]) != 1 or lets:
        raise ValueError('expected a scalar without shared subexpressions')
    return x[0][0]


def _names(v):
    return [x for row in _sym(v)[0] for x in row]


def _equations(v):
    """Matrix-valued field -> {'rows': [...], 'let': [...]} ('let' only if needed)."""
    rows, lets = _sym(v)
    d = {'rows': rows}
    if lets:
        d['let'] = lets
    return d


def _text(v):
    a = np.asarray(v).ravel()
    return str(a[0]) if a.size else ''


def _numbers(v, dtype=float):
    a = np.asarray(v, dtype=float).ravel()
    return [dtype(x) for x in a]


def convert_network(e, index):
    net = {'id': index, 'name': _text(e['name'])}
    for f in ('symV', 'symVn', 'symC'):
        net[f] = _names(e[f]) if _is_sym(e[f]) else []
    net['A'] = _scalar(e['symA'])
    net['B'] = _scalar(e['symB'])
    net['EL'] = _equations(e['symEL']) if _is_sym(e['symEL']) else None
    net['testEL'] = _numbers(e['testEL'])
    net['flagSig'] = _text(e['flagSig'])
    net['nodes'] = _numbers(e['nodes'], int)
    same_ids = _numbers(e['sameId'], int)
    cells = np.asarray(e['symSame'], dtype=object).ravel()
    if len(cells) != len(same_ids):
        raise ValueError(f'{net["name"]}: {len(cells)} transforms for {len(same_ids)} sameId')
    net['same'] = []
    for to, c in zip(same_ids, cells):
        if _is_sym(c):
            net['same'].append({'to': to, **_equations(c)})
        else:
            net['same'].append({'to': to, 'status': _text(c)})
    return net


def convert_catalog(path, source):
    import matio
    props = matio.load_from_mat(str(path))['theCat'].properties
    lst = props['list']
    nets = []
    for i in range(lst.shape[1]):
        nets.append(convert_network(lst[0, i], i + 1))
    units = {k: _text(props[k]) for k in ('fUnit', 'resUnit', 'indUnit', 'capUnit')}
    units['fMult'] = float(np.asarray(props['fMult']).ravel()[0])
    return {'source': source, 'converter': f'tools/convert_mat.py {__version__}',
            'units': units, 'networks': nets}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('crcat', type=pathlib.Path, help='CRCat checkout (folder with @CRCat)')
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path(__file__).parents[1] / 'src' / 'crcat' / 'data')
    ap.add_argument('--version', action='version', version=__version__)
    args = ap.parse_args()
    cat_dir = args.crcat / '@CRCat' / 'Catalogs'
    try:
        commit = subprocess.run(['git', '-C', str(args.crcat), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    print(f'CRCat folder: {args.crcat}\ncommit: {commit}\noutput: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    for mat, out in (('ABaseCatalog.mat', 'catalog_base.json'),
                     ('ACustomCatalog.mat', 'catalog_custom.json')):
        source = {'repository': 'https://github.com/Sonnet-Software/CRCat',
                  'commit': commit, 'file': f'@CRCat/Catalogs/{mat}'}
        cat = convert_catalog(cat_dir / mat, source)
        with open(args.out / out, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(cat, f, indent=1)
            f.write('\n')
        print(f'wrote {args.out / out}: {len(cat["networks"])} networks')


if __name__ == '__main__':
    main()
