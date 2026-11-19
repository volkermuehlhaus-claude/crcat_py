# crcat: CRCat in Python

Python port of [CRCat](https://github.com/Sonnet-Software/CRCat) by James C. Rautio (Sonnet
Software): a catalog of every RLC network with up to five elements, closed-form synthesis
equations from rational-polynomial coefficients to element values, transforms between
equivalent networks, and a direct (non-iterative) least-squares rational-polynomial fit.
Together they find the best small lumped model for measured or simulated data.

The theory is in J. C. Rautio, "Direct Least-Squares Rational-Polynomial Lumped-Circuit Model
Extraction and Group Theory," IEEE Journal of Microwaves, 2025,
doi:[10.1109/JMW.2025.3598986](https://doi.org/10.1109/JMW.2025.3598986).

## Status

Work in progress.

- Done: the complete CRCat catalog as JSON (`src/crcat/data/`), converted from the MATLAB
  `.mat` files without MATLAB, and a numeric evaluator for its equations
  (`crcat.symbolic`, mpmath, no SymPy at run time).
- Next: the rational-polynomial fit (`CRFit`), components (fit, evaluate, transform,
  passivity check) and the catalog search (`CRSeed`), reading Touchstone files with
  scikit-rf.

## The catalog

`catalog_base.json` holds the 623 networks of CRCat's base catalog, `catalog_custom.json`
the 8 example networks of its custom catalog. For each network:

| Field | Meaning |
|---|---|
| `id`, `name` | CRCat Id (1-based index, "CR500") and name ("XXXII-39": topology XXXII, 39th way to fill it) |
| `symVn`, `nodes` | element names and netlist: element k connects `nodes[2k]` and `nodes[2k+1]`; port is node 1, ground node 0 |
| `symV` | elements that have synthesis equations (the others are fixed, given by the user) |
| `A`, `B` | admittance Y(s) = A/B as polynomials in s |
| `symC`, `flagSig` | rational-polynomial coefficients used (`a`k: s^k in A, `b`k: s^k in B) and their signature |
| `EL` | synthesis equations: element values `symV` from `symC` (and fixed elements); several rows = several solutions; `null` if none exist |
| `testEL` | test element values for `symVn` |
| `same` | for each network with the same signature: transform equations (element values of the other network from this one's `symVn`) or a status (`NoXfrm`, `Nil`, `Trivial`, `EqnNotFound`) |

Expressions are Python-syntax text that `sympy.sympify` also reads. Non-integers are written
`Rational(p, q)` or `Float('...', 40)`; shared subexpressions are listed under `let` as `t1`,
`t2`, ... and evaluated first. MATLAB's `root(p, z, k)` (k-th root of polynomial p) is written
`root([c_n, ..., c_0], k)` with the coefficients of p. MATLAB's numbering of the roots is not
documented, so `crcat.symbolic` tries every root and returns all candidate solutions.

The zeros and poles of CRCat's catalog (`symZeros`, `symPoles`) are not converted; compute them
numerically from A and B.

### How it was converted

MATLAB stores symbolic values in `.mat` files as MuPAD's binary serialization. Its format is
not documented; `tools/mupad_decode.py` decodes the part of it the catalogs use, and
`tools/convert_mat.py` writes the JSON files (needs `pip install mat-io`):

```
python tools/convert_mat.py <CRCat checkout>
```

The JSON files were made from CRCat commit `c805a25` (2025-11-22).

### Validation

`tools/validate_catalog.py` checks every network against its own netlist (about 1 minute):

| Check | Result |
|---|---|
| A/B equals a nodal analysis of the netlist (random values, random complex s) | 631 of 631 networks |
| Synthesis round trip: coefficients of A/B at `testEL` through `EL` give `testEL` back | 499 of 499 networks with equations |
| Transforms: transformed values give the same admittance | 4844 of 4844 (16 more in the custom catalog have fixed elements and are not checked) |

The counts match the paper: 491 base networks with synthesis equations (plus 35 whose
equations are all zero), and 4651 transforms (4836 sets of equations, less 491 to the network
itself, plus 306 marked `EqnNotFound`).

## Installation

```
pip install -e .            # run time: numpy, mpmath
pip install -e ".[dev]"     # tests and the converter: pytest, sympy, mat-io
python -m pytest            # fast tests
python -m pytest -m slow    # catalog validation
```

## License

MIT, like the original. CRCat and its catalog data are Copyright (c) 2025 Sonnet Software Inc.
If you use this work, please acknowledge CRCat and cite the paper above, as its author asks.
