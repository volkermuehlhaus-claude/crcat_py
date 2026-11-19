# crcat: CRCat in Python

Python port of [CRCat](https://github.com/Sonnet-Software/CRCat) by James C. Rautio (Sonnet
Software). CRCat finds small lumped RLC models for measured or simulated data:

- a catalog of every RLC network with up to five elements (623 networks), with closed-form
  synthesis equations from rational-polynomial coefficients to element values and transforms
  between equivalent networks,
- a direct (non-iterative) least-squares fit of a rational polynomial, in high-precision
  arithmetic,
- a search that fits every network of the catalog to the data and ranks them.

The theory is in J. C. Rautio, "Direct Least-Squares Rational-Polynomial Lumped-Circuit Model
Extraction and Group Theory," IEEE Journal of Microwaves, 2025,
doi:[10.1109/JMW.2025.3598986](https://doi.org/10.1109/JMW.2025.3598986).

No MATLAB is needed: the catalog was converted from CRCat's `.mat` files to JSON (see
[The catalog](#the-catalog)), and the numerics use mpmath instead of MATLAB's variable
precision arithmetic.

## Installation

```
pip install -e .            # numpy, mpmath, scikit-rf
pip install -e ".[plot]"    # also matplotlib, for plots
```

## Quick start

Search the catalog for the best model of the pi-model shunt admittance at port 1:

```
crcat-seed tests/data/XXXII-39.y1p --top 10 --plot fit.png
```

```
Up to the 10 best fits (Id, RMS error; * = passive):
  (500* 3.13e-13) (624* 3.13e-13) (413* 3.13e-13) (239* 3.13e-13) (202* 0.00995) ...
Best passive models:
  CR500 XXXII-39, rms 3.13e-13: La=500pH Lb=400pH Rc=10Ohm Ld=200pH Cd=3pF
  CR413 XXXI-33, rms 3.13e-13: La=900pH Rb=50.63Ohm Lc=2.137nH Ld=2.375nH Cd=133fF
  ...
```

The data are a Sonnet netlist simulation of network XXXII-39 (CR500), so the search finds
it and its equivalent networks with the exact element values. `--fit` selects the data:
`y11` (pi-model admittance port 1 to ground, default), `y21` (pi-model admittance between
ports 1 and 2), `z11` / `z21` (tee model), `Y11`, `Z11`, `S11` (the parameter itself). A
leading number fits only the first frequencies, e.g. `--fit 200y11`. `crcat-seed --help`
lists all options.

In Python:

```python
import numpy as np
import crcat

cat = crcat.load()                         # 623 base + 8 custom networks
result = crcat.seed('measured.s2p', cat, fit='y21')
print(result.report())
best = result.best(1, passive=True)[0].component

c = crcat.Component('XXXII-39', cat)       # catalog test values (GHz, nH, nF, ohm)
f = np.linspace(1, 20, 40)
y = c.eval(f)[0]                           # admittance (one row per solution)
c.fit(y, f)                                # fit: RMS error per solution, values from the fit
c.transform('XXXI-33')                     # an equivalent network
c.is_passive()
cat.relations(500)                         # [(239, 'Equiv'), (413, 'Equiv'), (500, 'Self')]

r = crcat.crfit(y, f, [1] * 6, [1] * 6)    # rational fit with 6 + 6 terms
r, digits = crcat.fit_auto(y, f)           # order and precision chosen automatically
s_new = crcat.interpolate(network, f_new)  # scikit-rf Network at new frequencies (Hz)
```

Units are any consistent set (GHz, nH, nF, ohm or Hz, H, F, ohm). `seed` works in Hz, so
its element values are in H, F and ohm; `Component.format()` assumes these units.

## What is ported

| CRCat (MATLAB) | crcat |
|---|---|
| `CRFit` | `crcat.fit.crfit` |
| `CREval` | `crcat.fit.evaluate` |
| `CRFitAuto`, `CRInterp` | `crcat.fitauto.fit_auto`, `interpolate` |
| `@CRCat` (catalog, `Relationship`, `NetworkRelations`, `FindOrder`) | `crcat.catalog` |
| `CRComp` (`Eval`, `Fit`, `Transform`, `IsGain`, `ElementValues`) | `crcat.component.Component` |
| `CRSeed`, `CRSeedAddNoise` | `crcat.seed.seed`, `crcat-seed` |
| `CRSprout` | `crcat.sprout.sprout` |
| `SnP` (`Get`, `Pull`, `AddNoise`) | scikit-rf, `crcat.data.pull`, `seed(noise_floor_db=...)` |

Not ported: generating the catalog and its equations (`FillCat_*`, MATLAB `solve`), the
schematic drawings (`DrawRLC`, `Draw`, `DrawArray`), `CRFitClean`, and the interactive
prompts of `CRSeed`. Networks can therefore not be added to the custom catalog in Python yet.

### Differences from CRCat

- **More networks fit.** The normal equations of the rational fit span many decades when
  frequencies are in Hz. CRCat's solver declares some of them singular (code -3; the README
  example lists CR485). This port scales the matrix first, which does not change the
  solution, so these networks fit too. Their errors come from evaluating the synthesized
  network and agree with a nodal analysis. The search can therefore list networks that
  CRCat's list does not have (CR202 in the example above).
- **Passivity is checked for the best solution.** CRCat reports the smallest error of all
  solutions but checks passivity of solution 1. The check here decides by the sign of the
  real part between its zeros, so a real part that only touches zero counts as passive. A
  model whose real part is zero in exact arithmetic (CR505 in the example) can come out
  either way.
- **`root(p, z, k)`**: 33 networks have synthesis equations with the k-th root of a
  polynomial. MATLAB's numbering of the roots is not documented, so all roots are tried; a
  fit returns all solutions, sorted by error.
- **`Y12`-type data selection** returns the Y-parameter itself, as CRCat's documentation of
  `SnP.Pull` says (its code returns the negative).
- **The default catalog** is base + custom without the transforms between base and custom
  networks (CRCat recomputes those).
- The 35 base networks whose synthesis equations are all zero are treated as having none.

## The catalog

`src/crcat/data/catalog_base.json` holds the 623 networks of CRCat's base catalog,
`catalog_custom.json` the 8 example networks of its custom catalog. For each network:

| Field | Meaning |
|---|---|
| `id`, `name` | CRCat Id (1-based index, "CR500") and name ("XXXII-39": topology XXXII, 39th way to fill it) |
| `symVn`, `nodes` | element names and netlist: element k connects `nodes[2k]` and `nodes[2k+1]`; port is node 1, ground node 0 |
| `symV` | elements that have synthesis equations (the others are fixed, given by the user) |
| `A`, `B`, `Acoef`, `Bcoef` | admittance Y(s) = A/B; the coefficient lists are highest power of s first |
| `symC`, `flagSig` | rational-polynomial coefficients used (`a`k: s^k in A, `b`k: s^k in B) and their signature |
| `EL` | synthesis equations: element values `symV` from `symC` (and fixed elements); several rows = several solutions; `null` if none exist |
| `testEL` | test element values for `symVn` |
| `same` | for each network with the same signature: transform equations (element values of the other network from this one's `symVn`) or a status (`NoXfrm`, `Nil`, `Trivial`, `EqnNotFound`) |

Expressions are Python-syntax text that `sympy.sympify` also reads. Non-integers are written
`Rational(p, q)` or `Float('...', 40)`; shared subexpressions are listed under `let` as `t1`,
`t2`, ... and evaluated first. MATLAB's `root(p, z, k)` is written `root([c_n, ..., c_0], k)`
with the coefficients of p. `crcat.symbolic` compiles the text into plain Python functions on
mpmath numbers.

The zeros and poles of CRCat's catalog (`symZeros`, `symPoles`) are not converted; compute them
numerically from `Acoef` and `Bcoef`.

### How it was converted

MATLAB stores symbolic values in `.mat` files in MuPAD's binary serialization. That format
is not documented; `tools/mupad_decode.py` decodes the part of it the catalogs use, and
`tools/convert_mat.py` writes the JSON files:

```
pip install -e ".[dev]"     # adds sympy and mat-io
python tools/convert_mat.py <CRCat checkout>
```

The JSON files were made from CRCat commit `c805a25` (2025-11-22).

### Validation

`tools/validate_catalog.py` checks every network against its own netlist:

| Check | Result |
|---|---|
| A/B (and `Acoef`/`Bcoef`) equal a nodal analysis of the netlist (random values, random complex s) | 631 of 631 networks |
| Synthesis round trip: coefficients of A/B at `testEL` through `EL` give `testEL` back | 499 of 499 networks with equations |
| Transforms: transformed values give the same admittance | 4844 of 4844 (16 more in the custom catalog have fixed elements and are not checked) |

The counts match the paper: 491 base networks with synthesis equations (plus 35 whose
equations are all zero), and 4651 transforms (4836 sets of equations, less 491 to the network
itself, plus 306 marked `EqnNotFound`).

## Tests

```
pip install -e ".[dev]"
python -m pytest            # fast tests (about 5 s)
python -m pytest -m slow    # catalog validation and full catalog search (about 35 s)
```

The tests check against CRCat's README example and its MATLAB tests (`CRCompTest.m`,
`CRSproutTest.m`, `CRFitAutoTest.m`); the test data in `tests/data` come from CRCat.

## License

MIT, like the original. CRCat, its catalog data and the test data are Copyright (c) 2025
Sonnet Software Inc. If you use this work, please acknowledge CRCat and cite the paper above,
as its author asks.
