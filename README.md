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
pip install -e ".[gui]"     # also PySide6 and matplotlib: window and plots
pip install -e ".[fast]"    # also gmpy2: the searches run about 1.4 times faster
```

## From S-parameters to a netlist

```
crcat-netlist mim_170fF.s2p --fmax 100
crcat-gui                       # the same in a window
```

A reciprocal n-port is exactly a set of branches: one from each port to ground and one
between each pair of ports (for a 2-port, the pi network). Each branch is a one-port, so the
catalog can model it with up to five R, L, C, and the netlist is these networks wired
together.

For each branch, a candidate network is judged by what it does to the S-parameters: it
replaces the branch in the data, and the RMS difference of all S-parameters is its error.
The default choice is the network with the fewest elements, all positive, whose error is
within the limit (`--target-db`, default -40 dB). "Open" counts as a candidate, so weak
branches disappear. A model in two other forms is tried when this pi form fails, see
[Tee form](#tee-form) and [Coupled coils](#coupled-coils).

Example, a 170 fF MIM capacitor from a Palace simulation (1-300 GHz data, fitted to
100 GHz):

```
p1 - gnd   CR3 C          -41.4 dB  C=9.006fF
p2 - gnd   CR3 C          -41.6 dB  C=0.6499fF
p1 - p2    CR11 SerRLC    -56.0 dB  R=1.177Ohm L=6.605pH C=172.3fF
model (pi form): 5 elements, S-parameter error -49.5 dB rms, -42.8 dB max
```

```
.SUBCKT mim_170fF_model p1 p2
C_p1 p1 0 9.00639e-15
C_p2 p2 0 6.49939e-16
R_p1p2 p1 p1p2_n2 1.17698
L_p1p2 p1p2_n2 p1p2_n3 6.60537e-12
C_p1p2 p1p2_n3 p2 1.7233e-13
.ENDS mim_170fF_model
```

Written next to the input file: the netlist (`_crcat.sp`; with `--format spectre` or `both`
also `_crcat.scs`), the S-parameters of the model (`_crcat.sNp`), the result as JSON and a
plot of data against model.

![crcat-gui](doc/gui.png)

In the window, the Network column lists the other networks tried for each branch; choosing
one updates netlist, error and plot at once.

### Tee form

A device with a common path to ground is a tee by nature: two series legs to a center node
and one leg from there to ground. Forced into a pi, its loss can land in a branch with a
negative real part, which no positive resistor can give. For a 2-port, the tool therefore
also tries the tee form when a pi branch misses its limit, and keeps the form with the
lower error. A shorted series leg is a candidate there, like an open branch in the pi form.

Example: the same MIM capacitor over the full 1-300 GHz. In the pi form the port-2 branch of
the data has a negative real part in 96 % of the band (model error -19.3 dB). The tee form:

```
p1 - c     CR11 SerRLC    -59.9 dB  R=625.8mOhm L=3.286pH C=938.3fF
p2 - c     CR11 SerRLC    -52.6 dB  R=769.9mOhm L=2.994pH C=216.9fF
c - gnd    CR3 C          -57.5 dB  C=9.554fF
model (tee form): 7 elements, S-parameter error -49.2 dB rms, -40.0 dB max
```

### Coupled coils

Magnetic coupling between two branches shows in the pi form as a negative inductance, and
often a negative resistance, in the branch between them. When such a branch misses its limit
(or got a negative element), the tool tries to replace it and its two neighbours by two
coils with a coupling factor (SPICE `K`, Spectre `mutual_inductor`), plus a common resistor
if that helps. What the coils do not explain of the branch between the two ports stays a
branch of its own. The replacement is kept if it lowers the error.

Example: a center-tapped inductor (3-port, center tap at port 3), fitted to 30 GHz. As a pi
of positive elements over its full band the error was -12.5 dB. With coupled coils:

```
p1 - gnd   CR3 C          -65.9 dB  C=6.015fF
p2 - gnd   CR3 C          -65.2 dB  C=6.354fF
p3 - gnd   open           -40.7 dB  no elements
p1 - p2    open           -42.2 dB  no elements
p1, p2 - p3 coupled coils: CR7 SerRL + CR7 SerRL   -42.2 dB
           R=842.6mOhm L=165.4pH | R=1.111Ohm L=167pH | k=-0.3873
model (pi form): 7 elements, S-parameter error -38.2 dB rms, -29.9 dB max
```

```
R_k12a p1 k12a_n2 0.842632
L_k12a k12a_n2 p3 1.65399e-10
R_k12b p2 k12b_n2 1.11083
L_k12b k12b_n2 p3 1.67024e-10
K_k12 L_k12a L_k12b -0.387331
```

The coupling factor is one number for the whole band. Towards the resonance the apparent
mutual inductance of this inductor grows (from -59 pH to -109 pH at 60 GHz), so over the full
0.01-60 GHz the same model reaches -29.7 dB only.

Each coil must have an inductor that carries the whole coil current, so coils with a
capacitor across their terminals (a transformer whose only other branches go to ground) are
not replaced.

### Options

Main options (`crcat-netlist --help` lists all):

| Option | Meaning |
|---|---|
| `--fmin`, `--fmax` (GHz) | fit band; the plot still shows all data |
| `--require positive` | all element values positive (default) |
| `--require positive-r` | resistors positive, L and C of either sign. Still passive, since only resistors dissipate |
| `--require passive`, `any` | any signs if the branch stays passive; no restriction |
| `--pick simplest` | fewest elements within the error limit (default) |
| `--pick best` | lowest error |
| `--target-db` | error limit per branch, RMS S-parameter difference |
| `--max-elements` | most elements per branch (1..5) |
| `--form auto`, `pi`, `tee` | auto (default): pi, and the tee form for a 2-port if a pi branch misses the limit |
| `--no-coupling` | never use coupled coils |

Limits:

- **Five elements per branch.** A branch with several resonances in the band needs a
  narrower band.
- **Tee form for 2-ports only.** A tee has too few branches for a general n-port.
- **One coupling factor per coil pair,** see above; and coupled loss that needs a negative
  resistance stays out unless `--require any`.
- The tool warns when a branch misses its limit and when the data of that branch have a
  negative real part.
- The model is reciprocal; non-reciprocal data are symmetrized, and the difference is
  reported.

In Python, without any GUI or plotting import:

```python
import crcat

model = crcat.extract('dut.s2p', crcat.Options(fmax=100e9))   # or a scikit-rf Network
print(model.summary())
text = crcat.to_spice(model, 'dut')
s = model.s_matrix(freq)              # S-parameters of the model
model.choose(2, 1)                    # another candidate for branch 2
```

`extract` takes `progress(fraction, text)` and `cancel()` callbacks; `crcat.gui` is only a
window on top of this.

## Searching the catalog for one branch

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
it and its equivalent networks with the exact element values. "Passive" means Re Y >= 0 at
all frequencies, which still allows negative elements; the report therefore also lists the
best models whose element values are all real and positive. For the series branch of a
170 fF MIM capacitor simulated with Palace (1-300 GHz, `--fit y21`), for example, that list
starts with 5-element models (C plus a frequency-dependent loss) and contains the plain
series RLC (CR11: R 1.36 ohm, L 6.16 pH, C 180.7 fF). `--fit` selects the data:
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
| (not in CRCat) | `crcat.model.extract`, `crcat.netlist`, `crcat-netlist`, `crcat-gui` |

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
- **`Y12`-type data selection** returns the Y-parameter itself. CRCat's code (commit `c805a25`)
  returns the negative; its author confirmed that this is an error there (the minus sign
  belongs only to the lower-case `y12`, the pi-model admittance between two ports).
- **The default catalog** is base + custom without the transforms between base and custom
  networks (CRCat recomputes those).
- The 35 base networks whose synthesis equations are all zero are treated as having none.
- Added: the filter for models with positive real element values (`best(physical=True)`).

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
python -m pytest            # fast tests (about 45 s, GUI offscreen)
python -m pytest -m slow    # catalog validation, full searches (about 1.5 min)
```

The tests check against CRCat's README example and its MATLAB tests (`CRCompTest.m`,
`CRSproutTest.m`, `CRFitAutoTest.m`); the test data in `tests/data` come from CRCat.

## License

MIT, like the original. CRCat, its catalog data and the test data are Copyright (c) 2025
Sonnet Software Inc. If you use this work, please acknowledge CRCat and cite the paper above,
as its author asks.
