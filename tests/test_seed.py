"""Catalog search (CRSeed) and automatic fit / interpolation (CRFitAuto, CRInterp)."""

import pathlib

import numpy as np
import pytest
import skrf

from crcat import catalog
from crcat.fitauto import fit_auto, interpolate
from crcat.seed import main, seed

DATA = pathlib.Path(__file__).parent / 'data'

# CRCat README, search of XXXII-39.y1p: (Id, RMS error, passive). CR505's best solution
# has Re Y = c * w**4 exactly, so its passivity depends on rounding: not checked (None).
README = {500: (3.13e-13, True), 239: (3.13e-13, True), 413: (3.13e-13, True),
          4: (0.0118, True), 2: (0.0158, True), 505: (0.0183, None), 573: (0.0196, True),
          604: (0.0196, True), 164: (0.0197, True)}


def check_readme(entries):
    for e in entries:
        if e.id in README:
            err, passive = README[e.id]
            if err < 1e-10:
                assert e.error < 1e-11, e
            else:
                assert abs(e.error - err) < 0.0006, e
            if passive is not None:
                assert e.passive == passive, e


def test_seed_subset():
    r = seed(DATA / 'XXXII-39.y1p', ids=list(README))
    assert {e.id for e in r.entries[:3]} == {500, 239, 413}
    check_readme(r.entries)
    assert r.best(1, passive=True)[0].id in (500, 239, 413)
    assert r.best(1, physical=True)[0].id in (500, 239, 413)
    assert all(e.physical == all(isinstance(v, float) and v > 0 for v in e.component.values[0])
               for e in r.entries)
    report = r.report()
    assert 'Pi admit port 1 to gnd' in report and 'positive element values' in report


def test_seed_upper_case_y_is_not_negated():
    # Upper-case 'Y11' fits Y11 itself, as CRCat's documentation of SnP.Pull says; CRCat's
    # code returns -Y11 (question to J. Rautio open). On a 1-port, Y11 equals the pi-model
    # y11, so both selections must give the README result with positive element values.
    ids = [500, 239, 413, 4]
    upper = seed(DATA / 'XXXII-39.y1p', fit='Y11', ids=ids)
    lower = seed(DATA / 'XXXII-39.y1p', fit='y11', ids=ids)
    for r in (upper, lower):
        best = r.entries[0]
        assert best.id in (500, 239, 413) and best.error < 1e-11
        assert best.passive and best.physical
    assert [e.id for e in upper.entries] == [e.id for e in lower.entries]
    # With -Y11 (CRCat's code) the same circuit comes out with every element negated.
    neg = seed((upper.freq, -upper.y), ids=[500])
    assert neg.entries[0].error < 1e-11 and not neg.entries[0].passive
    assert all(v < 0 for v in neg.entries[0].component.values[0])


@pytest.mark.slow
def test_seed_full_catalog():
    r = seed(DATA / 'XXXII-39.y1p')
    assert {e.id for e in r.entries[:4]} == {500, 624, 413, 239}
    check_readme(r.entries)
    assert len(r.entries) + len(r.skipped) == len(catalog.load())


def test_seed_cli(capsys, tmp_path):
    main([str(DATA / 'XXXII-39.y1p'), '--ids', '2,4,500', '--top', '3',
          '--plot', str(tmp_path / 'fit.png')])
    out = capsys.readouterr().out
    assert 'crcat-seed' in out and 'CR500 XXXII-39' in out
    assert (tmp_path / 'fit.png').stat().st_size > 1000


def test_seed_two_port_fit_limit():
    # fit only the first 100 frequencies of the hairpin S11 pi shunt admittance
    r = seed(DATA / 'hairpin_InBand_Nf251.s2p', fit='100y11', ids=[2, 4, 5, 6])
    assert r.nfit == 100 and len(r.freq) == 251
    assert r.entries


def test_fit_auto_and_interpolate():
    # CRFitAutoTest test 4: from the pass band of a hairpin filter to 0.1 .. 6 GHz
    full = skrf.Network(str(DATA / 'hairpin_BroadBand_Nf991.s2p'))
    band, ref = full[375:421], full[0:592]
    r, digits = fit_auto(band.s[:, 1, 0], band.f / 1e9)
    assert r.ok and digits >= 32
    out = interpolate(band, ref.f)
    assert np.abs(out.s[:, 1, 0] - ref.s[:, 1, 0]).max() < 1e-7
    assert np.abs(out.s[:, 0, 0] - ref.s[:, 0, 0]).max() < 1e-2
    with pytest.raises(ValueError):
        fit_auto(band.s[:5, 0, 0], band.f[:5])
