"""Full check of the converted catalog (tools/validate_catalog.py), about 1 minute."""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[1]


@pytest.mark.slow
def test_catalog_validation():
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'validate_catalog.py')],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert 'all checks passed' in r.stdout
