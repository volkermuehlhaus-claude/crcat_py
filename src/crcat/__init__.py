"""crcat: Python port of CRCat (J. C. Rautio, Sonnet Software).

Direct least-squares rational-polynomial fitting and the catalog of all RLC networks with up
to five elements, with closed-form synthesis equations and transforms between equivalent
networks.
"""

__version__ = '0.10.0'

from .catalog import Catalog, Network, load                       # noqa: E402
from .component import Component, format_value                   # noqa: E402
from .fit import crfit, evaluate                                 # noqa: E402
from .fitauto import fit_auto, interpolate                       # noqa: E402
from .model import Options, extract                              # noqa: E402
from .netlist import to_spectre, to_spice                        # noqa: E402
from .seed import seed                                           # noqa: E402
from .sprout import sprout                                       # noqa: E402

__all__ = ['Catalog', 'Network', 'load', 'Component', 'format_value', 'crfit', 'evaluate',
           'fit_auto', 'interpolate', 'seed', 'sprout', 'Options', 'extract', 'to_spice',
           'to_spectre']
