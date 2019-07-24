from .algorithms import *
from .utils import *
from .nbs import *
from .version import __version__, __version_info__
from .citations import BCTPY, RUBINOV2010

from .due import due, BibTeX

__citation__ = BCTPY

due.cite(BibTeX(__citation__), description="Brain Connectivity Toolbox for Python", path="bct")
due.cite(BibTeX(RUBINOV2010), description="Brain Connectivity Toolbox", path="bct")
