from importlib.metadata import version
__version__ = version(__name__)  

from . import spatially_variable
from . import plotting 
from . import preprocess

from . import spatially_variable as sv
from . import plotting as pl
from . import preprocess as pp

__all__ = [
    "preprocess", "plotting", "spatially_variable",
    "pp", "pl", "sv",
]
