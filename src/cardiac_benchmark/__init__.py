from importlib.metadata import metadata

from . import benchmark
from . import cli
from . import geometry
from . import material
from . import microstructure
from . import postprocess
from . import pressure_model
from . import problem
from . import solver
from . import step1

meta = metadata("cardiac-benchmark")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = [
    "benchmark",
    "cli",
    "geometry",
    "material",
    "microstructure",
    "postprocess",
    "pressure_model",
    "problem",
    "solver",
    "step1",
]
