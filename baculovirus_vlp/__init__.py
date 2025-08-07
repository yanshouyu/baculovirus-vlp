"""Baculovirus VLP - Implementation of Hu & Bentley's article 
(DOI: 10.1016/S0009-2509(99)00579-5)."""

__version__ = "0.1.0"
__author__ = "Shouyu Yan"

# Import main classes/functions to make them available at package level
from .utils import pinf, jmin_jmax, td
from .parameters import Parameters
from .simulation import Simulation

# Define what gets imported with "from my_package import *"
__all__ = ["pinf", "jmin_jmax", "td", "Parameters", "Simulation"]