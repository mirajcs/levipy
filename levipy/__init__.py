"""
LeviPy — Levi-Civita connection & differential geometry toolkit
===============================================================
A pure-Python, SymPy-backed library for symbolic differential geometry computations.

Quick start
-----------
>>> from levipy import Manifold
>>> import sympy as sp
>>> r, theta = sp.symbols('r theta', positive=True)
>>> M = Manifold('S2', coords=[r, theta])
>>> g = M.metric([[1, 0], [0, r**2]])
>>> g.christoffel()          # Christoffel symbols Γ^k_ij
>>> g.riemann()              # Riemann curvature tensor
>>> g.ricci_tensor()         # Ricci tensor
>>> g.ricci_scalar()         # Ricci scalar
"""

from levipy.geometry.covariant_derivative import CovariantDerivative
from levipy.geometry.lie_bracket import lie_bracket
from levipy.geometry.manifold import Manifold
from levipy.gr.geodesic import GeodesicEquations
from levipy.gr.parallel import ParallelTransport
from levipy.lean4 import MathlibLinker, SymPyToLean4, TheoremBuilder
from levipy.tensors.christoffel import ChristoffelSymbols
from levipy.tensors.einstein import EinsteinTensor
from levipy.tensors.metric import MetricTensor
from levipy.tensors.ricci import RicciScalar, RicciTensor
from levipy.tensors.riemann import RiemannTensor

__version__ = "0.1.0"
__author__ = "Miraj Samarakkody"
__all__ = [
    "Manifold",
    "MetricTensor",
    "ChristoffelSymbols",
    "RiemannTensor",
    "RicciTensor",
    "RicciScalar",
    "EinsteinTensor",
    "GeodesicEquations",
    "ParallelTransport",
    "CovariantDerivative",
    "lie_bracket",
]
