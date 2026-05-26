"""
levipy.lean4 — Python → Lean 4 formal verification bridge.

Modules
-------
translator      SymPy expression → Lean 4 term string
theorem_builder Emits .lean files with proof obligations
mathlib_linker  Maps expressions to relevant Mathlib lemmas

Quick usage
-----------
>>> from levipy import Manifold
>>> from levipy.lean4 import TheoremBuilder
>>> import sympy as sp
>>> th, ph = sp.symbols('theta phi', real=True)
>>> M = Manifold('S2', [th, ph])
>>> g = M.metric([[1, 0], [0, sp.sin(th)**2]])
>>> builder = TheoremBuilder(g, manifold_name='Sphere2',
...     extra_hypotheses=['(hth : 0 < θ)', '(hth2 : θ < π)'])
>>> builder.save('Sphere2.lean')
"""

from levipy.lean4.mathlib_linker import MathlibLinker
from levipy.lean4.theorem_builder import TheoremBuilder
from levipy.lean4.translator import SymPyToLean4

__all__ = ["SymPyToLean4", "TheoremBuilder", "MathlibLinker"]
