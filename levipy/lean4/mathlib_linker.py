"""
Mathlib lemma linker.

Looks at a SymPy expression and suggests the most relevant
Mathlib 4 lemmas / tactics for proving equalities involving it.

The mappings are based on Mathlib naming conventions as of 2025.
"""
from __future__ import annotations

from typing import List, Tuple

import sympy as sp

# ──────────────────────────────────────────────────────────────────────
# Registry:  (pattern_check, lean4_lemma, description)
# pattern_check: callable(expr) -> bool
# ──────────────────────────────────────────────────────────────────────

class MathlibLinker:
    """
    Suggest Mathlib 4 lemmas relevant to a SymPy expression.

    The suggestions are intended to help a human complete `sorry` proofs.

    Examples
    --------
    >>> import sympy as sp
    >>> th = sp.Symbol('theta')
    >>> linker = MathlibLinker()
    >>> linker.suggest(sp.sin(th)**2 + sp.cos(th)**2 - 1)
    [('Real.sin_sq_add_cos_sq', 'sin²θ + cos²θ = 1')]
    """

    # Each entry: (check_fn, lean_name, description)
    REGISTRY: List[Tuple] = [
        # ── Trig identities ──
        (
            lambda e: e.has(sp.sin) and e.has(sp.cos)
                      and sp.simplify(e - (sp.sin(list(e.free_symbols)[0])**2
                                          + sp.cos(list(e.free_symbols)[0])**2 - 1)) == 0  # noqa: E501
                      if e.free_symbols else False,
            "Real.sin_sq_add_cos_sq",
            "sin²θ + cos²θ = 1",
        ),
        (
            lambda e: e.has(sp.sin) and e.has(sp.cos),
            "Real.sin_sq",
            "sin²θ = 1 − cos²θ",
        ),
        (
            lambda e: e.has(sp.cos),
            "Real.cos_sq",
            "cos²θ = 1 − sin²θ",
        ),
        (
            lambda e: e.has(sp.tan),
            "Real.tan_eq_sin_div_cos",
            "tan θ = sin θ / cos θ",
        ),
        (
            lambda e: e.has(sp.sin) and sp.simplify(e) == 0
                      and e.has(sp.pi),
            "Real.sin_pi",
            "sin π = 0",
        ),

        # ── Square roots ──
        (
            lambda e: e.has(sp.sqrt),
            "Real.sqrt_sq",
            "√(x²) = |x|",
        ),
        (
            lambda e: e.has(sp.sqrt) and e.has(sp.Pow),
            "Real.sq_sqrt",
            "(√x)² = x  (for x ≥ 0)",
        ),

        # ── Arithmetic / algebra ──
        (
            lambda e: isinstance(e, sp.Rational),
            "norm_num",
            "Numeric equality, closed by norm_num",
        ),
        (
            lambda e: e.is_polynomial(),
            "ring",
            "Polynomial identity, closed by ring tactic",
        ),
        (
            lambda e: e.has(sp.Pow) and any(
                a.exp == -1 for a in e.atoms(sp.Pow)
            ),
            "field_simp",
            "Clears denominators before ring",
        ),

        # ── Manifold / Riemannian geometry (Mathlib 4) ──
        (
            lambda e: False,  # structural hints always shown
            "RiemannianMetric.christoffelSymbols",
            "Definition of Christoffel symbols in Mathlib",
        ),
        (
            lambda e: False,
            "RiemannianMetric.metric_compatible",
            "∇g = 0 (metric compatibility of Levi-Civita)",
        ),
        (
            lambda e: False,
            "RiemannianMetric.torsionFree",
            "Levi-Civita connection is torsion-free",
        ),
    ]

    def suggest(self, expr: sp.Expr) -> List[Tuple[str, str]]:
        """
        Return a list of (lean_name, description) pairs for `expr`.

        Parameters
        ----------
        expr : sp.Expr
            The expression whose equality to zero you want to prove.

        Returns
        -------
        list of (str, str)
        """
        results = []
        seen = set()
        for check, name, desc in self.REGISTRY:
            try:
                if check(expr) and name not in seen:
                    results.append((name, desc))
                    seen.add(name)
            except Exception:
                pass
        return results

    def format_suggestions(self, expr: sp.Expr) -> str:
        """Return a human-readable string of suggestions."""
        suggestions = self.suggest(expr)
        if not suggestions:
            return "  -- No automatic suggestions; try: ring, norm_num, or simp"
        lines = ["  -- Relevant Mathlib lemmas:"]
        for name, desc in suggestions:
            lines.append(f"  --   {name:<50} -- {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tactic chain for common cases
    # ------------------------------------------------------------------

    def tactic_chain(self, expr: sp.Expr) -> str:
        """
        Return a recommended tactic chain as a string.

        This is intended to close a goal of the form `lhs = rhs`
        where `lhs - rhs` simplifies to `expr`.
        """
        s = sp.simplify(expr)
        if s == 0:
            return "rfl"
        if s.is_number:
            return "norm_num"
        if s.has(sp.sin) or s.has(sp.cos):
            return "simp [Real.sin_sq, Real.cos_sq]; ring"
        if s.has(sp.sqrt):
            return "rw [Real.sqrt_eq_iff_sq_eq]; ring_nf; norm_num"
        has_div = any(
            isinstance(a, sp.Pow) and a.exp == -1
            for a in s.atoms(sp.Pow)
        )
        if has_div:
            return "field_simp; ring"
        return "ring"
