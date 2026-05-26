"""
SymPy → Lean 4 / Mathlib general-purpose expression translator.

Handles the full SymPy expression universe:
  Numbers      Integer, Rational, Float, pi, E, I, oo, nan
  Symbols      Symbol (with assumption-based type inference)
  Arithmetic   Add, Mul, Pow (neg exponents, sqrt, cbrt)
  Comparison   Eq, Ne, Lt, Le, Gt, Ge
  Logic        And, Or, Not, Implies, Xor, BooleanTrue/False
  Algebra      Abs, sign, floor, ceiling, Max, Min, Mod,
               factorial, binomial
  Trig         sin cos tan cot sec csc + inverses + atan2
  Hyperbolic   sinh cosh tanh + inverses
  Analysis     exp, log (with base), sqrt, cbrt
  Complex      re, im, conjugate, I
  Calculus     Derivative → HasDerivAt stub
               Integral   → ∫ Lean notation
               Limit      → Filter.Tendsto stub
  Linear alg   Matrix, Trace, Transpose
  Sums/prods   Sum → ∑, Product → ∏  (Finset.Icc range)
  Piecewise    → if/else chain
  Sets         Interval (Icc/Ioc/Ico/Ioo), FiniteSet, Union, Intersection
  Polynomials  Poly → Polynomial.C / .X
  Unknown      → (sorry /- unsupported ... -/)
"""
from __future__ import annotations

import sympy as sp
from sympy import (
    Abs,
    Add,
    Derivative,
    Eq,
    FiniteSet,
    Ge,
    Gt,
    Integral,
    Intersection,
    Interval,
    Le,
    Limit,
    Lt,
    Matrix,
    MatrixSymbol,
    Max,
    Min,
    Mod,
    Mul,
    Ne,
    Piecewise,
    Poly,
    Pow,
    Product,
    S,
    Sum,
    Symbol,
    Transpose,
    Union,
    acos,
    acosh,
    acot,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    binomial,
    ceiling,
    conjugate,
    cos,
    cosh,
    cot,
    csc,
    exp,
    factorial,
    floor,
    im,
    log,
    re,
    sec,
    sign,
    sin,
    sinh,
    tan,
    tanh,
)
from sympy.core.numbers import (
    Float,
    Integer,
    Rational,
)
from sympy.matrices.matrixbase import MatrixBase as _MatrixBase

# Trace is in sympy.matrices
try:
    from sympy.matrices.expressions.trace import Trace
    HAS_TRACE = True
except ImportError:
    HAS_TRACE = False

from typing import Dict, List, Optional

from sympy.logic.boolalg import (
    And,
    BooleanFalse,
    BooleanTrue,
    Implies,
    Not,
    Or,
    Xor,
)

# ──────────────────────────────────────────────────────────────────────
# Mathlib import catalogue
# ──────────────────────────────────────────────────────────────────────

LEAN_TYPE = {
    "real":    "ℝ",
    "complex": "ℂ",
    "int":     "ℤ",
    "nat":     "ℕ",
    "rat":     "ℚ",
    "bool":    "Bool",
}

IMPORTS: Dict[str, str] = {
    "trig":       "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
    "trig_inv":   "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse",
    "trig_deriv": "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv",
    "hyp":        "import Mathlib.Analysis.SpecialFunctions.Complex.Analytic",
    "exp_log":    "import Mathlib.Analysis.SpecialFunctions.ExpDeriv",
    "log_basic":  "import Mathlib.Analysis.SpecialFunctions.Log.Basic",
    "pow_real":   "import Mathlib.Analysis.SpecialFunctions.Pow.Real",
    "complex":    "import Mathlib.Analysis.SpecialFunctions.Complex.Circle",
    "factorial":  "import Mathlib.Data.Nat.Factorial.Basic",
    "polynomial": "import Mathlib.RingTheory.Polynomial.Basic",
    "mvpoly":     "import Mathlib.Algebra.MvPolynomial.Basic",
    "matrix":     "import Mathlib.LinearAlgebra.Matrix.Determinant.Basic",
    "finset":     "import Mathlib.Algebra.BigOperators.Group.Finset",
    "integral":   "import Mathlib.MeasureTheory.Integral.IntervalIntegral",
    "tactic":     "import Mathlib.Tactic",
}

BASE_IMPORTS = [
    IMPORTS["trig"],
    IMPORTS["exp_log"],
    IMPORTS["log_basic"],
    IMPORTS["factorial"],
    IMPORTS["tactic"],
]

# Trig function dispatch table: (sympy_class, lean4_name, import_tag)
TRIG_TABLE = [
    (sin,   "Real.sin",    "trig"),
    (cos,   "Real.cos",    "trig"),
    (tan,   "Real.tan",    "trig"),
    (cot,   "Real.cot",    "trig"),
    (asin,  "Real.arcsin", "trig_inv"),
    (acos,  "Real.arccos", "trig_inv"),
    (atan,  "Real.arctan", "trig_inv"),
    (acot,  "Real.arccot", "trig_inv"),
    (sinh,  "Real.sinh",   "hyp"),
    (cosh,  "Real.cosh",   "hyp"),
    (tanh,  "Real.tanh",   "hyp"),
    (asinh, "Real.arcsinh","hyp"),
    (acosh, "Real.arccosh","hyp"),
    (atanh, "Real.arctanh","hyp"),
]


class SymPyToLean4:
    """
    General-purpose SymPy → Lean 4 translator.

    Parameters
    ----------
    domain : 'real' | 'complex' | 'int' | 'nat'
        Default numeric domain.
    track_imports : bool
        Collect Mathlib imports as translation proceeds.
    var_types : dict, optional
        Explicit overrides  {symbol_name: 'real'|'complex'|...}
    real_vars : list of str, optional
        Compatibility alias, marks these symbols as real.

    Examples
    --------
    >>> t = SymPyToLean4()
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> t.translate(sp.sin(x)**2 + sp.cos(x)**2)
    'Real.sin x ^ 2 + Real.cos x ^ 2'
    >>> t.translate(sp.Eq(x**2, 1))
    'x ^ 2 = 1'
    >>> t.translate(sp.Integral(sp.sin(x), (x, 0, sp.pi)), simplify=False)
    '∫ x in Real.pi..0, Real.sin x'
    """

    def __init__(
        self,
        domain: str = "real",
        track_imports: bool = True,
        var_types: Optional[Dict[str, str]] = None,
        real_vars: Optional[List[str]] = None,
    ):
        self.domain = domain
        self.track_imports = track_imports
        self.var_types: Dict[str, str] = dict(var_types or {})
        self._used_imports: set = set()

        if real_vars:
            for v in real_vars:
                self.var_types.setdefault(v, "real")

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def translate(self, expr: sp.Basic, simplify: bool = True) -> str:
        """Translate a SymPy expression to a Lean 4 string."""
        if simplify:
            try:
                expr = sp.simplify(expr)
            except Exception:
                pass
        return self._e(expr)

    def required_imports(self) -> List[str]:
        """Return Mathlib import lines needed for the last translation."""
        result = list(BASE_IMPORTS)
        for tag in sorted(self._used_imports):
            line = IMPORTS.get(tag)
            if line and line not in result:
                result.append(line)
        return result

    def suggest_tactic(self, expr: sp.Basic) -> str:
        """Suggest a Lean 4 closing tactic for the goal `lhs - rhs = expr`."""
        try:
            s = sp.simplify(expr)
        except Exception:
            s = expr
        if s == S.Zero:
            return "ring"
        if isinstance(s, (Integer, Rational, Float)) or s.is_number:
            return "norm_num"
        if s.has(sin) or s.has(cos):
            lemmas = []
            if s.has(sin):
                lemmas.append("Real.sin_sq")
            if s.has(cos):
                lemmas.append("Real.cos_sq")
            if s.has(tan):
                lemmas.append("Real.tan_eq_sin_div_cos")
            return f"simp [{', '.join(lemmas)}]; ring"
        if s.has(exp) or s.has(log):
            return "simp [Real.exp_add, Real.log_mul]; ring"
        if s.has(sp.sqrt):
            return "rw [Real.sqrt_eq_iff_sq_eq (by positivity) (by positivity)]; ring"
        if any(
            isinstance(a, Pow) and a.exp == S.NegativeOne
            for a in s.atoms(Pow)
        ):
            return "field_simp; ring"
        return "ring"

    def variable_declarations(self, symbols: list) -> str:
        """Emit Lean 4 `variable` lines for a list of SymPy symbols."""
        lines = []
        for sym in symbols:
            t = self._sym_type(sym)
            lines.append(f"variable ({sym} : {LEAN_TYPE[t]})")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Core dispatcher
    # ──────────────────────────────────────────────────────────────────

    def _e(self, e: sp.Basic) -> str:  # noqa: C901
        """Translate expression e (no outer parenthesisation)."""

        # ── Boolean literals ──
        if isinstance(e, BooleanTrue):
            return "true"
        if isinstance(e, BooleanFalse):
            return "false"

        # ── Numeric constants ──
        if e is S.Zero:
            return "0"
        if e is S.One:
            return "1"
        if e is S.NegativeOne:
            return "-1"
        if e is S.Half:
            return f"(1 : {LEAN_TYPE[self.domain]}) / 2"
        if e is S.Pi:
            self._use("trig")
            return "Real.pi"
        if e is S.Exp1:
            self._use("exp_log")
            return "Real.exp 1"
        if e is S.ImaginaryUnit:
            self._use("complex")
            return "Complex.I"
        if e is S.Infinity:
            return "⊤"
        if e is S.NegativeInfinity:
            return "⊥"
        if e is S.ComplexInfinity:
            return "(⊤ : ℂ)"
        if e is S.NaN:
            return "default"

        # ── Numeric types ──
        if isinstance(e, Integer):
            n = int(e)
            if abs(n) <= 1:
                return str(n)
            typ = LEAN_TYPE[self.domain]
            return f"({n} : {typ})" if n >= 0 else f"(-{-n} : {typ})"
        if isinstance(e, Rational):
            typ = LEAN_TYPE[self.domain]
            return f"({e.p} : {typ}) / {e.q}"
        if isinstance(e, Float):
            return f"({float(e)} : {LEAN_TYPE[self.domain]})"

        # ── Symbol ──
        if isinstance(e, Symbol):
            return str(e)

        # ── Arithmetic ──
        if isinstance(e, Add):
            return self._add(e)
        if isinstance(e, Mul):
            return self._mul(e)
        if isinstance(e, Pow):
            return self._pow(e)

        # ── Relational ──
        if isinstance(e, Eq):
            return f"{self._e(e.lhs)} = {self._e(e.rhs)}"
        if isinstance(e, Ne):
            return f"{self._e(e.lhs)} ≠ {self._e(e.rhs)}"
        if isinstance(e, Lt):
            return f"{self._e(e.lhs)} < {self._e(e.rhs)}"
        if isinstance(e, Le):
            return f"{self._e(e.lhs)} ≤ {self._e(e.rhs)}"
        if isinstance(e, Gt):
            return f"{self._e(e.lhs)} > {self._e(e.rhs)}"
        if isinstance(e, Ge):
            return f"{self._e(e.lhs)} ≥ {self._e(e.rhs)}"

        # ── Logic ──
        if isinstance(e, And):
            return " ∧ ".join(f"({self._e(a)})" for a in e.args)
        if isinstance(e, Or):
            return " ∨ ".join(f"({self._e(a)})" for a in e.args)
        if isinstance(e, Not):
            return f"¬ ({self._e(e.args[0])})"
        if isinstance(e, Implies):
            return f"({self._e(e.args[0])}) → ({self._e(e.args[1])})"
        if isinstance(e, Xor):
            return " ⊕ ".join(f"({self._e(a)})" for a in e.args)

        # ── Trig / Hyperbolic ──
        for cls, lean_name, tag in TRIG_TABLE:
            if type(e) is cls:
                self._use(tag)
                return f"{lean_name} {self._arg(e.args[0])}"

        # ── atan2 ──
        if type(e) is atan2:
            self._use("trig_inv")
            return f"Real.arctan2 {self._arg(e.args[0])} {self._arg(e.args[1])}"

        # ── sec / csc (expressed as 1/cos, 1/sin) ──
        if type(e) is sec:
            self._use("trig")
            return f"1 / Real.cos {self._arg(e.args[0])}"
        if type(e) is csc:
            self._use("trig")
            return f"1 / Real.sin {self._arg(e.args[0])}"

        # ── exp / log ──
        if type(e) is exp:
            self._use("exp_log")
            return f"Real.exp {self._arg(e.args[0])}"
        if type(e) is log:
            self._use("log_basic")
            if len(e.args) == 2:
                return f"Real.log {self._arg(e.args[0])} / Real.log {self._arg(e.args[1])}"  # noqa: E501
            return f"Real.log {self._arg(e.args[0])}"

        # ── Abs ──
        if type(e) is Abs:
            return f"|{self._e(e.args[0])}|"

        # ── sign ──
        if type(e) is sign:
            return f"Int.sign ({self._e(e.args[0])})"

        # ── floor / ceiling ──
        if type(e) is floor:
            return f"⌊{self._e(e.args[0])}⌋"
        if type(e) is ceiling:
            return f"⌈{self._e(e.args[0])}⌉"

        # ── Max / Min ──
        if type(e) is Max:
            return self._fold("max", e.args)
        if type(e) is Min:
            return self._fold("min", e.args)

        # ── Mod ──
        if type(e) is Mod:
            return f"{self._paren(self._e(e.args[0]))} % {self._paren(self._e(e.args[1]))}"  # noqa: E501

        # ── re / im / conjugate ──
        if type(e) is re:
            self._use("complex")
            return f"Complex.re ({self._e(e.args[0])})"
        if type(e) is im:
            self._use("complex")
            return f"Complex.im ({self._e(e.args[0])})"
        if type(e) is conjugate:
            self._use("complex")
            return f"starRingEnd ℂ ({self._e(e.args[0])})"

        # ── factorial ──
        if type(e) is factorial:
            self._use("factorial")
            # factorial(5) simplifies to 120 — check if already a number
            inner = e.args[0]
            return f"Nat.factorial {self._arg(inner)}"

        # ── binomial ──
        if type(e) is binomial:
            self._use("factorial")
            return f"Nat.choose {self._arg(e.args[0])} {self._arg(e.args[1])}"

        # ── Piecewise ──
        if isinstance(e, Piecewise):
            return self._piecewise(e)

        # ── Sum / Product ──
        if isinstance(e, Sum):
            return self._sum(e)
        if isinstance(e, Product):
            return self._product(e)

        # ── Derivative ──
        if isinstance(e, Derivative):
            return self._derivative(e)

        # ── Integral ──
        if isinstance(e, Integral):
            return self._integral(e)

        # ── Limit ──
        if isinstance(e, Limit):
            return self._limit(e)

        # ── Matrix ──
        if isinstance(e, (Matrix, _MatrixBase)):
            return self._matrix(e)
        if isinstance(e, MatrixSymbol):
            return str(e.name)
        if isinstance(e, Transpose):
            self._use("matrix")
            return f"({self._e(e.args[0])})ᵀ"
        if HAS_TRACE and isinstance(e, Trace):
            self._use("matrix")
            return f"Matrix.trace ({self._e(e.args[0])})"

        # ── Sets ──
        if isinstance(e, FiniteSet):
            return "{" + ", ".join(self._e(a) for a in e.args) + "}"
        if isinstance(e, Interval):
            a_s = self._e(e.args[0])
            b_s = self._e(e.args[1])
            if e.left_open and e.right_open:
                return f"Set.Ioo {a_s} {b_s}"
            if e.left_open:
                return f"Set.Ioc {a_s} {b_s}"
            if e.right_open:
                return f"Set.Ico {a_s} {b_s}"
            return f"Set.Icc {a_s} {b_s}"
        if isinstance(e, Union):
            return " ∪ ".join(f"({self._e(a)})" for a in e.args)
        if isinstance(e, Intersection):
            return " ∩ ".join(f"({self._e(a)})" for a in e.args)

        # ── Polynomial ──
        if isinstance(e, Poly):
            return self._poly(e)

        # ── Fallback ──
        return f"(sorry  /- unsupported: {type(e).__name__} — {e} -/)"

    # ──────────────────────────────────────────────────────────────────
    # Arithmetic helpers
    # ──────────────────────────────────────────────────────────────────

    def _add(self, e: Add) -> str:
        terms = e.as_ordered_terms()
        parts: list[str] = []
        for t in terms:
            s = self._e(t)
            if not parts:
                parts.append(s)
            elif s.startswith("-"):
                parts.append(f"- {s[1:].lstrip()}")
            else:
                parts.append(f"+ {s}")
        return " ".join(parts)

    def _mul(self, e: Mul) -> str:
        factors = e.as_ordered_factors()
        negate = False
        num_coeff = S.One
        sym_factors = []

        for f in factors:
            if f == S.NegativeOne:
                negate = not negate
            elif isinstance(f, (Integer, Rational, Float)):
                if f < 0:
                    negate = not negate
                    num_coeff = num_coeff * (-f)
                else:
                    num_coeff = num_coeff * f
            else:
                sym_factors.append(f)

        parts: list[str] = []
        if num_coeff != S.One:
            parts.append(self._e(num_coeff))
        for f in sym_factors:
            s = self._e(f)
            if isinstance(f, Add):
                s = f"({s})"
            parts.append(s)

        joined = " * ".join(parts) if parts else "1"
        return f"-{joined}" if negate else joined

    def _pow(self, e: Pow) -> str:
        base, exp_val = e.args

        if exp_val == S.NegativeOne:
            return f"1 / {self._paren(self._e(base))}"
        if exp_val == sp.Rational(1, 2):
            self._use("pow_real")
            return f"Real.sqrt {self._arg(base)}"
        if exp_val == sp.Rational(1, 3):
            self._use("pow_real")
            return f"{self._arg(base)} ^ ((1 : ℝ) / 3)"

        b  = self._arg(base)
        ex = self._arg(exp_val)
        return f"{b} ^ {ex}"

    def _paren(self, s: str) -> str:
        if " " in s and not (s.startswith("(") and s.endswith(")")):
            return f"({s})"
        return s

    def _arg(self, e: sp.Basic) -> str:
        """Format e as a function argument — parenthesise if compound."""
        s = self._e(e)
        if isinstance(e, (Add, Mul)) or (
            isinstance(e, Pow) and len(s) > 4
        ):
            return f"({s})"
        return s

    def _fold(self, fn: str, args) -> str:
        if len(args) == 1:
            return self._arg(args[0])
        return f"{fn} {self._arg(args[0])} ({self._fold(fn, args[1:])})"

    # ──────────────────────────────────────────────────────────────────
    # Piecewise
    # ──────────────────────────────────────────────────────────────────

    def _piecewise(self, e: Piecewise) -> str:
        pieces = list(e.args)
        def build(ps):
            if not ps:
                return "default"
            val, cond = ps[0]
            if cond is S.true or cond is True:
                return self._e(val)
            return f"if {self._e(cond)} then {self._e(val)} else {build(ps[1:])}"
        return build(pieces)

    # ──────────────────────────────────────────────────────────────────
    # Sum / Product
    # ──────────────────────────────────────────────────────────────────

    def _sum(self, e: Sum) -> str:
        self._use("finset")
        body = self._e(e.function)
        if not e.limits:
            return body
        sym, lo, hi = e.limits[0]
        lo_s, hi_s = self._e(lo), self._e(hi)
        return f"∑ {sym} ∈ Finset.Icc {lo_s} {hi_s}, {body}"

    def _product(self, e: Product) -> str:
        self._use("finset")
        body = self._e(e.function)
        if not e.limits:
            return body
        sym, lo, hi = e.limits[0]
        lo_s, hi_s = self._e(lo), self._e(hi)
        return f"∏ {sym} ∈ Finset.Icc {lo_s} {hi_s}, {body}"

    # ──────────────────────────────────────────────────────────────────
    # Calculus
    # ──────────────────────────────────────────────────────────────────

    def _derivative(self, e: Derivative) -> str:
        self._use("trig_deriv")
        func = e.expr
        if not e.variables:
            return f"(sorry /- derivative: {e} -/)"
        var_info = e.variables[0]
        var, order = (var_info if isinstance(var_info, tuple) else (var_info, 1))
        v = str(var)
        f_s = self._e(func)
        if order == 1:
            return f"HasDerivAt (fun {v} => {f_s}) (sorry /- d/d{v} -/) {v}"
        return f"iteratedDeriv {order} (fun {v} => {f_s}) {v}"

    def _integral(self, e: Integral) -> str:
        self._use("integral")
        body = self._e(e.function)
        if not e.limits:
            return f"∫ x, {body}"
        sym, *bounds = e.limits[0]
        v = str(sym)
        if len(bounds) == 2:
            a_s, b_s = self._e(bounds[0]), self._e(bounds[1])
            return f"∫ {v} in {a_s}..{b_s}, {body}"
        return f"∫ {v}, {body}"

    def _limit(self, e: Limit) -> str:
        f_s = self._e(e.args[0])
        v   = str(e.args[1])
        pt  = self._e(e.args[2])
        return (f"Filter.Tendsto (fun {v} => {f_s}) "
                f"(nhds {pt}) (nhds (sorry /- lim -/))")

    # ──────────────────────────────────────────────────────────────────
    # Matrix
    # ──────────────────────────────────────────────────────────────────

    def _matrix(self, e: Matrix) -> str:
        self._use("matrix")
        rows = [
            "![" + ", ".join(self._e(e[i, j]) for j in range(e.cols)) + "]"
            for i in range(e.rows)
        ]
        return "![" + ", ".join(rows) + "]"

    # ──────────────────────────────────────────────────────────────────
    # Polynomial
    # ──────────────────────────────────────────────────────────────────

    def _poly(self, e: Poly) -> str:
        self._use("polynomial")
        coeffs = e.all_coeffs()
        gens = e.gens
        if len(gens) == 1:
            deg = len(coeffs) - 1
            terms = []
            for i, c in enumerate(coeffs):
                d = deg - i
                c_s = self._e(sp.sympify(c))
                if d == 0:
                    terms.append(f"Polynomial.C {self._paren(c_s)}")
                elif d == 1:
                    terms.append(f"Polynomial.C {self._paren(c_s)} * Polynomial.X")
                else:
                    terms.append(f"Polynomial.C {self._paren(c_s)} * Polynomial.X ^ {d}")  # noqa: E501
            return " + ".join(terms) if terms else "0"
        self._use("mvpoly")
        return f"(sorry /- MvPolynomial: {e} -/)"

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _sym_type(self, sym: Symbol) -> str:
        name = str(sym)
        if name in self.var_types:
            return self.var_types[name]
        if sym.is_integer:
            return "int"
        if sym.is_real:
            return "real"
        if sym.is_complex:
            return "complex"
        return self.domain

    def _use(self, tag: str) -> None:
        if self.track_imports:
            self._used_imports.add(tag)
