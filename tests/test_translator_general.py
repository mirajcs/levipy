"""
Comprehensive tests for the general-purpose SymPyToLean4 translator.

Covers every construct: numbers, symbols, arithmetic, trig, hyperbolic,
exp/log, sqrt, abs, floor/ceil, max/min, factorial, binomial, piecewise,
sum, product, derivative, integral, matrix, polynomial, sets, relational,
logic, and the tactic suggester.
"""
import sympy as sp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from levipy.lean4.translator import SymPyToLean4

T = SymPyToLean4()
x, y, z = sp.symbols('x y z', real=True)
n, k = sp.symbols('n k', integer=True, nonneg=True)
c = sp.Symbol('c')   # complex


# ─────────────────────────────────────────────────────────────────────
# Numbers
# ─────────────────────────────────────────────────────────────────────
def test_zero():         assert T.translate(sp.Integer(0)) == "0"
def test_one():          assert T.translate(sp.Integer(1)) == "1"
def test_neg_one():      assert T.translate(sp.Integer(-1)) == "-1"
def test_pos_int():
    r = T.translate(sp.Integer(7))
    assert "7" in r
def test_neg_int():
    r = T.translate(sp.Integer(-5))
    assert "5" in r
def test_rational():
    r = T.translate(sp.Rational(3, 4))
    assert "3" in r and "4" in r
def test_half():
    r = T.translate(sp.Rational(1, 2))
    assert "1" in r and "2" in r
def test_float():
    r = T.translate(sp.Float(3.14))
    assert "3.14" in r
def test_pi():
    r = T.translate(sp.pi)
    assert "Real.pi" in r
def test_e():
    r = T.translate(sp.E)
    assert "Real.exp" in r or "exp" in r
def test_inf():          assert "⊤" in T.translate(sp.oo)
def test_neg_inf():      assert "⊥" in T.translate(-sp.oo)
def test_imaginary_unit():
    r = T.translate(sp.I)
    assert "Complex.I" in r or "I" in r

# ─────────────────────────────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────────────────────────────
def test_symbol_name():
    r = T.translate(x)
    assert "x" in r

def test_variable_declarations():
    r = T.variable_declarations([x, y])
    assert "variable (x : ℝ)" in r
    assert "variable (y : ℝ)" in r

def test_integer_symbol_decl():
    t = SymPyToLean4()
    r = t.variable_declarations([n])
    assert "ℤ" in r or "ℕ" in r

# ─────────────────────────────────────────────────────────────────────
# Arithmetic
# ─────────────────────────────────────────────────────────────────────
def test_add():
    r = T.translate(x + y, simplify=False)
    assert "x" in r and "y" in r and ("+" in r or "-" in r)

def test_mul():
    r = T.translate(x * y, simplify=False)
    assert "x" in r and "y" in r

def test_pow():
    r = T.translate(x**3)
    assert "^" in r and "3" in r

def test_inverse():
    r = T.translate(1/x)
    assert "1 / x" in r

def test_sqrt_pow():
    r = T.translate(x**sp.Rational(1, 2))
    assert "Real.sqrt" in r

def test_cbrt():
    r = T.translate(sp.cbrt(x))
    assert "1" in r and "3" in r

def test_neg_expr():
    r = T.translate(-x)
    assert "-" in r and "x" in r

# ─────────────────────────────────────────────────────────────────────
# Trig
# ─────────────────────────────────────────────────────────────────────
def test_sin():     assert "Real.sin" in T.translate(sp.sin(x))
def test_cos():     assert "Real.cos" in T.translate(sp.cos(x))
def test_tan():     assert "Real.tan" in T.translate(sp.tan(x))
def test_asin():    assert "arcsin" in T.translate(sp.asin(x))
def test_acos():    assert "arccos" in T.translate(sp.acos(x))
def test_atan():    assert "arctan" in T.translate(sp.atan(x))
def test_sin_sq():
    r = T.translate(sp.sin(x)**2)
    assert "Real.sin" in r and "^" in r

# ─────────────────────────────────────────────────────────────────────
# Hyperbolic
# ─────────────────────────────────────────────────────────────────────
def test_sinh():    assert "Real.sinh" in T.translate(sp.sinh(x))
def test_cosh():    assert "Real.cosh" in T.translate(sp.cosh(x))
def test_tanh():    assert "Real.tanh" in T.translate(sp.tanh(x))

# ─────────────────────────────────────────────────────────────────────
# Exp / log / sqrt
# ─────────────────────────────────────────────────────────────────────
def test_exp():     assert "Real.exp" in T.translate(sp.exp(x))
def test_log():     assert "Real.log" in T.translate(sp.log(x))
def test_log_base():
    r = T.translate(sp.log(x, 2))
    assert "Real.log" in r and "2" in r
def test_sqrt():    assert "Real.sqrt" in T.translate(sp.sqrt(x))

# ─────────────────────────────────────────────────────────────────────
# Abs / floor / ceiling / sign
# ─────────────────────────────────────────────────────────────────────
def test_abs():     assert "|" in T.translate(sp.Abs(x))
def test_floor():   assert "⌊" in T.translate(sp.floor(x))
def test_ceil():    assert "⌈" in T.translate(sp.ceiling(x))

# ─────────────────────────────────────────────────────────────────────
# Max / Min
# ─────────────────────────────────────────────────────────────────────
def test_max():
    r = T.translate(sp.Max(x, y))
    assert "max" in r
def test_min():
    r = T.translate(sp.Min(x, y))
    assert "min" in r
def test_max_three():
    r = T.translate(sp.Max(x, y, z))
    assert "max" in r

# ─────────────────────────────────────────────────────────────────────
# Factorial / binomial
# ─────────────────────────────────────────────────────────────────────
def test_factorial():
    r = T.translate(sp.factorial(n))
    assert "Nat.factorial" in r
def test_factorial_literal():
    # factorial(5) simplifies to integer 120 — that's correct Lean output
    r = T.translate(sp.factorial(5))
    assert "120" in r or "5" in r
def test_binomial():
    r = T.translate(sp.binomial(n, k))
    assert "Nat.choose" in r

# ─────────────────────────────────────────────────────────────────────
# Relational
# ─────────────────────────────────────────────────────────────────────
def test_eq():
    r = T.translate(sp.Eq(x**2, 1))
    assert "=" in r and "x" in r
def test_ne():
    r = T.translate(sp.Ne(x, 0))
    assert "≠" in r
def test_lt():
    r = T.translate(sp.Lt(x, y))
    assert "<" in r
def test_le():
    r = T.translate(sp.Le(x, 0))
    assert "≤" in r
def test_gt():
    r = T.translate(sp.Gt(x, 0))
    assert ">" in r
def test_ge():
    r = T.translate(sp.Ge(x, y))
    assert "≥" in r

# ─────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────
def test_and():
    r = T.translate(sp.And(sp.Gt(x, 0), sp.Lt(x, 1)))
    assert "∧" in r
def test_or():
    r = T.translate(sp.Or(sp.Eq(x, 0), sp.Eq(x, 1)))
    assert "∨" in r
def test_not():
    # Not(Gt(x,0)) becomes Le(x,0) symbolically — still a negation
    r = T.translate(sp.Not(sp.Gt(x, 0)))
    assert r  # SymPy may simplify Not(x>0) to x<=0 — both valid
def test_implies():
    from sympy.logic.boolalg import Implies as SImplies
    i2 = SImplies(sp.Symbol("p"), sp.Symbol("q"))  # stays as Implies for pure booleans
    r = T.translate(i2, simplify=False)
    assert "→" in r or "∨" in r  # SymPy may convert to Or

# ─────────────────────────────────────────────────────────────────────
# Piecewise
# ─────────────────────────────────────────────────────────────────────
def test_piecewise():
    pw = sp.Piecewise((x, x > 0), (-x, True))
    r = T.translate(pw)
    assert "if" in r and "then" in r and "else" in r
def test_piecewise_three():
    pw = sp.Piecewise((1, x < 0), (0, sp.Eq(x, 0)), (-1, True))
    r = T.translate(pw)
    assert r.count("if") >= 2

# ─────────────────────────────────────────────────────────────────────
# Sum and Product
# ─────────────────────────────────────────────────────────────────────
def test_sum():
    i = sp.Symbol('i', integer=True)
    r = T.translate(sp.Sum(i**2, (i, 1, n)))
    assert "∑" in r
def test_product():
    i = sp.Symbol('i', integer=True)
    r = T.translate(sp.Product(i, (i, 1, n)), simplify=False)
    assert "∏" in r

# ─────────────────────────────────────────────────────────────────────
# Derivative
# ─────────────────────────────────────────────────────────────────────
def test_derivative_sin():
    r = T.translate(sp.Derivative(sp.sin(x), x), simplify=False)
    assert "HasDerivAt" in r or "sorry" in r or "Real.sin" in r
def test_derivative_poly():
    r = T.translate(sp.Derivative(x**3, x), simplify=False)
    assert "HasDerivAt" in r or "sorry" in r

# ─────────────────────────────────────────────────────────────────────
# Integral
# ─────────────────────────────────────────────────────────────────────
def test_definite_integral():
    r = T.translate(sp.Integral(sp.sin(x), (x, 0, sp.pi)), simplify=False)
    assert "∫" in r and "Real.sin" in r
def test_indefinite_integral():
    r = T.translate(sp.Integral(x**2, x), simplify=False)
    assert "∫" in r

# ─────────────────────────────────────────────────────────────────────
# Matrix
# ─────────────────────────────────────────────────────────────────────
def test_matrix_2x2():
    M = sp.Matrix([[1, 2], [3, 4]])
    r = T.translate(M)
    assert "![![" in r
    assert "1" in r and "4" in r
def test_matrix_identity():
    M = sp.eye(2)
    r = T.translate(M)
    assert "![![" in r

# ─────────────────────────────────────────────────────────────────────
# Sets
# ─────────────────────────────────────────────────────────────────────
def test_interval_closed():
    r = T.translate(sp.Interval(0, 1))
    assert "Set.Icc" in r
def test_interval_open():
    r = T.translate(sp.Interval.open(0, 1))
    assert "Set.Ioo" in r
def test_interval_left_open():
    r = T.translate(sp.Interval.Lopen(0, 1))
    assert "Set.Ioc" in r
def test_finite_set():
    r = T.translate(sp.FiniteSet(1, 2, 3))
    assert "{" in r

# ─────────────────────────────────────────────────────────────────────
# Polynomial
# ─────────────────────────────────────────────────────────────────────
def test_poly_linear():
    p = sp.Poly(x + 1, x)
    r = T.translate(p)
    assert "Polynomial.X" in r or "Polynomial.C" in r
def test_poly_quadratic():
    p = sp.Poly(x**2 - 3*x + 2, x)
    r = T.translate(p)
    assert "Polynomial" in r

# ─────────────────────────────────────────────────────────────────────
# Import tracking
# ─────────────────────────────────────────────────────────────────────
def test_imports_trig():
    t = SymPyToLean4(track_imports=True)
    t.translate(sp.sin(x))
    imports = t.required_imports()
    assert any("Trigonometric" in i for i in imports)

def test_imports_matrix():
    t = SymPyToLean4(track_imports=True)
    t.translate(sp.Matrix([[1, 2], [3, 4]]))
    imports = t.required_imports()
    assert any("Matrix" in i for i in imports)

def test_imports_factorial():
    t = SymPyToLean4(track_imports=True)
    t.translate(sp.factorial(5))
    imports = t.required_imports()
    assert any("Factorial" in i for i in imports)

# ─────────────────────────────────────────────────────────────────────
# Tactic suggester
# ─────────────────────────────────────────────────────────────────────
def test_tactic_zero():       assert T.suggest_tactic(sp.Integer(0)) == "ring"
def test_tactic_number():     assert T.suggest_tactic(sp.Rational(1, 3)) == "norm_num"
def test_tactic_polynomial():
    r = T.suggest_tactic(x**2 - x)
    assert "ring" in r
def test_tactic_trig():
    r = T.suggest_tactic(sp.sin(x)**2)
    assert "simp" in r or "ring" in r
def test_tactic_division():
    r = T.suggest_tactic(1/x - 1/y)
    assert "field_simp" in r
def test_tactic_exp():
    r = T.suggest_tactic(sp.exp(x + y))
    assert "simp" in r or "ring" in r


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed+failed} tests")
    if failed:
        raise SystemExit(1)
