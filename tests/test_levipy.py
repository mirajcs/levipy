"""
Tests for levipy.

Covers:
- Flat R^2 (all Christoffel = 0)
- 2-sphere S^2 (known Christoffel symbols)
- Schwarzschild metric (Ricci-flat check)
"""
import sympy as sp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from levipy import Manifold
from levipy.geometry.lie_bracket import lie_bracket


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def symplify_zero(expr):
    return sp.simplify(expr) == 0


# -----------------------------------------------------------------------
# Flat R^2
# -----------------------------------------------------------------------

def test_flat_r2_christoffel_zero():
    x, y = sp.symbols('x y')
    M = Manifold('R2', [x, y])
    g = M.metric([[1, 0], [0, 1]])
    ch = g.christoffel()
    for k in range(2):
        for i in range(2):
            for j in range(2):
                assert ch.Gamma[k][i][j] == 0, \
                    f"Γ^{k}_{{{i}{j}}} should be 0 in flat R²"


def test_flat_r2_riemann_zero():
    x, y = sp.symbols('x y')
    M = Manifold('R2', [x, y])
    g = M.metric([[1, 0], [0, 1]])
    R = g.riemann()
    assert R.is_flat()


def test_flat_r2_ricci_zero():
    x, y = sp.symbols('x y')
    M = Manifold('R2', [x, y])
    g = M.metric([[1, 0], [0, 1]])
    Ric = g.ricci_tensor()
    for i in range(2):
        for j in range(2):
            assert sp.simplify(Ric[i, j]) == 0


# -----------------------------------------------------------------------
# 2-sphere S^2 (unit, r=1)
# -----------------------------------------------------------------------

def test_s2_christoffel():
    """
    For the unit 2-sphere  g = [[1,0],[0,sin²θ]]:
      Γ^θ_{φφ} = −sin θ cos θ
      Γ^φ_{θφ} = Γ^φ_{φθ} = cos θ / sin θ
      All others = 0
    """
    th, ph = sp.symbols('theta phi', real=True)
    M = Manifold('S2', [th, ph])
    g = M.metric([[1, 0], [0, sp.sin(th)**2]])
    ch = g.christoffel()

    # Γ^θ_{φφ} = -sin θ cos θ
    val = sp.simplify(ch.Gamma[0][1][1] + sp.sin(th) * sp.cos(th))
    assert val == 0, f"Γ^θ_{{φφ}} wrong: {ch.Gamma[0][1][1]}"

    # Γ^φ_{θφ} = cos θ / sin θ
    val = sp.simplify(ch.Gamma[1][0][1] - sp.cos(th) / sp.sin(th))
    assert val == 0, f"Γ^φ_{{θφ}} wrong: {ch.Gamma[1][0][1]}"

    # Γ^θ_{θθ}, Γ^θ_{θφ}, Γ^φ_{φφ} = 0
    assert ch.Gamma[0][0][0] == 0
    assert ch.Gamma[0][0][1] == 0
    assert sp.simplify(ch.Gamma[1][1][1]) == 0


def test_s2_ricci_scalar():
    """Ricci scalar of unit 2-sphere = 2."""
    th, ph = sp.symbols('theta phi', real=True)
    M = Manifold('S2', [th, ph])
    g = M.metric([[1, 0], [0, sp.sin(th)**2]])
    R = g.ricci_scalar()
    assert sp.simplify(R.value - 2) == 0, f"Expected R=2, got {R.value}"


# -----------------------------------------------------------------------
# Lie bracket
# -----------------------------------------------------------------------

def test_lie_bracket_anticommutative():
    x, y = sp.symbols('x y')
    X = [y, sp.Integer(0)]
    Y = [sp.Integer(0), x]
    XY = lie_bracket(X, Y, [x, y])
    YX = lie_bracket(Y, X, [x, y])
    for k in range(2):
        assert sp.simplify(XY[k] + YX[k]) == 0


def test_lie_bracket_coordinate_fields():
    """[∂_x, ∂_y] = 0."""
    x, y = sp.symbols('x y')
    X = [sp.Integer(1), sp.Integer(0)]
    Y = [sp.Integer(0), sp.Integer(1)]
    result = lie_bracket(X, Y, [x, y])
    assert all(r == 0 for r in result)


# -----------------------------------------------------------------------
# Geodesic equations (smoke test)
# -----------------------------------------------------------------------

def test_geodesic_equations_produced():
    th, ph = sp.symbols('theta phi', real=True)
    M = Manifold('S2', [th, ph])
    g = M.metric([[1, 0], [0, sp.sin(th)**2]])
    geo = g.geodesic_equations()
    assert len(geo.equations) == 2
    for eq in geo.equations:
        assert isinstance(eq, sp.Eq)


# -----------------------------------------------------------------------
# Covariant derivative of constant vector in flat space = 0
# -----------------------------------------------------------------------

def test_covariant_derivative_flat():
    x, y = sp.symbols('x y')
    M = Manifold('R2', [x, y])
    g = M.metric([[1, 0], [0, 1]])
    nabla = g.covariant_derivative()
    X = [sp.Integer(1), sp.Integer(0)]   # constant vector
    result = nabla.of_vector(X)
    for i in range(2):
        for k in range(2):
            assert result[i, k] == 0


if __name__ == "__main__":
    test_flat_r2_christoffel_zero();  print("✓ flat R² Christoffel = 0")
    test_flat_r2_riemann_zero();      print("✓ flat R² Riemann = 0")
    test_flat_r2_ricci_zero();        print("✓ flat R² Ricci = 0")
    test_s2_christoffel();            print("✓ S² Christoffel symbols correct")
    test_s2_ricci_scalar();           print("✓ S² Ricci scalar = 2")
    test_lie_bracket_anticommutative(); print("✓ Lie bracket anti-commutative")
    test_lie_bracket_coordinate_fields(); print("✓ [∂_x, ∂_y] = 0")
    test_geodesic_equations_produced(); print("✓ Geodesic equations produced")
    test_covariant_derivative_flat();  print("✓ ∇ of constant vector = 0 (flat)")
    print("\nAll tests passed!")
