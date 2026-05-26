"""
Tests for the Lean 4 bridge (translator, theorem_builder, mathlib_linker).
"""
import sympy as sp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from levipy.lean4.translator import SymPyToLean4
from levipy.lean4.theorem_builder import TheoremBuilder
from levipy.lean4.mathlib_linker import MathlibLinker
from levipy import Manifold


# ──────────────────────────────────────────────────────────────────────
# Translator tests
# ──────────────────────────────────────────────────────────────────────

def test_translate_zero():
    t = SymPyToLean4()
    assert t.translate(sp.Integer(0)) == "0"

def test_translate_one():
    t = SymPyToLean4()
    assert t.translate(sp.Integer(1)) == "1"

def test_translate_integer():
    t = SymPyToLean4()
    result = t.translate(sp.Integer(5))
    assert "5" in result

def test_translate_rational():
    t = SymPyToLean4()
    result = t.translate(sp.Rational(1, 2))
    assert "1" in result and "2" in result

def test_translate_symbol():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    assert t.translate(x) == "x"

def test_translate_sin():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    result = t.translate(sp.sin(x))
    assert "Real.sin" in result
    assert "x" in result

def test_translate_cos():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    result = t.translate(sp.cos(x))
    assert "Real.cos" in result

def test_translate_power():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    result = t.translate(x**2)
    assert "^" in result

def test_translate_inverse():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    result = t.translate(1 / x)
    assert "1 / x" in result

def test_translate_sqrt():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    result = t.translate(sp.sqrt(x))
    assert "Real.sqrt" in result

def test_suggest_tactic_zero():
    t = SymPyToLean4()
    assert t.suggest_tactic(sp.Integer(0)) == "ring"

def test_suggest_tactic_number():
    t = SymPyToLean4()
    assert t.suggest_tactic(sp.Rational(1, 3)) == "norm_num"

def test_suggest_tactic_trig():
    t = SymPyToLean4()
    x = sp.Symbol('x')
    result = t.suggest_tactic(sp.sin(x)**2 + sp.cos(x)**2 - 1)
    assert "simp" in result or "ring" in result

# ──────────────────────────────────────────────────────────────────────
# TheoremBuilder tests
# ──────────────────────────────────────────────────────────────────────

def _s2_builder():
    th, ph = sp.symbols('theta phi', real=True)
    M = Manifold('S2', [th, ph])
    g = M.metric([[1, 0], [0, sp.sin(th)**2]])
    return TheoremBuilder(g, manifold_name='Sphere2',
                          extra_hypotheses=['(hth : 0 < theta)'])

def test_emit_returns_string():
    lean_src = _s2_builder().emit()
    assert isinstance(lean_src, str)
    assert len(lean_src) > 100

def test_emit_has_imports():
    lean_src = _s2_builder().emit()
    assert "import Mathlib" in lean_src

def test_emit_has_variable_decls():
    lean_src = _s2_builder().emit()
    assert "variable (theta : ℝ)" in lean_src
    assert "variable (phi : ℝ)" in lean_src

def test_emit_has_christoffel_theorems():
    lean_src = _s2_builder().emit()
    assert "christoffel" in lean_src
    assert "theorem" in lean_src

def test_emit_has_ricci_scalar():
    lean_src = _s2_builder().emit()
    assert "ricci_scalar" in lean_src

def test_emit_has_sorry():
    lean_src = _s2_builder().emit()
    assert "sorry" in lean_src

def test_emit_flat_manifold():
    x, y = sp.symbols('x y')
    M = Manifold('R2', [x, y])
    g = M.metric([[1, 0], [0, 1]])
    builder = TheoremBuilder(g, manifold_name='FlatR2')
    lean_src = builder.emit()
    assert "is_flat" in lean_src or "vanish" in lean_src.lower()

def test_save_creates_file(tmp_path):
    path = str(tmp_path / "Sphere2.lean")
    _s2_builder().save(path)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "import Mathlib" in content

# ──────────────────────────────────────────────────────────────────────
# MathlibLinker tests
# ──────────────────────────────────────────────────────────────────────

def test_linker_rational():
    linker = MathlibLinker()
    suggestions = linker.suggest(sp.Rational(1, 2))
    names = [s[0] for s in suggestions]
    assert "norm_num" in names

def test_linker_trig():
    th = sp.Symbol('theta')
    linker = MathlibLinker()
    # sin(th)**2 alone is not zero, tactic_chain should suggest trig path
    result = linker.tactic_chain(sp.sin(th)**2)
    assert isinstance(result, str) and len(result) > 0

def test_linker_tactic_chain_zero():
    linker = MathlibLinker()
    assert linker.tactic_chain(sp.Integer(0)) == "rfl"

def test_linker_tactic_chain_number():
    linker = MathlibLinker()
    result = linker.tactic_chain(sp.Rational(3, 7))
    assert result == "norm_num"

def test_linker_tactic_chain_division():
    x = sp.Symbol('x')
    linker = MathlibLinker()
    result = linker.tactic_chain(1/x)
    assert "field_simp" in result


if __name__ == "__main__":
    # Translator
    test_translate_zero();      print("✓ translate 0")
    test_translate_one();       print("✓ translate 1")
    test_translate_integer();   print("✓ translate integer")
    test_translate_rational();  print("✓ translate rational")
    test_translate_symbol();    print("✓ translate symbol")
    test_translate_sin();       print("✓ translate sin")
    test_translate_cos();       print("✓ translate cos")
    test_translate_power();     print("✓ translate power")
    test_translate_inverse();   print("✓ translate inverse")
    test_translate_sqrt();      print("✓ translate sqrt")
    test_suggest_tactic_zero();   print("✓ suggest tactic: zero → ring")
    test_suggest_tactic_number(); print("✓ suggest tactic: number → norm_num")
    test_suggest_tactic_trig();   print("✓ suggest tactic: trig")
    # Builder
    test_emit_returns_string();      print("✓ emit returns string")
    test_emit_has_imports();         print("✓ emit has Mathlib imports")
    test_emit_has_variable_decls();  print("✓ emit has variable decls")
    test_emit_has_christoffel_theorems(); print("✓ emit has Christoffel theorems")
    test_emit_has_ricci_scalar();    print("✓ emit has Ricci scalar theorem")
    test_emit_has_sorry();           print("✓ emit has sorry skeletons")
    test_emit_flat_manifold();       print("✓ flat manifold emits flatness theorem")
    # Linker
    test_linker_rational();      print("✓ linker: rational → norm_num")
    test_linker_trig();          print("✓ linker: trig suggestions")
    test_linker_tactic_chain_zero();   print("✓ tactic chain: 0 → rfl")
    test_linker_tactic_chain_number(); print("✓ tactic chain: number → norm_num")
    test_linker_tactic_chain_division(); print("✓ tactic chain: division → field_simp")
    print("\nAll Lean 4 bridge tests passed!")
