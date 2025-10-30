# scripts/test_cec2017.py
import importlib, numpy as np, sys, inspect

# --- import ---
ok = False
for name in ("cec2017", "cec2017.functions"):
    try:
        importlib.import_module(name)
        print(f"[OK] imported {name}")
        ok = True
    except Exception as e:
        print(f"[WARN] {name}: {e}")
if not ok:
    sys.exit(1)

import cec2017.functions as fns

D = 10  # CEC-2017 thường dùng 10/30/50. Chọn 10 cho smoke test.
X = np.zeros((1, D), dtype=float)  # <- quan trọng: 2D (batch_size, dim)

def eval_any(f):
    y = f(X)          # nhiều hàm yêu cầu 2D
    y = np.asarray(y, dtype=float).ravel()
    assert np.isfinite(y).all()
    return float(y[0])

# --- đường 1: get_function ---
getf = getattr(fns, "get_function", None)
if getf is not None:
    try:
        sig = inspect.signature(getf)
        if len(sig.parameters) >= 2:
            f = getf(1, D)        # (idx, dim)
        else:
            f = getf(1)           # (idx)
        v = eval_any(f)
        print(f"[OK] get_function -> f1({D}D) = {v:.6g}")
    except Exception as e:
        print(f"[WARN] get_function path failed: {e}")

# --- đường 2: gọi trực tiếp f1 ---
try:
    v = eval_any(fns.f1)
    print(f"[OK] direct f1({D}D) = {v:.6g}")
except Exception as e:
    print(f"[FAIL] direct f1 failed: {e}")
    sys.exit(2)

print("[PASS] cec2017 smoke test ✓")
