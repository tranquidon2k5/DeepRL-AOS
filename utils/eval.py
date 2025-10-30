# utils/eval.py
import numpy as np

def eval_pop_safe(f, X: np.ndarray) -> np.ndarray:
    """Đánh giá fitness cho quần thể X (N,D) an toàn với mọi biến thể CEC:
       - Ưu tiên gọi batch nếu f hỗ trợ (trả về N giá trị)
       - Fallback: gọi từng cá thể nếu batch không dùng được
    """
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2, "X must be (N, D)"
    try:
        y = f(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] == X.shape[0]:
            return y
    except Exception:
        pass
    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        out[i] = float(f(X[i]))
    return out
