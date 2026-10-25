from itertools import combinations
import numpy as np

try:
    from ._core import cscv as _cscv_cpp
except ImportError:
    _cscv_cpp = None


def sharpe(x):
    # an all-zero column (a config that never trades in this window) scores 0, not NaN
    flat = ~x.any(axis=0)
    m, v = x.mean(axis=0), x.var(axis=0, ddof=1)
    return np.where(flat, 0.0, m / np.sqrt(np.where(flat, 1.0, v)))


def cscv(M, S=16, metric=sharpe):
    """Probability of Backtest Overfitting via CSCV (Bailey et al., Algorithm 2.3).

    M: T x N array, column n = P&L series of trial n.
    Returns (pbo, logits). Runs the C++ kernel when it is built and metric is sharpe.
    """
    if metric is sharpe and _cscv_cpp is not None:
        return _cscv_cpp(np.asarray(M, dtype=float), S)
    return cscv_reference(M, S, metric)


def cscv_reference(M, S=16, metric=sharpe):
    """NumPy implementation of cscv, for any metric. The C++ kernel is tested against it."""
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if S % 2 != 0:
        raise ValueError("S must be even")

    blocks = np.array_split(np.arange(T - T % S), S)

    logits = []
    for c in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[i] for i in c])
        oos_rows = np.concatenate([blocks[i] for i in range(S) if i not in c])

        R = metric(M[is_rows])
        R_bar = metric(M[oos_rows])

        n_star = np.argmax(R)
        rank = 1 + np.sum(R_bar < R_bar[n_star])  # 1..N, higher = better; ties rank low
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))

    logits = np.array(logits)
    return (logits <= 0).mean(), logits
