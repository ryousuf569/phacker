from itertools import combinations
import numpy as np


def sharpe(x):
    return x.mean(axis=0) / x.std(axis=0, ddof=1)

def cscv(M, S=16, metric=sharpe):
    """Probability of Backtest Overfitting via CSCV (Bailey et al., Algorithm 2.3).

    M: T x N array, column n = P&L series of trial n.
    Returns (pbo, logits).
    """
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    assert S % 2 == 0, "S must be even"

    blocks = np.array_split(np.arange(T - T % S), S)

    logits = []
    for c in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[i] for i in c])
        oos_rows = np.concatenate([blocks[i] for i in range(S) if i not in c])

        R = metric(M[is_rows])
        R_bar = metric(M[oos_rows])

        n_star = np.argmax(R)
        rank = np.argsort(np.argsort(R_bar))[n_star] + 1  # 1..N, higher = better
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))

    logits = np.array(logits)
    return (logits <= 0).mean(), logits