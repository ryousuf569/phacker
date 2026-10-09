import numpy as np


def sharpe_se(x):
    """Delta-method standard error of the Sharpe ratio, per column.

    var(SR) = (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1), g4 = non-excess kurtosis.
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    d = x - x.mean(axis=0)
    s = d.std(axis=0)
    sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
    g3 = (d**3).mean(axis=0) / s**3
    g4 = (d**4).mean(axis=0) / s**4
    return np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (T - 1))


def logistic_cdf(x):
    """Exact null CDF of the CSCV logit: omega ~ U(0,1) => logit(omega) ~ Logistic(0,1)."""
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def logistic_pdf(x):
    p = logistic_cdf(x)
    return p * (1 - p)


def logit_fit(logits):
    """KS distance from the logistic null and mean logit. Descriptive only:
    the CSCV logits are dependent, so no p-value is reported."""
    lam = np.sort(np.asarray(logits, dtype=float))
    n = len(lam)
    F = logistic_cdf(lam)
    i = np.arange(1, n + 1)
    ks = max((i / n - F).max(), (F - (i - 1) / n).max())
    return ks, lam.mean()


def n_eff(M):
    """Effective number of trials: participation ratio of the centred SVD spectrum."""
    M = np.asarray(M, dtype=float)
    s2 = np.linalg.svd(M - M.mean(axis=0), compute_uv=False) ** 2
    return s2.sum() ** 2 / (s2**2).sum()
