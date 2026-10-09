"""Checks of cscv.py and stats.py against the definitions in Bailey et al.,
"The Probability of Backtest Overfitting" (section numbers refer to the paper)."""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from src.cscv import cscv, sharpe
from src.stats import logistic_cdf, logistic_pdf, logit_fit, n_eff, sharpe_se


def traced(metric):
    """Wrap a metric so every call's input and output are recorded.
    cscv calls the metric on J then on J-bar, so calls alternate IS, OOS."""
    inputs, outputs = [], []

    def m(x):
        v = metric(x)
        inputs.append(x.copy())
        outputs.append(v)
        return v

    return m, inputs, outputs


# Algorithm 2.3, steps 2-4b: the partition


@pytest.mark.parametrize("S, expected", [(4, 6), (12, 924), (16, 12_870)])
def test_number_of_combinations(S, expected):
    # eq. (2.3). The paper prints 12,780 for S=16; C(16,8) is 12,870.
    assert comb(S, S // 2) == expected
    rng = np.random.default_rng(0)
    _, logits = cscv(rng.standard_normal((S * 4, 3)), S=S)
    assert len(logits) == expected


def test_figure_1_splits():
    # Figure 1: S=4 blocks A,B,C,D give exactly these six IS/OOS pairs, in this order
    T, S = 8, 4
    M = np.c_[np.arange(T, dtype=float), np.random.default_rng(1).standard_normal((T, 2))]
    metric, inputs, _ = traced(sharpe)
    cscv(M, S=S, metric=metric)

    letters = "ABCD"
    to_blocks = lambda x: "".join(letters[i] for i in sorted({int(t) // (T // S) for t in x[:, 0]}))
    got = [(to_blocks(inputs[2 * k]), to_blocks(inputs[2 * k + 1])) for k in range(len(inputs) // 2)]
    assert got == [("AB", "CD"), ("AC", "BD"), ("AD", "BC"), ("BC", "AD"), ("BD", "AC"), ("CD", "AB")]


def test_partition_structure():
    # steps 4a/4b: J is S/2 whole blocks in original order, J-bar its complement,
    # and every training set is reused as a testing set (end of section 2.2)
    T, S = 96, 8
    M = np.c_[np.arange(T, dtype=float), np.random.default_rng(2).standard_normal((T, 3))]
    metric, inputs, _ = traced(sharpe)
    cscv(M, S=S, metric=metric)

    is_sets, oos_sets = [], []
    for k in range(len(inputs) // 2):
        J, Jbar = inputs[2 * k][:, 0], inputs[2 * k + 1][:, 0]
        assert len(J) == len(Jbar) == T // 2
        assert np.all(np.diff(J) > 0) and np.all(np.diff(Jbar) > 0)
        assert np.array_equal(np.sort(np.r_[J, Jbar]), np.arange(T))
        assert len({int(t) // (T // S) for t in J}) == S // 2
        is_sets.append(frozenset(J.astype(int)))
        oos_sets.append(frozenset(Jbar.astype(int)))

    assert len(set(is_sets)) == comb(S, S // 2)
    assert set(is_sets) == set(oos_sets)


def test_trailing_rows_dropped():
    # rows beyond the last full block are discarded, so T need not divide by S
    M = np.random.default_rng(3).standard_normal((103, 5))
    pbo_a, lg_a = cscv(M, S=10)
    pbo_b, lg_b = cscv(M[:100], S=10)
    assert pbo_a == pbo_b and np.array_equal(lg_a, lg_b)


def test_odd_S_rejected():
    with pytest.raises(AssertionError):
        cscv(np.zeros((30, 2)), S=5)


# Algorithm 2.3, steps 4c-4g: ranks and logits


def test_logits_match_definition():
    # rebuild every logit from the recorded R^c, R-bar^c using steps e-g literally
    M = np.random.default_rng(4).standard_normal((120, 7))
    metric, _, out = traced(sharpe)
    pbo, logits = cscv(M, S=8, metric=metric)

    N = M.shape[1]
    expected = []
    for R, Rbar in zip(out[0::2], out[1::2]):
        n_star = np.argmax(R)
        rbar = 1 + np.sum(Rbar < Rbar[n_star])  # OOS rank, 1 = worst, N = best
        w = rbar / (N + 1)
        expected.append(np.log(w / (1 - w)))
    np.testing.assert_allclose(logits, expected, rtol=0, atol=1e-12)
    assert pbo == np.mean(np.array(expected) <= 0)


def test_section_2_1_worked_example():
    # N=3, R^c=(0.5, 1.1, 0.7) -> r^c=(1,3,2); R-bar^c=(0.6, 0.7, 1.3) -> r-bar^c=(1,2,3).
    # n* is strategy 2, OOS rank 2, so w = 2/4 and the logit is exactly 0.
    calls = iter([np.array([0.5, 1.1, 0.7]), np.array([0.6, 0.7, 1.3])] * 6)
    pbo, logits = cscv(np.zeros((8, 3)), S=4, metric=lambda x: next(calls))
    np.testing.assert_allclose(logits, 0.0, atol=1e-15)
    # section 3.1: only lambda > 0 counts as the IS choice beating the OOS median
    assert pbo == 1.0


def test_consistent_family_has_zero_pbo():
    # strategy n has drift n: the IS winner is always the OOS winner, rank N, logit ln(N)
    rng = np.random.default_rng(5)
    N = 6
    M = 0.1 * rng.standard_normal((160, N)) + np.arange(N)
    pbo, logits = cscv(M, S=8)
    assert pbo == 0.0
    np.testing.assert_allclose(logits, np.log(N))


def test_compensation_effect_gives_pbo_one():
    # section 3.2: if every strategy has the same full-sample mean, the IS mean and OOS
    # mean of a column sum to a constant, so the IS winner is always the OOS loser
    rng = np.random.default_rng(6)
    M = rng.standard_normal((160, 9))
    M -= M.mean(axis=0)
    pbo, logits = cscv(M, S=8, metric=lambda x: x.mean(axis=0))
    assert pbo == 1.0
    np.testing.assert_allclose(logits, np.log(1 / 9))


# invariances (spec section 4)


def test_invariant_to_column_permutation():
    rng = np.random.default_rng(7)
    M = rng.standard_normal((200, 12)) + 0.05 * rng.standard_normal(12)
    pbo, logits = cscv(M, S=10)
    pbo_p, logits_p = cscv(M[:, rng.permutation(12)], S=10)
    assert pbo == pbo_p
    np.testing.assert_allclose(logits, logits_p)


def test_invariant_to_positive_scaling():
    rng = np.random.default_rng(8)
    M = rng.standard_normal((200, 12)) + 0.05 * rng.standard_normal(12)
    pbo, logits = cscv(M, S=10)
    pbo_s, logits_s = cscv(3.7 * M, S=10)
    assert pbo == pbo_s
    np.testing.assert_allclose(logits, logits_s)


def test_duplicated_columns():
    # with even N, duplicating every column leaves PBO unchanged and N_eff unchanged
    # (N doubles, the spectrum's shape does not)
    rng = np.random.default_rng(9)
    M = rng.standard_normal((200, 10)) + 0.05 * rng.standard_normal(10)
    D = np.repeat(M, 2, axis=1)
    assert cscv(M, S=10)[0] == cscv(D, S=10)[0]
    assert n_eff(D) == pytest.approx(n_eff(M), rel=1e-10)


def test_noise_family_pbo_near_half():
    # section 3.1/4: with no skill the selected strategy's OOS rank is uniform, so
    # PBO ~ 0.5 and the mean logit ~ 0. A single family is noisy (the logits are
    # dependent), so average over independent families. N even so the median is not a rank.
    pbos, means = [], []
    for seed in range(40):
        M = np.random.default_rng(100 + seed).standard_normal((200, 20))
        pbo, logits = cscv(M, S=8)
        pbos.append(pbo)
        means.append(logits.mean())
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.08)
    assert np.mean(means) == pytest.approx(0.0, abs=0.3)


# stats.py


def test_sharpe_se_gaussian_case():
    # gamma3 = 0, gamma4 = 3 collapses to (1 + SR^2/2) / (T-1)
    rng = np.random.default_rng(10)
    T = 200_000
    x = 0.3 + rng.standard_normal((T, 1))
    sr = sharpe(x)[0]
    assert sharpe_se(x)[0] == pytest.approx(np.sqrt((1 + sr**2 / 2) / (T - 1)), rel=5e-3)


@pytest.mark.parametrize("dist", ["normal", "skewed"])
def test_sharpe_se_matches_sampling_distribution(dist):
    # the delta-method SE should match the spread of Sharpe across many samples
    rng = np.random.default_rng(11)
    T, reps = 500, 4000
    if dist == "normal":
        x = 0.1 + rng.standard_normal((T, reps))
    else:
        x = rng.exponential(1.0, (T, reps)) - 0.9  # skew 2, excess kurtosis 6
    empirical = sharpe(x).std(ddof=1)
    predicted = sharpe_se(x).mean()
    assert predicted == pytest.approx(empirical, rel=0.06)


def test_logistic_null():
    # w ~ U(0,1)  =>  logit(w) ~ standard logistic: density integrates to 1,
    # mean 0, variance pi^2/3, and the CDF inverts the logit
    x = np.linspace(-40, 40, 400_001)
    f = logistic_pdf(x)
    dx = x[1] - x[0]
    assert f.sum() * dx == pytest.approx(1.0, abs=1e-8)
    assert (x * f).sum() * dx == pytest.approx(0.0, abs=1e-8)
    assert (x**2 * f).sum() * dx == pytest.approx(np.pi**2 / 3, rel=1e-6)

    w = np.linspace(0.01, 0.99, 99)
    np.testing.assert_allclose(logistic_cdf(np.log(w / (1 - w))), w)


def test_logit_fit_on_exact_quantiles():
    # the i-th of n midpoint quantiles sits (i-0.5)/n up the CDF, so KS = 1/(2n)
    n = 500
    w = (np.arange(1, n + 1) - 0.5) / n
    ks, mean = logit_fit(np.log(w / (1 - w)))
    assert ks == pytest.approx(0.5 / n, rel=1e-9)
    assert mean == pytest.approx(0.0, abs=1e-12)


def test_logit_fit_detects_shift():
    w = (np.arange(1, 501) - 0.5) / 500
    ks, mean = logit_fit(np.log(w / (1 - w)) - 1.0)
    assert ks > 0.2 and mean == pytest.approx(-1.0)


def test_n_eff_counts_independent_directions():
    # k orthonormal centred columns, each repeated 12 times: the spectrum has k
    # equal eigenvalues, so the participation ratio is exactly k while N = 60
    rng = np.random.default_rng(12)
    k = 5
    A = rng.standard_normal((300, k))
    Q, _ = np.linalg.qr(A - A.mean(axis=0))
    assert n_eff(np.tile(Q, 12)) == pytest.approx(k, rel=1e-10)


def test_n_eff_bounds():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((500, 1))
    assert n_eff(np.repeat(x, 30, axis=1)) == pytest.approx(1.0)
    iid = n_eff(rng.standard_normal((5000, 30)))
    assert 25 < iid <= 30
