"""Section 6 of Bailey et al.: the seasonal-strategy experiment, as a labelled
ground-truth check on cscv.
"""

import numpy as np
import pandas as pd
import pytest

from src.cscv import cscv, sharpe

T = 1000
ENTRY_DAYS = range(1, 23)
HOLDING = range(1, 21)
SIDES = (1, -1)
SEEDS = range(8)
S = 10


def month_bounds(T):
    days = pd.bdate_range("2000-01-03", periods=T)
    key = days.year * 12 + days.month
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return starts, np.r_[starts[1:], T]


def daily_returns(seed, effect):
    r = np.random.default_rng(seed).standard_normal(T)
    for s in month_bounds(T)[0]:
        r[s : s + 5] += effect
    return r


def seasonal_family(r):
    """T x N P&L matrix: enter at the close of business day `entry` of each month,
    hold `hold` days, long or short."""
    starts, ends = month_bounds(len(r))
    cols = []
    for e in ENTRY_DAYS:
        for h in HOLDING:
            pos = np.zeros(len(r))
            for s, end in zip(starts, ends):
                if s + e - 1 < end:  # month has an entry day e
                    pos[s + e : s + e + h] = 1.0
            cols.extend(side * pos * r for side in SIDES)
    return np.column_stack(cols)


def run(seed, effect):
    M = seasonal_family(daily_returns(seed, effect))
    out = []

    def metric(x):
        v = sharpe(x)
        out.append(v)
        return v

    pbo, logits = cscv(M, S=S, metric=metric)
    R, Rbar = np.array(out[0::2]), np.array(out[1::2])
    n_star = R.argmax(axis=1)
    rows = np.arange(len(n_star))
    is_best, oos_best = R[rows, n_star], Rbar[rows, n_star]
    beta = np.polyfit(is_best, oos_best, 1)[0]  # section 3.2 degradation slope
    return dict(
        pbo=pbo,
        logits=logits,
        best_sr=sharpe(M).max() * np.sqrt(252),
        prob_loss=np.mean(oos_best < 0),
        beta=beta,
        oos_selected=oos_best,
        oos_all=Rbar.ravel(),
    )


@pytest.fixture(scope="module")
def negative():
    return [run(seed, 0.0) for seed in SEEDS]


@pytest.fixture(scope="module")
def positive():
    return [run(seed, 0.25) for seed in SEEDS]


def avg(runs, key):
    return np.mean([r[key] for r in runs])


def test_family_shape():
    M = seasonal_family(daily_returns(0, 0.0))
    assert M.shape == (T, len(ENTRY_DAYS) * len(HOLDING) * len(SIDES))
    # long and short columns are exact mirrors
    np.testing.assert_array_equal(M[:, 0], -M[:, 1])


def test_planted_effect_is_found_in_sample():
    # with the effect, the full-sample winner should be a long trade over the
    # first days of the month (paper: entry day 1, hold 4, long)
    M = seasonal_family(daily_returns(0, 0.25))
    best = sharpe(M).argmax()
    e_idx, rest = divmod(best, len(HOLDING) * len(SIDES))
    side = SIDES[rest % len(SIDES)]
    assert ENTRY_DAYS[e_idx] <= 2 and side == 1


def test_noise_backtests_look_good_in_sample(negative):
    # selection over 880 configs manufactures a high Sharpe from a random walk
    assert avg(negative, "best_sr") > 1.0


def test_negative_control_is_overfit(negative):
    assert 0.3 < avg(negative, "pbo") < 0.7
    assert avg(negative, "prob_loss") > 0.3
    assert avg(negative, "beta") < 0


def test_positive_control_is_not_overfit(positive):
    assert avg(positive, "pbo") < 0.3
    assert avg(positive, "prob_loss") < 0.3


def test_cscv_separates_the_controls(negative, positive):
    assert avg(positive, "pbo") < avg(negative, "pbo") - 0.15
    assert np.mean([r["logits"].mean() for r in positive]) > np.mean([r["logits"].mean() for r in negative])


def test_stochastic_dominance(negative, positive):
    # section 3.3: with a real effect, the selected strategy's OOS Sharpe dominates
    # the OOS Sharpe of a strategy picked at random; on noise it does not
    def sd2_gap(run):
        grid = np.linspace(-3, 3, 241)
        F_sel = np.searchsorted(np.sort(run["oos_selected"]), grid, side="right") / len(run["oos_selected"])
        F_all = np.searchsorted(np.sort(run["oos_all"]), grid, side="right") / len(run["oos_all"])
        # second-order: integral of (F_sel - F_all) <= 0 everywhere means dominance
        return np.cumsum(F_sel - F_all).max() * (grid[1] - grid[0])

    pos_gaps = [sd2_gap(r) for r in positive]
    neg_gaps = [sd2_gap(r) for r in negative]
    assert np.median(pos_gaps) < np.median(neg_gaps)
    assert np.mean([g <= 1e-9 for g in pos_gaps]) >= 0.5
