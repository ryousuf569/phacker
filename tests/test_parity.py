"""The compiled kernel against the NumPy reference in cscv.py."""

import numpy as np
import pytest

import phacker
from phacker.cscv import cscv_reference as cscv

pytestmark = pytest.mark.skipif(phacker.backend() != "cpp", reason="C++ kernel not built")


def kernel(M, S):
    return phacker._core.cscv(M, S)


@pytest.mark.parametrize("T, N, S", [(103, 5, 10), (64, 1, 4), (400, 37, 8), (1000, 200, 12), (2000, 7, 2)])
def test_matches_reference(T, N, S):
    rng = np.random.default_rng(T + N + S)
    M = rng.standard_normal((T, N)) * rng.uniform(0.1, 3, N) + rng.normal(0, 0.05, N)
    pbo, logits = cscv(M, S=S)
    pbo_k, logits_k = kernel(M, S)
    assert pbo_k == pbo
    np.testing.assert_allclose(logits_k, logits, rtol=0, atol=1e-12)


def test_large_mean_small_spread():
    # where unshifted b - a^2/n cancels catastrophically
    M = 1e3 + 1e-4 * np.random.default_rng(0).standard_normal((800, 20))
    np.testing.assert_allclose(kernel(M, 16)[1], cscv(M, S=16)[1], rtol=0, atol=1e-12)


def test_noise_family_has_nontrivial_pbo():
    M = np.random.default_rng(1).standard_normal((400, 50))
    pbo, logits = cscv(M, S=8)
    assert 0 < pbo < 1
    np.testing.assert_allclose(kernel(M, 8)[1], logits, rtol=0, atol=1e-12)


def test_any_input_layout():
    M = np.random.default_rng(2).standard_normal((400, 30))
    pbo = kernel(M, 8)[0]
    assert kernel(np.asfortranarray(M), 8)[0] == pbo
    assert kernel(M[::2], 8)[0] == cscv(M[::2], S=8)[0]
    Mi = np.random.default_rng(3).integers(-5, 5, (400, 30))
    assert kernel(Mi, 8)[0] == cscv(Mi, S=8)[0]


@pytest.mark.parametrize("S", [5, 0, -4, 100])
def test_rejects_bad_S(S):
    with pytest.raises(ValueError):
        kernel(np.zeros((400, 3)), S)


def test_rejects_too_few_rows():
    with pytest.raises(ValueError):
        kernel(np.zeros((3, 4)), 4)


def test_flat_columns():
    # configs that never trade, or trade in only part of the sample, give flat halves
    # and exact Sharpe ties; both backends score a flat half 0 and rank ties low
    rng = np.random.default_rng(4)
    M = rng.standard_normal((480, 40)) * 0.01
    M[:, :10] = 0.0
    M[:240, 10:20] = 0.0
    M[::3, 20:25] = 0.0
    pbo, logits = cscv(M, S=8)
    assert np.isfinite(logits).all()
    pbo_k, logits_k = kernel(M, 8)
    assert pbo_k == pbo
    np.testing.assert_allclose(logits_k, logits, rtol=0, atol=1e-12)


def test_dispatch_uses_kernel():
    from phacker.cscv import cscv as dispatched
    M = np.random.default_rng(5).standard_normal((400, 30))
    np.testing.assert_array_equal(dispatched(M, S=8)[1], kernel(M, 8)[1])
