#include <algorithm>
#include <bit>
#include <cmath>
#include <vector>

#include "cscv.hpp"

namespace phacker {

namespace {

constexpr int64_t CHUNK = 256;  // combinations per task; also bounds accumulator drift

int64_t binom(int n, int k) {
    if (k < 0 || k > n) return 0;
    int64_t c = 1;
    for (int i = 1; i <= k; ++i) c = c * (n - k + i) / i;
    return c;
}

// Revolving-door order (Kreher & Stinson, Alg. 2.12/2.13): consecutive subsets
// differ by one element in and one out. t[1..k] holds the subset of {1..S}
// sorted, t[k+1] = S+1 is a sentinel and t[0] is scratch.
void rd_unrank(int64_t r, int S, int k, int* t) {
    int x = S;
    for (int i = k; i >= 1; --i) {
        while (binom(x, i) > r) --x;
        t[i] = x + 1;
        r = binom(x + 1, i) - r - 1;
    }
    t[k + 1] = S + 1;
}

void rd_successor(int k, int* t) {
    int j = 1;
    while (j <= k && t[j] == j) ++j;
    if ((k - j) % 2 != 0) {
        if (j == 1) {
            --t[1];
        } else {
            t[j - 1] = j;
            if (j > 2) t[j - 2] = j - 1;
        }
    } else if (t[j + 1] != t[j] + 1) {
        t[j - 1] = t[j];
        ++t[j];
    } else {
        t[j + 1] = t[j];
        t[j] = j;
    }
}

uint32_t to_mask(const int* t, int k) {
    uint32_t m = 0;
    for (int i = 1; i <= k; ++i) m |= 1u << (t[i] - 1);
    return m;
}

// Lexicographic rank via the combinatorial number system, so logits land in the
// same order as the itertools.combinations loop in cscv.py.
int64_t lex_rank(uint32_t mask, int S, int k) {
    int64_t r = binom(S, k) - 1;
    for (int i = 0; mask; ++i, mask &= mask - 1)
        r -= binom(S - 1 - std::countr_zero(mask), k - i);
    return r;
}

// A flat half (a config that never trades there) scores 0, as in cscv.sharpe.
// b - a^2/n carries rounding of order eps * b, so flatness is judged against b.
inline double sharpe(double mu, double a, double b, double n) {
    const double ss = b - a * a / n;
    return ss <= 1e-12 * b ? 0.0 : (mu + a / n) / std::sqrt(ss / (n - 1));
}

}  // namespace

int64_t n_combinations(int S) { return binom(S, S / 2); }

// Each task unranks the first combination of its chunk, sums its blocks once,
// then Gray-walks the rest at one add and one subtract per column. The OOS half
// is the complement, so its sums come free from the totals.
double cscv(const BlockStats& bs, double* logits) {
    const int S = bs.S, k = S / 2;
    const int64_t N = bs.N;
    const int64_t total = binom(S, k);
    const double len = double(bs.rows * k);
    const double* A = bs.A.data();
    const double* B = bs.B.data();
    const double* mu = bs.shift.data();

    std::vector<double> At(N, 0.0), Bt(N, 0.0);
    for (int s = 0; s < S; ++s)
        for (int64_t n = 0; n < N; ++n) {
            At[n] += A[s * N + n];
            Bt[n] += B[s * N + n];
        }

    #pragma omp parallel
    {
        std::vector<double> a(N), b(N), is(N), oos(N);
        int t[34];

        #pragma omp for schedule(dynamic)
        for (int64_t r0 = 0; r0 < total; r0 += CHUNK) {
            rd_unrank(r0, S, k, t);
            uint32_t mask = to_mask(t, k);
            std::fill(a.begin(), a.end(), 0.0);
            std::fill(b.begin(), b.end(), 0.0);
            for (uint32_t m = mask; m; m &= m - 1) {
                const int64_t s = std::countr_zero(m);
                #pragma omp simd
                for (int64_t n = 0; n < N; ++n) {
                    a[n] += A[s * N + n];
                    b[n] += B[s * N + n];
                }
            }

            for (int64_t r = r0; r < std::min(total, r0 + CHUNK); ++r) {
                if (r > r0) {
                    rd_successor(k, t);
                    const uint32_t next = to_mask(t, k), diff = mask ^ next;
                    const int64_t in = std::countr_zero(next & diff), out = std::countr_zero(mask & diff);
                    #pragma omp simd
                    for (int64_t n = 0; n < N; ++n) {
                        a[n] += A[in * N + n] - A[out * N + n];
                        b[n] += B[in * N + n] - B[out * N + n];
                    }
                    mask = next;
                }

                #pragma omp simd
                for (int64_t n = 0; n < N; ++n) {
                    is[n] = sharpe(mu[n], a[n], b[n], len);
                    oos[n] = sharpe(mu[n], At[n] - a[n], Bt[n] - b[n], len);
                }
                int64_t best = 0;  // first max, as np.argmax
                for (int64_t n = 1; n < N; ++n)
                    if (is[n] > is[best]) best = n;
                const double target = oos[best];
                int64_t below = 0;
                #pragma omp simd reduction(+ : below)
                for (int64_t n = 0; n < N; ++n) below += oos[n] < target;

                // OOS rank is below + 1, so w / (1 - w) = (below + 1) / (N - below)
                logits[lex_rank(mask, S, k)] = std::log(double(below + 1) / double(N - below));
            }
        }
    }

    int64_t overfit = 0;
    for (int64_t c = 0; c < total; ++c) overfit += logits[c] <= 0;
    return double(overfit) / double(total);
}

}  // namespace phacker
