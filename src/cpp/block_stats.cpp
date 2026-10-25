#include <stdexcept>

#include "cscv.hpp"

namespace phacker {

// Sums are shifted by the column mean first: b - a^2/n cancels catastrophically
// when the mean is large next to the spread, and variance is shift-invariant.
// Welford would also fix that, but its state is not additive across blocks.
BlockStats block_stats(const double* M, int64_t T, int64_t N, int S) {
    if (S < 2 || S > 32 || S % 2 != 0) throw std::invalid_argument("S must be even, 2..32");
    if (N < 1) throw std::invalid_argument("need at least one column");
    if ((T / S) * (S / 2) < 2) throw std::invalid_argument("each half needs at least 2 rows");

    BlockStats bs;
    bs.S = S;
    bs.N = N;
    bs.rows = T / S;
    bs.shift.assign(N, 0.0);
    bs.A.assign(S * N, 0.0);
    bs.B.assign(S * N, 0.0);
    const int64_t L = bs.rows;

    #pragma omp parallel for
    for (int s = 0; s < S; ++s) {
        double* a = &bs.A[s * N];
        for (int64_t t = s * L; t < (s + 1) * L; ++t) {
            const double* r = M + t * N;
            #pragma omp simd
            for (int64_t n = 0; n < N; ++n) a[n] += r[n];
        }
    }

    for (int s = 0; s < S; ++s)
        for (int64_t n = 0; n < N; ++n) bs.shift[n] += bs.A[s * N + n];
    for (int64_t n = 0; n < N; ++n) bs.shift[n] /= double(S * L);

    const double* mu = bs.shift.data();
    #pragma omp parallel for
    for (int s = 0; s < S; ++s) {
        double* a = &bs.A[s * N];
        double* b = &bs.B[s * N];
        for (int64_t n = 0; n < N; ++n) a[n] = 0.0;
        for (int64_t t = s * L; t < (s + 1) * L; ++t) {
            const double* r = M + t * N;
            #pragma omp simd
            for (int64_t n = 0; n < N; ++n) {
                const double d = r[n] - mu[n];
                a[n] += d;
                b[n] += d * d;
            }
        }
    }
    return bs;
}

}
