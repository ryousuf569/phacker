#pragma once

#include <cstdint>
#include <vector>

namespace phacker {

// Per-block sufficient statistics of a T x N P&L matrix. A and B are S x N,
// row-major, so the accumulator loop over n is unit-stride.
struct BlockStats {
    int S = 0;
    int64_t N = 0;
    int64_t rows = 0;            // rows per block; the trailing T % S rows are dropped
    std::vector<double> shift;   // column means
    std::vector<double> A;       // A[s*N + n] = sum over block s of (r - shift[n])
    std::vector<double> B;       // B[s*N + n] = sum over block s of (r - shift[n])^2
};

int64_t n_combinations(int S);

// M is row-major T x N (a C-contiguous numpy array).
BlockStats block_stats(const double* M, int64_t T, int64_t N, int S);

// Fills logits[n_combinations(S)] in itertools.combinations order and returns PBO.
double cscv(const BlockStats& bs, double* logits);

}  // namespace phacker
