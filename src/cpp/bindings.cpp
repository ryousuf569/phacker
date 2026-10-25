#include <memory>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include "cscv.hpp"

namespace nb = nanobind;

// Non-float64 or non-C-contiguous input is copied into this layout by nanobind.
using Matrix = nb::ndarray<const double, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Vector = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

static std::pair<double, Vector> cscv(Matrix M, int S) {
    const int64_t T = M.shape(0), N = M.shape(1);
    std::unique_ptr<double[]> logits;
    size_t C;
    double pbo;
    {
        nb::gil_scoped_release release;
        const auto bs = phacker::block_stats(M.data(), T, N, S);  // validates S before we allocate
        C = size_t(phacker::n_combinations(S));
        logits.reset(new double[C]);
        pbo = phacker::cscv(bs, logits.get());
    }
    double* data = logits.release();
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return {pbo, Vector(data, {C}, owner)};
}

NB_MODULE(_core, m) {
    m.doc() = "C++ CSCV kernel: block sufficient statistics, Gray-code walk, OpenMP.";

    m.def("cscv", &cscv, nb::arg("M"), nb::arg("S") = 16,
          "Same contract as cscv.cscv with metric=sharpe: returns (pbo, logits),\n"
          "logits in itertools.combinations order.");
    m.def("n_combinations", &phacker::n_combinations, nb::arg("S"), "C(S, S/2).");
}
