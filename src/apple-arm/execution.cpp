#include "apple-arm/execution.hpp"

#include "common/internal.hpp"

// Framework umbrella header provides the BLAS declarations.
#include <Accelerate/Accelerate.h> // IWYU pragma: keep
#include <climits>

namespace tsac::apple_arm
{
void matrix_product(bool transposed, std::size_t m, std::size_t n, std::size_t k, const float* a,
                    const float* b, float* c, const Execution* execution)
{
    detail::require(m && n && k && m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                    "invalid BLAS matrix dimensions");
    const auto multiply = [&]
    {
        cblas_sgemm(CblasColMajor, transposed ? CblasTrans : CblasNoTrans, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1, a,
                    static_cast<int>(transposed ? k : m), b, static_cast<int>(k), 0, c,
                    static_cast<int>(m));
    };
    if (!execution || !execution->single_threaded_blas())
    {
        multiply();
        return;
    }
    // Parallelism belongs to the outer job pool. Force BLAS to one thread on
    // this calling thread, then restore its policy so nested GEMMs do not create
    // an additional internal pool beyond the configured codec budget.
    const auto previous = BLASGetThreading();
    detail::require(BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0,
                    "cannot configure single-threaded Accelerate");
    multiply();
    BLASSetThreading(previous);
}
} // namespace tsac::apple_arm
