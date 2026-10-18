#pragma once
#include "common/execution.hpp"

#include <memory>

namespace tsac::apple_arm
{
unsigned physical_threads();

// A recording's reusable workers borrow the analyzer's immutable thread budget.
class Execution final : public ParallelExecutor
{
public:
    explicit Execution(const unsigned& threads);
    Execution(unsigned&&) = delete;
    ~Execution();
    bool parallel() const override;
    bool single_threaded_blas() const;
    void parallel_for(std::size_t count,
                      const std::function<void(std::size_t)>& task) const override;

private:
    class Impl;
    const unsigned& threads_;
    std::unique_ptr<Impl> impl_;
};

// Column-major A(m,k) or transpose(A(k,m)), B(k,n), C(m,n).
void matrix_product(bool transpose_a, std::size_t m, std::size_t n, std::size_t k, const float* a,
                    const float* b, float* c, const Execution* execution = nullptr);
} // namespace tsac::apple_arm
