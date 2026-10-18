#pragma once
#include "common/common.hpp"

#include <functional>

namespace tsac
{
class ParallelExecutor
{
public:
    virtual ~ParallelExecutor() = default;
    virtual bool parallel() const = 0;
    virtual void parallel_for(std::size_t count,
                              const std::function<void(std::size_t)>& task) const = 0;
};
} // namespace tsac
