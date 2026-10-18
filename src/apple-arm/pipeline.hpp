#pragma once
#include "apple-arm/kernels.hpp"
#include "common/analysis.hpp"
#include "common/cpu_transformer.hpp"
#include "common/packed_linear.hpp"
#include "common/rvq.hpp"
#include "common/synthesis.hpp"

namespace tsac::apple_arm
{
using DacAnalysis = tsac::DacAnalysis<Kernels>;
using DacSynthesis = tsac::DacSynthesis<Kernels>;
using CpuTransformer = tsac::CpuTransformer<Kernels>;
using PackedLinear = tsac::PackedLinear<Kernels>;
using RvqEncoder = tsac::RvqEncoder<Kernels>;
} // namespace tsac::apple_arm
