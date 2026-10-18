#include "apple-arm/pipeline.hpp"

#include "common/analysis.tpp"
#include "common/cpu_transformer.tpp"
#include "common/packed_linear.tpp"
#include "common/rvq.tpp"
#include "common/synthesis.tpp"

template class tsac::DacAnalysis<tsac::apple_arm::Kernels>;
template class tsac::DacSynthesis<tsac::apple_arm::Kernels>;
template class tsac::CpuTransformer<tsac::apple_arm::Kernels>;
template class tsac::PackedLinear<tsac::apple_arm::Kernels>;
template class tsac::RvqEncoder<tsac::apple_arm::Kernels>;
