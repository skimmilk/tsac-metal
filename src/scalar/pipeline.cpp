#include "scalar/pipeline.hpp"

#include "common/analysis.tpp"
#include "common/cpu_transformer.tpp"
#include "common/packed_linear.tpp"
#include "common/rvq.tpp"
#include "common/synthesis.tpp"

template class tsac::DacAnalysis<tsac::scalar::Kernels>;
template class tsac::DacSynthesis<tsac::scalar::Kernels>;
template class tsac::CpuTransformer<tsac::scalar::Kernels>;
template class tsac::PackedLinear<tsac::scalar::Kernels>;
template class tsac::RvqEncoder<tsac::scalar::Kernels>;
