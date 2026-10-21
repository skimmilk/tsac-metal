#include "common/dsp.hpp"
#include "dac_preparation_fixture.hpp"
#include "models/dac_analysis_model.hpp"
#include "models/dac_synthesis_model.hpp"
#include "scalar/pipeline.hpp"
#include "weights/preparation.hpp"

#if TSAC_ENABLE_APPLE_ARM
#include "apple-arm/pipeline.hpp"
#endif

#include <array>
#include <bit>
#include <chrono>
#include <iostream>
#include <sys/resource.h>

using namespace tsac;

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

bool same(float a, float b)
{
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

// Inspect the common source-layout preparation without exposing private model
// weights in the production convolution interface or executing scalar inference.
struct ColumnProbe : Convolution<ColumnProbe>
{
    std::span<const float> values() const
    {
        return weights_;
    }
};

void columns(const ConvolutionWeights& c, const ParallelExecutor* execution, bool transposed,
             std::size_t stride = 1, std::size_t dilation = 1)
{
    const auto axes = c.values.shape().axes;
    const auto expected =
        fold_dac_weight_norm(weights::expand(c.values), c.gain.values(), axes[0] * axes[1]);
    const auto prepared = ColumnProbe::from_weights(c, {}, stride, dilation, transposed, execution);
    check(std::equal(expected.begin(), expected.end(), prepared.values().begin(), same),
          "CPU DAC preparation changed weight bits");
    Limits bounded;
    bounded.tensor_elements = expected.size() - 1;
    bool rejected = false;
    try
    {
        ColumnProbe::from_weights(c, bounded, stride, dilation, transposed, execution);
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, "CPU DAC preparation ignored the tensor limit");
}

template <class Model>
void model_columns(const Model& model, bool synthesis, const ParallelExecutor* execution)
{
    columns(model.entry_convolution(), execution, false);
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        const auto stride = synthesis ? std::array<unsigned, 4>{8, 8, 4, 2}[stage]
                                      : std::array<unsigned, 4>{2, 4, 8, 8}[stage];
        columns(model.resample_convolution(stage), execution, synthesis, stride);
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            columns(model.residual_convolution(stage, unit, ResidualConvolution::dilated),
                    execution, false, 1, std::array<unsigned, 3>{1, 3, 9}[unit]);
            columns(model.residual_convolution(stage, unit, ResidualConvolution::pointwise),
                    execution, false);
        }
    }
    columns(model.exit_convolution(), execution, false);
}

#if TSAC_ENABLE_APPLE_ARM
void phase_matrix(const ConvolutionWeights& c, std::size_t stride, std::size_t dilation,
                  const apple_arm::Execution& execution)
{
    const auto axes = c.values.shape().axes;
    const Conv1dConfig config{axes[2],
                              axes[0],
                              axes[1],
                              stride,
                              stride == 1 ? (axes[1] - 1) * dilation / 2 : (stride + 1) / 2,
                              dilation,
                              ConvolutionKind::transposed};
    const auto folded =
        fold_dac_weight_norm(weights::expand(c.values), c.gain.values(), axes[0] * axes[1]);
    apple_arm::Conv1d reference(config, folded, c.bias.values());
    reference.set_execution(&execution);
    auto actual = apple_arm::Kernels{&execution}.convolution(c, {}, stride, dilation, true);
    const SignalShape shape{config.input_channels, 3, 2};
    std::vector<float> input(shape.elements());
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = i % 19 == 0 ? -0.0f : float(int((i * 73 + i / 7) % 257) - 128) / 32;
    }
    const auto expected = reference.apply(input, shape);
    const auto output = actual.apply(input, shape);
    check(std::equal(expected.values.begin(), expected.values.end(), output.values.begin(), same),
          "direct CPU phase packing changed convolution bits");
}

void phase_matrices(const DacSynthesisModel& model)
{
    for (unsigned budget : {1u, 4u})
    {
        apple_arm::Execution execution(budget);
        for (unsigned stage = 0; stage < 4; ++stage)
        {
            phase_matrix(model.resample_convolution(stage),
                         std::array<unsigned, 4>{8, 8, 4, 2}[stage], 1, execution);
        }
    }
}

void preparation_budgets(const ConvolutionWeights& weights)
{
    struct Executor : ParallelExecutor
    {
        bool enabled;
        mutable std::size_t jobs = 0;

        explicit Executor(bool enabled) : enabled(enabled)
        {
        }

        bool parallel() const override
        {
            return enabled;
        }

        void parallel_for(std::size_t count,
                          const std::function<void(std::size_t)>& task) const override
        {
            jobs += count;
            for (std::size_t i = 0; i < count; ++i)
            {
                task(i);
            }
        }
    } serial(false), parallel(true);

    ColumnProbe::from_weights(weights, {}, 1, 1, false, &serial);
    check(serial.jobs == 0, "serial preparation submitted parallel work");
    ColumnProbe::from_weights(weights, {}, 1, 1, false, &parallel);
    check(parallel.jobs > 1, "parallel preparation did not split columns");
}
#endif

void preparation_failures(const ModelDirectory& directory, const ParallelExecutor* execution)
{
    test::dac_preparation_fixtures(
        directory,
        [&](const ConvolutionWeights& c, bool valid)
        {
            const auto bias = weights::View(c.bias);
            const ConvolutionWeights transposed{
                c.values, c.gain,
                TensorView<float>(c.bias.region(), bias.offset, {{c.values.shape().axes[0]}})};
            if (valid)
            {
                columns(c, execution, false);
                columns(transposed, execution, true, 4, 2);
#if TSAC_ENABLE_APPLE_ARM
                phase_matrix(transposed, 4, 2,
                             *static_cast<const apple_arm::Execution*>(execution));
#endif
                return;
            }
            const auto rejects = [&](auto prepare)
            {
                bool rejected = false;
                try
                {
                    prepare();
                }
                catch (const Error&)
                {
                    rejected = true;
                }
                check(rejected, "CPU preparation lost a parallel normalization failure");
            };
            rejects(
                [&]
                {
                    ColumnProbe::from_weights(c, {}, 1, 1, false, execution);
                });
#if TSAC_ENABLE_APPLE_ARM
            rejects(
                [&]
                {
                    apple_arm::Conv1d::from_weights(transposed, {}, 4, 2, true, execution);
                });
#endif
        });
}

void benchmark(const ModelDirectory& directory, const std::string& role, int trials,
               unsigned threads)
{
    check(trials > 0 && threads <= 256, "invalid preparation benchmark budget");
#if TSAC_ENABLE_APPLE_ARM
    apple_arm::Execution execution(threads);
#endif
    for (int trial = 0; trial < trials; ++trial)
    {
        const auto start = std::chrono::steady_clock::now();
#if TSAC_ENABLE_APPLE_ARM
        if (role == "analysis")
        {
            auto model = apple_arm::DacAnalysis::from_model(
                DacAnalysisModel(directory, ModelVariant::stereo), {}, {&execution});
        }
        else if (role == "synthesis")
        {
            auto model = apple_arm::DacSynthesis::from_model(
                DacSynthesisModel(directory, ModelVariant::stereo), {}, {&execution});
        }
        else
#endif
            if (role == "scalar-analysis")
        {
            auto model =
                scalar::DacAnalysis::from_model(DacAnalysisModel(directory, ModelVariant::stereo));
        }
        else if (role == "scalar-synthesis")
        {
            auto model = scalar::DacSynthesis::from_model(
                DacSynthesisModel(directory, ModelVariant::stereo));
        }
        else
        {
            throw Error("unknown preparation benchmark role");
        }
        const auto seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << role << " trial=" << trial + 1 << " threads=" << threads
                  << " seconds=" << seconds << " peak_bytes=" << usage.ru_maxrss << std::endl;
    }
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        check(argc == 2 || argc == 5, "usage: cpu-preparation MODELS [ROLE TRIALS THREADS]");
        const ModelDirectory directory(argv[1]);
        if (argc == 5)
        {
            benchmark(directory, argv[2], std::stoi(argv[3]),
                      static_cast<unsigned>(std::stoul(argv[4])));
            return 0;
        }
        const ParallelExecutor* executor = nullptr;
#if TSAC_ENABLE_APPLE_ARM
        const unsigned budget = 4;
        apple_arm::Execution execution(budget);
        executor = &execution;
#endif
        for (auto variant : {ModelVariant::mono, ModelVariant::stereo})
        {
            model_columns(DacAnalysisModel(directory, variant), false, executor);
            const DacSynthesisModel synthesis(directory, variant);
            model_columns(synthesis, true, executor);
#if TSAC_ENABLE_APPLE_ARM
            phase_matrices(synthesis);
#endif
        }
        preparation_failures(directory, executor);
#if TSAC_ENABLE_APPLE_ARM
        preparation_budgets(DacAnalysisModel(directory, ModelVariant::stereo).exit_convolution());
#endif
        std::cout << "CPU DAC preparation exact checks passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
