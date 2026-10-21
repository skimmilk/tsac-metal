#pragma once

#include "models/types.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <unistd.h>

namespace tsac::test
{
// Ragged normalization columns, signed zero, and failures in distinct worker
// ranges. Reuse a private fixture filename; no supplied weight file is modified.
template <class F> void dac_preparation_fixtures(const ModelDirectory& models, F test)
{
    const auto require = [](bool condition, const char* message)
    {
        if (!condition)
        {
            throw Error(message);
        }
    };

    struct Fixture
    {
        std::string path =
            (std::filesystem::temp_directory_path() / "tsac-preparation-XXXXXX").string();

        Fixture()
        {
            if (!mkdtemp(path.data()))
            {
                throw Error("cannot create preparation fixture");
            }
        }

        ~Fixture()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } fixture;

    std::filesystem::path filename;
    for (const auto& file : std::filesystem::directory_iterator(models.path()))
    {
        if (file.path().filename().string().starts_with("tsac-mono-dac-analysis-source-"))
        {
            filename = file.path().filename();
            break;
        }
    }
    require(!filename.empty(), "missing preparation fixture filename");
    constexpr std::size_t fan_in = 64 * 7, columns = 257, count = fan_in * columns;
    const auto bytes = (count + 2 * columns) * sizeof(float);
    const auto mapped_bytes = (bytes + 16383) / 16384 * 16384;
    for (unsigned failure = 0; failure < 6; ++failure)
    {
        std::vector<float> payload(mapped_bytes / sizeof(float), 0.125f);
        payload[3] = -0.0f;
        if (failure == 1)
        {
            payload[fan_in * 64] = std::numeric_limits<float>::infinity();
        }
        else if (failure == 2)
        {
            payload[count + 128] = std::numeric_limits<float>::quiet_NaN();
        }
        else if (failure == 3)
        {
            std::fill_n(payload.data() + fan_in * 192, fan_in, 0.0f);
        }
        else if (failure == 4)
        {
            payload[fan_in * 256] = std::numeric_limits<float>::max();
        }
        // Every previous mapping is gone before replacing this fixture's file.
        {
            std::ofstream file(std::filesystem::path(fixture.path) / filename, std::ios::binary);
            file.write(reinterpret_cast<const char*>(payload.data()),
                       static_cast<std::streamsize>(mapped_bytes));
            require(bool(file), "cannot write preparation fixture");
        }
        const auto region = detail::ModelDirectoryAccess::map(
            ModelDirectory(fixture.path), ModelFile::mono_dac_analysis_source, 0, mapped_bytes);
        const ConvolutionWeights weights{
            EncodedTensorView(region, 0, count * sizeof(float), {{64, 7, columns}},
                              TensorEncoding::float32),
            TensorView<float>(region, count * sizeof(float), {{columns}}),
            TensorView<float>(region, (count + columns) * sizeof(float), {{columns}})};
        test(weights, failure == 0 || failure == 5);
    }
}
} // namespace tsac::test
