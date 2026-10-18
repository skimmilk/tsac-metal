#include "audio.hpp"
#include "cli.hpp"
#include "format/container.hpp"
#include "format/range.hpp"
#include "weights/source_model.hpp"

#include <array>
#include <iostream>
#include <limits>

namespace
{
template <class F> void rejects(F&& call)
{
    try
    {
        call();
    }
    catch (const tsac::Error&)
    {
        return;
    }
    throw tsac::Error("invalid input was accepted");
}

void word(tsac::Bytes& bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
    {
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void containers()
{
    const tsac::Bytes fast{'F', 'B', 'A', 'Z', 0, 0, 1, 1, 0xf6, 0xc5, 0xa0};
    for (std::size_t size = 0; size < fast.size(); ++size)
    {
        if (size != 8)
        {
            rejects(
                [&]
                {
                    tsac::parse_header(tsac::ByteView(fast).first(size));
                });
        }
    }
    for (auto [offset, value] : std::array<std::pair<std::size_t, std::uint8_t>, 7>{
             {{0, 0}, {4, 1}, {5, 2}, {6, 0}, {7, 0}, {7, 10}, {10, 0xa1}}})
    {
        auto bad = fast;
        bad[offset] = value;
        rejects(
            [&]
            {
                tsac::parse_header(bad);
            });
    }
    auto limits = tsac::Limits{};
    limits.symbols = 1;
    rejects(
        [&]
        {
            tsac::unpack_fast(fast, limits);
        });
    limits = {};
    limits.pcm_bytes = 2047;
    rejects(
        [&]
        {
            tsac::parse_header(fast, limits);
        });
    const auto header = tsac::parse_header(fast);
    const std::array<std::uint16_t, 2> invalid{1024, 0};
    rejects(
        [&]
        {
            tsac::pack_fast(header, invalid);
        });
    rejects(
        [&]
        {
            tsac::pack_fast(header, std::span(invalid).first(1));
        });

    tsac::Bytes normal{'F', 'B', 'A', 'Z', 0, 1, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0x89, 0x5c, 0xe0, 0x95};
    for (std::size_t size = 0; size < normal.size(); ++size)
    {
        rejects(
            [&]
            {
                tsac::parse_header(tsac::ByteView(normal).first(size));
            });
    }
    normal.back() ^= 1;
    rejects(
        [&]
        {
            tsac::verify_token_crc(tsac::parse_header(normal), tsac::unpack_fast(fast));
        });
    normal[13] = 0;
    rejects(
        [&]
        {
            tsac::parse_header(normal);
        });
    rejects(
        []
        {
            tsac::checked_mul(std::numeric_limits<std::size_t>::max(), 2);
        });
    rejects(
        []
        {
            tsac::checked_add(std::numeric_limits<std::size_t>::max(), 1);
        });
}

void models()
{
    tsac::Bytes bytes;
    word(bytes, 0x23f4aefb);
    word(bytes, 2);
    bytes.insert(bytes.end(), {'{', '}'});
    const auto record = bytes.size();
    word(bytes, 0x23f4aefa);
    word(bytes, static_cast<std::uint32_t>(tsac::packing::Storage::f32));
    word(bytes, 1);
    word(bytes, 1);
    word(bytes, 1);
    bytes.push_back('f');
    word(bytes, 0x3f800000);
    for (std::size_t size = 0; size < bytes.size(); ++size)
    {
        if (size != record)
        {
            rejects(
                [&]
                {
                    tsac::packing::SourceModel::parse(tsac::Bytes(
                        bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(size)));
                });
        }
    }
    const auto model = tsac::packing::SourceModel::parse(bytes);
    rejects(
        [&]
        {
            model.dac_config();
        });
    rejects(
        [&]
        {
            model.transformer_config();
        });
    rejects(
        [&]
        {
            model.dequantize("absent");
        });
    for (auto [offset, value] : std::array<std::pair<std::size_t, std::uint32_t>, 5>{
             {{4, 0xffffffff}, {14, 0x10000}, {18, 9}, {26, 0}, {26, 0xffffffff}}})
    {
        auto bad = bytes;
        for (unsigned shift = 0; shift < 32; shift += 8)
        {
            bad[offset + shift / 8] = static_cast<std::uint8_t>(value >> shift);
        }
        rejects(
            [&]
            {
                tsac::packing::SourceModel::parse(bad);
            });
    }
    auto bad = bytes;
    bad.push_back(0);
    rejects(
        [&]
        {
            tsac::packing::SourceModel::parse(bad);
        });
    bad = bytes;
    bad.insert(bad.end(), bytes.begin() + static_cast<std::ptrdiff_t>(record), bytes.end());
    rejects(
        [&]
        {
            tsac::packing::SourceModel::parse(bad);
        });
    auto limits = tsac::Limits{};
    limits.file_bytes = bytes.size() - 1;
    rejects(
        [&]
        {
            tsac::packing::SourceModel::parse(bytes, limits);
        });
    limits = {};
    limits.metadata_bytes = 1;
    rejects(
        [&]
        {
            tsac::packing::SourceModel::parse(bytes, limits);
        });
}

void audio_and_range()
{
    const std::vector<float> samples(8, 0);
    rejects(
        [&]
        {
            tsac::pcm16_wave(samples, {2, 3, 2});
        });
    auto limits = tsac::Limits{};
    limits.pcm_bytes = 15;
    rejects(
        [&]
        {
            tsac::pcm16_wave(samples, {2, 2, 2}, limits);
        });
    for (const auto value :
         {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    {
        const std::array<float, 1> bad{value};
        rejects(
            [&]
            {
                tsac::pcm16_wave(bad, {1, 1, 1});
            });
    }
    rejects(
        []
        {
            tsac::pcm16_input({}, 0);
        });
    rejects(
        []
        {
            tsac::pcm16_input({}, 128);
        });
    tsac::RangeEncoder encoder;
    rejects(
        [&]
        {
            encoder.put(0, false);
        });
    rejects(
        [&]
        {
            encoder.put(32768, false);
        });
    encoder.finish();
    rejects(
        [&]
        {
            encoder.put(16384, false);
        });
    rejects(
        [&]
        {
            encoder.finish();
        });
    tsac::RangeEncoder limited(0);
    rejects(
        [&]
        {
            limited.finish();
        });
    const tsac::Bytes invalid{0xff, 0xff, 0xff};
    rejects(
        [&]
        {
            tsac::RangeDecoder decoder(invalid);
        });
    tsac::RangeDecoder zero({});
    rejects(
        [&]
        {
            zero.get(0);
        });
}

void cli()
{
    for (const auto option : {"--cpu", "--scalar", "--full", "--cuda", "--model", "--trf_model"})
    {
        const std::array<std::string_view, 4> args{option, "c", "input.wav", "output.tsac"};
        rejects(
            [&]
            {
                tsac::parse_cli(args);
            });
    }
    for (const auto command : {"inspect", "tokens", "export-audio", "decode-fast"})
    {
        const std::array<std::string_view, 3> args{command, "input", "output"};
        rejects(
            [&]
            {
                tsac::parse_cli(args);
            });
    }
    const std::array<std::string_view, 4> bad_integer{"-q", "1x", "c", "input"};
    rejects(
        [&]
        {
            tsac::parse_cli(bad_integer);
        });
}
} // namespace

int main()
{
    try
    {
        containers();
        models();
        audio_and_range();
        cli();
        std::cout << "Sanitation checks passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
