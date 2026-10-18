#include "common/probability.hpp"
#include "scalar/kernels.hpp"

#include <array>
#include <iostream>

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw tsac::Error(message);
    }
}
} // namespace

int main()
{
    try
    {
        using namespace tsac;
        check(sha256({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "empty SHA256");
        const Bytes abc{'a', 'b', 'c'};
        check(sha256(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "SHA256 abc");
        const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        check(sha256(Bytes(two_blocks.begin(), two_blocks.end())) ==
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
              "SHA256 two-block padding");
        const Bytes many(1000000, 'a');
        check(sha256(many) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
              "multi-block SHA256");
        const std::array<float, 3> input{1, 2, 3};
        const std::array<float, 2> weights{2, 3};
        const std::array<float, 1> bias{1};
        scalar::Conv1d ordinary({1, 1, 2, 1, 0, 1}, weights, bias);
        check(ordinary.apply(input, {1, 3, 1}).values == std::vector<float>({9, 14}),
              "scalar convolution");
        scalar::Conv1d transposed({1, 1, 2, 2, 0, 1, ConvolutionKind::transposed}, weights, bias);
        check(transposed.apply(input, {1, 3, 1}).values == std::vector<float>({3, 4, 5, 7, 7, 10}),
              "scalar transposed convolution");
        std::array<float, 256> values{};
        values[0] = 127;
        values[1] = 0.5f;
        values[2] = 1.5f;
        values[3] = -1.5f;
        const auto quantized = scalar::bf8l(values);
        check(quantized.scales == std::vector<std::uint16_t>({probability_bf16(1)}) &&
                  quantized.values[0] == 127 && quantized.values[1] == 0 &&
                  quantized.values[2] == 2 && quantized.values[3] == -2,
              "scalar BF8L nearest-even quantization");
        std::cout << "Scalar kernels passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
