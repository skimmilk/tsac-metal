#include "cli.hpp"

#include <charconv>

namespace tsac
{
CliOptions parse_cli(std::span<const std::string_view> arguments)
{
    CliOptions options;
    std::vector<std::string_view> positional;
    for (std::size_t i = 0; i < arguments.size(); ++i)
    {
        const auto arg = arguments[i];
        if (arg == "--")
        {
            positional.insert(positional.end(),
                              arguments.begin() + static_cast<std::ptrdiff_t>(i + 1),
                              arguments.end());
            break;
        }
        if (arg.size() <= 1 || arg[0] != '-')
        {
            positional.push_back(arg);
            continue;
        }
        const auto value = [&](std::string_view name)
        {
            if (++i == arguments.size())
            {
                throw Error("option '" + std::string(name) + "' must have an argument");
            }
            auto text = arguments[i];
            if (text.starts_with('+'))
            {
                text.remove_prefix(1);
            }
            int number = 0;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), number);
            if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
            {
                const auto canonical = name == "-q"   ? "--n_codebooks"
                                       : name == "-c" ? "--channels"
                                                      : name;
                throw Error("option " + std::string(canonical) + " expects an integer");
            }
            return number;
        };
        const auto flag = [&](std::string_view name)
        {
            if (name == "-h" || name == "--help")
            {
                options.help = true;
            }
            else if (name == "-f" || name == "--fast")
            {
                options.fast = true;
            }
            else if (name == "-s" || name == "--separate_channels")
            {
                options.separate = true;
            }
            else if (name == "-v")
            {
                ++options.verbose;
            }
            else if (name == "-T")
            {
                options.threads = value(name);
            }
            else if (name == "-q" || name == "--n_codebooks")
            {
                options.quality = value(name);
            }
            else if (name == "-c" || name == "--channels")
            {
                options.channels = value(name);
            }
            else if (name == "--batch_size")
            {
                options.batch = value(name);
            }
            else if (name == "--metal")
            {
                options.metal = true;
            }
            else
            {
                throw Error("unknown option: '" + std::string(name) + "'");
            }
        };
        if (arg.starts_with("--"))
        {
            flag(arg);
        }
        else
        {
            // Validate the whole short-option group before consuming values.
            // In particular, -q1 is an unknown -1 option, not an attached value.
            for (const char character : arg.substr(1))
            {
                if (std::string_view("hfsvTqc").find(character) == std::string_view::npos)
                {
                    throw Error("unknown option: '-" + std::string(1, character) + "'");
                }
            }
            for (std::size_t character = 1; character < arg.size(); ++character)
            {
                const std::string name = std::string("-") + arg[character];
                flag(name);
            }
        }
        if (options.help)
        {
            return options;
        }
    }
    if (positional.size() != 3)
    {
        options.help = true;
        return options;
    }
    if (positional[0] != "c" && positional[0] != "d" && positional[0] != "t")
    {
        throw Error("unknown command: " + std::string(positional[0]));
    }
    options.command = positional[0][0];
    options.input = positional[1];
    options.output = positional[2];
    return options;
}

std::string_view cli_help()
{
    return R"(TSAC for Apple Silicon

Very low bitrate audio compression

usage: tsac [options] c infile outfile     audio compression
                      d infile outfile     audio decompression
                      t infile outfile     audio compression then decompression to outfile
Uncompressed wave (.wav) and raw audio (.sw) are natively supported as input files.
FFmpeg is invoked to support other audio formats. Decompression outputs WAVE.
Weights load from generated hashed filenames beside the actual executable.

Available options:
-h --help               show the help
-T n                    number of threads (default=number of physical cores)
-q --n_codebooks n      quality (1-12 for joint stereo, 1-9 otherwise; 0=max)
-s --separate_channels  encode stereo as separate channels
-c --channels n         force input channels through FFmpeg
-f --fast               disable the transformer model
--batch_size n          force the entropy batch size (default=auto)
-v                      display statistics
--metal                 use Metal execution (default: CPU)
)";
}
} // namespace tsac
