#include "codec.hpp"

#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i)
        {
            arguments.emplace_back(argv[i]);
        }
        const auto options = tsac::parse_cli(arguments);
        if (options.help)
        {
            std::cout << tsac::cli_help();
            return 1;
        }
        tsac::run_codec(options);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "tsac: " << error.what() << '\n';
        return 1;
    }
}
