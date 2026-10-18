foreach(argument INPUT OUTPUT NAME)
  if(NOT DEFINED ${argument} OR "${${argument}}" STREQUAL "")
    message(FATAL_ERROR "Missing ${argument} for Metal library embedding")
  endif()
endforeach()
if(NOT NAME MATCHES "^(probability|dac)$")
  message(FATAL_ERROR "Unknown Metal library: ${NAME}")
endif()
file(READ "${INPUT}" data HEX)
if(data STREQUAL "")
  message(FATAL_ERROR "Cannot embed empty Metal library: ${INPUT}")
endif()
string(REGEX REPLACE "(..)" "0x\\1," data "${data}")
# Bound generated line lengths without adding a build dependency on Python/xxd.
string(REPEAT "0x[0-9a-f][0-9a-f]," 16 row_pattern)
string(REGEX REPLACE "(${row_pattern})" "\\1\n" data "${data}")
file(WRITE "${OUTPUT}"
  "// Generated from ${NAME}.metallib; do not edit.\n"
  "#pragma once\n\nnamespace tsac::metal::embedded\n{\n"
  "alignas(16) inline constexpr unsigned char ${NAME}[] = {\n${data}\n};\n"
  "} // namespace tsac::metal::embedded\n")
