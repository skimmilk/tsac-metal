cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compilation database for editor tooling")
project(tsac_cpu VERSION 0.1.0 LANGUAGES CXX)

set(tsac_apple_default OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(tsac_apple_default ON)
endif()
option(TSAC_ENABLE_APPLE_ARM "Build Apple ARM Accelerate/NEON analyzers" ${tsac_apple_default})
option(TSAC_ENABLE_METAL "Build Metal 4 runtime and shaders" ${TSAC_ENABLE_APPLE_ARM})
option(TSAC_BUILD_CLI "Build the Apple command-line application" ${TSAC_ENABLE_APPLE_ARM})
if(TSAC_ENABLE_APPLE_ARM OR TSAC_ENABLE_METAL OR TSAC_BUILD_CLI)
  if(NOT tsac_apple_default)
    message(FATAL_ERROR "Apple backends and the CLI require macOS on Apple M1 or later; disable TSAC_ENABLE_APPLE_ARM, TSAC_ENABLE_METAL and TSAC_BUILD_CLI for the portable scalar library")
  endif()
  if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    message(FATAL_ERROR "Apple backends support only arm64")
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version" FORCE)
  endif()
endif()
if(TSAC_BUILD_CLI AND NOT TSAC_ENABLE_APPLE_ARM)
  message(FATAL_ERROR "The CLI requires TSAC_ENABLE_APPLE_ARM; use ScalarAnalyzer directly for reference builds")
endif()
if(TSAC_ENABLE_APPLE_ARM)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "15.0")
    message(FATAL_ERROR "The Accelerate backend requires macOS 15 or later")
  endif()
endif()
if(TSAC_ENABLE_METAL)
  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

option(TSAC_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Keep the CLI and tests under the same warning policy as the library.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

# The compatibility target contains the shared pipeline and portable reference
# kernels, plus optional native implementations. Consumers need only include/.
add_library(tsac_cpu
  src/common/audio_analyzer.cpp
  src/common/audio_pipeline.hpp
  src/common/common.cpp
  src/common/signal.cpp
  src/common/probability.cpp
  src/common/rvq.cpp
  src/common/transformer.cpp
  src/format/categorical.cpp
  src/format/container.cpp
  src/format/entropy.cpp
  src/format/range.cpp
  src/weights/normalization.cpp
  src/weights/preparation.cpp
  src/scalar/pipeline.cpp
  src/scalar/audio_analyzer.cpp
  src/scalar/dsp.cpp
  src/scalar/attention.cpp
  src/scalar/packed_linear.cpp
  src/scalar/probability.cpp
)
add_library(tsac::scalar ALIAS tsac_cpu)
if(TSAC_ENABLE_APPLE_ARM)
  target_sources(tsac_cpu PRIVATE
    src/apple-arm/pipeline.cpp
    src/apple-arm/audio_analyzer.cpp
    src/apple-arm/attention.cpp
    src/apple-arm/dsp.cpp
    src/apple-arm/execution.cpp
    src/apple-arm/workers.cpp
    src/apple-arm/packed_linear.cpp
    src/apple-arm/probability.cpp)
  set_source_files_properties(
    src/apple-arm/pipeline.cpp src/apple-arm/audio_analyzer.cpp src/apple-arm/attention.cpp
    src/apple-arm/dsp.cpp src/apple-arm/execution.cpp src/apple-arm/workers.cpp src/apple-arm/packed_linear.cpp
    src/apple-arm/probability.cpp PROPERTIES COMPILE_OPTIONS "-mcpu=apple-m1")
  target_compile_definitions(tsac_cpu PRIVATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(tsac_cpu PUBLIC "${ACCELERATE_FRAMEWORK}")
else()
  target_sources(tsac_cpu PRIVATE src/apple-arm/disabled.cpp)
endif()
if(TSAC_BUILD_CLI)
  target_sources(tsac_cpu PRIVATE src/audio.cpp src/cli.cpp src/codec.cpp)
endif()

if(TSAC_ENABLE_METAL)
  target_sources(tsac_cpu PRIVATE
    src/metal/audio_analyzer.cpp
    src/metal/audio_dac.cpp
    src/metal/audio_rvq.cpp
    src/metal/weight_preparation.cpp
    src/metal/transformer.mm
    src/metal/transformer/model.mm
    src/metal/transformer/session.mm
    src/metal/transformer/execution.mm
    src/metal/transformer/encoding.mm
    src/metal/transformer/decoding.mm
    src/metal/transformer/replay.mm
    src/metal/runtime.mm
    src/metal/dac_graph.mm
    src/metal/prediction.mm
  )
else()
  target_sources(tsac_cpu PRIVATE src/metal/disabled.cpp)
endif()
target_include_directories(tsac_cpu PUBLIC include PRIVATE src)
target_compile_features(tsac_cpu PUBLIC cxx_std_23)
target_compile_definitions(tsac_cpu PRIVATE TSAC_ENABLE_APPLE_ARM=$<BOOL:${TSAC_ENABLE_APPLE_ARM}>)
find_package(Threads REQUIRED)
target_link_libraries(tsac_cpu PUBLIC Threads::Threads)
if(TSAC_ENABLE_METAL)
  target_link_libraries(tsac_cpu PUBLIC "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
endif()
target_compile_options(tsac_cpu PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
set_target_properties(tsac_cpu PROPERTIES CXX_EXTENSIONS OFF OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED YES)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(tsac_cpu PRIVATE -ffp-contract=off)
  if(TSAC_SANITIZERS)
    target_compile_options(tsac_cpu PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(tsac_cpu PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(TSAC_BUILD_CLI)
  add_executable(tsac src/main.cpp)
  target_link_libraries(tsac PRIVATE tsac_cpu)
  target_include_directories(tsac PRIVATE src)
endif()

if(TSAC_ENABLE_METAL)
  include(cmake/MetalShaders.cmake)
  tsac_add_metal_shaders(tsac_cpu)
endif()

include(cmake/Weights.cmake)
tsac_add_weights()
include(cmake/ModelAdapters.cmake)
target_link_libraries(tsac_cpu PRIVATE tsac-model-adapters)
target_compile_definitions(tsac_cpu PRIVATE TSAC_ENABLE_TRANSFORMER=$<BOOL:${TSAC_ENABLE_TRANSFORMER}>)
if(TSAC_BUILD_CLI)
  install(TARGETS tsac RUNTIME DESTINATION bin)
endif()
install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prune_installed_weights.cmake")
install(FILES README.md WEIGHTS.md AUDIO_API.md SCALAR.md DESTINATION share/doc/tsac)

include(CTest)
if(BUILD_TESTING)
  add_executable(tsac-numeric-test tests/numeric.cpp)
  target_link_libraries(tsac-numeric-test PRIVATE tsac_cpu)
  target_include_directories(tsac-numeric-test PRIVATE src)
  add_test(NAME numeric COMMAND tsac-numeric-test)

  # External-style consumer: deliberately has no private include directories.
  add_executable(tsac-public-api-test tests/public_api.cpp)
  target_link_libraries(tsac-public-api-test PRIVATE tsac_cpu)
  add_test(NAME public-api COMMAND "${CMAKE_COMMAND}"
    "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
    "-DPROGRAM=$<TARGET_FILE:tsac-public-api-test>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")

  target_compile_definitions(tsac-public-api-test PRIVATE TSAC_TEST_TRANSFORMER=$<BOOL:${TSAC_ENABLE_TRANSFORMER}> TSAC_TEST_SCALAR=$<NOT:$<BOOL:${TSAC_ENABLE_APPLE_ARM}>>)
  if(TSAC_ENABLE_APPLE_ARM)
    add_executable(tsac-scalar-api-test tests/public_api.cpp)
    target_link_libraries(tsac-scalar-api-test PRIVATE tsac::scalar)
    target_compile_definitions(tsac-scalar-api-test PRIVATE TSAC_TEST_SCALAR=1 TSAC_TEST_TRANSFORMER=$<BOOL:${TSAC_ENABLE_TRANSFORMER}>)
    add_test(NAME public-api-scalar COMMAND "${CMAKE_COMMAND}"
      "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
      "-DPROGRAM=$<TARGET_FILE:tsac-scalar-api-test>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
  endif()

  # Compile each public header on its own through the same public-only boundary.
  set(public_header_sources)
  foreach(header types pcm observer model_directory audio_analyzer apple_arm_analyzer
          metal_analyzer scalar_analyzer)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/public-headers/${header}.cpp")
    file(GENERATE OUTPUT "${source}" CONTENT "#include <tsac/${header}.hpp>\n")
    list(APPEND public_header_sources "${source}")
  endforeach()
  add_library(tsac-public-headers-check OBJECT ${public_header_sources})
  target_link_libraries(tsac-public-headers-check PRIVATE tsac_cpu)

  add_executable(tsac-scalar-kernels-test tests/scalar.cpp)
  target_link_libraries(tsac-scalar-kernels-test PRIVATE tsac::scalar)
  target_include_directories(tsac-scalar-kernels-test PRIVATE src)
  add_test(NAME scalar-kernels COMMAND tsac-scalar-kernels-test)
  add_executable(tsac-scalar-attention-test tests/attention.cpp)
  target_link_libraries(tsac-scalar-attention-test PRIVATE tsac::scalar)
  target_include_directories(tsac-scalar-attention-test PRIVATE src)
  target_compile_definitions(tsac-scalar-attention-test PRIVATE TSAC_ENABLE_APPLE_ARM=0)
  add_test(NAME attention-scalar COMMAND tsac-scalar-attention-test)

  if(TSAC_ENABLE_APPLE_ARM)
    add_executable(tsac-apple-cpu-test tests/apple_cpu.cpp)
    target_link_libraries(tsac-apple-cpu-test PRIVATE tsac_cpu)
    target_include_directories(tsac-apple-cpu-test PRIVATE src)
    add_test(NAME apple-cpu COMMAND tsac-apple-cpu-test)
    add_executable(tsac-attention-test tests/attention.cpp)
    target_link_libraries(tsac-attention-test PRIVATE tsac_cpu)
    target_compile_definitions(tsac-attention-test PRIVATE TSAC_ENABLE_APPLE_ARM=$<BOOL:${TSAC_ENABLE_APPLE_ARM}>)
    add_test(NAME attention-cpu COMMAND tsac-attention-test)
  endif()
  if(TSAC_ENABLE_APPLE_ARM)
    add_executable(tsac-audio-analyzer-test tests/audio_analyzer.cpp)
    target_link_libraries(tsac-audio-analyzer-test PRIVATE tsac_cpu)
    target_compile_definitions(tsac-audio-analyzer-test PRIVATE
      TSAC_TEST_TRANSFORMER=$<BOOL:${TSAC_ENABLE_TRANSFORMER}>)
    add_test(NAME audio-analyzer-cpu COMMAND "${CMAKE_COMMAND}"
      "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
      "-DPROGRAM=$<TARGET_FILE:tsac-audio-analyzer-test>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
    if(TSAC_ENABLE_METAL)
      add_test(NAME audio-analyzer-metal COMMAND "${CMAKE_COMMAND}"
        "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
        "-DPROGRAM=$<TARGET_FILE:tsac-audio-analyzer-test>" -DEXTRA_ARGUMENTS=--metal
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
      set_tests_properties(audio-analyzer-metal PROPERTIES LABELS metal)
    endif()
  endif()
  if(UNIX)
    add_executable(tsac-cpu-preparation tests/cpu_preparation.cpp)
    target_link_libraries(tsac-cpu-preparation PRIVATE tsac_cpu)
    target_include_directories(tsac-cpu-preparation PRIVATE src)
    target_compile_definitions(tsac-cpu-preparation PRIVATE
      TSAC_ENABLE_APPLE_ARM=$<BOOL:${TSAC_ENABLE_APPLE_ARM}>)
    target_compile_options(tsac-cpu-preparation PRIVATE "$<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-ffp-contract=off>")
    add_test(NAME cpu-preparation COMMAND "${CMAKE_COMMAND}"
      "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
      "-DPROGRAM=$<TARGET_FILE:tsac-cpu-preparation>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
    add_executable(tsac-model-adapters-test
      tests/model_adapters.cpp src/weights/normalization.cpp src/weights/preparation.cpp)
    target_compile_options(tsac-model-adapters-test PRIVATE "$<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-ffp-contract=off>")
    target_compile_definitions(tsac-model-adapters-test PRIVATE
      TSAC_TEST_TRANSFORMER=$<BOOL:${TSAC_ENABLE_TRANSFORMER}>)
    target_link_libraries(tsac-model-adapters-test PRIVATE tsac-model-source tsac-model-adapters)
    add_test(NAME model-adapters COMMAND "${CMAKE_COMMAND}"
      "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
      "-DPROGRAM=$<TARGET_FILE:tsac-model-adapters-test>" "-DORIGINALS=${TSAC_MODEL_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
  endif()
  if(TSAC_BUILD_CLI)
    add_executable(tsac-sanitation tests/sanitation.cpp)
    target_link_libraries(tsac-sanitation PRIVATE tsac-model-source tsac_cpu)
    add_test(NAME sanitation COMMAND tsac-sanitation)
    if(TSAC_ENABLE_TRANSFORMER)
      add_executable(tsac-weights-test tests/weights.cpp)
      target_link_libraries(tsac-weights-test PRIVATE tsac-model-source tsac_cpu)
      add_test(NAME weights COMMAND "${CMAKE_COMMAND}"
        "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
        "-DPROGRAM=$<TARGET_FILE:tsac-weights-test>" "-DORIGINALS=${TSAC_MODEL_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
    endif()
  endif()
  if(TSAC_ENABLE_METAL)
    add_executable(tsac-metal-range tests/metal_range.cpp)
    target_link_libraries(tsac-metal-range PRIVATE tsac_cpu)
    add_test(NAME metal-range COMMAND tsac-metal-range)
    set_tests_properties(metal-range PROPERTIES LABELS metal)
    add_executable(tsac-metal-abi tests/metal_abi.cpp)
    target_link_libraries(tsac-metal-abi PRIVATE tsac_cpu)
    add_test(NAME metal-abi COMMAND tsac-metal-abi
      "${CMAKE_CURRENT_BINARY_DIR}/metal/$<CONFIG>/probability.metallib")
    set_tests_properties(metal-abi PROPERTIES LABELS metal)
    if(TSAC_ENABLE_TRANSFORMER AND TSAC_ENABLE_APPLE_ARM)
      add_executable(tsac-metal-preparation tests/metal_preparation.cpp)
      target_link_libraries(tsac-metal-preparation PRIVATE tsac_cpu)
      target_include_directories(tsac-metal-preparation PRIVATE src)
      target_compile_options(tsac-metal-preparation PRIVATE -ffp-contract=off)
      add_test(NAME metal-preparation COMMAND "${CMAKE_COMMAND}"
        "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
        "-DPROGRAM=$<TARGET_FILE:tsac-metal-preparation>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
      add_executable(tsac-metal-optimized tests/metal_optimized.cpp)
      target_link_libraries(tsac-metal-optimized PRIVATE tsac_cpu)
      add_test(NAME metal-optimized COMMAND "${CMAKE_COMMAND}"
        "-DROOT=${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>"
        "-DPROGRAM=$<TARGET_FILE:tsac-metal-optimized>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_weights.cmake")
      set_tests_properties(metal-optimized PROPERTIES LABELS metal)
    endif()
  endif()
  foreach(test_target tsac-attention-test tsac-audio-analyzer-test tsac-model-adapters-test
          tsac-sanitation tsac-weights-test tsac-metal-range tsac-metal-abi tsac-metal-optimized)
    if(TARGET ${test_target})
      target_include_directories(${test_target} PRIVATE include src)
      target_compile_definitions(${test_target} PRIVATE TSAC_ENABLE_APPLE_ARM=$<BOOL:${TSAC_ENABLE_APPLE_ARM}>)
    endif()
  endforeach()
endif()
