set(default_models "${CMAKE_CURRENT_SOURCE_DIR}/models")
foreach(name dac_mono_q8.bin dac_stereo_q8.bin tsac_mono_q8.bin tsac_stereo_q8.bin)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/${name}")
    set(default_models "${CMAKE_CURRENT_BINARY_DIR}/models/source")
  endif()
endforeach()
set(TSAC_MODEL_DIR "${default_models}" CACHE PATH "Original model inputs")
option(TSAC_DOWNLOAD_MODELS "Download missing original model files during build" ON)
option(TSAC_ENABLE_TRANSFORMER "Include normal entropy assets (fast mode otherwise)" ON)
set(TSAC_WEIGHT_BUDGET 400000000 CACHE STRING "Maximum staged executable and weights size in bytes")

function(tsac_add_weights)
  # Source-format parsing is private to the host packer and parity tests.
  # Inference does not link this library or expose its name-based API.
  add_library(tsac-model-source STATIC src/weights/source_model.cpp)
  target_include_directories(tsac-model-source PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(tsac-model-source PRIVATE cxx_std_20)
  target_compile_options(tsac-model-source PRIVATE "$<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-fno-fast-math;-ffp-contract=off>")
  if(TSAC_SANITIZERS)
    target_compile_options(tsac-model-source PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(tsac-model-source INTERFACE -fsanitize=address,undefined)
  endif()
  add_executable(tsac-pack-models
    src/weights/pack_main.cpp src/weights/preparation.cpp
    src/common/common.cpp src/weights/normalization.cpp src/format/categorical.cpp src/format/range.cpp
    src/models/contract.cpp src/models/tensor_view.cpp)
  target_link_libraries(tsac-pack-models PRIVATE tsac-model-source)
  target_include_directories(tsac-pack-models PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(tsac-pack-models PRIVATE cxx_std_20)
  target_compile_options(tsac-pack-models PRIVATE "$<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-fno-fast-math;-ffp-contract=off>")
  set(root "${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>")
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
  add_custom_target(tsac-weights-generate
    BYPRODUCTS "${generated}/adapter_offsets.hpp" "${root}/active-generation.cmake"
    COMMAND "${CMAKE_COMMAND}"
      "-DPACKER=$<TARGET_FILE:tsac-pack-models>"
      "-DMODELS=${TSAC_MODEL_DIR}" "-DROOT=${root}" "-DGENERATED=${generated}"
      "-DDOWNLOAD=${TSAC_DOWNLOAD_MODELS}" "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DTRANSFORMER=${TSAC_ENABLE_TRANSFORMER}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pack_weights.cmake"
    DEPENDS tsac-pack-models
    VERBATIM)
  if(TSAC_BUILD_CLI)
  add_custom_target(tsac-stage ALL
    COMMAND "${CMAKE_COMMAND}" "-DROOT=${root}" "-DBINARY=$<TARGET_FILE:tsac>"
      "-DDESTINATION=${CMAKE_CURRENT_SOURCE_DIR}" "-DBUDGET=${TSAC_WEIGHT_BUDGET}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stage_weights.cmake"
    DEPENDS tsac VERBATIM)
  endif()
  add_custom_target(tsac-clean-assets
    COMMAND "${CMAKE_COMMAND}" "-DDESTINATION=${CMAKE_CURRENT_SOURCE_DIR}" "-DROOT=${root}" -DALL=ON
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clean_weights.cmake" VERBATIM)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/weights" "${CMAKE_CURRENT_BINARY_DIR}/generated")
  install(CODE "set(TSAC_INSTALL_WEIGHT_ROOT \"${root}\")")
  install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_weights.cmake")
  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" "-DMANIFEST=${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake" VERBATIM)
endfunction()
