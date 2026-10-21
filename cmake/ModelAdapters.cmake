# Purpose-specific adapters shared by CPU and Metal preparation.
add_library(tsac-model-adapters STATIC EXCLUDE_FROM_ALL
  src/common/common.cpp
  src/weights/model_directory.cpp
  src/models/contract.cpp
  src/models/tensor_view.cpp
  src/models/dac_analysis_model.cpp
  src/models/dac_synthesis_model.cpp
  src/models/rvq_encoder_model.cpp
  src/models/rvq_decoder_model.cpp
  src/models/transformer_model.cpp)
target_include_directories(tsac-model-adapters PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>" "${CMAKE_CURRENT_SOURCE_DIR}/src/models")
target_compile_features(tsac-model-adapters PUBLIC cxx_std_23)
target_link_libraries(tsac-model-adapters PUBLIC Threads::Threads)
target_compile_options(tsac-model-adapters PRIVATE "$<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-ffp-contract=off>")
set_target_properties(tsac-model-adapters PROPERTIES CXX_EXTENSIONS OFF)
add_dependencies(tsac-model-adapters tsac-weights-generate)
target_compile_definitions(tsac-model-adapters PRIVATE
  "TSAC_BUILD_WEIGHT_ROOT=\"${CMAKE_CURRENT_BINARY_DIR}/weights/$<CONFIG>/generations\""
  "TSAC_BUILD_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\"")
if(TSAC_SANITIZERS)
  target_compile_options(tsac-model-adapters PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(tsac-model-adapters PUBLIC -fsanitize=address,undefined)
endif()
