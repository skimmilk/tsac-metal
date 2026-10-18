# Compile independent translation units, link each shader library, then embed it.
# Keep generated files in configuration-specific directories for multi-config builds.
find_program(TSAC_XCRUN xcrun REQUIRED)
set(TSAC_METAL_COMPILER "" CACHE FILEPATH "Optional override for Xcode's Metal compiler")
set(TSAC_METAL_LINKER "" CACHE FILEPATH "Optional override for Xcode's metallib linker")

function(tsac_add_metal_shaders target)
  set(shader_root "${CMAKE_CURRENT_SOURCE_DIR}/src/metal")
  set(output_root "${CMAKE_CURRENT_BINARY_DIR}/metal/$<CONFIG>")
  set(embed_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_library.cmake")
  set(sdk macosx)
  if(CMAKE_OSX_SYSROOT)
    set(sdk "${CMAKE_OSX_SYSROOT}")
  endif()
  set(compiler "${TSAC_XCRUN}" --sdk "${sdk}" metal)
  set(linker "${TSAC_XCRUN}" --sdk "${sdk}" metallib)
  if(TSAC_METAL_COMPILER)
    set(compiler "${TSAC_METAL_COMPILER}")
  endif()
  if(TSAC_METAL_LINKER)
    set(linker "${TSAC_METAL_LINKER}")
  endif()

  set(probability_sources
    attention cache elementwise embedding linear linear_matrix linear_prepared
    normalization quantization range softmax)
  set(probability_headers numeric.h parameters.h quantization.h)
  set(dac_sources
    convolution convolution_specialized convolution_tiled convolution_transposed
    elementwise rvq snake)
  set(dac_headers convolution.h parameters.h)

  set(optimization "$<IF:$<CONFIG:Debug>,-O0,$<IF:$<CONFIG:MinSizeRel>,-Os,-O2>>")
  set(debug_info "$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>")
  set(library_outputs)
  set(embedded_headers)
  foreach(library probability dac)
    set(headers)
    foreach(header IN LISTS ${library}_headers)
      list(APPEND headers "${shader_root}/${library}/${header}")
    endforeach()
    set(objects)
    foreach(name IN LISTS ${library}_sources)
      set(source "${shader_root}/${library}/${name}.metal")
      set(object "${output_root}/${library}/${name}.air")
      set(dependency_options)
      if(CMAKE_GENERATOR MATCHES "Ninja|Makefiles" OR CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        list(APPEND dependency_options DEPFILE "${object}.d")
      endif()
      add_custom_command(
        OUTPUT "${object}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_root}/${library}"
        COMMAND ${compiler}
          -std=metal4.0 -mmacosx-version-min=26.0
          -Wall -Wextra -Wuninitialized -fno-fast-math -ffp-contract=off
          -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise
          "${optimization}"
          "$<${debug_info}:-gline-tables-only>" "$<${debug_info}:-frecord-sources>"
          "-fmodules-cache-path=${output_root}/module-cache"
          -MMD -MF "${object}.d" -MT "${object}"
          -c "${source}" -o "${object}"
        DEPENDS "${source}" ${headers}
          "${shader_root}/abi/types.h" "${shader_root}/abi/bindings.h"
        ${dependency_options}
        COMMENT "Compiling Metal ${library}/${name} ($<CONFIG>)"
        COMMAND_EXPAND_LISTS
        VERBATIM)
      list(APPEND objects "${object}")
    endforeach()

    set(metallib "${output_root}/${library}.metallib")
    set(embedded_header "${output_root}/${library}_metallib.hpp")
    add_custom_command(
      OUTPUT "${metallib}"
      COMMAND ${linker} ${objects} -o "${metallib}"
      DEPENDS ${objects}
      COMMENT "Linking ${library}.metallib ($<CONFIG>)"
      COMMAND_EXPAND_LISTS
      VERBATIM)
    add_custom_command(
      OUTPUT "${embedded_header}"
      COMMAND "${CMAKE_COMMAND}"
        "-DINPUT=${metallib}" "-DOUTPUT=${embedded_header}" "-DNAME=${library}"
        -P "${embed_script}"
      DEPENDS "${metallib}" "${embed_script}"
      COMMENT "Embedding ${library}.metallib ($<CONFIG>)"
      VERBATIM)
    list(APPEND library_outputs "${metallib}")
    list(APPEND embedded_headers "${embedded_header}")
  endforeach()

  add_custom_target(tsac-metal-shaders DEPENDS ${library_outputs} ${embedded_headers})
  # Retain the existing convenience target; normal builds perform the same check.
  add_custom_target(tsac-metal-shader-check DEPENDS tsac-metal-shaders)
  add_dependencies(${target} tsac-metal-shaders)
  target_include_directories(${target} PRIVATE "${output_root}")
endfunction()
