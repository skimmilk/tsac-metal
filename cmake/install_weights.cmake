include("${TSAC_INSTALL_WEIGHT_ROOT}/active-generation.cmake")
include("${TSAC_GENERATION_DIRECTORY}/weights-manifest.cmake")
set(TSAC_INSTALL_PHYSICAL_BIN "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin")
set(TSAC_OLD_INSTALLED_WEIGHTS)
if(EXISTS "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake")
  include("${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake")
  set(TSAC_OLD_INSTALLED_WEIGHTS ${TSAC_INSTALLED_WEIGHT_FILES})
endif()
file(MAKE_DIRECTORY "${TSAC_INSTALL_PHYSICAL_BIN}")
# Track the union before installing, so interrupted upgrades remain recoverable.
file(WRITE "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake.tmp"
  "set(TSAC_INSTALLED_WEIGHT_FILES \"${TSAC_OLD_INSTALLED_WEIGHTS};${TSAC_WEIGHT_FILES}\")\n")
file(RENAME "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake.tmp"
  "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake")
foreach(name IN LISTS TSAC_WEIGHT_FILES)
  file(INSTALL "${TSAC_GENERATION_DIRECTORY}/${name}" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE)
endforeach()
list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${CMAKE_INSTALL_PREFIX}/bin/.tsac-installed-weights.cmake")
