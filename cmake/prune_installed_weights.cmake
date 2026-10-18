# Runs after the executable is installed, with the prepare script's variables.
foreach(name IN LISTS TSAC_OLD_INSTALLED_WEIGHTS)
  if(NOT name IN_LIST TSAC_WEIGHT_FILES AND name MATCHES "^tsac-[a-z0-9-]+[.]weights$")
    file(REMOVE "${TSAC_INSTALL_PHYSICAL_BIN}/${name}")
  endif()
endforeach()
file(WRITE "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake.tmp"
  "set(TSAC_INSTALLED_WEIGHT_FILES \"${TSAC_WEIGHT_FILES}\")\n")
file(RENAME "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake.tmp"
  "${TSAC_INSTALL_PHYSICAL_BIN}/.tsac-installed-weights.cmake")
