cmake_minimum_required(VERSION 3.10)
project(LadybugRFI CXX)

# Use C++17 consistently
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ladybug SDK paths
set(LADYBUG_LIB_DIR "/lib/ladybug")
set(LADYBUG_INCLUDE_DIR "/usr/include/ladybug")

find_library(LADYBUG_LIBRARY       NAMES ladybug         PATHS ${LADYBUG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FLYCAPTURE_LIBRARY    NAMES flycapture      PATHS ${LADYBUG_LIB_DIR} NO_DEFAULT_PATH)
find_library(FLYCAPTUREVIDEO_LIB   NAMES flycapturevideo PATHS ${LADYBUG_LIB_DIR} NO_DEFAULT_PATH)

if(NOT LADYBUG_LIBRARY)
  message(FATAL_ERROR "libladybug non trovata in ${LADYBUG_LIB_DIR}")
endif()
if(NOT FLYCAPTURE_LIBRARY)
  message(FATAL_ERROR "libflycapture non trovata in ${LADYBUG_LIB_DIR}")
endif()
if(NOT FLYCAPTUREVIDEO_LIB)
  message(FATAL_ERROR "libflycapturevideo non trovata in ${LADYBUG_LIB_DIR}")
endif()
if(NOT EXISTS "${LADYBUG_INCLUDE_DIR}/ladybug.h")
  message(FATAL_ERROR "Header ladybug.h non trovato in ${LADYBUG_INCLUDE_DIR}")
endif()

include_directories(${LADYBUG_INCLUDE_DIR})

add_executable(ladybugRFI main.cpp)

# Threads
find_package(Threads REQUIRED)
target_link_libraries(ladybugRFI
  PRIVATE
    ${FLYCAPTURE_LIBRARY}
    ${LADYBUG_LIBRARY}
    ${FLYCAPTUREVIDEO_LIB}
    Threads::Threads
)

# Warnings in Debug
target_compile_options(ladybugRFI PRIVATE
  $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
)

# Install
install(TARGETS ladybugRFI DESTINATION bin)

message(STATUS "Configurazione completata con successo.")