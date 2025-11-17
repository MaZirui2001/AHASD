CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# project setting
set(LIB_NAME "Simulator")

# set source and headers
# Recursively includes all source files, including:
# - AHASD Integration (AHASDIntegration.h)
# - Async Queue modules (async_queue/*.h)
# - EDC, TVC modules
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# build
add_executable(${LIB_NAME} ${SRC_FILES})
add_library(${LIB_NAME}_lib ${SRC_FILES})
