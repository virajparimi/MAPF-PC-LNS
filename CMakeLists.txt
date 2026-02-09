cmake_minimum_required(VERSION 3.16)
project(mapf_pc_lns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options system filesystem thread)
find_package(Threads REQUIRED)

# --- plog (logging) ---
# plog is header-only; point CMake at the directory that contains `plog/Log.h`.
set(PLOG_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../Dependencies/plog/include"
    CACHE PATH "Path to a directory containing plog/Log.h")

if(NOT EXISTS "${PLOG_INCLUDE_DIR}/plog/Log.h")
  message(FATAL_ERROR
    "plog not found at PLOG_INCLUDE_DIR='${PLOG_INCLUDE_DIR}'. "
    "Set -DPLOG_INCLUDE_DIR=/path/to/plog/include (must contain plog/Log.h)."
  )
endif()

add_library(plog INTERFACE)
target_include_directories(plog INTERFACE "${PLOG_INCLUDE_DIR}")
add_library(plog::plog ALIAS plog)

set(SOURCES
  src/astar.cpp
  src/constrainttable.cpp
  src/costchecker.cpp
  src/instance.cpp
  src/lns.cpp
  src/main.cpp
  src/mlastar.cpp
  src/sipps.cpp
  src/utils.cpp
)

add_executable(mapf_pc_lns ${SOURCES})
target_include_directories(mapf_pc_lns PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(mapf_pc_lns PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
target_link_libraries(mapf_pc_lns PRIVATE
  Boost::program_options
  Boost::system
  Boost::filesystem
  Boost::thread
  Threads::Threads
  plog::plog
)
