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
  src/report_exporter.cpp
  src/instance.cpp
  src/instance_generic.cpp
  src/instance_kiva.cpp
  src/lns.cpp
  src/lns_destroy.cpp
  src/lns_repair.cpp
  src/lns_run.cpp
  src/lns_validation.cpp
  src/main.cpp
  src/mlastar.cpp
  src/run_reporting.cpp
  src/sipps_helpers.cpp
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

set(PLANNER_ISOLATION_SOURCES
  src/planner_isolation.cpp
  src/astar.cpp
  src/constrainttable.cpp
  src/instance.cpp
  src/instance_generic.cpp
  src/instance_kiva.cpp
  src/mlastar.cpp
  src/sipps_helpers.cpp
  src/sipps.cpp
  src/utils.cpp
)

add_executable(planner_isolation ${PLANNER_ISOLATION_SOURCES})
target_include_directories(planner_isolation PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(planner_isolation PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
target_link_libraries(planner_isolation PRIVATE
  Boost::program_options
  Boost::system
  Boost::filesystem
  Boost::thread
  Threads::Threads
  plog::plog
)

enable_testing()

set(TEST_MAP_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_input/benchmark/maps/maze-32-32-4.map")
set(TEST_SCEN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_input/benchmark/scenarios/scen-even/maze-32-32-4-even-1.scen")
set(TEST_AGENT_COUNT 6)
set(TEST_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_artifacts")

add_test(
  NAME planner_correctness_mlastar
  COMMAND
    bash
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_planner_from_scen.sh"
    "$<TARGET_FILE:mapf_pc_lns>"
    "${TEST_MAP_FILE}"
    "${TEST_SCEN_FILE}"
    "${TEST_AGENT_COUNT}"
    "mlastar"
    "${TEST_ARTIFACT_DIR}"
)

add_test(
  NAME planner_correctness_sipps
  COMMAND
    bash
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_planner_from_scen.sh"
    "$<TARGET_FILE:mapf_pc_lns>"
    "${TEST_MAP_FILE}"
    "${TEST_SCEN_FILE}"
    "${TEST_AGENT_COUNT}"
    "sipps"
    "${TEST_ARTIFACT_DIR}"
)

set_tests_properties(planner_correctness_mlastar PROPERTIES TIMEOUT 120)
set_tests_properties(planner_correctness_sipps PROPERTIES TIMEOUT 120)
