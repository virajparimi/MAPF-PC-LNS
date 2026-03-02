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
  src/app/main.cpp
  src/core/astar.cpp
  src/core/constrainttable.cpp
  src/core/instance.cpp
  src/core/instance_generic.cpp
  src/core/utils.cpp
  src/lns/acceptance/lns_acceptance.cpp
  src/lns/acceptance/lns_acceptance_orchestrator.cpp
  src/lns/acceptance/lns_acceptance_policy.cpp
  src/lns/core/lns.cpp
  src/lns/core/lns_prepare.cpp
  src/lns/core/lns_run.cpp
  src/lns/core/lns_run_iteration.cpp
  src/lns/destroy/lns_destroy.cpp
  src/lns/destroy/lns_destroy_classic.cpp
  src/lns/destroy/lns_destroy_market.cpp
  src/lns/destroy/lns_destroy_orchestrator.cpp
  src/lns/destroy/lns_destroy_precedence.cpp
  src/lns/init/lns_initialization.cpp
  src/lns/init/lns_initialization_mapfpc.cpp
  src/lns/init/lns_initialization_reposition.cpp
  src/lns/iteration/lns_candidate_phase_orchestrator.cpp
  src/lns/iteration/lns_iteration_acceptance_bookkeeping_orchestrator.cpp
  src/lns/iteration/lns_iteration_candidate_postprocess_orchestrator.cpp
  src/lns/iteration/lns_iteration_diagnostics_orchestrator.cpp
  src/lns/iteration/lns_iteration_lifecycle_orchestrator.cpp
  src/lns/iteration/lns_iteration_outcome_orchestrator.cpp
  src/lns/iteration/lns_iteration_setup_orchestrator.cpp
  src/lns/mapfpc/lns_mapfpc_io.cpp
  src/lns/mapfpc/lns_mapfpc_repair.cpp
  src/lns/mapfpc/lns_mapfpc_runner.cpp
  src/lns/market/lns_market.cpp
  src/lns/market/lns_market_iteration_orchestrator.cpp
  src/lns/repair/lns_repair.cpp
  src/lns/repair/lns_repair_commit.cpp
  src/lns/repair/lns_repair_constraints.cpp
  src/lns/repair/lns_repair_engine.cpp
  src/lns/repair/lns_repair_insert.cpp
  src/lns/repair/lns_repair_nrr.cpp
  src/lns/repair/lns_repair_nrr_helpers.cpp
  src/lns/repair/lns_repair_regret_agent.cpp
  src/lns/repair/lns_repair_state.cpp
  src/lns/repair/lns_repair_workspace.cpp
  src/lns/validation/lns_validation.cpp
  src/lns/validation/lns_validation_orchestrator.cpp
  src/planners/mlastar.cpp
  src/planners/sipps.cpp
  src/planners/sipps_helpers.cpp
  src/reporting/report_exporter.cpp
  src/reporting/run_reporting.cpp
  src/reporting/run_reporting_common.cpp
  src/reporting/run_reporting_summary.cpp
)

add_executable(mapf_pc_lns ${SOURCES})
set(PROJECT_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/core"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/planners"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/core"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/acceptance"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/destroy"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/iteration"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/internal"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/mapfpc"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/market"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/repair"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/state"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lns/validation"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/reporting"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/internal"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/reporting"
)
target_include_directories(mapf_pc_lns PRIVATE ${PROJECT_INCLUDE_DIRS})
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
  src/tools/planner_isolation.cpp
  src/core/astar.cpp
  src/core/constrainttable.cpp
  src/core/instance.cpp
  src/core/instance_generic.cpp
  src/core/utils.cpp
  src/planners/mlastar.cpp
  src/planners/sipps.cpp
  src/planners/sipps_helpers.cpp
)

add_executable(planner_isolation ${PLANNER_ISOLATION_SOURCES})
target_include_directories(planner_isolation PRIVATE ${PROJECT_INCLUDE_DIRS})
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

add_executable(test_nrr_boundary_windows tests/test_nrr_boundary_windows.cpp)
target_include_directories(test_nrr_boundary_windows
                           PRIVATE ${PROJECT_INCLUDE_DIRS})
target_compile_options(test_nrr_boundary_windows PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

enable_testing()

set(TEST_MAP_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/datasets/benchmark/maps/maze-32-32-4.map")
set(TEST_SCEN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/datasets/benchmark/scenarios/scen-even/maze-32-32-4-even-1.scen")
set(TEST_AGENT_COUNT 6)
set(TEST_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_artifacts")
set(TEST_SMALL_TIER_CASES_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/small_tier_fixed_instances.tsv")

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

add_test(
  NAME nrr_boundary_windows_unit
  COMMAND $<TARGET_FILE:test_nrr_boundary_windows>
)

add_test(
  NAME nrr_mock_success
  COMMAND
    bash
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_nrr_mock_success.sh"
    "$<TARGET_FILE:mapf_pc_lns>"
    "${CMAKE_CURRENT_BINARY_DIR}/test_artifacts"
)

add_test(
  NAME nrr_mock_fallback
  COMMAND
    bash
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_nrr_mock_fallback.sh"
    "$<TARGET_FILE:mapf_pc_lns>"
    "${CMAKE_CURRENT_BINARY_DIR}/test_artifacts"
)

add_test(
  NAME lns_small_tier_fixed_suite
  COMMAND
    bash
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_lns_small_tier_suite.sh"
    "$<TARGET_FILE:mapf_pc_lns>"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${TEST_SMALL_TIER_CASES_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/test_artifacts/small_tier"
)

set_tests_properties(planner_correctness_mlastar PROPERTIES TIMEOUT 120)
set_tests_properties(planner_correctness_sipps PROPERTIES TIMEOUT 120)
set_tests_properties(nrr_boundary_windows_unit PROPERTIES TIMEOUT 30)
set_tests_properties(nrr_mock_success PROPERTIES TIMEOUT 60)
set_tests_properties(nrr_mock_fallback PROPERTIES TIMEOUT 60)
set_tests_properties(
  lns_small_tier_fixed_suite
  PROPERTIES
    TIMEOUT 1200
    ENVIRONMENT
      "LNS_ENABLE_NRR=1;LNS_NRR_MINI_SOLVER=cbs;LNS_LOW_LEVEL_PLANNER=sipps;LNS_NRR_FALLBACK_TO_STANDARD=0;LNS_REPAIR_MAPFPC_SOLVER=cbs"
)
