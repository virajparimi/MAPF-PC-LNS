#include "lns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>

#include "lns_internal_helpers.hpp"
#include "nrr_boundary_windows.hpp"
#include "lns_repair_nrr_helpers.hpp"

bool LNS::runNeighborhoodReoptimizationRepair() {
  // Design reference: docs/mini_solver_repair_NRR_TAPFPC.md
  // Implementation plan:
  // 1) Select neighborhood/frozen agents and compute boundary windows.
  // 2) Build regret-biased slot matching proposal (Hungarian).
  // 3) Mini-solve, stitch neighborhood, and globally validate.
  namespace fs = std::filesystem;
  const Time::time_point attemptStart = Time::now();
  auto finalizeAttempt = [&](bool success) {
    const double elapsed = ((fsec)(Time::now() - attemptStart)).count();
    nrrStats_.runtimeSecSum += elapsed;
    if (success) {
      nrrStats_.success++;
    }
    return success;
  };
  auto fail = [&](const string& reason) {
    nrrStats_.failureReasonHistogram
        [lns_nrr_helpers::normalizeNrrFailureReason(reason)]++;
    PLOGW << "nrr_repair: failed (" << reason << "); fallback required\n";
    return finalizeAttempt(false);
  };

  const int numAgents = instance_.getAgentNum();
  const int numTasks = instance_.getTasksNum();
  if (numAgents <= 0 || numTasks <= 0) {
    return fail("empty_instance");
  }
  if ((int)previousSolution_.agents.size() != numAgents) {
    return fail("missing_previous_solution");
  }

  vector<int> destroyedTasks;
  vector<char> destroyedMask(numTasks, 0);
  destroyedTasks.reserve(lnsNeighborhood_.immutableRemovedTasks.size());
  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    const int task = conflict.task;
    if (task < 0 || task >= numTasks) {
      return fail("removed_task_out_of_range");
    }
    if (!destroyedMask[task]) {
      destroyedMask[task] = 1;
      destroyedTasks.push_back(task);
    }
  }
  if (destroyedTasks.empty()) {
    return fail("empty_destroyed_set");
  }

  nrrStats_.attempts++;
  nrrStats_.removedTasksSum += (int64_t)destroyedTasks.size();
  nrrStats_.removedTasksMax =
      std::max(nrrStats_.removedTasksMax, (int64_t)destroyedTasks.size());
  nrrStats_.removedTasksHistogram[(int)destroyedTasks.size()]++;

  vector<vector<int>> baseAssignments(numAgents);
  vector<char> seenTask(numTasks, 0);
  int assignedCount = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    baseAssignments[agent] = previousSolution_.agents[agent].taskAssignments;
    for (int task : baseAssignments[agent]) {
      if (task < 0 || task >= numTasks) {
        return fail("previous_assignment_out_of_range");
      }
      if (seenTask[task]) {
        return fail("previous_assignment_duplicate");
      }
      seenTask[task] = 1;
      assignedCount++;
    }
  }
  if (assignedCount != numTasks) {
    return fail("previous_assignment_incomplete");
  }

  // Remove destroyed tasks from the base assignment before slot construction.
  for (int agent = 0; agent < numAgents; agent++) {
    auto& q = baseAssignments[agent];
    q.erase(std::remove_if(q.begin(), q.end(),
                           [&](int task) { return destroyedMask[task] != 0; }),
            q.end());
  }
  // 1) Choose neighborhood agent set A' (owners + helper agents until cap).
  const int neighborhoodCap =
      std::min(numAgents, std::max(2, 2 * (int)destroyedTasks.size()));
  vector<int> neighborhoodAgents;
  vector<char> neighborhoodMask(numAgents, 0);
  auto addNeighborhoodAgent = [&](int agent) {
    if (agent < 0 || agent >= numAgents || neighborhoodMask[agent]) {
      return;
    }
    neighborhoodMask[agent] = 1;
    neighborhoodAgents.push_back(agent);
  };
  for (int task : destroyedTasks) {
    int owner = UNASSIGNED;
    if (task >= 0 && task < (int)previousSolution_.taskAgentMap.size()) {
      owner = previousSolution_.taskAgentMap[task];
    }
    if (owner != UNASSIGNED) {
      addNeighborhoodAgent(owner);
    }
  }
  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    if ((int)neighborhoodAgents.size() >= neighborhoodCap) {
      break;
    }
    addNeighborhoodAgent(conflict.agent);
  }
  if (neighborhoodAgents.empty()) {
    return fail("empty_neighborhood_agents");
  }

  if ((int)neighborhoodAgents.size() < neighborhoodCap) {
    vector<pair<long long, int>> helperScores;
    helperScores.reserve(numAgents);
    for (int agent = 0; agent < numAgents; agent++) {
      if (neighborhoodMask[agent]) {
        continue;
      }
      const int anchor = instance_.getStartLocationsRef()[agent];
      long long best = lns_nrr_helpers::kNrrInfCost;
      for (int task : destroyedTasks) {
        const int taskLoc = instance_.getTaskLocations(task);
        best = std::min(
            best, lns_nrr_helpers::distOrInf(instance_, anchor, taskLoc));
      }
      helperScores.emplace_back(best, agent);
    }
    std::sort(helperScores.begin(), helperScores.end(),
              [](const pair<long long, int>& lhs,
                 const pair<long long, int>& rhs) {
                if (lhs.first != rhs.first) {
                  return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
              });
    for (const auto& entry : helperScores) {
      if ((int)neighborhoodAgents.size() >= neighborhoodCap) {
        break;
      }
      addNeighborhoodAgent(entry.second);
    }
  }
  std::sort(neighborhoodAgents.begin(), neighborhoodAgents.end());

  nrrStats_.neighborhoodAgentsSum += (int64_t)neighborhoodAgents.size();
  nrrStats_.neighborhoodAgentsMax =
      std::max(nrrStats_.neighborhoodAgentsMax,
               (int64_t)neighborhoodAgents.size());
  nrrStats_.neighborhoodAgentsHistogram[(int)neighborhoodAgents.size()]++;

  vector<int> frozenAgents;
  frozenAgents.reserve(numAgents);
  for (int agent = 0; agent < numAgents; agent++) {
    if (!neighborhoodMask[agent]) {
      frozenAgents.push_back(agent);
    }
  }

  // 2/3) Build hard CT from frozen agents under existing occupancy semantics.
  ConstraintTable frozenCt(instance_.numOfCols, instance_.mapSize);
  for (int agent : frozenAgents) {
    const auto& path = previousSolution_.agents[agent].path;
    if (path.empty()) {
      return fail("frozen_agent_missing_path");
    }
    reservePathWithGoalPolicy(frozenCt, path, true, false);
    if (goalOccupationMode_ == "reposition_true" &&
        previousSolution_.agents[agent].terminalPathActive &&
        !previousSolution_.agents[agent].terminalPath.empty()) {
      frozenCt.addPath(previousSolution_.agents[agent].terminalPath, true);
    }
  }

  // Incumbent completion times τ_S(task).
  vector<int> incumbentCompletion(numTasks, -1);
  for (int agent = 0; agent < numAgents; agent++) {
    const auto& tasks = previousSolution_.agents[agent].taskAssignments;
    const auto& taskPaths = previousSolution_.agents[agent].taskPaths;
    if (tasks.size() != taskPaths.size()) {
      return fail("previous_assignment_path_mismatch");
    }
    for (int i = 0; i < (int)tasks.size(); i++) {
      const int task = tasks[i];
      if (task < 0 || task >= numTasks || taskPaths[i].empty()) {
        return fail("invalid_previous_task_path");
      }
      incumbentCompletion[task] = taskPaths[i].endTime();
    }
  }

  // 4) Boundary precedence windows from fixed<->destroyed edges.
  const auto boundaryWindows = mapf_pc_lns::nrr::computeBoundaryWindows(
      numTasks, instance_.getInputPrecedenceConstraintsRef(), destroyedMask,
      incumbentCompletion);
  if (!boundaryWindows.feasible) {
    return fail("boundary_windows_" + boundaryWindows.failureReason);
  }

  // 5) Build precedence-safe iterative insertion proposal.
  vector<vector<int>> proposedAssignments = baseAssignments;
  {
    string proposalFailureReason;
    if (!lns_nrr_helpers::buildIterativeProposal(
            instance_, numAgents, numTasks, destroyedTasks, destroyedMask,
            neighborhoodAgents, boundaryWindows, frozenCt, proposedAssignments,
            proposalFailureReason)) {
      return fail(proposalFailureReason);
    }
  }

  {
    string temporalPrecheckFailureReason;
    if (!lns_nrr_helpers::runTemporalPrecheck(
            instance_, numTasks, neighborhoodAgents, destroyedTasks,
            proposedAssignments, incumbentCompletion, boundaryWindows,
            temporalPrecheckFailureReason)) {
      return fail(temporalPrecheckFailureReason);
    }
  }

  int neighborhoodGoals = 0;
  for (int agent : neighborhoodAgents) {
    neighborhoodGoals += (int)proposedAssignments[agent].size();
  }
  string miniSolver = "cbs";
  if (nrrMiniSolver_ == "pbs") {
    miniSolver = "pbs";
  }
  if (miniSolver == "pbs") {
    nrrStats_.solverPbs++;
  } else {
    nrrStats_.solverCbs++;
  }

  const int solverTimeoutSec =
      std::max(1, std::min(repairMapfpcTimeoutSec_,
                           (int)std::floor(std::max(
                               1.0, remainingRuntimeBudgetSec()))));
  if (solverTimeoutSec <= 0) {
    return fail("mini_solver_budget_exhausted");
  }

  std::error_code tempDirEc;
  fs::path tempDir = fs::temp_directory_path(tempDirEc);
  if (tempDirEc || tempDir.empty()) {
    tempDir = fs::current_path(tempDirEc);
  }
  if (tempDirEc || tempDir.empty()) {
    return fail("temp_directory_unavailable");
  }
  const string token =
      std::to_string(seed_) + "_" +
      std::to_string((long long)Time::now().time_since_epoch().count());
  const bool keepNrrInputs = std::getenv("MAPF_PC_KEEP_NRR_INPUTS") != nullptr;
  const fs::path assignmentPath = tempDir / ("mapfpc_nrr_assign_" + token + ".txt");
  const fs::path mutableAgentsPath =
      tempDir / ("mapfpc_nrr_mutable_" + token + ".txt");
  const fs::path mutableTasksPath =
      tempDir / ("mapfpc_nrr_mutable_tasks_" + token + ".txt");
  const fs::path initialPathsPath =
      tempDir / ("mapfpc_nrr_paths_" + token + ".txt");
  const fs::path manifestPath =
      tempDir / ("mapfpc_nrr_manifest_" + token + ".txt");
  auto cleanupTemp = [&]() {
    if (keepNrrInputs) {
      return;
    }
    std::error_code rmEc;
    fs::remove(assignmentPath, rmEc);
    fs::remove(mutableAgentsPath, rmEc);
    fs::remove(mutableTasksPath, rmEc);
    fs::remove(initialPathsPath, rmEc);
    fs::remove(manifestPath, rmEc);
  };

  string assignmentError;
  if (!writeMAPFPCAssignmentFile(proposedAssignments, assignmentPath.string(),
                                 assignmentError)) {
    cleanupTemp();
    return fail("assignment_file_write_failed:" + assignmentError);
  }

  {
    std::ofstream outMutable(mutableAgentsPath);
    if (!outMutable.is_open()) {
      cleanupTemp();
      return fail("mutable_agents_file_open_failed");
    }
    outMutable << "MUTABLE_AGENTS\n";
    for (int i = 0; i < (int)neighborhoodAgents.size(); i++) {
      if (i > 0) {
        outMutable << ' ';
      }
      outMutable << neighborhoodAgents[i];
    }
    outMutable << "\n";
    outMutable.flush();
    if (!outMutable.good()) {
      cleanupTemp();
      return fail("mutable_agents_file_write_failed");
    }
  }

  {
    std::ofstream outMutableTasks(mutableTasksPath);
    if (!outMutableTasks.is_open()) {
      cleanupTemp();
      return fail("mutable_tasks_file_open_failed");
    }
    outMutableTasks << "MUTABLE_GLOBAL_TASKS\n";
    for (int i = 0; i < (int)destroyedTasks.size(); i++) {
      if (i > 0) {
        outMutableTasks << ' ';
      }
      outMutableTasks << destroyedTasks[i];
    }
    outMutableTasks << "\n";
    outMutableTasks.flush();
    if (!outMutableTasks.good()) {
      cleanupTemp();
      return fail("mutable_tasks_file_write_failed");
    }
  }

  vector<vector<int>> pathLocations(numAgents);
  vector<vector<int>> pathTimestamps(numAgents);
  for (int agent = 0; agent < numAgents; agent++) {
    const auto& joined = previousSolution_.agents[agent].path;
    if (joined.empty()) {
      cleanupTemp();
      return fail("seed_initial_paths_missing_joined_path");
    }
    pathLocations[agent].reserve(joined.path.size());
    for (const auto& entry : joined.path) {
      pathLocations[agent].push_back(entry.location);
    }

    const auto& previousTasks = previousSolution_.agents[agent].taskAssignments;
    const auto& taskPaths = previousSolution_.agents[agent].taskPaths;
    if ((int)taskPaths.size() == (int)previousTasks.size()) {
      pathTimestamps[agent].reserve(taskPaths.size());
      for (const auto& segment : taskPaths) {
        if (segment.empty()) {
          cleanupTemp();
          return fail("seed_initial_paths_empty_task_segment");
        }
        pathTimestamps[agent].push_back(segment.endTime());
      }
    } else if ((int)joined.timeStamps.size() == (int)previousTasks.size()) {
      pathTimestamps[agent] = joined.timeStamps;
    } else {
      cleanupTemp();
      return fail("seed_initial_paths_timestamp_mismatch");
    }

    if (proposedAssignments[agent].empty() && pathTimestamps[agent].empty()) {
      pathTimestamps[agent].push_back(0);
    }
    int prevTs = -1;
    for (int ts : pathTimestamps[agent]) {
      if (ts < 0 || ts >= (int)pathLocations[agent].size() || ts < prevTs) {
        cleanupTemp();
        return fail("seed_initial_paths_invalid_timestamp");
      }
      prevTs = ts;
    }
  }

  {
    std::ofstream outPaths(initialPathsPath);
    if (!outPaths.is_open()) {
      cleanupTemp();
      return fail("seed_initial_paths_open_failed");
    }
    outPaths << "AGENT_PATHS\n";
    for (int agent = 0; agent < numAgents; agent++) {
      outPaths << "Agent " << agent << "\n";
      outPaths << "locations:";
      for (int loc : pathLocations[agent]) {
        outPaths << " " << loc;
      }
      outPaths << "\n";
      outPaths << "timestamps:";
      for (int ts : pathTimestamps[agent]) {
        outPaths << " " << ts;
      }
      outPaths << "\n";
    }
    outPaths.flush();
    if (!outPaths.good()) {
      cleanupTemp();
      return fail("seed_initial_paths_write_failed");
    }
  }

  {
    std::ofstream manifest(manifestPath);
    if (!manifest.is_open()) {
      cleanupTemp();
      return fail("nrr_manifest_open_failed");
    }
    manifest << "seed=" << seed_ << "\n";
    manifest << "mini_solver=" << miniSolver << "\n";
    manifest << "destroyed_count=" << destroyedTasks.size() << "\n";
    manifest << "destroyed_tasks=";
    for (int i = 0; i < (int)destroyedTasks.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << destroyedTasks[i];
    }
    manifest << "\n";
    manifest << "neighborhood_agent_count=" << neighborhoodAgents.size() << "\n";
    manifest << "neighborhood_agents=";
    for (int i = 0; i < (int)neighborhoodAgents.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << neighborhoodAgents[i];
    }
    manifest << "\n";
    manifest << "frozen_agent_count=" << frozenAgents.size() << "\n";
    manifest << "frozen_agents=";
    for (int i = 0; i < (int)frozenAgents.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << frozenAgents[i];
    }
    manifest << "\n";
    manifest << "assignment_file=" << assignmentPath.string() << "\n";
    manifest << "mutable_agents_file=" << mutableAgentsPath.string() << "\n";
    manifest << "mutable_tasks_file=" << mutableTasksPath.string() << "\n";
    manifest << "initial_paths_file=" << initialPathsPath.string() << "\n";
    manifest.flush();
    if (!manifest.good()) {
      cleanupTemp();
      return fail("nrr_manifest_write_failed");
    }
  }

  const Solution backupSolution = solution_;
  const string sourceLabel =
      "repair_mapfpc_nrr(" + miniSolver + ",removed=" +
      std::to_string(destroyedTasks.size()) + ",A'=" +
      std::to_string(neighborhoodAgents.size()) + ")";
  PLOGI << "nrr_repair: start (removed=" << destroyedTasks.size()
        << ", neighborhood_agents=" << neighborhoodAgents.size()
        << ", frozen_agents=" << frozenAgents.size()
        << ", mini_solver=" << miniSolver
        << ", timeout_sec=" << solverTimeoutSec << ")\n";

  const bool miniSolved = runMAPFPCForAgentFile(
      instance_.getAgentTaskFName(), miniSolver, solverTimeoutSec, sourceLabel,
      assignmentPath.string(), mutableAgentsPath.string(),
      initialPathsPath.string(), mutableTasksPath.string(), "sipps");
  if (keepNrrInputs) {
    PLOGE << "nrr_repair: kept inputs for verification: manifest='"
          << manifestPath.string() << "' assignment='"
          << assignmentPath.string() << "' mutable='"
          << mutableAgentsPath.string() << "' mutable_tasks='"
          << mutableTasksPath.string() << "' initial_paths='"
          << initialPathsPath.string() << "'\n";
  }
  cleanupTemp();
  if (!miniSolved) {
    solution_ = backupSolution;
    return fail("mini_solver_failed");
  }

  // 7) Stitch: keep frozen agents identical to incumbent.
  for (int frozenAgent : frozenAgents) {
    solution_.agents[frozenAgent] = previousSolution_.agents[frozenAgent];
  }

  // Strict NRR commit gate: mutable-vs-frozen soft conflicts must be zero.
  // This keeps global validity constraints explicit even when mini-solver uses
  // soft conflict minimization internally.
  auto countCrossSetConflicts = [&](const vector<int>& mutableAgents,
                                    const vector<int>& frozenAgentsSet,
                                    bool includeTerminal) -> int64_t {
    int64_t conflicts = 0;
    for (int mutableAgent : mutableAgents) {
      for (int frozenAgent : frozenAgentsSet) {
        const int horizon = std::max(
            getAgentOccupancyHorizon(mutableAgent, includeTerminal),
            getAgentOccupancyHorizon(frozenAgent, includeTerminal));
        for (int t = 0; t < horizon; t++) {
          const int mutableLoc =
              getAgentLocationAt(mutableAgent, t, includeTerminal);
          const int frozenLoc =
              getAgentLocationAt(frozenAgent, t, includeTerminal);
          if (mutableLoc != UNDEFINED && frozenLoc != UNDEFINED &&
              mutableLoc == frozenLoc) {
            conflicts++;
          }
          if (t + 1 >= horizon) {
            continue;
          }
          const int mutableNext =
              getAgentLocationAt(mutableAgent, t + 1, includeTerminal);
          const int frozenNext =
              getAgentLocationAt(frozenAgent, t + 1, includeTerminal);
          if (mutableLoc == UNDEFINED || frozenLoc == UNDEFINED ||
              mutableNext == UNDEFINED || frozenNext == UNDEFINED) {
            continue;
          }
          if (mutableLoc != mutableNext && frozenLoc != frozenNext &&
              mutableLoc == frozenNext && frozenLoc == mutableNext) {
            conflicts++;
          }
        }
      }
    }
    return conflicts;
  };
  const bool includeTerminalForSoftCheck =
      (goalOccupationMode_ == "reposition_true");
  const int64_t softConflicts = countCrossSetConflicts(
      neighborhoodAgents, frozenAgents, includeTerminalForSoftCheck);
  if (softConflicts > 0) {
    solution_ = backupSolution;
    return fail("soft_conflicts_nonzero:" + std::to_string(softConflicts));
  }

  solution_.taskAgentMap.assign(numTasks, UNASSIGNED);
  for (int agent = 0; agent < numAgents; agent++) {
    for (int task : solution_.agents[agent].taskAssignments) {
      if (task < 0 || task >= numTasks) {
        solution_ = backupSolution;
        return fail("stitched_assignment_out_of_range");
      }
      if (solution_.taskAgentMap[task] != UNASSIGNED) {
        solution_ = backupSolution;
        return fail("stitched_assignment_duplicate");
      }
      solution_.taskAgentMap[task] = agent;
    }
  }
  for (int task = 0; task < numTasks; task++) {
    if (solution_.taskAgentMap[task] == UNASSIGNED) {
      solution_ = backupSolution;
      return fail("stitched_assignment_incomplete");
    }
  }

  if (goalOccupationMode_ == "reposition_true") {
    vector<int> terminalReplanAgents = selectTerminalReplanAgents(neighborhoodAgents);
    if (!terminalReplanAgents.empty() &&
        !planTerminalReposition(terminalReplanAgents, false)) {
      solution_ = backupSolution;
      return fail("terminal_replan_failed_after_stitch");
    }
  }

  long long recomputedSoc = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    recomputedSoc +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts = clampSocToInt(recomputedSoc, "nrr_repair");

  // 8) Global validate with current collision + precedence semantics.
  const bool previousTerminalValidationFlag = useTerminalPathsInValidation_;
  useTerminalPathsInValidation_ = (goalOccupationMode_ == "reposition_true");
  ValidationStats stitchedStats;
  const bool stitchedValid = validateSolution(nullptr, &stitchedStats);
  useTerminalPathsInValidation_ = previousTerminalValidationFlag;
  if (!stitchedValid) {
    nrrStats_.stitchedInvalidAttempts++;
    nrrStats_.stitchedInvalidPrecedenceViolations +=
        stitchedStats.precedenceViolations;
    nrrStats_.stitchedInvalidVertexCollisions += stitchedStats.vertexCollisions;
    nrrStats_.stitchedInvalidEdgeSwapCollisions +=
        stitchedStats.edgeSwapCollisions;
    nrrStats_.stitchedInvalidStructuralViolations +=
        stitchedStats.structuralViolations;
    solution_ = backupSolution;
    return fail("stitched_solution_invalid");
  }

  PLOGI << "nrr_repair: success (mini_solver=" << miniSolver
        << ", removed=" << destroyedTasks.size()
        << ", neighborhood_agents=" << neighborhoodAgents.size()
        << ", soc=" << solution_.sumOfCosts << ")\n";
  return finalizeAttempt(true);
}
