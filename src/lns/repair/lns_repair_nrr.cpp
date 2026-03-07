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
#include "lns_temp_file_guard.hpp"

bool LNS::runNeighborhoodReoptimizationRepair() {
  // Design reference: docs/mini_solver_repair_NRR_TAPFPC.md
  // Implementation plan:
  // 1) Select neighborhood/frozen agents and compute boundary windows.
  // 2) Build regret-biased slot matching proposal (Hungarian).
  // 3) Mini-solve, stitch neighborhood, and globally validate.
  namespace fs = std::filesystem;
  const Time::time_point attemptStart = Time::now();
  const bool collectNrrDetailedTiming = []() {
    const char* timingLevel = std::getenv("MAPF_PC_TIMING_LEVEL");
    return timingLevel != nullptr && std::string(timingLevel) == "full";
  }();
  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };
  double attemptTimeSetupSec = 0.0;
  double attemptTimeBoundaryWindowsSec = 0.0;
  double attemptTimeProposalBuildSec = 0.0;
  double attemptTimeTemporalPrecheckSec = 0.0;
  double attemptTimeInputBuildSec = 0.0;
  double attemptTimeMiniSolverSec = 0.0;
  double attemptTimeStitchSec = 0.0;
  double attemptTimeTaskMapRebuildSec = 0.0;
  double attemptTimeTerminalReplanSec = 0.0;
  double attemptTimeRecomputeObjectiveSec = 0.0;
  double attemptTimeValidationSec = 0.0;
  enum class NrrPhase {
    none,
    setup,
    boundaryWindows,
    proposalBuild,
    temporalPrecheck,
    inputBuild,
    miniSolver,
    stitch,
    taskMapRebuild,
    terminalReplan,
    recomputeObjective,
    validation,
  };
  NrrPhase activePhase = NrrPhase::setup;
  Time::time_point activePhaseStart = Time::now();
  auto accumulatePhaseElapsed = [&](NrrPhase phase, double elapsed) {
    switch (phase) {
      case NrrPhase::setup:
        attemptTimeSetupSec += elapsed;
        break;
      case NrrPhase::boundaryWindows:
        attemptTimeBoundaryWindowsSec += elapsed;
        break;
      case NrrPhase::proposalBuild:
        attemptTimeProposalBuildSec += elapsed;
        break;
      case NrrPhase::temporalPrecheck:
        attemptTimeTemporalPrecheckSec += elapsed;
        break;
      case NrrPhase::inputBuild:
        attemptTimeInputBuildSec += elapsed;
        break;
      case NrrPhase::miniSolver:
        attemptTimeMiniSolverSec += elapsed;
        break;
      case NrrPhase::stitch:
        attemptTimeStitchSec += elapsed;
        break;
      case NrrPhase::taskMapRebuild:
        attemptTimeTaskMapRebuildSec += elapsed;
        break;
      case NrrPhase::terminalReplan:
        attemptTimeTerminalReplanSec += elapsed;
        break;
      case NrrPhase::recomputeObjective:
        attemptTimeRecomputeObjectiveSec += elapsed;
        break;
      case NrrPhase::validation:
        attemptTimeValidationSec += elapsed;
        break;
      case NrrPhase::none:
        break;
    }
  };
  auto closeActivePhase = [&]() {
    if (!collectNrrDetailedTiming) {
      return;
    }
    if (activePhase == NrrPhase::none) {
      return;
    }
    accumulatePhaseElapsed(activePhase, elapsedSecSince(activePhaseStart));
    activePhase = NrrPhase::none;
  };
  auto beginPhase = [&](NrrPhase nextPhase) {
    if (!collectNrrDetailedTiming) {
      return;
    }
    closeActivePhase();
    activePhase = nextPhase;
    activePhaseStart = Time::now();
  };
  auto finalizeAttempt = [&](bool success) {
    closeActivePhase();
    const double elapsed = ((fsec)(Time::now() - attemptStart)).count();
    nrrStats_.runtimeSecSum += elapsed;
    nrrStats_.timeSetupSecSum += attemptTimeSetupSec;
    nrrStats_.timeBoundaryWindowsSecSum += attemptTimeBoundaryWindowsSec;
    nrrStats_.timeProposalBuildSecSum += attemptTimeProposalBuildSec;
    nrrStats_.timeTemporalPrecheckSecSum += attemptTimeTemporalPrecheckSec;
    nrrStats_.timeInputBuildSecSum += attemptTimeInputBuildSec;
    nrrStats_.timeMiniSolverSecSum += attemptTimeMiniSolverSec;
    nrrStats_.timeStitchSecSum += attemptTimeStitchSec;
    nrrStats_.timeTaskMapRebuildSecSum += attemptTimeTaskMapRebuildSec;
    nrrStats_.timeTerminalReplanSecSum += attemptTimeTerminalReplanSec;
    nrrStats_.timeRecomputeObjectiveSecSum += attemptTimeRecomputeObjectiveSec;
    nrrStats_.timeValidationSecSum += attemptTimeValidationSec;
    if (success) {
      nrrStats_.success++;
    }
    return success;
  };
  auto fail = [&](const string& reason) {
    closeActivePhase();
    const string normalizedReason =
        lns_nrr_helpers::normalizeNrrFailureReason(reason);
    nrrStats_.failureReasonHistogram[normalizedReason]++;
    if (nrrGlobalReassign_ &&
        normalizedReason == "no_feasible_slot_for_destroyed_task_iterative_global") {
      nrrStats_.globalNoFeasibleSlotCount++;
    }
    PLOGW << "nrr_repair: failed (" << reason << "); fallback required\n";
    return finalizeAttempt(false);
  };
  nrrStats_.calls++;

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
  const bool useGlobalReassign = nrrGlobalReassign_;

  // 1) Choose mutable seed agents.
  vector<int> initialMutableAgents;
  vector<char> initialMutableMask(numAgents, 0);
  auto addInitialMutableAgent = [&](int agent) {
    if (agent < 0 || agent >= numAgents || initialMutableMask[agent]) {
      return;
    }
    initialMutableMask[agent] = 1;
    initialMutableAgents.push_back(agent);
  };

  for (int task : destroyedTasks) {
    int owner = UNASSIGNED;
    if (task >= 0 && task < (int)previousSolution_.taskAgentMap.size()) {
      owner = previousSolution_.taskAgentMap[task];
    }
    if (owner != UNASSIGNED) {
      addInitialMutableAgent(owner);
    }
  }
  if (!useGlobalReassign) {
    const int neighborhoodCap =
        std::min(numAgents, std::max(2, 2 * (int)destroyedTasks.size()));
    for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
      if ((int)initialMutableAgents.size() >= neighborhoodCap) {
        break;
      }
      addInitialMutableAgent(conflict.agent);
    }

    if ((int)initialMutableAgents.size() < neighborhoodCap) {
      vector<pair<long long, int>> helperScores;
      helperScores.reserve(numAgents);
      for (int agent = 0; agent < numAgents; agent++) {
        if (initialMutableMask[agent]) {
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
        if ((int)initialMutableAgents.size() >= neighborhoodCap) {
          break;
        }
        addInitialMutableAgent(entry.second);
      }
    }
  }

  if (initialMutableAgents.empty()) {
    return fail("empty_neighborhood_agents");
  }
  std::sort(initialMutableAgents.begin(), initialMutableAgents.end());
  initialMutableAgents.erase(
      std::unique(initialMutableAgents.begin(), initialMutableAgents.end()),
      initialMutableAgents.end());

  vector<int> mutableAgents = initialMutableAgents;
  vector<char> mutableMask(numAgents, 0);
  for (int agent : mutableAgents) {
    if (agent >= 0 && agent < numAgents) {
      mutableMask[agent] = 1;
    }
  }

  vector<int> frozenAgents;
  frozenAgents.reserve(numAgents);
  for (int agent = 0; agent < numAgents; agent++) {
    if (!mutableMask[agent]) {
      frozenAgents.push_back(agent);
    }
  }

  ConstraintTable frozenCt(instance_.numOfCols, instance_.mapSize);
  if (!useGlobalReassign) {
    // 2/3) Build hard CT from frozen agents under existing occupancy semantics.
    for (int agent : frozenAgents) {
      const auto& path = previousSolution_.agents[agent].path;
      if (path.empty()) {
        return fail("frozen_agent_missing_path");
      }
      reservePathWithGoalPolicy(frozenCt, path, true, false);
      if (isGoalOccupationRepositionTrue() &&
          previousSolution_.agents[agent].terminalPathActive &&
          !previousSolution_.agents[agent].terminalPath.empty()) {
        frozenCt.addPath(previousSolution_.agents[agent].terminalPath, true);
      }
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
  beginPhase(NrrPhase::boundaryWindows);
  const auto boundaryWindows = mapf_pc_lns::nrr::computeBoundaryWindows(
      numTasks, instance_.getInputPrecedenceConstraintsRef(), destroyedMask,
      incumbentCompletion);
  if (!boundaryWindows.feasible) {
    return fail("boundary_windows_" + boundaryWindows.failureReason);
  }

  // 5) Build precedence-safe iterative insertion proposal.
  const Time::time_point proposalBuildStart = Time::now();
  beginPhase(NrrPhase::proposalBuild);
  vector<vector<int>> proposedAssignments = baseAssignments;
  lns_nrr_helpers::ProposalMutationTrace proposalTrace;
  vector<int> mutableGlobalTasks = destroyedTasks;
  {
    string proposalFailureReason;
    if (!useGlobalReassign) {
      if (!lns_nrr_helpers::buildIterativeProposal(
              instance_, numAgents, numTasks, destroyedTasks, destroyedMask,
              mutableAgents, boundaryWindows, frozenCt, proposedAssignments,
              proposalFailureReason)) {
        return fail(proposalFailureReason);
      }
    } else {
      vector<int> candidateAgents(numAgents);
      std::iota(candidateAgents.begin(), candidateAgents.end(), 0);
      vector<int> incumbentTaskOwnerByTask(numTasks, UNASSIGNED);
      if ((int)previousSolution_.taskAgentMap.size() == numTasks) {
        incumbentTaskOwnerByTask = previousSolution_.taskAgentMap;
      } else {
        for (int agent = 0; agent < numAgents; agent++) {
          for (int task : previousSolution_.agents[agent].taskAssignments) {
            if (task >= 0 && task < numTasks) {
              incumbentTaskOwnerByTask[task] = agent;
            }
          }
        }
      }
      lns_nrr_helpers::NrrFrozenOccupancyIndex frozenOccupancy;
      frozenOccupancy.buildFromPreviousSolution(
          instance_, previousSolution_, isGoalOccupationStay());
      if (!lns_nrr_helpers::buildIterativeProposalGlobal(
              instance_, numAgents, numTasks, destroyedTasks, destroyedMask,
              candidateAgents, mutableAgents, incumbentTaskOwnerByTask,
              boundaryWindows, frozenOccupancy, proposedAssignments,
              proposalTrace, proposalFailureReason)) {
        return fail(proposalFailureReason);
      }

      if (!proposalTrace.touchedAgentsFinal.empty()) {
        mutableAgents = proposalTrace.touchedAgentsFinal;
      }
      mutableMask.assign(numAgents, 0);
      for (int agent : mutableAgents) {
        if (agent >= 0 && agent < numAgents) {
          mutableMask[agent] = 1;
        }
      }
      frozenAgents.clear();
      for (int agent = 0; agent < numAgents; agent++) {
        if (!mutableMask[agent]) {
          frozenAgents.push_back(agent);
        }
      }

      std::set<int> patchedSet(proposalTrace.patchedImmediateSuccessorTasks.begin(),
                               proposalTrace.patchedImmediateSuccessorTasks.end());
      for (int task : destroyedTasks) {
        int owner = UNASSIGNED;
        if (task >= 0 && task < (int)previousSolution_.taskAgentMap.size()) {
          owner = previousSolution_.taskAgentMap[task];
        }
        if (owner == UNASSIGNED || owner < 0 || owner >= numAgents) {
          continue;
        }
        const auto& prevQueue = previousSolution_.agents[owner].taskAssignments;
        auto it = std::find(prevQueue.begin(), prevQueue.end(), task);
        if (it == prevQueue.end()) {
          continue;
        }
        ++it;
        if (it != prevQueue.end() && *it >= 0 && *it < numTasks &&
            !destroyedMask[*it]) {
          patchedSet.insert(*it);
        }
      }
      for (int agent : mutableAgents) {
        if (agent < 0 || agent >= numAgents) {
          continue;
        }
        const auto& queue = proposedAssignments[agent];
        for (int i = 0; i + 1 < (int)queue.size(); i++) {
          const int task = queue[i];
          if (task < 0 || task >= numTasks || !destroyedMask[task]) {
            continue;
          }
          const int successor = queue[i + 1];
          if (successor >= 0 && successor < numTasks &&
              !destroyedMask[successor]) {
            patchedSet.insert(successor);
          }
        }
      }
      proposalTrace.patchedImmediateSuccessorTasks.assign(patchedSet.begin(),
                                                          patchedSet.end());
      for (int patchedTask : proposalTrace.patchedImmediateSuccessorTasks) {
        if (std::find(mutableGlobalTasks.begin(), mutableGlobalTasks.end(),
                      patchedTask) == mutableGlobalTasks.end()) {
          mutableGlobalTasks.push_back(patchedTask);
        }
      }
      std::sort(mutableGlobalTasks.begin(), mutableGlobalTasks.end());
      mutableGlobalTasks.erase(
          std::unique(mutableGlobalTasks.begin(), mutableGlobalTasks.end()),
          mutableGlobalTasks.end());

      nrrStats_.globalSlotEvalCount += proposalTrace.globalSlotEvaluations;
      nrrStats_.globalDemotionsCount += proposalTrace.globalDemotions;
      nrrStats_.globalPatchedTasksSum +=
          (int64_t)proposalTrace.patchedImmediateSuccessorTasks.size();
      if (proposalTrace.globalNoFeasibleSlotCount > 0) {
        nrrStats_.globalNoFeasibleSlotCount +=
            proposalTrace.globalNoFeasibleSlotCount;
      }
      nrrStats_.globalTouchedAgentsSum += (int64_t)mutableAgents.size();
      nrrStats_.globalTouchedAgentsMax =
          std::max(nrrStats_.globalTouchedAgentsMax, (int64_t)mutableAgents.size());
      nrrStats_.globalTouchedAgentsHistogram[(int)mutableAgents.size()]++;
    }
    if (mutableAgents.empty()) {
      return fail("empty_mutable_agents_after_proposal");
    }
    if ((int)mutableGlobalTasks.size() < (int)destroyedTasks.size()) {
      return fail("mutable_tasks_less_than_destroyed");
    }
    if (useGlobalReassign) {
      for (int task : destroyedTasks) {
        if (std::find(mutableGlobalTasks.begin(), mutableGlobalTasks.end(), task) ==
            mutableGlobalTasks.end()) {
          return fail("destroyed_task_missing_from_mutable_tasks");
        }
      }
    }
    nrrStats_.neighborhoodAgentsSum += (int64_t)mutableAgents.size();
    nrrStats_.neighborhoodAgentsMax =
        std::max(nrrStats_.neighborhoodAgentsMax, (int64_t)mutableAgents.size());
    nrrStats_.neighborhoodAgentsHistogram[(int)mutableAgents.size()]++;
  }
  if (!collectNrrDetailedTiming) {
    attemptTimeProposalBuildSec += elapsedSecSince(proposalBuildStart);
  }

  beginPhase(NrrPhase::temporalPrecheck);
  {
    string temporalPrecheckFailureReason;
    if (!lns_nrr_helpers::runTemporalPrecheck(
            instance_, numTasks, mutableAgents, destroyedTasks,
            proposedAssignments, incumbentCompletion, boundaryWindows,
            temporalPrecheckFailureReason)) {
      return fail(temporalPrecheckFailureReason);
    }
  }

  beginPhase(NrrPhase::inputBuild);
  string miniSolver = "cbs";
  if (isNrrMiniSolverPbs()) {
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
  lns_temp_file::ScopedPathCleanup tempCleanup(!keepNrrInputs);
  tempCleanup.add(assignmentPath);
  tempCleanup.add(mutableAgentsPath);
  tempCleanup.add(mutableTasksPath);
  tempCleanup.add(initialPathsPath);
  tempCleanup.add(manifestPath);

  string assignmentError;
  if (!writeMAPFPCAssignmentFile(proposedAssignments, assignmentPath.string(),
                                 assignmentError)) {
    return fail("assignment_file_write_failed:" + assignmentError);
  }

  {
    std::ofstream outMutable(mutableAgentsPath);
    if (!outMutable.is_open()) {
      return fail("mutable_agents_file_open_failed");
    }
    outMutable << "MUTABLE_AGENTS\n";
    for (int i = 0; i < (int)mutableAgents.size(); i++) {
      if (i > 0) {
        outMutable << ' ';
      }
      outMutable << mutableAgents[i];
    }
    outMutable << "\n";
    outMutable.flush();
    if (!outMutable.good()) {
      return fail("mutable_agents_file_write_failed");
    }
  }

  {
    std::ofstream outMutableTasks(mutableTasksPath);
    if (!outMutableTasks.is_open()) {
      return fail("mutable_tasks_file_open_failed");
    }
    outMutableTasks << "MUTABLE_GLOBAL_TASKS\n";
    for (int i = 0; i < (int)mutableGlobalTasks.size(); i++) {
      if (i > 0) {
        outMutableTasks << ' ';
      }
      outMutableTasks << mutableGlobalTasks[i];
    }
    outMutableTasks << "\n";
    outMutableTasks.flush();
    if (!outMutableTasks.good()) {
      return fail("mutable_tasks_file_write_failed");
    }
  }

  vector<vector<int>> pathLocations(numAgents);
  vector<vector<int>> pathTimestamps(numAgents);
  for (int agent = 0; agent < numAgents; agent++) {
    const auto& joined = previousSolution_.agents[agent].path;
    if (joined.empty()) {
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
          return fail("seed_initial_paths_empty_task_segment");
        }
        pathTimestamps[agent].push_back(segment.endTime());
      }
    } else if ((int)joined.timeStamps.size() == (int)previousTasks.size()) {
      pathTimestamps[agent] = joined.timeStamps;
    } else {
      return fail("seed_initial_paths_timestamp_mismatch");
    }

    if (proposedAssignments[agent].empty() && pathTimestamps[agent].empty()) {
      pathTimestamps[agent].push_back(0);
    }
    int prevTs = -1;
    for (int ts : pathTimestamps[agent]) {
      if (ts < 0 || ts >= (int)pathLocations[agent].size() || ts < prevTs) {
        return fail("seed_initial_paths_invalid_timestamp");
      }
      prevTs = ts;
    }
  }

  {
    std::ofstream outPaths(initialPathsPath);
    if (!outPaths.is_open()) {
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
      return fail("seed_initial_paths_write_failed");
    }
  }

  {
    std::ofstream manifest(manifestPath);
    if (!manifest.is_open()) {
      return fail("nrr_manifest_open_failed");
    }
    manifest << "seed=" << seed_ << "\n";
    manifest << "mini_solver=" << miniSolver << "\n";
    manifest << "cat_backend=" << nrrCatBackend_ << "\n";
    manifest << "global_reassign_mode=" << (useGlobalReassign ? 1 : 0) << "\n";
    manifest << "destroyed_count=" << destroyedTasks.size() << "\n";
    manifest << "destroyed_tasks=";
    for (int i = 0; i < (int)destroyedTasks.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << destroyedTasks[i];
    }
    manifest << "\n";
    manifest << "initial_touched_agents=";
    for (int i = 0; i < (int)initialMutableAgents.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << initialMutableAgents[i];
    }
    manifest << "\n";
    manifest << "final_touched_agents=";
    for (int i = 0; i < (int)mutableAgents.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << mutableAgents[i];
    }
    manifest << "\n";
    manifest << "patched_successor_tasks=";
    for (int i = 0; i < (int)proposalTrace.patchedImmediateSuccessorTasks.size();
         i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << proposalTrace.patchedImmediateSuccessorTasks[i];
    }
    manifest << "\n";
    manifest << "owner_changes=";
    for (int i = 0; i < (int)proposalTrace.destroyedTaskOwnerChanges.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      const auto& change = proposalTrace.destroyedTaskOwnerChanges[i];
      manifest << change.task << ":" << change.oldOwner << "->"
               << change.newOwner;
    }
    manifest << "\n";
    manifest << "neighborhood_agent_count=" << mutableAgents.size() << "\n";
    manifest << "neighborhood_agents=";
    for (int i = 0; i < (int)mutableAgents.size(); i++) {
      if (i > 0) {
        manifest << " ";
      }
      manifest << mutableAgents[i];
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
      return fail("nrr_manifest_write_failed");
    }
  }

  const Solution backupSolution = solution_;
  const string sourceLabel =
      "repair_mapfpc_nrr(" + miniSolver + ",removed=" +
      std::to_string(destroyedTasks.size()) + ",A'=" +
      std::to_string(mutableAgents.size()) + ")";
  PLOGI << "nrr_repair: start (removed=" << destroyedTasks.size()
        << ", neighborhood_agents=" << mutableAgents.size()
        << ", frozen_agents=" << frozenAgents.size()
        << ", mini_solver=" << miniSolver
        << ", cat_backend=" << nrrCatBackend_
        << ", timeout_sec=" << solverTimeoutSec << ")\n";

  const Time::time_point miniSolverStart = Time::now();
  beginPhase(NrrPhase::miniSolver);
  nrrStats_.miniSolverCalls++;
  const bool miniSolved = runMAPFPCForAgentFile(
      instance_.getAgentTaskFName(), miniSolver, solverTimeoutSec, sourceLabel,
      assignmentPath.string(), mutableAgentsPath.string(),
      initialPathsPath.string(), mutableTasksPath.string(), "sipps",
      nrrCatBackend_, 1);
  if (keepNrrInputs) {
    PLOGE << "nrr_repair: kept inputs for verification: manifest='"
          << manifestPath.string() << "' assignment='"
          << assignmentPath.string() << "' mutable='"
          << mutableAgentsPath.string() << "' mutable_tasks='"
          << mutableTasksPath.string() << "' initial_paths='"
          << initialPathsPath.string() << "'\n";
  }
  if (!collectNrrDetailedTiming) {
    attemptTimeMiniSolverSec += elapsedSecSince(miniSolverStart);
  }
  if (!miniSolved) {
    solution_ = backupSolution;
    invalidateCurrentTaskAssignmentIndexCache();
    return fail("mini_solver_failed");
  }

  // 7) Stitch: keep frozen agents identical to incumbent.
  beginPhase(NrrPhase::stitch);
  for (int frozenAgent : frozenAgents) {
    solution_.agents[frozenAgent] = previousSolution_.agents[frozenAgent];
  }

  // Compute mutable-vs-frozen soft conflicts under existing occupancy
  // semantics. Classification against full validation happens below.
  auto countCrossSetConflicts = [&](const vector<int>& mutableAgentsSet,
                                    const vector<int>& frozenAgentsSet,
                                    bool includeTerminal) -> int64_t {
    int64_t conflicts = 0;
    for (int mutableAgent : mutableAgentsSet) {
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
      isGoalOccupationRepositionTrue();
  const int64_t softConflicts = countCrossSetConflicts(
      mutableAgents, frozenAgents, includeTerminalForSoftCheck);

  beginPhase(NrrPhase::taskMapRebuild);
  solution_.taskAgentMap.assign(numTasks, UNASSIGNED);
  for (int agent = 0; agent < numAgents; agent++) {
    for (int task : solution_.agents[agent].taskAssignments) {
      if (task < 0 || task >= numTasks) {
        solution_ = backupSolution;
        invalidateCurrentTaskAssignmentIndexCache();
        return fail("stitched_assignment_out_of_range");
      }
      if (solution_.taskAgentMap[task] != UNASSIGNED) {
        solution_ = backupSolution;
        invalidateCurrentTaskAssignmentIndexCache();
        return fail("stitched_assignment_duplicate");
      }
      solution_.taskAgentMap[task] = agent;
    }
  }
  for (int task = 0; task < numTasks; task++) {
    if (solution_.taskAgentMap[task] == UNASSIGNED) {
      solution_ = backupSolution;
      invalidateCurrentTaskAssignmentIndexCache();
      return fail("stitched_assignment_incomplete");
    }
  }

  if (isGoalOccupationRepositionTrue()) {
    beginPhase(NrrPhase::terminalReplan);
    vector<int> terminalReplanAgents = selectTerminalReplanAgents(mutableAgents);
    if (!terminalReplanAgents.empty() &&
        !planTerminalReposition(terminalReplanAgents, false)) {
      solution_ = backupSolution;
      invalidateCurrentTaskAssignmentIndexCache();
      return fail("terminal_replan_failed_after_stitch");
    }
  }

  beginPhase(NrrPhase::recomputeObjective);
  long long recomputedSoc = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    recomputedSoc +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts = clampSocToInt(recomputedSoc, "nrr_repair");

  // 8) Global validate with current collision + precedence semantics.
  const Time::time_point validationStart = Time::now();
  beginPhase(NrrPhase::validation);
  const bool previousTerminalValidationFlag = useTerminalPathsInValidation_;
  useTerminalPathsInValidation_ = isGoalOccupationRepositionTrue();
  ValidationStats stitchedStats;
  const bool stitchedValid = validateSolution(nullptr, &stitchedStats);
  if (!collectNrrDetailedTiming) {
    attemptTimeValidationSec += elapsedSecSince(validationStart);
  }
  useTerminalPathsInValidation_ = previousTerminalValidationFlag;
  const bool hardSuccess = stitchedValid && softConflicts == 0;
  const bool softOnlyInvalid =
      (!stitchedValid && softConflicts > 0 &&
       stitchedStats.precedenceViolations == 0 &&
       stitchedStats.structuralViolations == 0);
  const bool softCandidateSuccess = softOnlyInvalid;
  if (!hardSuccess && !softCandidateSuccess) {
    nrrStats_.stitchedInvalidAttempts++;
    nrrStats_.stitchedInvalidPrecedenceViolations +=
        stitchedStats.precedenceViolations;
    nrrStats_.stitchedInvalidVertexCollisions += stitchedStats.vertexCollisions;
    nrrStats_.stitchedInvalidEdgeSwapCollisions +=
        stitchedStats.edgeSwapCollisions;
    nrrStats_.stitchedInvalidStructuralViolations +=
        stitchedStats.structuralViolations;
    solution_ = backupSolution;
    invalidateCurrentTaskAssignmentIndexCache();
    if (softConflicts > 0) {
      if (stitchedValid) {
        return fail("soft_conflicts_nonzero_validated:" +
                    std::to_string(softConflicts));
      }
      return fail("soft_conflicts_with_hard_invalid:" +
                  std::to_string(softConflicts));
    }
    return fail("stitched_solution_invalid");
  }

  if (softCandidateSuccess) {
    lastNrrSoftCandidate_ = true;
    lastNrrSoftConflictCount_ = softConflicts;
    lastNrrSoftOnlyInvalid_ = true;
    nrrStats_.softCandidatesProduced++;
    nrrStats_.softCandidateConflictSum += static_cast<double>(softConflicts);
    nrrStats_.softCandidateConflictSamples++;
    PLOGI << "nrr_repair: soft-candidate success (mini_solver=" << miniSolver
          << ", removed=" << destroyedTasks.size()
          << ", neighborhood_agents=" << mutableAgents.size()
          << ", soft_conflicts=" << softConflicts
          << ", soc=" << solution_.sumOfCosts << ")\n";
  } else {
    PLOGI << "nrr_repair: success (mini_solver=" << miniSolver
          << ", removed=" << destroyedTasks.size()
          << ", neighborhood_agents=" << mutableAgents.size()
          << ", soc=" << solution_.sumOfCosts << ")\n";
  }
  return finalizeAttempt(true);
  }
