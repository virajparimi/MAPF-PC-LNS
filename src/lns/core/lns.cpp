#include "lns.hpp"

#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

#include "common.hpp"
#include "lns_internal_helpers.hpp"
#include "utils.hpp"

namespace {
struct LowLevelStructuralDiagnostics {
  bool validStage = false;
  bool goalPermanentBeforeArrivalLb = false;
  bool startTrappedAtTPlus1 = false;
  bool staticDisconnectedPermanent = false;
  int startLocation = -1;
  int goalLocation = -1;
  int hDist = MAX_TIMESTEP;
  int earliestArrivalLb = MAX_TIMESTEP;
  int goalPermanentFrom = MAX_TIMESTEP;
  int feasibleImmediateMoves = -1;
  bool canWaitAtStart = false;
  int stableTime = -1;

  int certificateCount() const {
    return (goalPermanentBeforeArrivalLb ? 1 : 0) +
           (startTrappedAtTPlus1 ? 1 : 0) +
           (staticDisconnectedPermanent ? 1 : 0);
  }
};

LowLevelStructuralDiagnostics classifyLowLevelStructuralDiagnostics(
    const SingleAgentSolver& solver, const ConstraintTable& constraintTable,
    int startTime, int stage) {
  LowLevelStructuralDiagnostics diag;
  if (stage < 0 || stage >= (int)solver.goalLocations.size()) {
    return diag;
  }
  diag.validStage = true;
  diag.goalLocation = solver.goalLocations[stage];
  diag.startLocation =
      (stage == 0) ? solver.startLocation : solver.goalLocations[stage - 1];
  if (diag.startLocation < 0 || diag.startLocation >= solver.instance.mapSize ||
      diag.goalLocation < 0 || diag.goalLocation >= solver.instance.mapSize) {
    diag.validStage = false;
    return diag;
  }

  diag.hDist = solver.getStageGoalDistance(stage, diag.startLocation);
  diag.earliestArrivalLb = (diag.hDist >= MAX_TIMESTEP / 2)
                               ? MAX_TIMESTEP
                               : startTime + diag.hDist;
  const auto* goalIntervals =
      constraintTable.getConstraintIntervals(diag.goalLocation);
  diag.goalPermanentFrom = permanentOccupancyStart(goalIntervals);
  if (diag.goalPermanentFrom != MAX_TIMESTEP &&
      diag.goalPermanentFrom <= diag.earliestArrivalLb) {
    diag.goalPermanentBeforeArrivalLb = true;
  }

  diag.feasibleImmediateMoves = 0;
  for (int nxt : solver.instance.getNeighbors(diag.startLocation)) {
    const bool blockedVertex = constraintTable.constrained(nxt, startTime + 1);
    const bool blockedEdge =
        constraintTable.constrained(diag.startLocation, nxt, startTime + 1);
    if (!blockedVertex && !blockedEdge) {
      diag.feasibleImmediateMoves++;
    }
  }
  diag.canWaitAtStart =
      !constraintTable.constrained(diag.startLocation, startTime + 1);
  if (!diag.canWaitAtStart && diag.feasibleImmediateMoves == 0) {
    diag.startTrappedAtTPlus1 = true;
  }

  diag.stableTime = max(startTime, constraintTable.temporalExtent + 1);
  const bool staticReachable = reachableWithPermanentBlocksByTime(
      solver.instance, constraintTable, diag.startLocation, diag.goalLocation,
      diag.stableTime);
  if (!staticReachable) {
    diag.staticDisconnectedPermanent = true;
  }
  return diag;
}
}  // namespace

void LNS::buildFullPrecedenceConstraints(
    vector<pair<int, int>>& precedenceConstraints,
    bool includeIntraConstraints) const {
  const auto& inputConstraints = instance_.getInputPrecedenceConstraintsRef();
  size_t totalConstraints = inputConstraints.size();
  if (includeIntraConstraints) {
    for (int agent = 0; agent < instance_.getAgentNum(); ++agent) {
      const auto& assignments = solution_.agents[agent].taskAssignments;
      if (assignments.size() > 1) {
        totalConstraints += (assignments.size() - 1);
      }
    }
  }

  precedenceConstraints.clear();
  precedenceConstraints.reserve(totalConstraints);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputConstraints.begin(),
                               inputConstraints.end());
  if (includeIntraConstraints) {
    int mismatchLogs = 0;
    constexpr int kMaxMismatchLogs = 10;
    for (int agent = 0; agent < instance_.getAgentNum(); ++agent) {
      const auto& assignments = solution_.agents[agent].taskAssignments;
      const auto& storedIntra = solution_.agents[agent].intraPrecedenceConstraints;
      const bool intraDirty = solution_.agents[agent].intraPrecedenceDirty;

      // Correctness-first: derive intra-agent precedence directly from the
      // current assignment order so cycle checks and precedence constraints use
      // the same edge source.
      for (int pos = 1; pos < (int)assignments.size(); ++pos) {
        const int pred = assignments[pos - 1];
        const int succ = assignments[pos];
        if (pred >= 0 && pred < instance_.getTasksNum() &&
            succ >= 0 && succ < instance_.getTasksNum()) {
          precedenceConstraints.emplace_back(pred, succ);
        }
      }

      // Keep stored intra edges for diagnostics only. If they drift from the
      // assignment-derived edges, log warnings so stale state is visible.
      if (!intraDirty &&
          (storedIntra.size() + 1 < assignments.size() ||
           (assignments.empty() && !storedIntra.empty()))) {
        if (mismatchLogs < kMaxMismatchLogs) {
          PLOGW << "buildFullPrecedenceConstraints: stored intra-edge count ("
                << storedIntra.size() << ") differs from assignment-derived count ("
                << (assignments.empty() ? 0 : (int)assignments.size() - 1)
                << ") for agent " << agent << "\n";
          mismatchLogs++;
        }
      }
    }
  }
}

vector<pair<int, int>> LNS::buildFullPrecedenceConstraints(
    bool includeIntraConstraints) const {
  vector<pair<int, int>> precedenceConstraints;
  buildFullPrecedenceConstraints(precedenceConstraints, includeIntraConstraints);
  return precedenceConstraints;
}

AgentTaskPath LNS::runLowLevelSearch(SingleAgentSolver& solver,
                                     ConstraintTable& constraintTable,
                                     int startTime, int stage,
                                     int lowerBound) {
  const Time::time_point lowLevelStart = Time::now();
  auto accumulateLowLevelRuntime = [&]() {
    cumulativeLowLevelSearchSec_ +=
        ((fsec)(Time::now() - lowLevelStart)).count();
  };
  const double remainingBudget = remainingRuntimeBudgetSec();
  lowLevelState_.lastRemainingBudgetSec = remainingBudget;
  if (remainingBudget <= 0.0) {
    solver.setLastSearchOutcome(
        SingleAgentSolver::SearchOutcome::budget_exhausted);
    lowLevelState_.lastOutcome = solver.getLastSearchOutcome();
    lowLevelState_.lastEffectiveTimeoutSec = 0.0;
    lowLevelState_.counters.budgetExhausted++;
    accumulateLowLevelRuntime();
    return AgentTaskPath();
  }

  const double configuredTimeout = solver.getSegmentTimeout();
  const double effectiveTimeout = max(1e-6, min(configuredTimeout, remainingBudget));
  lowLevelState_.lastEffectiveTimeoutSec = effectiveTimeout;
  solver.setSegmentTimeout(effectiveTimeout);

  lowLevelState_.counters.calls++;
  if (lowLevelStructuralPrePrune_) {
    const auto diag = classifyLowLevelStructuralDiagnostics(
        solver, constraintTable, startTime, stage);
    if (diag.validStage && diag.certificateCount() > 0) {
      lowLevelState_.counters.structuralPrePruned++;
      if (diag.certificateCount() > 1) {
        lowLevelState_.counters.structuralPrePrunedMultiCertificate++;
      }
      if (diag.goalPermanentBeforeArrivalLb) {
        lowLevelState_.counters
            .structuralPrePrunedGoalPermanentBeforeArrivalLb++;
      }
      if (diag.startTrappedAtTPlus1) {
        lowLevelState_.counters.structuralPrePrunedStartTrappedAtTPlus1++;
      }
      if (diag.staticDisconnectedPermanent) {
        lowLevelState_.counters
            .structuralPrePrunedStaticDisconnectedPermanent++;
      }

      solver.setLastSearchOutcome(
          SingleAgentSolver::SearchOutcome::search_exhausted);
      lowLevelState_.lastOutcome = solver.getLastSearchOutcome();
      lowLevelState_.counters.searchExhausted++;
      if (debugImprovementDiagnostics_ &&
          lowLevelState_.timeoutDiagnosticsLogsEmitted < 200) {
        PLOGW << "LL-structural-preprune: stage=" << stage
              << ", start_time=" << startTime << ", lower_bound=" << lowerBound
              << ", start_loc=" << diag.startLocation
              << ", goal_loc=" << diag.goalLocation
              << ", hdist_to_goal=" << diag.hDist
              << ", earliest_arrival_lb=" << diag.earliestArrivalLb
              << ", goal_permanent_from="
              << (diag.goalPermanentFrom == MAX_TIMESTEP ? -1
                                                         : diag.goalPermanentFrom)
              << ", feasible_immediate_moves=" << diag.feasibleImmediateMoves
              << ", can_wait_at_start="
              << (diag.canWaitAtStart ? "true" : "false")
              << ", stable_time=" << diag.stableTime
              << ", cert_goal_permanent="
              << (diag.goalPermanentBeforeArrivalLb ? "true" : "false")
              << ", cert_start_trapped="
              << (diag.startTrappedAtTPlus1 ? "true" : "false")
              << ", cert_static_disconnected="
              << (diag.staticDisconnectedPermanent ? "true" : "false")
              << ", configured_timeout_sec=" << configuredTimeout
              << ", effective_timeout_sec=" << effectiveTimeout << "\n";
        lowLevelState_.timeoutDiagnosticsLogsEmitted++;
      }
      solver.setSegmentTimeout(configuredTimeout);
      accumulateLowLevelRuntime();
      return AgentTaskPath();
    }
  }

  const uint64_t expandedBefore = solver.numExpanded;
  const uint64_t generatedBefore = solver.numGenerated;
  AgentTaskPath path =
      solver.findPathSegment(constraintTable, startTime, stage, lowerBound);
  lowLevelState_.counters.expanded += (solver.numExpanded - expandedBefore);
  lowLevelState_.counters.generated += (solver.numGenerated - generatedBefore);
  lowLevelState_.lastOutcome = solver.getLastSearchOutcome();
  switch (lowLevelState_.lastOutcome) {
    case SingleAgentSolver::SearchOutcome::found:
      lowLevelState_.counters.found++;
      break;
    case SingleAgentSolver::SearchOutcome::timeout:
      lowLevelState_.counters.timeout++;
      recordLowLevelTimeoutDiagnostics(solver, constraintTable, startTime, stage,
                                       lowerBound, configuredTimeout,
                                       effectiveTimeout);
      break;
    case SingleAgentSolver::SearchOutcome::search_exhausted:
      lowLevelState_.counters.searchExhausted++;
      break;
    case SingleAgentSolver::SearchOutcome::invalid_input:
      lowLevelState_.counters.invalidInput++;
      break;
    case SingleAgentSolver::SearchOutcome::budget_exhausted:
      lowLevelState_.counters.budgetExhausted++;
      break;
    case SingleAgentSolver::SearchOutcome::unknown:
    default:
      lowLevelState_.counters.unknown++;
      break;
  }
  solver.setSegmentTimeout(configuredTimeout);
  accumulateLowLevelRuntime();
  return path;
}

void LNS::recordLowLevelTimeoutDiagnostics(
    const SingleAgentSolver& solver, const ConstraintTable& constraintTable,
    int startTime, int stage, int lowerBound, double configuredTimeout,
    double effectiveTimeout) {
  const bool reducedByGlobalBudget = effectiveTimeout + 1e-9 < configuredTimeout;
  if (reducedByGlobalBudget) {
    lowLevelState_.counters.timeoutReducedByGlobalBudget++;
  }

  const auto diag = classifyLowLevelStructuralDiagnostics(
      solver, constraintTable, startTime, stage);
  if (!diag.validStage) {
    lowLevelState_.counters.timeoutOther++;
    return;
  }

  if (diag.certificateCount() > 1) {
    lowLevelState_.counters.timeoutMultiCertificate++;
  }

  const char* reason = "other";
  if (diag.goalPermanentBeforeArrivalLb) {
    lowLevelState_.counters.timeoutGoalPermanentBeforeArrivalLb++;
    reason = "goal_permanent_before_arrival_lb";
  } else if (diag.startTrappedAtTPlus1) {
    lowLevelState_.counters.timeoutStartTrappedAtTPlus1++;
    reason = "start_trapped_at_t+1";
  } else if (diag.staticDisconnectedPermanent) {
    lowLevelState_.counters.timeoutStaticDisconnectedPermanent++;
    reason = "static_disconnected_under_permanent_blocks";
  } else {
    lowLevelState_.counters.timeoutOther++;
  }

  if (debugImprovementDiagnostics_ &&
      lowLevelState_.timeoutDiagnosticsLogsEmitted < 200) {
    PLOGW << "LL-timeout diagnostics: reason=" << reason
          << ", stage=" << stage << ", start_time=" << startTime
          << ", lower_bound=" << lowerBound << ", start_loc="
          << diag.startLocation << ", goal_loc=" << diag.goalLocation
          << ", hdist_to_goal=" << diag.hDist
          << ", earliest_arrival_lb=" << diag.earliestArrivalLb
          << ", goal_permanent_from="
          << (diag.goalPermanentFrom == MAX_TIMESTEP ? -1
                                                     : diag.goalPermanentFrom)
          << ", feasible_immediate_moves=" << diag.feasibleImmediateMoves
          << ", can_wait_at_start=" << (diag.canWaitAtStart ? "true" : "false")
          << ", stable_time=" << diag.stableTime
          << ", reduced_by_global_budget="
          << (reducedByGlobalBudget ? "true" : "false")
          << ", configured_timeout_sec=" << configuredTimeout
          << ", effective_timeout_sec=" << effectiveTimeout << "\n";
    lowLevelState_.timeoutDiagnosticsLogsEmitted++;
  }
}

double LNS::elapsedRuntimeSec() const {
  return ((fsec)(Time::now() - plannerStartTime_)).count();
}

double LNS::remainingRuntimeBudgetSec() const {
  return max(0.0, timeLimit_ - elapsedRuntimeSec());
}

bool LNS::runtimeBudgetExhausted() const { return elapsedRuntimeSec() >= timeLimit_; }

void LNS::reservePathWithGoalPolicy(ConstraintTable& constraintTable,
                                    const AgentTaskPath& path,
                                    bool isFinalTask,
                                    bool softOnly) const {
  if (path.empty()) {
    return;
  }
  if (softOnly) {
    constraintTable.addSoftPath(path, isFinalTask);
  } else {
    constraintTable.addPath(path, isFinalTask);
  }
}

void LNS::invalidateCurrentTaskAssignmentIndexCache() {
  currentTaskAssignmentIndexCacheValid_ = false;
}

const vector<int>& LNS::getCurrentTaskPositionIndexByTask() const {
  const int numTasks = instance_.getTasksNum();
  if (!currentTaskAssignmentIndexCacheValid_ ||
      (int)currentTaskPosByTaskCache_.size() != numTasks ||
      (int)currentTaskOwnerByTaskCache_.size() != numTasks) {
    currentTaskOwnerByTaskCache_.assign(numTasks, UNASSIGNED);
    currentTaskPosByTaskCache_.assign(numTasks, -1);
    int duplicateTaskOwners = 0;
    for (int agent = 0; agent < solution_.numOfAgents; agent++) {
      const auto& assignments = solution_.agents[agent].taskAssignments;
      for (int p = 0; p < (int)assignments.size(); p++) {
        const int task = assignments[p];
        if (task < 0 || task >= numTasks) {
          continue;
        }
        if (currentTaskOwnerByTaskCache_[task] != UNASSIGNED) {
          duplicateTaskOwners++;
          if (duplicateTaskOwners <= 5) {
            PLOGE << "Duplicate task ownership detected in current solution for task "
                  << task << " (existing agent="
                  << currentTaskOwnerByTaskCache_[task]
                  << ", new agent=" << agent << ")\n";
          }
        }
        currentTaskOwnerByTaskCache_[task] = agent;
        currentTaskPosByTaskCache_[task] = p;
      }
    }
    if (duplicateTaskOwners > 5) {
      PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
            << " tasks in current solution assignment index.\n";
    }
    currentTaskAssignmentIndexCacheValid_ = true;
  }
  return currentTaskPosByTaskCache_;
}

void LNS::restoreSolutionFromPrevious() {
  solutionRestoreStats_.restoreCalls++;
  solution_ = previousSolution_;
  invalidateCurrentTaskAssignmentIndexCache();
  solutionRestoreStats_.fullRestores++;
}

void LNS::restoreSolutionFromPrevious(const vector<int>& agentSubset) {
  solutionRestoreStats_.restoreCalls++;
  if (agentSubset.empty() ||
      solution_.agents.size() != previousSolution_.agents.size()) {
    solution_ = previousSolution_;
    invalidateCurrentTaskAssignmentIndexCache();
    solutionRestoreStats_.fullRestores++;
    return;
  }

  solution_.numOfTasks = previousSolution_.numOfTasks;
  solution_.numOfAgents = previousSolution_.numOfAgents;
  solution_.sumOfCosts = previousSolution_.sumOfCosts;
  solution_.utility = previousSolution_.utility;
  solution_.taskAgentMap = previousSolution_.taskAgentMap;

  const int agentCount = instance_.getAgentNum();
  vector<char> selected(agentCount, 0);
  int copiedAgents = 0;
  for (int agent : agentSubset) {
    if (agent < 0 || agent >= agentCount) {
      continue;
    }
    if (selected[agent]) {
      continue;
    }
    selected[agent] = 1;
    solution_.agents[agent] = previousSolution_.agents[agent];
    copiedAgents++;
  }

  if (copiedAgents == 0) {
    solution_ = previousSolution_;
    solutionRestoreStats_.fullRestores++;
  }
  invalidateCurrentTaskAssignmentIndexCache();
}

void LNS::snapshotPreviousFromCurrent(const vector<int>& agentSubset) {
  if (agentSubset.empty() ||
      solution_.agents.size() != previousSolution_.agents.size()) {
    previousSolution_ = solution_;
    return;
  }

  previousSolution_.numOfTasks = solution_.numOfTasks;
  previousSolution_.numOfAgents = solution_.numOfAgents;
  previousSolution_.sumOfCosts = solution_.sumOfCosts;
  previousSolution_.utility = solution_.utility;
  previousSolution_.taskAgentMap = solution_.taskAgentMap;

  const int agentCount = instance_.getAgentNum();
  vector<char> selected(agentCount, 0);
  int copiedAgents = 0;
  for (int agent : agentSubset) {
    if (agent < 0 || agent >= agentCount) {
      continue;
    }
    if (selected[agent]) {
      continue;
    }
    selected[agent] = 1;
    previousSolution_.agents[agent] = solution_.agents[agent];
    copiedAgents++;
  }

  if (copiedAgents == 0) {
    previousSolution_ = solution_;
  }
}

int LNS::computeObjectiveValue(const Solution& solution) const {
  if (isOptimizationObjectiveSoc()) {
    return solution.sumOfCosts;
  }
  long long makespan = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    makespan = max(makespan, static_cast<long long>(
                                 solution.agents[agent].path.endTimeOrZero()));
  }
  if (makespan > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(makespan);
}

int LNS::computeObjectiveValue(const FeasibleSolution& solution) const {
  if (isOptimizationObjectiveSoc()) {
    return solution.sumOfCosts;
  }
  long long makespan = 0;
  const int cappedAgents = std::min(instance_.getAgentNum(),
                                    static_cast<int>(solution.agentPaths.size()));
  for (int agent = 0; agent < cappedAgents; agent++) {
    makespan = max(makespan,
                   static_cast<long long>(solution.agentPaths[agent].endTimeOrZero()));
  }
  if (makespan > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(makespan);
}

int LNS::currentObjectiveValue() const {
  return computeObjectiveValue(solution_);
}

int LNS::previousObjectiveValue() const {
  return computeObjectiveValue(previousSolution_);
}

int LNS::incumbentObjectiveValueOrMax() const {
  if (incumbentSolution_.agentPaths.empty()) {
    return std::numeric_limits<int>::max();
  }
  return computeObjectiveValue(incumbentSolution_);
}

LNS::LNS(int numOfIterations, const Instance& instance,
         const LNSParams& parameters)
    : numOfIterations_(numOfIterations),
      instance_(instance),
      seed_(parameters.core.seed),
      rng_(parameters.core.seed),
      solution_(instance),
      previousSolution_(instance) {
  plannerStartTime_ = Time::now();
  neighborSize_ = parameters.core.neighborhoodSize;
  timeLimit_ = parameters.core.timeLimit;
  acceptanceState_.temperature = parameters.core.temperature;
  acceptanceState_.coolingCoefficient = parameters.core.coolingCoefficient;
  acceptanceState_.heatingCoefficient = parameters.core.heatingCoefficient;
  tolerance_ = parameters.core.tolerance;
  shawDistanceWeight_ = parameters.core.shawDistanceWeight;
  shawTemporalWeight_ = parameters.core.shawTemporalWeight;
  lnsConflictWeight_ = parameters.core.lnsConflictWeight;
  lnsCostWeight_ = parameters.core.lnsCostWeight;
  initialSolutionStrategy = parameters.core.initialSolutionStrategy;
  initialSeedFromMapfpcLog_ = parameters.core.initialSeedFromMapfpcLog;
  postRefineWithMapfpc_ = parameters.core.postRefineWithMapfpc;
  postRefineAssignmentSource_ = parameters.core.postRefineAssignmentSource;
  if (postRefineAssignmentSource_ != "solution" &&
      postRefineAssignmentSource_ != "log") {
    PLOGW << "Unknown postRefineAssignmentSource '"
          << postRefineAssignmentSource_
          << "'; defaulting to 'solution'\n";
    postRefineAssignmentSource_ = "solution";
  }
  postRefineAssignmentLog_ = parameters.core.postRefineAssignmentLog;
  postRefineSolver_ = parameters.core.postRefineSolver;
  if (postRefineSolver_ != "pbs" && postRefineSolver_ != "cbs") {
    PLOGW << "Unknown postRefineSolver '" << postRefineSolver_
          << "'; defaulting to 'pbs'\n";
    postRefineSolver_ = "pbs";
  }
  postRefineTimeoutSec_ = std::max(1, parameters.core.postRefineTimeoutSec);
  postRefineAcceptOnlyIfBetter_ =
      parameters.core.postRefineAcceptOnlyIfBetter;
  debugImprovementDiagnostics_ =
      parameters.core.debugImprovementDiagnostics;
  debugIterationTsvPath_ = parameters.core.debugIterationTsvPath;
  initialSolutionRequested_ = initialSolutionStrategy;
  initialSolutionEffective_ = initialSolutionStrategy;
  initialSolutionFallbackUsed_ = false;
  initialSolutionFallbackReason_ = "none";
  optimizationObjective_ = parameters.core.optimizationObjective;
  if (optimizationObjective_ != "soc" &&
      optimizationObjective_ != "makespan") {
    PLOGW << "Unknown optimizationObjective '" << optimizationObjective_
          << "'; defaulting to 'soc'\n";
    optimizationObjective_ = "soc";
  }
  optimizationObjectiveMode_ =
      (optimizationObjective_ == "makespan")
          ? OptimizationObjectiveMode::makespan
          : OptimizationObjectiveMode::soc;
  improvementDiagnosticsStats_.reset();
  acceptanceState_.acceptedSolutionFingerprints.clear();
  iterationDebugRecords_.clear();
  destroyHeuristic = parameters.core.destroyHeuristic;
  acceptanceCriteria = parameters.core.acceptanceCriteria;
  acceptanceState_.acceptOnlyValidCandidates = parameters.core.acceptOnlyValidCandidates;
  repairHeuristic = parameters.core.repairHeuristic;
  if (repairHeuristic != "regret") {
    PLOGW << "Unknown repairHeuristic '" << repairHeuristic
          << "'; defaulting to 'regret'\n";
    repairHeuristic = "regret";
  }
  repairHeuristicMode_ = RepairHeuristicMode::regret;
  enableNrrRepair_ = parameters.core.enableNrrRepair;
  nrrFallbackToStandard_ = parameters.core.nrrFallbackToStandard;
  nrrGlobalReassign_ = parameters.core.nrrGlobalReassign;
  acceptanceState_.forceNeighborhoodChangeOnReject =
      parameters.core.forceNeighborhoodChangeOnReject;
  nrrMiniSolver_ = parameters.core.nrrMiniSolver;
  if (nrrMiniSolver_ != "pbs" && nrrMiniSolver_ != "cbs" &&
      nrrMiniSolver_ != "auto") {
    PLOGW << "Unknown nrrMiniSolver '" << nrrMiniSolver_
          << "'; defaulting to 'cbs'\n";
    nrrMiniSolver_ = "cbs";
  }
  if (nrrMiniSolver_ == "pbs") {
    nrrMiniSolverMode_ = NrrMiniSolverMode::pbs;
  } else if (nrrMiniSolver_ == "auto") {
    nrrMiniSolverMode_ = NrrMiniSolverMode::auto_mode;
  } else {
    nrrMiniSolverMode_ = NrrMiniSolverMode::cbs;
  }
  nrrStats_.reset();
  const int destroyHeuristicCount = adaptiveLNS_.numDestroyHeuristics;
  nrrStats_.softModeSelectionsByDestroy.assign(destroyHeuristicCount, 0);
  nrrStats_.softModeAcceptedByDestroy.assign(destroyHeuristicCount, 0);
  nrrStats_.softModeBestUpdatesByDestroy.assign(destroyHeuristicCount, 0);
  improvementDiagnosticsStats_.softModeSelectionsByDestroy.assign(
      destroyHeuristicCount, 0);
  improvementDiagnosticsStats_.softModeAcceptedByDestroy.assign(
      destroyHeuristicCount, 0);
  improvementDiagnosticsStats_.softModeBestUpdatesByDestroy.assign(
      destroyHeuristicCount, 0);
  repairMapfpcSolver_ = parameters.core.repairMapfpcSolver;
  if (repairMapfpcSolver_ != "pbs" && repairMapfpcSolver_ != "cbs") {
    PLOGW << "Unknown repairMapfpcSolver '" << repairMapfpcSolver_
          << "'; defaulting to 'cbs'\n";
    repairMapfpcSolver_ = "cbs";
  }
  repairMapfpcTimeoutSec_ = std::max(1, parameters.core.repairMapfpcTimeoutSec);
  regretType = parameters.core.regretType;
  if (regretType == "relative") {
    regretTypeMode_ = RegretTypeMode::relative;
  } else {
    if (regretType != "absolute") {
      PLOGW << "Unknown regretType '" << regretType
            << "'; defaulting to 'absolute'\n";
      regretType = "absolute";
    }
    regretTypeMode_ = RegretTypeMode::absolute;
  }
  buildSuccessorPressureStaticSignals();
  alnsEnablePrecedenceAwareDestroy_ =
      parameters.core.alnsEnablePrecedenceAwareDestroy;
  softRecoveryState_.destroyMode = parameters.core.softRecoveryDestroyMode;
  softRecoveryState_.persistentConflictGraph = parameters.core.softPersistentConflictGraph;
  if (parameters.lowLevel.planner == "sipps") {
    lowLevelPlannerType_ = LowLevelPlannerType::sipps;
  } else {
    lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  }
  lowLevelSippsSuboptimality_ = max(1.0, parameters.lowLevel.sippsSuboptimality);
  lowLevelSegmentTimeout_ = max(0.0, parameters.lowLevel.segmentTimeout);
  lowLevelStructuralPrePrune_ = parameters.lowLevel.structuralPrePrune;
  plannerParityCheck_ = parameters.lowLevel.parityCheck;

  // Ensure both working solutions use the selected low-level planner.
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].pathPlanner = createSharedPlanner(agent);
    previousSolution_.agents[agent].pathPlanner = createSharedPlanner(agent);
  }

}

void LNS::buildSuccessorPressureStaticSignals() {
  // Fixed behavior: always use all precedence descendants for successor
  // pressure, with unweighted contributions and no depth cap.
  const int taskCount = instance_.getTasksNum();
  successorPressureStaticSignalsByTask_.assign(taskCount, {});
  const auto& successors = instance_.getSuccessorsRef();

  for (int task = 0; task < taskCount; task++) {
    auto& signals = successorPressureStaticSignalsByTask_[task];
    if (task < 0 || task >= (int)successors.size()) {
      continue;
    }

    vector<int> minDepth(taskCount, -1);
    std::deque<int> frontier;
    for (int successor : successors[task]) {
      if (successor < 0 || successor >= taskCount) {
        continue;
      }
      if (minDepth[successor] == -1 || minDepth[successor] > 1) {
        minDepth[successor] = 1;
        frontier.push_back(successor);
      }
    }

    while (!frontier.empty()) {
      const int current = frontier.front();
      frontier.pop_front();
      const int currentDepth = minDepth[current];
      if (currentDepth <= 0) {
        continue;
      }
      if (current < 0 || current >= (int)successors.size()) {
        continue;
      }
      for (int next : successors[current]) {
        if (next < 0 || next >= taskCount) {
          continue;
        }
        const int candidateDepth = currentDepth + 1;
        if (minDepth[next] == -1 || candidateDepth < minDepth[next]) {
          minDepth[next] = candidateDepth;
          frontier.push_back(next);
        }
      }
    }

    for (int descendant = 0; descendant < taskCount; descendant++) {
      const int depth = minDepth[descendant];
      if (depth <= 0) {
        continue;
      }
      signals.push_back({descendant, depth});
    }
  }
}

std::shared_ptr<SingleAgentSolver> LNS::createSharedPlanner(int agent) const {
  std::shared_ptr<SingleAgentSolver> planner;
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      planner = std::make_shared<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_shared<MultiLabelSpaceTimeAStar>(
          instance_, agent, true, true);
      break;
  }
  planner->setSegmentTimeout(lowLevelSegmentTimeout_);
  return planner;
}

std::unique_ptr<SingleAgentSolver> LNS::createLocalPlanner(int agent) const {
  std::unique_ptr<SingleAgentSolver> planner;
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      // Local repair planners always set explicit goals before search.
      planner = std::make_unique<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, false);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_unique<MultiLabelSpaceTimeAStar>(
          instance_, agent, false, true);
      break;
  }
  planner->setSegmentTimeout(lowLevelSegmentTimeout_);
  return planner;
}

SingleAgentSolver& LNS::getReusableLocalPlanner(int agent) {
  if (agent < 0 || agent >= instance_.getAgentNum()) {
    throw std::out_of_range("LNS::getReusableLocalPlanner: invalid agent");
  }
  if (lowLevelState_.reusableLocalPlanners.size() !=
      (size_t)instance_.getAgentNum()) {
    lowLevelState_.reusableLocalPlanners.resize(instance_.getAgentNum());
  }
  if (lowLevelState_.reusableLocalPlanners[agent] == nullptr) {
    lowLevelState_.reusableLocalPlanners[agent] = createLocalPlanner(agent);
  }
  return *lowLevelState_.reusableLocalPlanners[agent];
}
