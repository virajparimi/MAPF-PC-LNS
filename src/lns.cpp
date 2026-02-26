#include "lns.hpp"

#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
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
  const double remainingBudget = remainingRuntimeBudgetSec();
  lastLowLevelRemainingBudgetSec_ = remainingBudget;
  if (remainingBudget <= 0.0) {
    solver.setLastSearchOutcome(
        SingleAgentSolver::SearchOutcome::budget_exhausted);
    lastLowLevelOutcome_ = solver.getLastSearchOutcome();
    lastLowLevelEffectiveTimeoutSec_ = 0.0;
    lowLevelBudgetExhausted_++;
    return AgentTaskPath();
  }

  const double configuredTimeout = solver.getSegmentTimeout();
  const double effectiveTimeout = max(1e-6, min(configuredTimeout, remainingBudget));
  lastLowLevelEffectiveTimeoutSec_ = effectiveTimeout;
  solver.setSegmentTimeout(effectiveTimeout);

  lowLevelCalls_++;
  if (lowLevelStructuralPrePrune_) {
    const auto diag = classifyLowLevelStructuralDiagnostics(
        solver, constraintTable, startTime, stage);
    if (diag.validStage && diag.certificateCount() > 0) {
      lowLevelStructuralPrePruned_++;
      if (diag.certificateCount() > 1) {
        lowLevelStructuralPrePrunedMultiCertificate_++;
      }
      if (diag.goalPermanentBeforeArrivalLb) {
        lowLevelStructuralPrePrunedGoalPermanentBeforeArrivalLb_++;
      }
      if (diag.startTrappedAtTPlus1) {
        lowLevelStructuralPrePrunedStartTrappedAtTPlus1_++;
      }
      if (diag.staticDisconnectedPermanent) {
        lowLevelStructuralPrePrunedStaticDisconnectedPermanent_++;
      }

      solver.setLastSearchOutcome(
          SingleAgentSolver::SearchOutcome::search_exhausted);
      lastLowLevelOutcome_ = solver.getLastSearchOutcome();
      lowLevelSearchExhausted_++;
      if (debugImprovementDiagnostics_ &&
          lowLevelTimeoutDiagnosticsLogsEmitted_ < 200) {
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
        lowLevelTimeoutDiagnosticsLogsEmitted_++;
      }
      solver.setSegmentTimeout(configuredTimeout);
      return AgentTaskPath();
    }
  }

  const uint64_t expandedBefore = solver.numExpanded;
  const uint64_t generatedBefore = solver.numGenerated;
  AgentTaskPath path =
      solver.findPathSegment(constraintTable, startTime, stage, lowerBound);
  lowLevelExpanded_ += (solver.numExpanded - expandedBefore);
  lowLevelGenerated_ += (solver.numGenerated - generatedBefore);
  lastLowLevelOutcome_ = solver.getLastSearchOutcome();
  switch (lastLowLevelOutcome_) {
    case SingleAgentSolver::SearchOutcome::found:
      lowLevelFound_++;
      break;
    case SingleAgentSolver::SearchOutcome::timeout:
      lowLevelTimeout_++;
      recordLowLevelTimeoutDiagnostics(solver, constraintTable, startTime, stage,
                                       lowerBound, configuredTimeout,
                                       effectiveTimeout);
      break;
    case SingleAgentSolver::SearchOutcome::search_exhausted:
      lowLevelSearchExhausted_++;
      break;
    case SingleAgentSolver::SearchOutcome::invalid_input:
      lowLevelInvalidInput_++;
      break;
    case SingleAgentSolver::SearchOutcome::budget_exhausted:
      lowLevelBudgetExhausted_++;
      break;
    case SingleAgentSolver::SearchOutcome::unknown:
    default:
      lowLevelUnknown_++;
      break;
  }
  solver.setSegmentTimeout(configuredTimeout);
  return path;
}

void LNS::recordLowLevelTimeoutDiagnostics(
    const SingleAgentSolver& solver, const ConstraintTable& constraintTable,
    int startTime, int stage, int lowerBound, double configuredTimeout,
    double effectiveTimeout) {
  const bool reducedByGlobalBudget = effectiveTimeout + 1e-9 < configuredTimeout;
  if (reducedByGlobalBudget) {
    lowLevelTimeoutReducedByGlobalBudget_++;
  }

  const auto diag = classifyLowLevelStructuralDiagnostics(
      solver, constraintTable, startTime, stage);
  if (!diag.validStage) {
    lowLevelTimeoutOther_++;
    return;
  }

  if (diag.certificateCount() > 1) {
    lowLevelTimeoutMultiCertificate_++;
  }

  const char* reason = "other";
  if (diag.goalPermanentBeforeArrivalLb) {
    lowLevelTimeoutGoalPermanentBeforeArrivalLb_++;
    reason = "goal_permanent_before_arrival_lb";
  } else if (diag.startTrappedAtTPlus1) {
    lowLevelTimeoutStartTrappedAtTPlus1_++;
    reason = "start_trapped_at_t+1";
  } else if (diag.staticDisconnectedPermanent) {
    lowLevelTimeoutStaticDisconnectedPermanent_++;
    reason = "static_disconnected_under_permanent_blocks";
  } else {
    lowLevelTimeoutOther_++;
  }

  if (debugImprovementDiagnostics_ &&
      lowLevelTimeoutDiagnosticsLogsEmitted_ < 200) {
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
    lowLevelTimeoutDiagnosticsLogsEmitted_++;
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
  if (!isFinalTask || goalOccupationMode_ == "stay") {
    if (softOnly) {
      constraintTable.addSoftPath(path, isFinalTask);
    } else {
      constraintTable.addPath(path, isFinalTask);
    }
    return;
  }

  // In reposition_true, service path occupancy ends at task completion and
  // explicit terminalPath handles post-completion occupancy.
  if (softOnly) {
    constraintTable.addSoftPath(path, false);
  } else {
    constraintTable.addPath(path, false);
  }
}

void LNS::reserveTerminalPathIfActive(ConstraintTable& constraintTable,
                                      int agent,
                                      bool softOnly) const {
  if (goalOccupationMode_ != "reposition_true") {
    return;
  }
  if (agent < 0 || agent >= instance_.getAgentNum()) {
    return;
  }
  const auto& terminalPath = solution_.agents[agent].terminalPath;
  if (!solution_.agents[agent].terminalPathActive || terminalPath.empty()) {
    return;
  }
  // Terminal reposition path is part of the active occupancy model; keep the
  // final terminal location reserved to MAX_TIMESTEP for CT/validator parity.
  if (softOnly) {
    constraintTable.addSoftPath(terminalPath, true);
  } else {
    constraintTable.addPath(terminalPath, true);
  }
}

bool LNS::didAgentServicePathChange(int agent) const {
  if (agent < 0 || agent >= instance_.getAgentNum()) {
    return true;
  }
  const auto& currentPath = solution_.agents[agent].path;
  const auto& previousPath = previousSolution_.agents[agent].path;
  if (currentPath.empty() != previousPath.empty()) {
    return true;
  }
  if (currentPath.empty() && previousPath.empty()) {
    return false;
  }
  if (currentPath.beginTime != previousPath.beginTime ||
      currentPath.size() != previousPath.size()) {
    return true;
  }
  for (int i = 0; i < (int)currentPath.size(); i++) {
    if (currentPath.at(i).location != previousPath.at(i).location ||
        currentPath.at(i).isGoal != previousPath.at(i).isGoal) {
      return true;
    }
  }
  return false;
}

vector<int> LNS::selectTerminalReplanAgents(
    const vector<int>& candidateAgents) const {
  vector<int> replanAgents;
  if (goalOccupationMode_ != "reposition_true") {
    return replanAgents;
  }

  const int agentCount = instance_.getAgentNum();
  vector<char> marked(agentCount, 0);
  auto markAgent = [&](int agent) {
    if (agent < 0 || agent >= agentCount || marked[agent]) {
      return;
    }
    marked[agent] = 1;
    replanAgents.push_back(agent);
  };

  vector<int> changedAgents;
  changedAgents.reserve(candidateAgents.size());
  for (int agent : candidateAgents) {
    if (agent < 0 || agent >= agentCount) {
      continue;
    }
    const bool hasTerminalPath =
        solution_.agents[agent].terminalPathActive &&
        !solution_.agents[agent].terminalPath.empty();
    if (didAgentServicePathChange(agent) || !hasTerminalPath) {
      changedAgents.push_back(agent);
      markAgent(agent);
    }
  }

  if (changedAgents.empty()) {
    return replanAgents;
  }

  unordered_map<int, vector<int>> finalGoalOwners;
  finalGoalOwners.reserve((size_t)agentCount);
  for (int agent = 0; agent < agentCount; agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    if (assignments.empty() || solution_.agents[agent].path.empty()) {
      continue;
    }
    const int finalTask = assignments.back();
    if (finalTask < 0 || finalTask >= instance_.getTasksNum()) {
      continue;
    }
    finalGoalOwners[instance_.getTaskLocations(finalTask)].push_back(agent);
  }

  auto markOwnersAtLocation = [&](int location, int sourceAgent) {
    const auto it = finalGoalOwners.find(location);
    if (it == finalGoalOwners.end()) {
      return;
    }
    for (int ownerAgent : it->second) {
      if (ownerAgent == sourceAgent) {
        continue;
      }
      markAgent(ownerAgent);
    }
  };

  for (int changedAgent : changedAgents) {
    const auto& currentServicePath = solution_.agents[changedAgent].path;
    for (int t = 0; t < (int)currentServicePath.size(); t++) {
      markOwnersAtLocation(currentServicePath.at(t).location, changedAgent);
    }
    const auto& previousServicePath = previousSolution_.agents[changedAgent].path;
    for (int t = 0; t < (int)previousServicePath.size(); t++) {
      markOwnersAtLocation(previousServicePath.at(t).location, changedAgent);
    }
    if (previousSolution_.agents[changedAgent].terminalPathActive) {
      const auto& previousTerminalPath =
          previousSolution_.agents[changedAgent].terminalPath;
      for (int t = 0; t < (int)previousTerminalPath.size(); t++) {
        markOwnersAtLocation(previousTerminalPath.at(t).location, changedAgent);
      }
    }
  }
  return replanAgents;
}

const vector<int>& LNS::getParkingCandidatesForGoal(int finalGoal) {
  auto cacheIt = parkingCandidatesCache_.find(finalGoal);
  if (cacheIt != parkingCandidatesCache_.end()) {
    terminalRepositionStats_.candidateCacheHits++;
    return cacheIt->second;
  }

  terminalRepositionStats_.candidateCacheMisses++;
  vector<int> candidates;
  if (finalGoal >= 0 && finalGoal < instance_.mapSize &&
      !instance_.isObstacle(finalGoal)) {
    // Enumerate all reachable free cells in nondecreasing shortest-path
    // distance from finalGoal (BFS order).
    vector<char> visited(instance_.mapSize, 0);
    deque<int> frontier;
    visited[finalGoal] = 1;
    frontier.push_back(finalGoal);
    candidates.reserve((size_t)max(0, instance_.mapSize - 1));
    while (!frontier.empty()) {
      const int current = frontier.front();
      frontier.pop_front();
      for (int next : instance_.getNeighbors(current)) {
        if (next < 0 || next >= instance_.mapSize || visited[next] ||
            instance_.isObstacle(next)) {
          continue;
        }
        visited[next] = 1;
        frontier.push_back(next);
        if (next != finalGoal) {
          candidates.push_back(next);
        }
      }
    }
  }
  auto inserted =
      parkingCandidatesCache_.emplace(finalGoal, std::move(candidates));
  return inserted.first->second;
}

int LNS::cascadeTaskBudget() const {
  if (maxCascadeTasks_ > 0) {
    return maxCascadeTasks_;
  }
  if (maxCascadeFactor_ <= 0.0) {
    return std::numeric_limits<int>::max();
  }
  const int neighborhood = max(0, neighborSize_);
  const double scaledBudget = maxCascadeFactor_ * (double)neighborhood;
  const int factorBudget =
      (scaledBudget >= (double)std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : (int)std::ceil(scaledBudget);
  const int offsetBudget = neighborhood + 10;
  return max(factorBudget, offsetBudget);
}

void LNS::restoreSolutionFromPrevious() {
  solutionRestoreStats_.restoreCalls++;
  solution_ = previousSolution_;
  solutionRestoreStats_.fullRestores++;
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
  temperature_ = parameters.core.temperature;
  coolingCoefficient_ = parameters.core.coolingCoefficient;
  heatingCoefficient_ = parameters.core.heatingCoefficient;
  tolerance_ = parameters.core.tolerance;
  shawDistanceWeight_ = parameters.core.shawDistanceWeight;
  shawTemporalWeight_ = parameters.core.shawTemporalWeight;
  lnsConflictWeight_ = parameters.core.lnsConflictWeight;
  lnsCostWeight_ = parameters.core.lnsCostWeight;
  initialSolutionStrategy = parameters.core.initialSolutionStrategy;
  initialPortfolioTimeFraction_ =
      parameters.core.initialPortfolioTimeFraction;
  if (!std::isfinite(initialPortfolioTimeFraction_)) {
    initialPortfolioTimeFraction_ = 0.10;
  }
  initialPortfolioTimeFraction_ =
      min(1.0, max(0.0, initialPortfolioTimeFraction_));
  adaptiveInitialPortfolioBudget_ =
      parameters.core.adaptiveInitialPortfolioBudget;
  initialSeedFromPbsLog_ = parameters.core.initialSeedFromPbsLog;
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
  goalOccupationMode_ = parameters.core.goalOccupationMode;
  terminalRepositionStats_.reset();
  improvementDiagnosticsStats_.reset();
  acceptedSolutionFingerprints_.clear();
  iterationDebugRecords_.clear();
  parkingCandidatesCache_.clear();
  if (goalOccupationMode_ != "stay" &&
      goalOccupationMode_ != "reposition_true") {
    PLOGW << "Unknown goalOccupationMode '" << goalOccupationMode_
          << "'; defaulting to 'reposition_true'\n";
    goalOccupationMode_ = "reposition_true";
  }
  destroyHeuristic = parameters.core.destroyHeuristic;
  acceptanceCriteria = parameters.core.acceptanceCriteria;
  acceptOnlyValidCandidates_ = parameters.core.acceptOnlyValidCandidates;
  repairHeuristic = parameters.core.repairHeuristic;
  if (repairHeuristic != "regret" &&
      repairHeuristic != "market_shortlist_regret") {
    PLOGW << "Unknown repairHeuristic '" << repairHeuristic
          << "'; defaulting to 'regret'\n";
    repairHeuristic = "regret";
  }
  regretType = parameters.core.regretType;
  regretCandidateTopK_ = std::max(0, parameters.core.regretCandidateTopK);
  regretShortlistDiagnostics_ = parameters.core.regretShortlistDiagnostics;
  buildSuccessorPressureStaticSignals();
  maxCascadeFactor_ = parameters.core.maxCascadeFactor;
  if (!std::isfinite(maxCascadeFactor_)) {
    maxCascadeFactor_ = 0.0;
  }
  maxCascadeFactor_ = max(0.0, maxCascadeFactor_);
  maxCascadeTasks_ = std::max(0, parameters.core.maxCascadeTasks);
  adaptiveCascadeBudget_ = parameters.core.adaptiveCascadeBudget;
  adaptiveCascadeBudgetCurrent_ = cascadeTaskBudget();
  adaptiveCascadeBudgetLastUsed_ = adaptiveCascadeBudgetCurrent_;
  alnsEnablePrecedenceAwareDestroy_ =
      parameters.core.alnsEnablePrecedenceAwareDestroy;
  if (parameters.lowLevel.planner == "sipps") {
    lowLevelPlannerType_ = LowLevelPlannerType::sipps;
  } else {
    lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  }
  lowLevelSegmentTimeout_ = max(0.0, parameters.lowLevel.segmentTimeout);
  lowLevelStructuralPrePrune_ = parameters.lowLevel.structuralPrePrune;
  plannerParityCheck_ = parameters.lowLevel.parityCheck;
  market_.heuristics = parameters.market.heuristics;
  market_.bucketDt = max(1, parameters.market.bucketDt);
  market_.vertexBucketCapacity = max(1, parameters.market.vertexBucketCapacity);
  market_.edgeBucketCapacity = max(1, parameters.market.edgeBucketCapacity);
  market_.updateOnAcceptedOnly = parameters.market.updateOnAcceptedOnly;
  market_.updateFromCandidate = parameters.market.updateFromCandidate;
  market_.updatePeriodAccepted = max(1, parameters.market.updatePeriodAccepted);
  market_.eta = max(0.0, parameters.market.eta);
  market_.rho = parameters.market.rho;
  market_.priceCap = max(0.0, parameters.market.priceCap);
  market_.priceInit = max(0.0, parameters.market.priceInit);
  market_.gamma = max(0.0, parameters.market.gamma);
  market_.acceptanceGuards = parameters.market.acceptanceGuards;
  market_.tauP = max(0.0, parameters.market.tauP);
  market_.tauW = max(0.0, parameters.market.tauW);
  market_.destroyWeightPrice = max(0.0, parameters.market.destroyWeightPrice);
  market_.destroyWeightWait = max(0.0, parameters.market.destroyWeightWait);
  market_.destroyWeightRoot = max(0.0, parameters.market.destroyWeightRoot);
  market_.destroyWarmupUpdates = max(0, parameters.market.destroyWarmupUpdates);
  market_.destroyRequireStable = parameters.market.destroyRequireStable;
  market_.destroySoftGate = parameters.market.destroySoftGate;
  market_.destroyWarmupWeightScale =
      max(0.0, parameters.market.destroyWarmupWeightScale);
  market_.destroyUnstableWeightScale =
      max(0.0, parameters.market.destroyUnstableWeightScale);
  market_.destroyMinAlnsWeight =
      max(0.0, parameters.market.destroyMinAlnsWeight);
  market_.stabilityEmaAlpha =
      min(1.0, max(0.0, parameters.market.stabilityEmaAlpha));
  market_.stabilityMaxRelPriceDelta =
      max(0.0, parameters.market.stabilityMaxRelPriceDelta);
  market_.stabilityMaxTopMassDelta =
      max(0.0, parameters.market.stabilityMaxTopMassDelta);
  market_.stabilityMinContendedJaccard =
      min(1.0, max(0.0, parameters.market.stabilityMinContendedJaccard));
  market_.seedTopFrac = min(1.0, max(0.0, parameters.market.seedTopFrac));
  market_.randomDestroyQuota =
      min(1.0, max(0.0, parameters.market.randomDestroyQuota));
  market_.cooldownIters = max(0, parameters.market.cooldownIters);
  market_.dUp = max(0, parameters.market.dUp);
  market_.dDown = max(0, parameters.market.dDown);
  market_.closureCap = parameters.market.closureCap;
  market_.repairTieBreak = parameters.market.repairTieBreak;
  market_.repairBlend = parameters.market.repairBlend;
  market_.repairNormalizeByObservedPrice =
      parameters.market.repairNormalizeByObservedPrice;
  market_.tieBreakEpsSoc = max(0.0, parameters.market.tieBreakEpsSoc);
  market_.lambdaPrice = max(0.0, parameters.market.lambdaPrice);
  market_.lambdaWait = max(0.0, parameters.market.lambdaWait);
  if (market_.closureCap <= 0) {
    market_.closureCap = neighborSize_;
  }
  // Repair redesign: keep market influence as SoC tie-break only.
  if (market_.repairBlend) {
    PLOGW << "marketRepairBlend is deprecated in favor of tie-break-only "
             "repair scoring; enabling marketRepairTieBreak and disabling "
             "marketRepairBlend.\n";
    market_.repairBlend = false;
    market_.repairTieBreak = true;
  }
  market_.taskCooldownUntilIter.assign(instance_.getTasksNum(), 0);
  market_.stats.reset();
  market_.prevContendedVertices.clear();
  market_.prevContendedEdges.clear();
  market_.hasStabilityBaseline = false;
  market_.candidateUpdateConsumed = false;

  // Ensure both working solutions use the selected low-level planner.
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].pathPlanner = createSharedPlanner(agent);
    previousSolution_.agents[agent].pathPlanner = createSharedPlanner(agent);
  }

  incrementalRegret_ = parameters.core.incrementalRegret;
  if (parameters.core.incrementalRegretMode == "descendants") {
    incrementalRegretMode_ = IncrementalRegretMode::descendants;
  } else {
    incrementalRegretMode_ = IncrementalRegretMode::descendants_and_agent;
  }
  regretStamp_.assign(instance_.getTasksNum(), 0);
  regretBestOption_.assign(instance_.getTasksNum(), {UNASSIGNED, -1});
  regretSecondBestOption_.assign(instance_.getTasksNum(), {UNASSIGNED, -1});
  regretCandidateAgents_.assign(instance_.getTasksNum(), {});
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
