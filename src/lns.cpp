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
#include "utils.hpp"

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

  const uint64_t expandedBefore = solver.numExpanded;
  const uint64_t generatedBefore = solver.numGenerated;
  lowLevelCalls_++;
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

double LNS::elapsedRuntimeSec() const {
  return ((fsec)(Time::now() - plannerStartTime_)).count();
}

double LNS::remainingRuntimeBudgetSec() const {
  return max(0.0, timeLimit_ - elapsedRuntimeSec());
}

bool LNS::runtimeBudgetExhausted() const { return elapsedRuntimeSec() >= timeLimit_; }

void LNS::reservePathWithGoalPolicy(ConstraintTable& constraintTable,
                                    const AgentTaskPath& path,
                                    bool isFinalTask) const {
  if (path.empty()) {
    return;
  }
  if (!isFinalTask || goalOccupationMode_ == "stay") {
    constraintTable.addPath(path, isFinalTask);
    return;
  }

  // Tail/reposition modes reserve the traversal but release the terminal goal
  // after a bounded hold to avoid permanent bottlenecks.
  constraintTable.addPath(path, false);
  if (goalTailSteps_ <= 0) {
    return;
  }

  const int holdStart = path.endTimeChecked() + 1;
  if (holdStart >= MAX_TIMESTEP) {
    return;
  }
  const int holdEnd = min(MAX_TIMESTEP, holdStart + goalTailSteps_);
  if (holdEnd > holdStart) {
    constraintTable.insert2CT(path.back().location, holdStart, holdEnd);
  }
}

void LNS::reserveTerminalPathIfActive(ConstraintTable& constraintTable,
                                      int agent) const {
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
  const int serviceHorizon = computeActiveServiceHorizon();
  const int cappedEndExclusive =
      serviceHorizon + max(0, repositionReservationSlack_);
  if (cappedEndExclusive <= terminalPath.beginTime) {
    return;
  }
  if (terminalPath.endTimeChecked() < cappedEndExclusive) {
    constraintTable.addPath(terminalPath, false);
    return;
  }

  AgentTaskPath clippedTerminal;
  clippedTerminal.beginTime = terminalPath.beginTime;
  const int clippedSize =
      min((int)terminalPath.size(), cappedEndExclusive - terminalPath.beginTime);
  if (clippedSize <= 0) {
    return;
  }
  clippedTerminal.path.insert(clippedTerminal.path.end(),
                              terminalPath.path.begin(),
                              terminalPath.path.begin() + clippedSize);
  constraintTable.addPath(clippedTerminal, false);

  // Hold the terminal end location through the capped horizon so CT occupancy
  // matches validator semantics (agent remains at terminal endpoint).
  const int holdStart = clippedTerminal.beginTime + (int)clippedTerminal.size();
  if (holdStart < cappedEndExclusive) {
    constraintTable.insert2CT(clippedTerminal.back().location, holdStart,
                              cappedEndExclusive);
  }
}

int LNS::computeActiveServiceHorizon() const {
  int horizon = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (solution_.agents[agent].path.empty()) {
      continue;
    }
    horizon = max(horizon, solution_.agents[agent].path.endTimeOrZero() + 1);
  }
  return horizon;
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
  vector<pair<int, int>> rankedCandidates;
  rankedCandidates.reserve(instance_.getTaskLocationsRef().size());
  unordered_set<int> seenLocations;
  for (int location : instance_.getTaskLocationsRef()) {
    if (location < 0 || location >= instance_.mapSize || location == finalGoal ||
        instance_.isObstacle(location) || seenLocations.count(location) > 0) {
      continue;
    }
    seenLocations.insert(location);
    rankedCandidates.emplace_back(
        instance_.getManhattanDistance(finalGoal, location), location);
  }
  std::sort(rankedCandidates.begin(), rankedCandidates.end(),
            [](const pair<int, int>& lhs, const pair<int, int>& rhs) {
              if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });
  vector<int> candidates;
  const int maxCandidates = max(1, repositionMaxCandidates_);
  candidates.reserve((size_t)maxCandidates);
  for (const auto& [distance, location] : rankedCandidates) {
    (void)distance;
    candidates.push_back(location);
    if ((int)candidates.size() >= maxCandidates) {
      break;
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

vector<int> LNS::buildRollbackAgentHints(const vector<int>& baseAgents) const {
  const int agentCount = instance_.getAgentNum();
  vector<char> marked(agentCount, 0);
  vector<int> result;
  result.reserve(baseAgents.size() + (size_t)agentCount / 4 + 1);

  auto markAgent = [&](int agent) {
    if (agent < 0 || agent >= agentCount || marked[agent]) {
      return;
    }
    marked[agent] = 1;
    result.push_back(agent);
  };

  for (int agent : baseAgents) {
    markAgent(agent);
  }

  // Cross-agent insertions can introduce touched agents outside the destroy
  // closure; assignment drift is a robust signal for that case.
  for (int agent = 0; agent < agentCount; agent++) {
    if (solution_.agents[agent].taskAssignments !=
        previousSolution_.agents[agent].taskAssignments) {
      markAgent(agent);
    }
  }

  return result;
}

void LNS::restoreSolutionFromPrevious(const vector<int>* agentHints) {
  solutionRestoreStats_.restoreCalls++;

  auto fullRestore = [&]() {
    solution_ = previousSolution_;
    solutionRestoreStats_.fullRestores++;
  };

  if (!partialSolutionRestore_) {
    fullRestore();
    return;
  }

  if (agentHints == nullptr || agentHints->empty()) {
    solutionRestoreStats_.partialRestoreFallbacks++;
    fullRestore();
    return;
  }

  if (solution_.agents.size() != previousSolution_.agents.size() ||
      solution_.taskAgentMap.size() != previousSolution_.taskAgentMap.size()) {
    solutionRestoreStats_.partialRestoreFallbacks++;
    fullRestore();
    return;
  }

  const int agentCount = instance_.getAgentNum();
  vector<char> marked(agentCount, 0);
  int touchedAgents = 0;
  for (int agent : *agentHints) {
    if (agent < 0 || agent >= agentCount || marked[agent]) {
      continue;
    }
    marked[agent] = 1;
    touchedAgents++;
  }

  if (touchedAgents == 0) {
    solutionRestoreStats_.partialRestoreFallbacks++;
    fullRestore();
    return;
  }

  solution_.numOfTasks = previousSolution_.numOfTasks;
  solution_.numOfAgents = previousSolution_.numOfAgents;
  solution_.sumOfCosts = previousSolution_.sumOfCosts;
  solution_.utility = previousSolution_.utility;
  solution_.taskAgentMap = previousSolution_.taskAgentMap;

  for (int agent = 0; agent < agentCount; agent++) {
    if (!marked[agent]) {
      continue;
    }
    solution_.agents[agent] = previousSolution_.agents[agent];
  }

  solutionRestoreStats_.partialRestores++;
  solutionRestoreStats_.partialAgentsRestored += touchedAgents;
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
  rejectInvalidCandidates_ = parameters.core.rejectInvalidCandidates;
  utilityUseConflictEventCount_ = parameters.core.utilityUseConflictEventCount;
  acceptanceFeasibilityFirstPrecedenceDebt_ =
      parameters.core.acceptanceFeasibilityFirstPrecedenceDebt;
  acceptanceInvalidSpatialWeight_ =
      parameters.core.acceptanceInvalidSpatialWeight;
  acceptanceInvalidPrecedenceDebtWeight_ =
      parameters.core.acceptanceInvalidPrecedenceDebtWeight;
  acceptanceInvalidSocTieBreakWeight_ =
      parameters.core.acceptanceInvalidSocTieBreakWeight;
  acceptanceUseDedicatedInvalidTemperature_ =
      parameters.core.acceptanceUseDedicatedInvalidTemperature;
  acceptanceInvalidTemperatureScale_ =
      parameters.core.acceptanceInvalidTemperatureScale;
  acceptanceInvalidTemperatureFloor_ =
      parameters.core.acceptanceInvalidTemperatureFloor;
  if (!std::isfinite(acceptanceInvalidSpatialWeight_)) {
    acceptanceInvalidSpatialWeight_ = 1.0;
  }
  if (!std::isfinite(acceptanceInvalidPrecedenceDebtWeight_)) {
    acceptanceInvalidPrecedenceDebtWeight_ = 1.0;
  }
  if (!std::isfinite(acceptanceInvalidSocTieBreakWeight_)) {
    acceptanceInvalidSocTieBreakWeight_ = 0.0;
  }
  if (!std::isfinite(acceptanceInvalidTemperatureScale_)) {
    acceptanceInvalidTemperatureScale_ = 0.25;
  }
  if (!std::isfinite(acceptanceInvalidTemperatureFloor_)) {
    acceptanceInvalidTemperatureFloor_ = 1e-3;
  }
  acceptanceInvalidSpatialWeight_ = max(0.0, acceptanceInvalidSpatialWeight_);
  acceptanceInvalidPrecedenceDebtWeight_ =
      max(0.0, acceptanceInvalidPrecedenceDebtWeight_);
  acceptanceInvalidSocTieBreakWeight_ =
      max(0.0, acceptanceInvalidSocTieBreakWeight_);
  acceptanceInvalidTemperatureScale_ = max(1e-9, acceptanceInvalidTemperatureScale_);
  acceptanceInvalidTemperatureFloor_ = max(1e-9, acceptanceInvalidTemperatureFloor_);
  invalidTemperatureInitialized_ = false;
  invalidTemperature_ = 0.0;
  invalidInitialTemperature_ = 0.0;
  invalidMaxTemperature_ = std::numeric_limits<double>::infinity();
  invalidGreatDelugeDecay_ = 0.0;
  if (acceptanceFeasibilityFirstPrecedenceDebt_ &&
      acceptanceInvalidSpatialWeight_ == 0.0 &&
      acceptanceInvalidPrecedenceDebtWeight_ == 0.0 &&
      acceptanceInvalidSocTieBreakWeight_ == 0.0) {
    PLOGW << "acceptanceFeasibilityFirstPrecedenceDebt enabled with all "
             "invalid-score weights at 0; defaulting precedence-debt weight "
             "to 1.0\n";
    acceptanceInvalidPrecedenceDebtWeight_ = 1.0;
  }
  initialSolutionStrategy = parameters.core.initialSolutionStrategy;
  initialSolutionFallback = parameters.core.initialSolutionFallback;
  initialPortfolioTimeFraction_ =
      parameters.core.initialPortfolioTimeFraction;
  initialPortfolioMinArmTimeSec_ =
      parameters.core.initialPortfolioMinArmTimeSec;
  initialPortfolioStopOnFirstFeasible_ =
      parameters.core.initialPortfolioStopOnFirstFeasible;
  if (!std::isfinite(initialPortfolioTimeFraction_)) {
    initialPortfolioTimeFraction_ = 0.10;
  }
  if (!std::isfinite(initialPortfolioMinArmTimeSec_)) {
    initialPortfolioMinArmTimeSec_ = 1.0;
  }
  initialPortfolioTimeFraction_ =
      min(1.0, max(0.0, initialPortfolioTimeFraction_));
  initialPortfolioMinArmTimeSec_ = max(1e-6, initialPortfolioMinArmTimeSec_);
  initialSolutionRequested_ = initialSolutionStrategy;
  initialSolutionEffective_ = initialSolutionStrategy;
  initialSolutionFallbackUsed_ = false;
  initialSolutionFallbackReason_ = "none";
  goalOccupationMode_ = parameters.core.goalOccupationMode;
  goalTailSteps_ = max(0, parameters.core.goalTailSteps);
  repositionMaxCandidates_ = max(1, parameters.core.repositionMaxCandidates);
  repositionDemandLookahead_ =
      max(0, parameters.core.repositionDemandLookahead);
  repositionReservationSlack_ =
      max(0, parameters.core.repositionReservationSlack);
  greedySegmentDiagnostics_ = parameters.core.greedySegmentDiagnostics;
  greedySegmentDiagnosticsTopK_ =
      max(1, parameters.core.greedySegmentDiagnosticsTopK);
  terminalRepositionStats_.reset();
  parkingCandidatesCache_.clear();
  if (goalOccupationMode_ != "stay" && goalOccupationMode_ != "tail" &&
      goalOccupationMode_ != "reposition" &&
      goalOccupationMode_ != "reposition_true") {
    PLOGW << "Unknown goalOccupationMode '" << goalOccupationMode_
          << "'; defaulting to 'stay'\n";
    goalOccupationMode_ = "stay";
  }
  if (goalOccupationMode_ == "reposition" && goalTailSteps_ <= 0) {
    goalTailSteps_ = 1;
    PLOGW << "goalOccupationMode='reposition' currently uses finite tail "
             "release; applying goalTailSteps=1 by default.\n";
  }
  destroyHeuristic = parameters.core.destroyHeuristic;
  acceptanceCriteria = parameters.core.acceptanceCriteria;
  repairHeuristic = parameters.core.repairHeuristic;
  if (repairHeuristic != "regret" &&
      repairHeuristic != "market_shortlist_regret") {
    PLOGW << "Unknown repairHeuristic '" << repairHeuristic
          << "'; defaulting to 'regret'\n";
    repairHeuristic = "regret";
  }
  regretType = parameters.core.regretType;
  regretCandidateTopK_ = std::max(0, parameters.core.regretCandidateTopK);
  adaptiveRegretTopK_ = parameters.core.adaptiveRegretTopK;
  if (adaptiveRegretTopK_ && regretCandidateTopK_ <= 0) {
    PLOGW << "adaptiveRegretTopK requires regretCandidateTopK > 0; disabling adaptive regret Top-K\n";
    adaptiveRegretTopK_ = false;
  }
  adaptiveRegretTopKCurrent_ = regretCandidateTopK_;
  adaptiveRegretTopKLastUsed_ = adaptiveRegretTopKCurrent_;
  regretShortlistUseNormalizedWaitProxy_ =
      parameters.core.regretShortlistUseNormalizedWaitProxy;
  regretShortlistUseNormalizedSuccessorPressure_ =
      parameters.core.regretShortlistUseNormalizedSuccessorPressure;
  regretShortlistClampSuccessorToPrecedenceRelease_ =
      parameters.core.regretShortlistClampSuccessorToPrecedenceRelease;
  regretShortlistUseDescendantWeightedSuccessorPressure_ =
      parameters.core.regretShortlistUseDescendantWeightedSuccessorPressure;
  regretShortlistSuccessorPressureDepthDecay_ =
      parameters.core.regretShortlistSuccessorPressureDepthDecay;
  regretShortlistSuccessorPressureMaxDepth_ =
      std::max(0, parameters.core.regretShortlistSuccessorPressureMaxDepth);
  regretShortlistDiagnostics_ = parameters.core.regretShortlistDiagnostics;
  if (!std::isfinite(regretShortlistSuccessorPressureDepthDecay_)) {
    regretShortlistSuccessorPressureDepthDecay_ = 0.5;
  }
  regretShortlistSuccessorPressureDepthDecay_ =
      min(1.0, max(0.0, regretShortlistSuccessorPressureDepthDecay_));
  buildSuccessorPressureStaticSignals();
  maxCascadeFactor_ = parameters.core.maxCascadeFactor;
  if (!std::isfinite(maxCascadeFactor_)) {
    maxCascadeFactor_ = 3.0;
  }
  maxCascadeFactor_ = max(0.0, maxCascadeFactor_);
  maxCascadeTasks_ = std::max(0, parameters.core.maxCascadeTasks);
  adaptiveCascadeBudget_ = parameters.core.adaptiveCascadeBudget;
  adaptiveCascadeBudgetCurrent_ = cascadeTaskBudget();
  adaptiveCascadeBudgetLastUsed_ = adaptiveCascadeBudgetCurrent_;
  repairIncludeNonAncestorAgents_ =
      parameters.core.repairIncludeNonAncestorAgents;
  alnsEnablePrecedenceAwareDestroy_ =
      parameters.core.alnsEnablePrecedenceAwareDestroy;
  if (parameters.lowLevel.planner == "sipps") {
    lowLevelPlannerType_ = LowLevelPlannerType::sipps;
  } else {
    lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  }
  lowLevelSegmentTimeout_ = max(0.0, parameters.lowLevel.segmentTimeout);
  plannerParityCheck_ = parameters.lowLevel.parityCheck;
  plannerParityMaxLogs_ = parameters.lowLevel.parityMaxLogs;
  mlastarIncrementalFocalRefresh_ =
      parameters.lowLevel.mlastarIncrementalFocalRefresh;
  partialSolutionRestore_ = parameters.core.partialSolutionRestore;
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
}

void LNS::buildSuccessorPressureStaticSignals() {
  const int taskCount = instance_.getTasksNum();
  successorPressureStaticSignalsByTask_.assign(taskCount, {});
  const auto& successors = instance_.getSuccessorsRef();

  for (int task = 0; task < taskCount; task++) {
    auto& signals = successorPressureStaticSignalsByTask_[task];
    if (task < 0 || task >= (int)successors.size()) {
      continue;
    }

    if (!regretShortlistUseDescendantWeightedSuccessorPressure_) {
      signals.reserve(successors[task].size());
      for (int successor : successors[task]) {
        if (successor < 0 || successor >= taskCount) {
          continue;
        }
        signals.push_back({successor, 1, 1.0});
      }
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
      if (regretShortlistSuccessorPressureMaxDepth_ > 0 &&
          currentDepth >= regretShortlistSuccessorPressureMaxDepth_) {
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
        if (regretShortlistSuccessorPressureMaxDepth_ > 0 &&
            candidateDepth > regretShortlistSuccessorPressureMaxDepth_) {
          continue;
        }
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
      const double weight =
          std::pow(regretShortlistSuccessorPressureDepthDecay_,
                   static_cast<double>(depth - 1));
      if (weight <= 0.0) {
        continue;
      }
      signals.push_back({descendant, depth, weight});
    }
  }
}

std::shared_ptr<SingleAgentSolver> LNS::createSharedPlanner(int agent) const {
  std::shared_ptr<SingleAgentSolver> planner;
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      planner = std::make_shared<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_shared<MultiLabelSpaceTimeAStar>(
          instance_, agent, true, mlastarIncrementalFocalRefresh_);
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
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_, false);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_unique<MultiLabelSpaceTimeAStar>(
          instance_, agent, false, mlastarIncrementalFocalRefresh_);
      break;
  }
  planner->setSegmentTimeout(lowLevelSegmentTimeout_);
  return planner;
}

