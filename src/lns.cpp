#include "lns.hpp"
#include <boost/process.hpp>
#if defined(__has_include)
#if __has_include(<boost/process/null.hpp>)
#include <boost/process/null.hpp>
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 1
#else
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 0
#endif
#else
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 0
#endif
#include <cmath>
#include <limits>
#include <filesystem>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <utility>
#include "common.hpp"
#include "utils.hpp"

namespace {
struct TaskAssignmentIndex {
  vector<int> owner;
  vector<int> pos;
};

int clampSocToInt(long long soc, const char* context) {
  if (soc > std::numeric_limits<int>::max()) {
    PLOGW << context << ": sum of costs overflowed int; clamping to INT_MAX\n";
    return std::numeric_limits<int>::max();
  }
  if (soc < std::numeric_limits<int>::min()) {
    PLOGW << context << ": sum of costs underflowed int; clamping to INT_MIN\n";
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(soc);
}

TaskAssignmentIndex buildTaskAssignmentIndex(
    const vector<vector<int>>& assignments, int numTasks) {
  TaskAssignmentIndex index;
  index.owner.assign(numTasks, UNASSIGNED);
  index.pos.assign(numTasks, -1);
  int duplicateTaskOwners = 0;
  for (int agent = 0; agent < (int)assignments.size(); agent++) {
    for (int p = 0; p < (int)assignments[agent].size(); p++) {
      const int task = assignments[agent][p];
      if (task < 0 || task >= numTasks) {
        continue;
      }
      if (index.owner[task] != UNASSIGNED) {
        duplicateTaskOwners++;
        if (duplicateTaskOwners <= 5) {
          PLOGE << "Duplicate task ownership detected while building assignment index"
                << " for task " << task << " (existing agent="
                << index.owner[task] << ", new agent=" << agent << ")\n";
        }
      }
      index.owner[task] = agent;
      index.pos[task] = p;
    }
  }
  if (duplicateTaskOwners > 5) {
    PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
          << " tasks while building assignment index.\n";
  }
  return index;
}

TaskAssignmentIndex buildCurrentTaskAssignmentIndex(const Solution& solution,
                                                    int numTasks) {
  TaskAssignmentIndex index;
  index.owner.assign(numTasks, UNASSIGNED);
  index.pos.assign(numTasks, -1);
  int duplicateTaskOwners = 0;
  for (int agent = 0; agent < solution.numOfAgents; agent++) {
    const auto& assignments = solution.agents[agent].taskAssignments;
    for (int p = 0; p < (int)assignments.size(); p++) {
      const int task = assignments[p];
      if (task < 0 || task >= numTasks) {
        continue;
      }
      if (index.owner[task] != UNASSIGNED) {
        duplicateTaskOwners++;
        if (duplicateTaskOwners <= 5) {
          PLOGE << "Duplicate task ownership detected in current solution for task "
                << task << " (existing agent=" << index.owner[task]
                << ", new agent=" << agent << ")\n";
        }
      }
      index.owner[task] = agent;
      index.pos[task] = p;
    }
  }
  if (duplicateTaskOwners > 5) {
    PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
          << " tasks in current solution assignment index.\n";
  }
  return index;
}

string summarizeIntervals(const vector<pair<int, int>>* intervals,
                          int maxIntervals = 6) {
  if (intervals == nullptr || intervals->empty()) {
    return "(none)";
  }
  std::ostringstream oss;
  const int limit = min((int)intervals->size(), maxIntervals);
  for (int i = 0; i < limit; i++) {
    if (i > 0) {
      oss << ", ";
    }
    oss << "[" << (*intervals)[i].first << ", ";
    if ((*intervals)[i].second == MAX_TIMESTEP) {
      oss << "INF";
    } else {
      oss << (*intervals)[i].second;
    }
    oss << ")";
  }
  if ((int)intervals->size() > limit) {
    oss << ", ... (" << intervals->size() << " total)";
  }
  return oss.str();
}

int permanentOccupancyStart(const vector<pair<int, int>>* intervals) {
  if (intervals == nullptr) {
    return MAX_TIMESTEP;
  }
  int start = MAX_TIMESTEP;
  for (const auto& interval : *intervals) {
    if (interval.second == MAX_TIMESTEP) {
      start = min(start, interval.first);
    }
  }
  return start;
}

int firstFreeTimeAtOrAfter(const vector<pair<int, int>>* intervals,
                           int timestep) {
  if (intervals == nullptr) {
    return timestep;
  }
  int t = timestep;
  for (const auto& interval : *intervals) {
    if (t < interval.first) {
      break;
    }
    if (interval.first <= t && t < interval.second) {
      if (interval.second == MAX_TIMESTEP) {
        return MAX_TIMESTEP;
      }
      t = interval.second;
    }
  }
  return t;
}

bool reachableWithPermanentBlocksByTime(const Instance& instance,
                                        const ConstraintTable& constraintTable,
                                        int start, int goal, int byTime) {
  if (start < 0 || goal < 0 || start >= instance.mapSize ||
      goal >= instance.mapSize) {
    return false;
  }

  const auto isBlocked = [&](int loc) {
    if (instance.isObstacle(loc)) {
      return true;
    }
    const int blockedFrom =
        permanentOccupancyStart(constraintTable.getConstraintIntervals(loc));
    return blockedFrom != MAX_TIMESTEP && blockedFrom <= byTime;
  };

  if (isBlocked(start) || isBlocked(goal)) {
    return false;
  }

  vector<char> seen(instance.mapSize, 0);
  std::queue<int> q;
  q.push(start);
  seen[start] = 1;
  while (!q.empty()) {
    const int curr = q.front();
    q.pop();
    if (curr == goal) {
      return true;
    }
    for (int nxt : instance.getNeighbors(curr)) {
      if (nxt < 0 || nxt >= instance.mapSize || seen[nxt] || isBlocked(nxt)) {
        continue;
      }
      seen[nxt] = 1;
      q.push(nxt);
    }
  }
  return false;
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

void logInitialSegmentFailureDiagnostics(
    const Instance& instance, const Solution& solution,
    const ConstraintTable& constraintTable, int agent, int task, int taskPosition,
    int startTime, SingleAgentSolver& solver, double remainingBudgetSec,
    double effectiveTimeoutSec) {
  const int goalLocation = instance.getTaskLocations(task);
  int startLocation = instance.getStartLocationsRef()[agent];
  if (taskPosition > 0) {
    const int prevTask = solution.getAgentGlobalTasks(agent, taskPosition - 1);
    startLocation = instance.getTaskLocations(prevTask);
  }

  const auto* goalIntervals = constraintTable.getConstraintIntervals(goalLocation);
  const auto* startIntervals = constraintTable.getConstraintIntervals(startLocation);
  const int hDist = solver.getStageGoalDistance(taskPosition, startLocation);
  const int earliestArrivalLb =
      (hDist >= MAX_TIMESTEP / 2) ? MAX_TIMESTEP : startTime + hDist;
  const int goalFreeAt = firstFreeTimeAtOrAfter(goalIntervals, earliestArrivalLb);
  const int goalPermanentFrom = permanentOccupancyStart(goalIntervals);
  const int stableTime = max(startTime, constraintTable.temporalExtent + 1);
  int constrainedVerticesAtStable = 0;
  int permanentVerticesByStable = 0;
  for (int loc = 0; loc < instance.mapSize; loc++) {
    if (constraintTable.constrained(loc, stableTime)) {
      constrainedVerticesAtStable++;
    }
    const int blockedFrom =
        permanentOccupancyStart(constraintTable.getConstraintIntervals(loc));
    if (blockedFrom != MAX_TIMESTEP && blockedFrom <= stableTime) {
      permanentVerticesByStable++;
    }
  }
  const bool permanentStaticReachable = reachableWithPermanentBlocksByTime(
      instance, constraintTable, startLocation, goalLocation, stableTime);

  int feasibleImmediateMoves = 0;
  for (int nxt : instance.getNeighbors(startLocation)) {
    const bool blockedVertex = constraintTable.constrained(nxt, startTime + 1);
    const bool blockedEdge =
        constraintTable.constrained(startLocation, nxt, startTime + 1);
    if (!blockedVertex && !blockedEdge) {
      feasibleImmediateMoves++;
    }
  }
  const bool canWaitAtStart =
      !constraintTable.constrained(startLocation, startTime + 1);

  PLOGE << "Init LL-failure diagnostics: agent=" << agent << ", task=" << task
        << ", stage=" << taskPosition << ", start_loc=" << startLocation
        << ", goal_loc=" << goalLocation << ", start_time=" << startTime
        << ", hdist_to_goal=" << hDist
        << ", earliest_arrival_lb=" << earliestArrivalLb
        << ", goal_first_free_at_or_after_lb="
        << (goalFreeAt == MAX_TIMESTEP ? -1 : goalFreeAt)
        << ", goal_permanent_block_from="
        << (goalPermanentFrom == MAX_TIMESTEP ? -1 : goalPermanentFrom)
        << ", stable_time=" << stableTime
        << ", constrained_vertices_at_stable=" << constrainedVerticesAtStable
        << ", permanent_vertices_by_stable=" << permanentVerticesByStable
        << ", permanent_static_reachable="
        << (permanentStaticReachable ? "true" : "false")
        << ", feasible_immediate_moves_from_start=" << feasibleImmediateMoves
        << ", can_wait_at_start=" << (canWaitAtStart ? "true" : "false")
        << ", remaining_budget_sec=" << remainingBudgetSec
        << ", effective_timeout_sec=" << effectiveTimeoutSec << "\n";
  PLOGE << "Init LL-failure diagnostics: goal_intervals="
        << summarizeIntervals(goalIntervals)
        << ", start_intervals=" << summarizeIntervals(startIntervals) << "\n";

  if (goalPermanentFrom != MAX_TIMESTEP &&
      goalPermanentFrom <= earliestArrivalLb) {
    PLOGE << "Init LL-failure diagnostics: certificate=goal_permanently_occupied"
          << "_before_earliest_arrival_lb\n";
  }
  if (!canWaitAtStart && feasibleImmediateMoves == 0) {
    PLOGE << "Init LL-failure diagnostics: certificate=start_trapped_at_t+1\n";
  }
  if (!permanentStaticReachable) {
    PLOGE << "Init LL-failure diagnostics: certificate=static_disconnected_under_"
             "permanent_blocks\n";
  }

  MultiLabelSIPPS sippsCrossCheck(instance, agent, false);
  sippsCrossCheck.setGoalLocations(instance.getTaskLocations(
      solution.getAgentGlobalTasks(agent)));
  sippsCrossCheck.setSegmentTimeout(max(1e-6, min(30.0, remainingBudgetSec)));
  ConstraintTable sippsCt(constraintTable);
  const AgentTaskPath sippsPath = sippsCrossCheck.findPathSegment(
      sippsCt, startTime, taskPosition, 0);
  PLOGE << "Init LL-failure diagnostics: sipps_crosscheck_outcome="
        << sippsCrossCheck.getLastSearchOutcomeName()
        << ", sipps_path_size=" << sippsPath.size() << "\n";
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
  initialSolutionStrategy = parameters.core.initialSolutionStrategy;
  initialSolutionFallback = parameters.core.initialSolutionFallback;
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
  regretType = parameters.core.regretType;
  regretCandidateTopK_ = std::max(0, parameters.core.regretCandidateTopK);
  maxCascadeFactor_ = parameters.core.maxCascadeFactor;
  if (!std::isfinite(maxCascadeFactor_)) {
    maxCascadeFactor_ = 3.0;
  }
  maxCascadeFactor_ = max(0.0, maxCascadeFactor_);
  maxCascadeTasks_ = std::max(0, parameters.core.maxCascadeTasks);
  repairIncludeNonAncestorAgents_ =
      parameters.core.repairIncludeNonAncestorAgents;
  if (parameters.lowLevel.planner == "sipps") {
    lowLevelPlannerType_ = LowLevelPlannerType::sipps;
  } else {
    lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  }
  lowLevelSegmentTimeout_ = max(0.0, parameters.lowLevel.segmentTimeout);
  plannerParityCheck_ = parameters.lowLevel.parityCheck;
  plannerParityMaxLogs_ = parameters.lowLevel.parityMaxLogs;
  partialSolutionRestore_ = parameters.core.partialSolutionRestore;
  market_.heuristics = parameters.market.heuristics;
  market_.bucketDt = max(1, parameters.market.bucketDt);
  market_.vertexBucketCapacity = max(1, parameters.market.vertexBucketCapacity);
  market_.edgeBucketCapacity = max(1, parameters.market.edgeBucketCapacity);
  market_.updateOnAcceptedOnly = parameters.market.updateOnAcceptedOnly;
  market_.updatePeriodAccepted = max(1, parameters.market.updatePeriodAccepted);
  market_.eta = max(0.0, parameters.market.eta);
  market_.rho = parameters.market.rho;
  market_.priceCap = max(0.0, parameters.market.priceCap);
  market_.gamma = max(0.0, parameters.market.gamma);
  market_.acceptanceGuards = parameters.market.acceptanceGuards;
  market_.tauP = max(0.0, parameters.market.tauP);
  market_.tauW = max(0.0, parameters.market.tauW);
  market_.destroyWeightPrice = max(0.0, parameters.market.destroyWeightPrice);
  market_.destroyWeightWait = max(0.0, parameters.market.destroyWeightWait);
  market_.destroyWeightRoot = max(0.0, parameters.market.destroyWeightRoot);
  market_.destroyWarmupUpdates = max(0, parameters.market.destroyWarmupUpdates);
  market_.seedTopFrac = min(1.0, max(0.0, parameters.market.seedTopFrac));
  market_.randomDestroyQuota =
      min(1.0, max(0.0, parameters.market.randomDestroyQuota));
  market_.cooldownIters = max(0, parameters.market.cooldownIters);
  market_.dUp = max(0, parameters.market.dUp);
  market_.dDown = max(0, parameters.market.dDown);
  market_.closureCap = parameters.market.closureCap;
  market_.repairTieBreak = parameters.market.repairTieBreak;
  market_.repairBlend = parameters.market.repairBlend;
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

std::shared_ptr<SingleAgentSolver> LNS::createSharedPlanner(int agent) const {
  std::shared_ptr<SingleAgentSolver> planner;
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      planner = std::make_shared<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_shared<MultiLabelSpaceTimeAStar>(instance_, agent);
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
      planner = std::make_unique<MultiLabelSpaceTimeAStar>(instance_, agent, false);
      break;
  }
  planner->setSegmentTimeout(lowLevelSegmentTimeout_);
  return planner;
}

int LNS::marketTimeBucket(int timestep) const {
  if (market_.bucketDt <= 1) {
    return timestep;
  }
  return timestep / market_.bucketDt;
}

uint64_t LNS::makeMarketVertexKey(int location, int bucket) const {
  uint64_t x = 0;
  x ^= (uint64_t)(uint32_t)location;
  x ^= ((uint64_t)(uint32_t)bucket) << 32;
  x ^= 0x9E3779B97F4A7C15ULL;
  return LLNode::mix64(x);
}

uint64_t LNS::makeMarketEdgeKey(int from, int to, int bucket) const {
  uint64_t x = 0xD1B54A32D192ED03ULL;
  x = LLNode::mix64(x ^ ((uint64_t)(uint32_t)from + 0x9E3779B97F4A7C15ULL));
  x = LLNode::mix64(x ^ ((uint64_t)(uint32_t)to + 0xC2B2AE3D27D4EB4FULL));
  x = LLNode::mix64(x ^ ((uint64_t)(uint32_t)bucket + 0x165667B19E3779F9ULL));
  return x;
}

void LNS::computeTaskScheduleMetricsFromIndex(
    const vector<int>& taskPosByTask, vector<TaskScheduleMetrics>& perTask,
    vector<double>* blockedWaitSum) const {
  const int taskCount = instance_.getTasksNum();
  perTask.assign(taskCount, TaskScheduleMetrics());
  if (blockedWaitSum != nullptr) {
    blockedWaitSum->assign(taskCount, 0.0);
  }
  const auto& predecessors = instance_.getAncestorsRef();

  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    const int pos = (task >= 0 && task < (int)taskPosByTask.size())
                        ? taskPosByTask[task]
                        : -1;
    if (pos < 0) {
      continue;
    }
    if (pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
    if (taskPath.empty()) {
      continue;
    }

    TaskScheduleMetrics metric;
    metric.valid = true;
    metric.end = taskPath.endTime();
    const int taskLocation = instance_.getTaskLocations(task);
    int arrive = taskPath.endTime();
    for (int i = 0; i < (int)taskPath.size(); i++) {
      if (taskPath[i].location == taskLocation) {
        arrive = taskPath.beginTime + i;
        break;
      }
    }
    metric.arrive = arrive;
    metric.release = 0;
    metric.blocker = UNASSIGNED;

    if (task >= 0 && task < (int)predecessors.size()) {
      for (int pred : predecessors[task]) {
        const int predAgent =
            (pred >= 0 && pred < (int)solution_.taskAgentMap.size())
                ? solution_.taskAgentMap[pred]
                : UNASSIGNED;
        if (predAgent == UNASSIGNED) {
          continue;
        }
        const int predPos = (pred >= 0 && pred < (int)taskPosByTask.size())
                                ? taskPosByTask[pred]
                                : -1;
        if (predPos < 0) {
          continue;
        }
        if (predPos < 0 ||
            predPos >= (int)solution_.agents[predAgent].taskPaths.size()) {
          continue;
        }
        const AgentTaskPath& predPath = solution_.agents[predAgent].taskPaths[predPos];
        if (predPath.empty()) {
          continue;
        }
        const int predEnd = predPath.endTime();
        if (predEnd > metric.release) {
          metric.release = predEnd;
          metric.blocker = pred;
        }
      }
    }
    metric.start = max(metric.arrive, metric.release);
    metric.waitPrec = max(0, metric.start - metric.arrive);
    perTask[task] = metric;
    if (blockedWaitSum != nullptr && metric.blocker != UNASSIGNED) {
      (*blockedWaitSum)[metric.blocker] += metric.waitPrec;
    }
  }
}

void LNS::computeTaskScheduleMetrics(vector<TaskScheduleMetrics>& perTask,
                                     vector<double>* blockedWaitSum) const {
  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  computeTaskScheduleMetricsFromIndex(currentIndex.pos, perTask, blockedWaitSum);
}

double LNS::computeTaskMarketExposure(int task, bool normalized) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0.0;
  }
  const int agent = solution_.taskAgentMap[task];
  if (agent == UNASSIGNED) {
    return 0.0;
  }
  const int pos = solution_.getLocalTaskIndex(agent, task);
  if (pos == UNASSIGNED) {
    return 0.0;
  }
  if (pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
    return 0.0;
  }
  const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
  if (taskPath.empty()) {
    return 0.0;
  }
  return computeMarketExposureFromPath(taskPath, normalized);
}

double LNS::computeMarketExposureFromPath(const AgentTaskPath& taskPath,
                                          bool normalized) const {
  if (taskPath.empty()) {
    return 0.0;
  }

  auto weightedResourcePrice =
      [](const unordered_map<uint64_t, double>& prices,
         const unordered_map<uint64_t, double>& excessHat,
         uint64_t key) -> double {
    const auto pIt = prices.find(key);
    if (pIt == prices.end()) {
      return 0.0;
    }
    const auto eIt = excessHat.find(key);
    if (eIt == excessHat.end()) {
      return 0.0;
    }
    const double excessWeight = max(0.0, eIt->second);
    if (excessWeight <= 0.0) {
      return 0.0;
    }
    return pIt->second * excessWeight;
  };

  double totalExposure = 0.0;
  int activeResourceCount = 0;
  for (int i = 0; i < (int)taskPath.size(); i++) {
    const int timestep = taskPath.beginTime + i;
    const int bucket = marketTimeBucket(timestep);
    const int location = taskPath[i].location;
    const uint64_t vKey = makeMarketVertexKey(location, bucket);
    const double vertexContribution =
        weightedResourcePrice(market_.vertexPrices, market_.vertexExcessHat,
                              vKey);
    if (vertexContribution > 0.0) {
      totalExposure += vertexContribution;
      activeResourceCount++;
    }

    if (i > 0) {
      const int prevLocation = taskPath[i - 1].location;
      const uint64_t eKey = makeMarketEdgeKey(prevLocation, location, bucket);
      const double edgeContribution =
          weightedResourcePrice(market_.edgePrices, market_.edgeExcessHat, eKey);
      if (edgeContribution > 0.0) {
        totalExposure += edgeContribution;
        activeResourceCount++;
      }
    }
  }
  if (!normalized) {
    return totalExposure;
  }
  // Normalize only over resources with positive congestion-weighted
  // contribution, so long uncongested paths do not dominate the score.
  return totalExposure / max(1, activeResourceCount);
}

int LNS::computeTaskPrecedenceWaitFromState(
    int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
    const vector<vector<AgentTaskPath>>& agentTaskPaths) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }

  const TaskAssignmentIndex stateIndex =
      buildTaskAssignmentIndex(agentTaskAssignments, instance_.getTasksNum());
  const int taskAgent =
      (task >= 0 && task < (int)stateIndex.owner.size()) ? stateIndex.owner[task]
                                                          : UNASSIGNED;
  const int taskPos =
      (task >= 0 && task < (int)stateIndex.pos.size()) ? stateIndex.pos[task] : -1;
  if (taskAgent == UNASSIGNED || taskPos < 0 ||
      taskPos >= (int)agentTaskPaths[taskAgent].size()) {
    return 0;
  }

  const AgentTaskPath& taskPath = agentTaskPaths[taskAgent][taskPos];
  if (taskPath.empty()) {
    return 0;
  }

  int arrive = taskPath.endTime();
  for (int i = 0; i < (int)taskPath.size(); i++) {
    if (taskPath[i].location == taskLocation) {
      arrive = taskPath.beginTime + i;
      break;
    }
  }

  int release = 0;
  const int taskCount = instance_.getTasksNum();
  vector<char> seenPredecessor(taskCount, 0);
  auto consumePredecessor = [&](int pred) {
    if (pred < 0 || pred >= taskCount) {
      return;
    }
    if (seenPredecessor[pred]) {
      return;
    }
    seenPredecessor[pred] = 1;
    const int predAgent = (pred >= 0 && pred < (int)stateIndex.owner.size())
                              ? stateIndex.owner[pred]
                              : UNASSIGNED;
    const int predPos = (pred >= 0 && pred < (int)stateIndex.pos.size())
                            ? stateIndex.pos[pred]
                            : -1;
    if (predAgent != UNASSIGNED && predPos >= 0 &&
        predPos < (int)agentTaskPaths[predAgent].size() &&
        !agentTaskPaths[predAgent][predPos].empty()) {
      release = max(release, agentTaskPaths[predAgent][predPos].endTime());
    }
  };

  // Base input precedence predecessors can be queried in O(in-degree(task)).
  const auto& baseAncestors = instance_.getAncestorsRef();
  if (task >= 0 && task < (int)baseAncestors.size()) {
    for (int pred : baseAncestors[task]) {
      consumePredecessor(pred);
    }
  }

  // Dynamic intra-agent predecessor in the current assignment state.
  if (taskPos > 0 && taskAgent >= 0 &&
      taskAgent < (int)agentTaskAssignments.size() &&
      taskPos - 1 < (int)agentTaskAssignments[taskAgent].size()) {
    consumePredecessor(agentTaskAssignments[taskAgent][taskPos - 1]);
  }

  return max(0, release - arrive);
}

void LNS::buildMarketDemandFromCurrentOccupancy(
    unordered_map<uint64_t, int>& vertexDemand,
    unordered_map<uint64_t, int>& edgeDemand) const {
  vertexDemand.clear();
  edgeDemand.clear();

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const int horizon = getAgentOccupancyHorizon(agent, true);
    if (horizon <= 0) {
      continue;
    }

    int prevLocation = UNDEFINED;
    for (int timestep = 0; timestep < horizon; timestep++) {
      const int location = getAgentLocationAt(agent, timestep, true);
      if (location == UNDEFINED) {
        prevLocation = UNDEFINED;
        continue;
      }
      const int bucket = marketTimeBucket(timestep);
      vertexDemand[makeMarketVertexKey(location, bucket)]++;
      if (prevLocation != UNDEFINED) {
        edgeDemand[makeMarketEdgeKey(prevLocation, location, bucket)]++;
      }
      prevLocation = location;
    }
  }
}

int LNS::computeTaskPrecedenceWaitInCurrentSolution(int task) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }
  const int taskAgent = solution_.taskAgentMap[task];
  if (taskAgent == UNASSIGNED) {
    return 0;
  }

  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  const int agent = taskAgent;
  const int taskPos =
      (task >= 0 && task < (int)currentIndex.pos.size()) ? currentIndex.pos[task]
                                                          : -1;
  if (taskPos < 0) {
    return 0;
  }
  if (taskPos < 0 || taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
    return 0;
  }
  const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[taskPos];
  if (taskPath.empty()) {
    return 0;
  }

  const int taskLocation = instance_.getTaskLocations(task);
  int arrive = taskPath.endTime();
  for (int i = 0; i < (int)taskPath.size(); i++) {
    if (taskPath[i].location == taskLocation) {
      arrive = taskPath.beginTime + i;
      break;
    }
  }

  int release = 0;
  const int taskCount = instance_.getTasksNum();
  vector<char> seenPredecessor(taskCount, 0);
  auto consumePredecessor = [&](int pred) {
    if (pred < 0 || pred >= taskCount || seenPredecessor[pred]) {
      return;
    }
    seenPredecessor[pred] = 1;
    const int predAgent = (pred >= 0 && pred < (int)solution_.taskAgentMap.size())
                              ? solution_.taskAgentMap[pred]
                              : UNASSIGNED;
    const int predPos = (pred >= 0 && pred < (int)currentIndex.pos.size())
                            ? currentIndex.pos[pred]
                            : -1;
    if (predAgent != UNASSIGNED && predPos >= 0 &&
        predPos < (int)solution_.agents[predAgent].taskPaths.size() &&
        !solution_.agents[predAgent].taskPaths[predPos].empty()) {
      release = max(release, solution_.agents[predAgent].taskPaths[predPos].endTime());
    }
  };

  const auto& ancestors = instance_.getAncestorsRef();
  if (task >= 0 && task < (int)ancestors.size()) {
    for (int pred : ancestors[task]) {
      consumePredecessor(pred);
    }
  }
  if (taskPos > 0 && agent >= 0 && agent < instance_.getAgentNum() &&
      taskPos - 1 < (int)solution_.agents[agent].taskAssignments.size()) {
    consumePredecessor(solution_.agents[agent].taskAssignments[taskPos - 1]);
  }

  return max(0, release - arrive);
}

double LNS::computeSolutionMarketPressure() const {
  auto weightedResourcePrice =
      [](const unordered_map<uint64_t, double>& prices,
         const unordered_map<uint64_t, double>& excessHat,
         uint64_t key) -> double {
    const auto pIt = prices.find(key);
    if (pIt == prices.end()) {
      return 0.0;
    }
    const auto eIt = excessHat.find(key);
    if (eIt == excessHat.end()) {
      return 0.0;
    }
    const double excessWeight = max(0.0, eIt->second);
    if (excessWeight <= 0.0) {
      return 0.0;
    }
    return pIt->second * excessWeight;
  };

  unordered_map<uint64_t, int> vertexDemand;
  unordered_map<uint64_t, int> edgeDemand;
  buildMarketDemandFromCurrentOccupancy(vertexDemand, edgeDemand);

  double pressure = 0.0;
  for (const auto& kv : vertexDemand) {
    const double unitContribution =
        weightedResourcePrice(market_.vertexPrices, market_.vertexExcessHat, kv.first);
    if (unitContribution <= 0.0) {
      continue;
    }
    pressure += unitContribution * kv.second;
  }
  for (const auto& kv : edgeDemand) {
    const double unitContribution =
        weightedResourcePrice(market_.edgePrices, market_.edgeExcessHat, kv.first);
    if (unitContribution <= 0.0) {
      continue;
    }
    pressure += unitContribution * kv.second;
  }
  return pressure;
}

double LNS::computeSolutionPrecedenceWaitFromIndex(
    const vector<int>& taskPosByTask) const {
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetricsFromIndex(taskPosByTask, perTask, nullptr);
  double totalWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
  }
  return totalWait;
}

double LNS::computeSolutionPrecedenceWait() const {
  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  return computeSolutionPrecedenceWaitFromIndex(currentIndex.pos);
}

bool LNS::passMarketAcceptanceGuards(double previousPressure,
                                     double candidatePressure,
                                     double previousWait,
                                     double candidateWait,
                                     bool candidateIsWorse) const {
  if (!market_.acceptanceGuards) {
    return true;
  }
  // Keep exploration intact: apply guards only for non-improving candidates.
  if (!candidateIsWorse) {
    return true;
  }
  if (!std::isfinite(previousPressure) || !std::isfinite(previousWait)) {
    return true;
  }
  const bool pressureOK = candidatePressure <= previousPressure + market_.tauP;
  const bool waitOK = candidateWait <= previousWait + market_.tauW;
  return pressureOK && waitOK;
}

void LNS::updateMarketStateFromCurrentSolution() {
  if (!market_.heuristics) {
    return;
  }

  unordered_map<uint64_t, int> vertexDemand;
  unordered_map<uint64_t, int> edgeDemand;
  buildMarketDemandFromCurrentOccupancy(vertexDemand, edgeDemand);

  const double eta =
      market_.eta / std::sqrt(1.0 + (double)market_.stats.updates);

  int64_t contendedResources = 0;
  double contendedPriceSum = 0.0;
  double maxPrice = 0.0;

  auto updateCategory = [&](const unordered_map<uint64_t, int>& demand,
                            unordered_map<uint64_t, double>& prices,
                            unordered_map<uint64_t, double>& excessHat,
                            int bucketCapacity) {
    vector<uint64_t> keys;
    keys.reserve(demand.size() + prices.size() + excessHat.size());
    for (const auto& kv : demand) {
      keys.push_back(kv.first);
    }
    for (const auto& kv : prices) {
      keys.push_back(kv.first);
    }
    for (const auto& kv : excessHat) {
      keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    unordered_map<uint64_t, double> nextPrices;
    unordered_map<uint64_t, double> nextExcessHat;
    nextPrices.reserve(prices.size() + demand.size());
    nextExcessHat.reserve(excessHat.size() + demand.size());

    for (uint64_t key : keys) {
      const auto itDemand = demand.find(key);
      const int d = (itDemand == demand.end()) ? 0 : itDemand->second;
      const int excess = max(0, d - max(1, bucketCapacity));
      const auto itOldPrice = prices.find(key);
      const auto itOldHat = excessHat.find(key);
      const double oldPrice = (itOldPrice == prices.end()) ? 0.0 : itOldPrice->second;
      const double oldHat = (itOldHat == excessHat.end()) ? 0.0 : itOldHat->second;
      const double newHat = market_.rho * oldHat + (1.0 - market_.rho) * excess;

      constexpr double kPriceInit = 1e-3;
      constexpr double kExcessHatEps = 1e-9;
      double newPrice = oldPrice;
      if (newHat > kExcessHatEps) {
        const double basePrice = (oldPrice > 0.0) ? oldPrice : kPriceInit;
        newPrice = min(market_.priceCap, basePrice * std::exp(eta * newHat));
      } else {
        newPrice = oldPrice * max(0.0, 1.0 - market_.gamma);
      }

      if (newPrice > 1e-9 || newHat > 1e-9) {
        nextPrices[key] = newPrice;
        nextExcessHat[key] = newHat;
      }
      if (excess > 0) {
        contendedResources++;
        contendedPriceSum += newPrice;
        maxPrice = max(maxPrice, newPrice);
      }
    }

    prices.swap(nextPrices);
    excessHat.swap(nextExcessHat);
  };

  updateCategory(vertexDemand, market_.vertexPrices, market_.vertexExcessHat,
                 market_.vertexBucketCapacity);
  updateCategory(edgeDemand, market_.edgePrices, market_.edgeExcessHat,
                 market_.edgeBucketCapacity);

  vector<double> allPrices;
  allPrices.reserve(market_.vertexPrices.size() + market_.edgePrices.size());
  double totalPriceMass = 0.0;
  for (const auto& kv : market_.vertexPrices) {
    if (kv.second > 0.0) {
      allPrices.push_back(kv.second);
      totalPriceMass += kv.second;
    }
  }
  for (const auto& kv : market_.edgePrices) {
    if (kv.second > 0.0) {
      allPrices.push_back(kv.second);
      totalPriceMass += kv.second;
    }
  }
  double topPriceMassFrac = 0.0;
  if (!allPrices.empty() && totalPriceMass > 1e-12) {
    std::sort(allPrices.begin(), allPrices.end(), std::greater<double>());
    const int topK = max(1, (int)std::ceil((double)allPrices.size() * 0.01));
    double topMass = 0.0;
    for (int i = 0; i < topK && i < (int)allPrices.size(); i++) {
      topMass += allPrices[i];
    }
    topPriceMassFrac = topMass / totalPriceMass;
  }

  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetricsFromIndex(currentIndex.pos, perTask, nullptr);
  double totalWait = 0.0;
  double maxWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
    maxWait = max(maxWait, (double)metric.waitPrec);
  }

  market_.stats.updates++;
  market_.stats.contendedResources = contendedResources;
  market_.stats.meanPriceContended =
      contendedResources > 0 ? contendedPriceSum / (double)contendedResources
                             : 0.0;
  market_.stats.maxPrice = maxPrice;
  market_.stats.topPriceMassFrac = topPriceMassFrac;
  market_.stats.totalPrecedenceWait = totalWait;
  market_.stats.maxPrecedenceWait = maxWait;
}

void LNS::maybeUpdateMarketState(bool accepted) {
  if (!market_.heuristics) {
    return;
  }
  if (market_.updateOnAcceptedOnly) {
    if (!accepted) {
      return;
    }
    market_.acceptedCounter++;
    if (market_.acceptedCounter % market_.updatePeriodAccepted != 0) {
      return;
    }
  } else {
    market_.updateCounter++;
    if (market_.updateCounter % market_.updatePeriodAccepted != 0) {
      return;
    }
  }
  updateMarketStateFromCurrentSolution();
}


bool LNS::buildGreedySolutionWithMAPFPC(const string& variant) {

  // Reset solution state in case this is called more than once.
  Solution freshSolution(instance_);
  solution_ = freshSolution;

  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  bool readingTaskAssignments = false, readingTaskPaths = false;

  auto parseInt = [](const std::string& s) -> std::optional<int> {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      return std::nullopt;
    }
    const size_t end = s.find_last_not_of(" \t\r\n");
    try {
      return std::stoi(s.substr(begin, end - begin + 1));
    } catch (...) {
      return std::nullopt;
    }
  };

  string solver;
  if (variant == "sota_cbs") {
    solver = "CBS";
  } else if (variant == "sota_pbs") {
    solver = "PBS";
  } else {
    PLOGE << "Initial solution solver using MAPF-PC variant not supported\n";
    return false;
  }

  // Run a child process to spawn the MAPC-PC codebase with the current map and agent informations
  namespace bp = boost::process;
  bp::ipstream inputStream;
  std::string taskAssignmentExe;
  if (std::filesystem::exists("./MAPF-PC/build_local/bin/task_assignment")) {
    taskAssignmentExe = "./MAPF-PC/build_local/bin/task_assignment";
  } else {
    taskAssignmentExe = "./MAPF-PC/build/bin/task_assignment";
  }
  const std::vector<std::string> args = {
      "-m", instance_.getMapName(),
      "-a", instance_.getAgentTaskFName(),
      "-k", std::to_string(instance_.getAgentNum()),
      "-t", "120",
      "-d", std::to_string(seed_),
      "--solver", solver,
  };
#if MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL
  bp::child child(taskAssignmentExe, bp::args(args), bp::std_out > inputStream,
                  bp::std_err > bp::null);
#else
  // Some Boost.Process installations don't ship <boost/process/null.hpp>.
  // In that case, don't suppress stderr.
  bp::child child(taskAssignmentExe, bp::args(args), bp::std_out > inputStream);
#endif

  // The output sequence of the MAPF-PC codebase is as follows:
  // 1. Output TASK ASSIGNMENTS
  // Output Agent # and then followed by the task sequences in order
  // 2. Output some other internal stuff
  // 3. Output TASK PATHS
  // Output Agent # and then followed by the task locations (non-linearized) with @ after the first location with begin time right after and -> between each location

  int agent = -1;
  string line;
  auto parseAgentHeader = [](const std::string& s) -> bool {
    // Expected format: "Agent <id>"
    // Avoid matching unrelated stderr/log lines that merely contain "Agent".
    std::istringstream iss(s);
    std::string tag;
    int id = -1;
    if (!(iss >> tag >> id)) {
      return false;
    }
    return tag == "Agent";
  };
  while (std::getline(inputStream, line)) {
    if (line.empty()) {
      continue;
    }

    // If the agent variable exceeds the total number of agents we are working with then we have read all the task assignments or all the task paths
    if (agent >= instance_.getAgentNum()) {
      if (readingTaskAssignments) {
        readingTaskAssignments = false;
      } else if (readingTaskPaths) {
        readingTaskPaths = false;
      }
    }

    // Check if the following set of lines will be for task assignments or task paths
    if (line == "TASK ASSIGNMENTS") {
      readingTaskAssignments = true;
      readingTaskPaths = false;
      agent = -1;
      continue;
    } else if (line == "TASK PATHS") {
      readingTaskAssignments = false;
      readingTaskPaths = true;
      agent = -1;
      continue;
    }

    // Use the Agent # to increment the agent variable
    if (parseAgentHeader(line)) {
      agent++;
    }
    // Otherwise if we are supposed to read the task assignments then we split that line using ',' as the delimiter and extract the tokens one by one
    // Eg: Agent 1
    //     1, 2, 3, 4,
    else if (readingTaskAssignments && agent > -1) {
      string token;
      size_t pos = 0;
      while ((pos = line.find(',')) != string::npos) {
        token = line.substr(0, pos);
        const auto task = parseInt(token);
        if (!task.has_value()) {
          PLOGE << "Failed to parse task assignment token: '" << token << "'\n";
          child.terminate();
          child.wait();
          return false;
        }
        solution_.assignTaskToAgent(agent, *task);
        line.erase(0, pos + 1);
      }
      // Handle the last token if the line doesn't end with a comma.
      if (const auto task = parseInt(line); task.has_value()) {
        solution_.assignTaskToAgent(agent, *task);
      }
      solution_.agents[agent].taskPaths.resize(
          solution_.agents[agent].taskAssignments.size(), AgentTaskPath());
    }
    // If we are not reading the task assignments then we must be reading the task paths.
    // Eg: Agent 1
    //     6 @ 0 -> 22 -> 23 @ 2 -> 24 @ 3 -> 25 -> 26 ->
    else if (readingTaskPaths && agent > -1) {
      bool ok = true;
      auto isWhitespaceOnly = [](const std::string& s) -> bool {
        return s.find_first_not_of(" \t\r\n") == std::string::npos;
      };
      auto hasDigit = [](const std::string& s) -> bool {
        return s.find_first_of("0123456789") != std::string::npos;
      };
      auto consumeToken = [&](const std::string& rawToken, AgentTaskPath& taskPath,
                              int& taskIndex) {
        if (rawToken.empty() || isWhitespaceOnly(rawToken)) {
          return;
        }
        string token = rawToken;
        if (isWhitespaceOnly(token)) {
          return;
        }
        if (token.find('@') != string::npos) {
          if (!taskPath.empty()) {
            // Mark the last location of the previous task as goal.
            taskPath.path.back().isGoal = true;
            solution_.agents[agent].taskPaths[taskIndex] = taskPath;
            initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
                taskPath;
            taskPath = AgentTaskPath();
          }
          taskIndex++;

          const size_t atPos = token.find('@');
          const auto loc = parseInt(token.substr(0, atPos));
          if (!loc.has_value()) {
            if (!hasDigit(token.substr(0, atPos))) {
              return;
            }
            PLOGE << "Failed to parse path location token: '" << token << "'\n";
            ok = false;
            return;
          }
          PathEntry pEntry = {false, *loc};

          if (taskIndex > 0) {
            if (taskIndex - 1 >=
                    (int)solution_.agents[agent].taskPaths.size() ||
                solution_.agents[agent].taskPaths[taskIndex - 1].empty()) {
              PLOGE << "buildGreedySolutionWithMAPFPC: previous task path "
                       "missing/empty for agent "
                    << agent << ", taskIndex " << taskIndex << "\n";
              ok = false;
              return;
            }
            const int previousLocation =
                solution_.agents[agent].taskPaths[taskIndex - 1]
                    .back()
                    .location;
            taskPath.path.push_back(PathEntry{false, previousLocation});
          }
          taskPath.path.push_back(pEntry);

          token.erase(0, atPos + 1);
          if (taskIndex > 0) {
            const auto beginTime = parseInt(token);
            if (!beginTime.has_value()) {
              PLOGE << "Failed to parse task begin time token: '" << token
                    << "'\n";
              ok = false;
              return;
            }
            taskPath.beginTime = *beginTime - 1;
          } else {
            taskPath.beginTime = 0;
          }
        } else {
          const auto loc = parseInt(token);
          if (!loc.has_value()) {
            if (!hasDigit(token)) {
              return;
            }
            PLOGE << "Failed to parse path location token: '" << token << "'\n";
            ok = false;
            return;
          }
          taskPath.path.push_back(PathEntry{false, *loc});
        }
      };

      AgentTaskPath taskPath;
      size_t pos = 0;
      int taskIndex = -1;
      while ((pos = line.find("->")) != string::npos) {
        consumeToken(line.substr(0, pos), taskPath, taskIndex);
        line.erase(0, pos + 2);
      }

      // Process any trailing token after the last "->".
      consumeToken(line, taskPath, taskIndex);

      // Add the last task path.
      if (!taskPath.empty() && taskIndex >= 0) {
        taskPath.path.back().isGoal = true;
        solution_.agents[agent].taskPaths[taskIndex] = taskPath;
        initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
            taskPath;
      }
      if (!ok) {
        child.terminate();
        child.wait();
        return false;
      }
    }
  }

  child.wait();
  if (child.exit_code() != 0) {
    PLOGE << "MAPF-PC task_assignment exited with code " << child.exit_code()
          << "\n";
    return false;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(agent));
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolutionWithMAPFPC: failed to join agent paths\n";
    return false;
  }

  // Gather the information
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildGreedySolutionWithMAPFPC");
  return true;
}

bool LNS::buildGreedySolution() {

  struct GreedySegmentDiag {
    int agent = UNASSIGNED;
    int task = UNASSIGNED;
    int taskPosition = -1;
    int startTime = 0;
    int pathLength = 0;
    double runtimeMs = 0.0;
    uint64_t expanded = 0;
    uint64_t generated = 0;
    const char* outcome = "unknown";
  };
  vector<GreedySegmentDiag> segmentDiags;
  segmentDiags.reserve((size_t)instance_.getTasksNum());

  const auto printGreedyDiagSummary = [&](const char* context) {
    if (!greedySegmentDiagnostics_ || segmentDiags.empty()) {
      return;
    }
    vector<GreedySegmentDiag> sorted = segmentDiags;
    std::sort(sorted.begin(), sorted.end(),
              [](const GreedySegmentDiag& lhs, const GreedySegmentDiag& rhs) {
                if (lhs.runtimeMs == rhs.runtimeMs) {
                  return lhs.expanded > rhs.expanded;
                }
                return lhs.runtimeMs > rhs.runtimeMs;
              });
    const int topK = min((int)sorted.size(), greedySegmentDiagnosticsTopK_);
    std::cout << "Greedy segment diagnostics (" << context
              << "): total_segments=" << sorted.size() << ", top_k=" << topK
              << '\n';
    for (int i = 0; i < topK; i++) {
      const auto& diag = sorted[i];
      std::cout << "  [" << i << "] agent=" << diag.agent
                << ", task=" << diag.task
                << ", local_pos=" << diag.taskPosition
                << ", start_t=" << diag.startTime
                << ", path_len=" << diag.pathLength
                << ", ll_ms=" << diag.runtimeMs
                << ", expanded=" << diag.expanded
                << ", generated=" << diag.generated
                << ", outcome=" << diag.outcome << '\n';
    }
  };

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].terminalPath = AgentTaskPath();
    solution_.agents[agent].terminalPathActive = false;
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
    solution_.agents[agent].intraPrecedenceDirty = false;
  }

  // Assign tasks
  if (!greedyTaskAssignment(&instance_, &solution_)) {
    PLOGE << "Failed to compute greedy task assignment\n";
    return false;
  }
  size_t expectedIntraSize = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    vector<int> taskLocations = instance_.getTaskLocations(agentTasks);
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(agentTasks.size(), AgentTaskPath());
    if (!agentTasks.empty()) {
      expectedIntraSize += agentTasks.size() - 1;
    }
  }

  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.reserve(inputPrecedenceConstraints.size() +
                                expectedIntraSize);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());

  // Compute the precedence constraints based on current task assignments
  // Intra-agent precedence constraints
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
      precedenceConstraints.emplace_back(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Find paths based on the task assignments
  // First we need to sort the tasks based on the precedence constraints
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return success;
  }

  // Following the topological order we find the paths for each task
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  vector<char> plannedTasks(instance_.getTasksNum(), 0);
  for (int id : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    const int task = id;
    int startTime = 0;
    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildGreedySolution: task " << task
            << " is not assigned to any agent\n";
      return false;
    }
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildGreedySolution: invalid local task index " << taskPosition
            << " for task " << task << " on agent " << agent << "\n";
      return false;
    }
    if (taskPosition != 0) {
      int previousTask =
          solution_.agents[agent].taskAssignments[taskPosition - 1];
      if (initialPaths_[previousTask].empty()) {
        PLOGE << "buildGreedySolution: missing path for predecessor task "
              << previousTask << "\n";
        return false;
      }
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    PLOGI << "Planning for agent " << agent << " and task " << task << "\n";

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    if (!buildConstraintTable(constraintTable, task)) {
      PLOGE << "buildGreedySolution: failed to build constraint table for task "
            << task << " (agent " << agent << ")\n";
      return false;
    }
    // Strengthen greedy initialization with collision constraints from tasks
    // that have already been planned in this pass. This keeps initialization
    // robust across low-level planners that may choose different but valid
    // shortest paths under precedence-only constraints.
    for (int plannedTask = 0; plannedTask < instance_.getTasksNum();
         plannedTask++) {
      if (!plannedTasks[plannedTask]) {
        continue;
      }
      const int plannedAgent = solution_.getAgentWithTask(plannedTask);
      if (plannedAgent == UNASSIGNED) {
        continue;
      }
      const int plannedTaskPos =
          (plannedTask >= 0 && plannedTask < (int)assignmentIndex.pos.size())
              ? assignmentIndex.pos[plannedTask]
              : UNASSIGNED;
      if (plannedTaskPos == UNASSIGNED ||
          plannedTaskPos >=
              (int)solution_.agents[plannedAgent].taskPaths.size()) {
        continue;
      }
      const auto& plannedPath =
          solution_.agents[plannedAgent].taskPaths[plannedTaskPos];
      if (plannedPath.empty()) {
        continue;
      }
      const bool isFinalTask =
          (plannedTaskPos + 1 ==
           (int)solution_.agents[plannedAgent].taskAssignments.size());
      reservePathWithGoalPolicy(constraintTable, plannedPath, isFinalTask);
    }

    uint64_t expandedBefore = 0;
    uint64_t generatedBefore = 0;
    double callStartSec = 0.0;
    if (greedySegmentDiagnostics_) {
      expandedBefore = lowLevelExpanded_;
      generatedBefore = lowLevelGenerated_;
      callStartSec = elapsedRuntimeSec();
    }
    initialPaths_[id] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (greedySegmentDiagnostics_) {
      const double callEndSec = elapsedRuntimeSec();
      GreedySegmentDiag diag;
      diag.agent = agent;
      diag.task = task;
      diag.taskPosition = taskPosition;
      diag.startTime = startTime;
      diag.pathLength = (int)initialPaths_[id].size();
      diag.runtimeMs = max(0.0, (callEndSec - callStartSec) * 1000.0);
      diag.expanded = lowLevelExpanded_ - expandedBefore;
      diag.generated = lowLevelGenerated_ - generatedBefore;
      diag.outcome = getLastLowLevelOutcomeName();
      segmentDiags.push_back(diag);
    }
    if (initialPaths_[id].empty()) {
      PLOGE << "No path exists for agent " << agent << " and task " << task
            << " (ll_outcome=" << getLastLowLevelOutcomeName()
            << ", remaining_budget_sec=" << getLastLowLevelRemainingBudgetSec()
            << ", effective_timeout_sec=" << getLastLowLevelEffectiveTimeoutSec()
            << ")\n";
      printGreedyDiagSummary("failure");
      logInitialSegmentFailureDiagnostics(
          instance_, solution_, constraintTable, agent, task, taskPosition,
          startTime, *solution_.agents[agent].pathPlanner,
          getLastLowLevelRemainingBudgetSec(),
          getLastLowLevelEffectiveTimeoutSec());
      return false;
    }
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[id];
    plannedTasks[task] = 1;
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolution: failed to join agent paths\n";
    return false;
  }

  // Gather the information
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts = clampSocToInt(initialSumOfCosts, "buildGreedySolution");
  printGreedyDiagSummary("success");
  return true;
}

bool LNS::planTerminalReposition(const vector<int>& agentsToPlan,
                                 bool fullRebuild) {
  if (goalOccupationMode_ != "reposition_true") {
    return true;
  }
  terminalRepositionStats_.replansRequested++;

  const int agentCount = instance_.getAgentNum();
  vector<char> shouldPlan(agentCount, 0);
  for (int agent : agentsToPlan) {
    if (agent >= 0 && agent < agentCount) {
      shouldPlan[agent] = 1;
    }
  }
  if (fullRebuild) {
    for (int agent = 0; agent < agentCount; agent++) {
      shouldPlan[agent] = 1;
      solution_.agents[agent].terminalPath = AgentTaskPath();
      solution_.agents[agent].terminalPathActive = false;
    }
  } else {
    for (int agent = 0; agent < agentCount; agent++) {
      if (shouldPlan[agent]) {
        solution_.agents[agent].terminalPath = AgentTaskPath();
        solution_.agents[agent].terminalPathActive = false;
      }
    }
  }

  vector<char> plannedTerminal(agentCount, 0);
  if (!fullRebuild) {
    for (int agent = 0; agent < agentCount; agent++) {
      if (!shouldPlan[agent] && solution_.agents[agent].terminalPathActive &&
          !solution_.agents[agent].terminalPath.empty()) {
        plannedTerminal[agent] = 1;
      }
    }
  }

  vector<pair<int, int>> planningOrder;
  planningOrder.reserve(agentCount);
  for (int agent = 0; agent < agentCount; agent++) {
    if (!shouldPlan[agent]) {
      continue;
    }
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& joinedPath = solution_.agents[agent].path;
    if (assignments.empty() || joinedPath.empty()) {
      continue;
    }
    const int completionTime = joinedPath.endTimeOrZero();
    planningOrder.emplace_back(completionTime, agent);
  }
  std::sort(planningOrder.begin(), planningOrder.end(),
            [](const pair<int, int>& lhs, const pair<int, int>& rhs) {
              if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });

  const auto reserveOtherAgents = [&](ConstraintTable& constraintTable,
                                      int planningAgent) {
    for (int otherAgent = 0; otherAgent < agentCount; otherAgent++) {
      if (otherAgent == planningAgent) {
        continue;
      }
      const auto& otherPath = solution_.agents[otherAgent].path;
      if (!otherPath.empty()) {
        constraintTable.addPath(otherPath, false);
      }
      if (plannedTerminal[otherAgent] &&
          solution_.agents[otherAgent].terminalPathActive &&
          !solution_.agents[otherAgent].terminalPath.empty()) {
        constraintTable.addPath(solution_.agents[otherAgent].terminalPath,
                                false);
      }
    }
  };

  for (const auto& [completionTime, agent] : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    terminalRepositionStats_.agentsEvaluated++;
    const auto& assignments = solution_.agents[agent].taskAssignments;
    if (assignments.empty()) {
      continue;
    }
    const int finalTask = assignments.back();
    if (finalTask < 0 || finalTask >= instance_.getTasksNum()) {
      PLOGE << "planTerminalReposition: invalid final task index " << finalTask
            << " for agent " << agent << "\n";
      return false;
    }
    const int finalGoal = instance_.getTaskLocations(finalTask);

    int lastDemandTime = -1;
    for (int otherAgent = 0; otherAgent < agentCount; otherAgent++) {
      if (otherAgent == agent) {
        continue;
      }
      const auto& otherPath = solution_.agents[otherAgent].path;
      if (otherPath.empty()) {
        continue;
      }
      const int fromTime = max(0, completionTime + 1);
      int demandScanEnd = (int)otherPath.size();
      if (repositionDemandLookahead_ > 0) {
        demandScanEnd =
            min(demandScanEnd, completionTime + 1 + repositionDemandLookahead_);
      }
      for (int timestep = fromTime; timestep < demandScanEnd;
           timestep++) {
        if (otherPath.at(timestep).location == finalGoal) {
          lastDemandTime = max(lastDemandTime, timestep);
        }
      }
    }

    if (lastDemandTime < completionTime + 1) {
      terminalRepositionStats_.skippedNoDemand++;
      // Even without demand, keep explicit terminal occupancy at final goal.
      AgentTaskPath holdPath;
      holdPath.beginTime = completionTime + 1;
      holdPath.path.push_back(PathEntry{false, finalGoal});
      solution_.agents[agent].terminalPath = std::move(holdPath);
      solution_.agents[agent].terminalPathActive = true;
      plannedTerminal[agent] = 1;
      continue;
    }

    const vector<int>& parkingCandidates = getParkingCandidatesForGoal(finalGoal);
    if (parkingCandidates.empty()) {
      terminalRepositionStats_.planningFailures++;
      PLOGE << "planTerminalReposition: no parking candidate found for agent "
            << agent << " final goal " << finalGoal << "\n";
      return false;
    }

    bool planned = false;
    for (int parkingLocation : parkingCandidates) {
      auto planner = createLocalPlanner(agent);
      planner->setGoalLocations(vector<int>{finalGoal, parkingLocation});
      ConstraintTable evacConstraints(instance_.numOfCols, instance_.mapSize);
      reserveOtherAgents(evacConstraints, agent);
      AgentTaskPath evacuationPath = runLowLevelSearch(
          *planner, evacConstraints, completionTime, 1, 0);
      if (evacuationPath.empty()) {
        continue;
      }

      const int evacuationEnd = evacuationPath.endTimeChecked();
      const int returnStart = max(evacuationEnd, lastDemandTime + 1);
      planner->setGoalLocations(vector<int>{parkingLocation, finalGoal});
      ConstraintTable returnConstraints(instance_.numOfCols, instance_.mapSize);
      reserveOtherAgents(returnConstraints, agent);
      AgentTaskPath returnPath = runLowLevelSearch(*planner, returnConstraints,
                                                   returnStart, 1, 0);
      if (returnPath.empty()) {
        continue;
      }

      AgentTaskPath terminalPath = evacuationPath;
      while (!terminalPath.empty() &&
             terminalPath.endTimeChecked() < returnStart) {
        terminalPath.path.push_back(PathEntry{false, parkingLocation});
      }
      for (int step = 1; step < (int)returnPath.size(); step++) {
        terminalPath.path.push_back(returnPath.at(step));
      }

      if (terminalPath.empty()) {
        continue;
      }
      solution_.agents[agent].terminalPath = std::move(terminalPath);
      solution_.agents[agent].terminalPathActive = true;
      plannedTerminal[agent] = 1;
      terminalRepositionStats_.agentsPlanned++;
      planned = true;
      break;
    }

    if (!planned) {
      terminalRepositionStats_.planningFailures++;
      PLOGE << "planTerminalReposition: failed to construct terminal path for "
            << "agent " << agent << " (goal " << finalGoal << ")\n";
      return false;
    }
  }

  return true;
}

bool LNS::buildPrioritizedInitialSolution() {

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].terminalPath = AgentTaskPath();
    solution_.agents[agent].terminalPathActive = false;
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
    solution_.agents[agent].intraPrecedenceDirty = false;
  }

  // Reuse existing greedy task assignment for now; prioritized initialization
  // focuses on robust path construction under inter-agent reservations.
  if (!greedyTaskAssignment(&instance_, &solution_)) {
    PLOGE << "Failed to compute greedy task assignment\n";
    return false;
  }

  size_t expectedIntraSize = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    vector<int> taskLocations = instance_.getTaskLocations(agentTasks);
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(agentTasks.size(), AgentTaskPath());
    if (!agentTasks.empty()) {
      expectedIntraSize += agentTasks.size() - 1;
    }
  }

  // Global precedence constraints = input + intra-agent ordering.
  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.reserve(inputPrecedenceConstraints.size() +
                                expectedIntraSize);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
      precedenceConstraints.emplace_back(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Task-level topological order used when traversing each agent's chain.
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return false;
  }
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());

  // Build predecessor adjacency to enforce precedence timing.
  vector<vector<int>> predecessors(instance_.getTasksNum());
  for (const auto& edge : precedenceConstraints) {
    if (edge.first < 0 || edge.first >= instance_.getTasksNum() ||
        edge.second < 0 || edge.second >= instance_.getTasksNum()) {
      continue;
    }
    predecessors[edge.second].push_back(edge.first);
  }

  const int agentCount = instance_.getAgentNum();
  const int taskCount = instance_.getTasksNum();
  initialPaths_.assign(taskCount, AgentTaskPath());
  vector<char> plannedTasks(taskCount, 0);

  // Plan tasks directly in global topological order. This avoids the
  // cross-agent ordering deadlocks caused by agent-level priority DAG cycles.
  for (int task : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildPrioritizedInitialSolution: task " << task
            << " is not assigned to any agent\n";
      return false;
    }

    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 || taskPosition >= (int)agentTasks.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildPrioritizedInitialSolution: invalid local task index "
            << taskPosition << " for task " << task << " on agent " << agent
            << "\n";
      return false;
    }

    int startTime = 0;
    if (taskPosition > 0) {
      const int previousTask = agentTasks[taskPosition - 1];
      if (previousTask < 0 || previousTask >= taskCount ||
          initialPaths_[previousTask].empty()) {
        PLOGE << "buildPrioritizedInitialSolution: missing prior task path "
              << "for task " << task << " on agent " << agent << "\n";
        return false;
      }
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    int earliestGoalTime = 0;
    for (int pred : predecessors[task]) {
      if (pred < 0 || pred >= taskCount) {
        continue;
      }
      if (initialPaths_[pred].empty()) {
        PLOGE << "buildPrioritizedInitialSolution: predecessor task " << pred
              << " for task " << task
              << " is unexpectedly unplanned in topological order\n";
        return false;
      }
      earliestGoalTime =
          max(earliestGoalTime, initialPaths_[pred].endTimeChecked() + 1);
    }

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    constraintTable.goalLocation = instance_.getTaskLocations(task);
    constraintTable.lengthMin = max(constraintTable.lengthMin, earliestGoalTime);
    constraintTable.latestTimestep =
        max(constraintTable.latestTimestep, constraintTable.lengthMin);

    // Reserve all already planned tasks, regardless of owning agent.
    for (int reservedTask = 0; reservedTask < taskCount; reservedTask++) {
      if (!plannedTasks[reservedTask]) {
        continue;
      }
      const int reservedAgent = solution_.getAgentWithTask(reservedTask);
      if (reservedAgent == UNASSIGNED) {
        PLOGE << "buildPrioritizedInitialSolution: planned task " << reservedTask
              << " has no assigned agent\n";
        return false;
      }
      const int reservedTaskPos =
          (reservedTask >= 0 && reservedTask < (int)assignmentIndex.pos.size())
              ? assignmentIndex.pos[reservedTask]
              : UNASSIGNED;
      if (reservedTaskPos < 0 ||
          reservedTaskPos >=
              (int)solution_.agents[reservedAgent].taskPaths.size()) {
        PLOGE << "buildPrioritizedInitialSolution: invalid reserved task index "
              << reservedTaskPos << " for task " << reservedTask << " (agent "
              << reservedAgent << ")\n";
        return false;
      }
      const auto& reservedPath =
          solution_.agents[reservedAgent].taskPaths[reservedTaskPos];
      if (reservedPath.empty()) {
        PLOGE << "buildPrioritizedInitialSolution: planned task "
              << reservedTask << " has empty path when reserving\n";
        return false;
      }
      const bool isFinalTask =
          (reservedTaskPos + 1 ==
           (int)solution_.agents[reservedAgent].taskAssignments.size());
      reservePathWithGoalPolicy(constraintTable, reservedPath, isFinalTask);
    }

    initialPaths_[task] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[task].empty()) {
      PLOGE << "buildPrioritizedInitialSolution: no path for agent " << agent
            << " and task " << task << " (ll_outcome="
            << getLastLowLevelOutcomeName()
            << ", remaining_budget_sec=" << getLastLowLevelRemainingBudgetSec()
            << ", effective_timeout_sec=" << getLastLowLevelEffectiveTimeoutSec()
            << ")\n";
      logInitialSegmentFailureDiagnostics(
          instance_, solution_, constraintTable, agent, task, taskPosition,
          startTime, *solution_.agents[agent].pathPlanner,
          getLastLowLevelRemainingBudgetSec(),
          getLastLowLevelEffectiveTimeoutSec());
      return false;
    }
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[task];
    plannedTasks[task] = 1;
  }

  // Join per-task paths into per-agent paths.
  vector<int> agentsToCompute(agentCount);
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildPrioritizedInitialSolution: failed to join agent paths\n";
    return false;
  }

  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < agentCount; agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildPrioritizedInitialSolution");
  return true;
}

bool LNS::buildGreedySolutionPrecedenceOnly() {

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].terminalPath = AgentTaskPath();
    solution_.agents[agent].terminalPathActive = false;
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
    solution_.agents[agent].intraPrecedenceDirty = false;
  }

  // Assign tasks (greedy), but do not build collision constraints.
  if (!greedyTaskAssignment(&instance_, &solution_)) {
    PLOGE << "Failed to compute greedy task assignment\n";
    return false;
  }
  size_t expectedIntraSize = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    vector<int> taskLocations = instance_.getTaskLocations(agentTasks);
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(agentTasks.size(), AgentTaskPath());
    if (!agentTasks.empty()) {
      expectedIntraSize += agentTasks.size() - 1;
    }
  }

  // Global precedence constraints = input + intra-agent ordering.
  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.reserve(inputPrecedenceConstraints.size() +
                                expectedIntraSize);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
      precedenceConstraints.emplace_back(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Compute a topological planning order.
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return false;
  }
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());

  // Build predecessor adjacency to enforce precedence via earliest goal times.
  vector<vector<int>> predecessors(instance_.getTasksNum());
  for (const auto& e : precedenceConstraints) {
    predecessors[e.second].push_back(e.first);
  }

  // Plan each task segment with only precedence timing constraints (no paths
  // from other agents in the constraint table).
  initialPaths_.assign(instance_.getTasksNum(), AgentTaskPath());
  for (int task : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildGreedySolutionPrecedenceOnly: task " << task
            << " is not assigned to any agent\n";
      return false;
    }
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildGreedySolutionPrecedenceOnly: invalid local task index "
            << taskPosition << " for task " << task << " on agent " << agent
            << "\n";
      return false;
    }

    int startTime = 0;
    if (taskPosition != 0) {
      int previousTask = solution_.agents[agent].taskAssignments[taskPosition - 1];
      if (initialPaths_[previousTask].empty()) {
        PLOGE << "buildGreedySolutionPrecedenceOnly: missing path for predecessor task "
              << previousTask << "\n";
        return false;
      }
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    int earliestGoalTime = 0;
    for (int pred : predecessors[task]) {
      if (initialPaths_[pred].empty()) {
        PLOGE << "buildGreedySolutionPrecedenceOnly: missing path for precedence predecessor task "
              << pred << "\n";
        return false;
      }
      assert(!initialPaths_[pred].empty());
      earliestGoalTime =
          max(earliestGoalTime, initialPaths_[pred].endTimeChecked() + 1);
    }

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    constraintTable.goalLocation = instance_.getTaskLocations(task);
    constraintTable.lengthMin = max(constraintTable.lengthMin, earliestGoalTime);
    constraintTable.latestTimestep =
        max(constraintTable.latestTimestep, constraintTable.lengthMin);

    initialPaths_[task] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[task].empty()) {
      PLOGE << "No path exists for agent " << agent << " and task " << task
            << " (ll_outcome=" << getLastLowLevelOutcomeName()
            << ", remaining_budget_sec=" << getLastLowLevelRemainingBudgetSec()
            << ", effective_timeout_sec=" << getLastLowLevelEffectiveTimeoutSec()
            << ")\n";
      logInitialSegmentFailureDiagnostics(
          instance_, solution_, constraintTable, agent, task, taskPosition,
          startTime, *solution_.agents[agent].pathPlanner,
          getLastLowLevelRemainingBudgetSec(),
          getLastLowLevelEffectiveTimeoutSec());
      return false;
    }
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[task];
  }

  // Join the individual task paths to form the agent's path.
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolutionPrecedenceOnly: failed to join agent paths\n";
    return false;
  }

  // Gather the information.
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      initialSumOfCosts +=
          static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
    }
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildGreedySolutionPrecedenceOnly");
  return true;
}

bool LNS::extractFeasibleSolution() {

  // Only update the feasible solution if the new solution has better cost!
  if (incumbentSolution_.agentPaths.empty() ||
      incumbentSolution_.sumOfCosts > solution_.sumOfCosts) {
    incumbentSolution_.numOfCols = instance_.numOfCols;
    incumbentSolution_.sumOfCosts = solution_.sumOfCosts;
    incumbentSolution_.agentPaths.resize(instance_.getAgentNum());
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      if (!solution_.agents[agent].taskAssignments.empty()) {
        incumbentSolution_.agentPaths[agent] = solution_.agents[agent].path;
      } else {
        incumbentSolution_.agentPaths[agent] = AgentTaskPath();
      }
    }
    return true;
  }
  return false;
}
