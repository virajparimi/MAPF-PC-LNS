#include "lns_candidate_phase_orchestrator.hpp"

#include "lns.hpp"
#include "lns_validation_orchestrator.hpp"
#include "utils.hpp"

#include <limits>
#include <numeric>
#include <stack>
#include <unordered_set>

std::vector<int> CandidatePhaseOrchestrator::collectImpactedAgents(
    const Instance& instance, const Solution& candidate, const Solution& previous,
    const ConflictMap& removedTasks) {
  std::vector<int> agentsToCompute;
  std::vector<char> agentMarked(instance.getAgentNum(), 0);
  auto markAgent = [&](int agent) {
    if (agent >= 0 && agent < instance.getAgentNum() && !agentMarked[agent]) {
      agentMarked[agent] = 1;
      agentsToCompute.push_back(agent);
    }
  };

  std::vector<char> visitedTask(instance.getTasksNum(), 0);
  std::stack<int> taskStack;
  for (const auto& [_, conflict] : removedTasks) {
    taskStack.push(conflict.task);
    markAgent(conflict.agent);
  }

  const auto& ancestors = instance.getAncestorsRef();
  const auto& successors = instance.getSuccessorsRef();
  while (!taskStack.empty()) {
    const int task = taskStack.top();
    taskStack.pop();
    if (task < 0 || task >= instance.getTasksNum() || visitedTask[task]) {
      continue;
    }
    visitedTask[task] = 1;
    const int curAgent =
        (task >= 0 && task < (int)candidate.taskAgentMap.size())
            ? candidate.taskAgentMap[task]
            : UNASSIGNED;
    if (curAgent != UNASSIGNED) {
      markAgent(curAgent);
    }
    const int prevAgent =
        (task >= 0 && task < (int)previous.taskAgentMap.size())
            ? previous.taskAgentMap[task]
            : UNASSIGNED;
    if (prevAgent != UNASSIGNED) {
      markAgent(prevAgent);
    }
    for (int parent : ancestors[task]) {
      if (parent >= 0 && parent < instance.getTasksNum() &&
          !visitedTask[parent]) {
        taskStack.push(parent);
      }
    }
    for (int child : successors[task]) {
      if (child >= 0 && child < instance.getTasksNum() &&
          !visitedTask[child]) {
        taskStack.push(child);
      }
    }
  }

  if (agentsToCompute.empty()) {
    agentsToCompute.resize(instance.getAgentNum());
    std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  }

  // Join only dirty agents when available (prepareNextIteration clears dirty
  // service paths; assignment changes also imply dirty).
  std::vector<int> filteredAgentsToCompute;
  filteredAgentsToCompute.reserve(agentsToCompute.size());
  for (int agent : agentsToCompute) {
    if (agent < 0 || agent >= instance.getAgentNum()) {
      continue;
    }
    const bool assignmentsChanged =
        (candidate.agents[agent].taskAssignments !=
         previous.agents[agent].taskAssignments);
    if (candidate.agents[agent].path.empty() || assignmentsChanged) {
      filteredAgentsToCompute.push_back(agent);
    }
  }
  if (!filteredAgentsToCompute.empty()) {
    agentsToCompute.swap(filteredAgentsToCompute);
  }

  return agentsToCompute;
}

CandidatePhaseResult CandidatePhaseOrchestrator::run(
    LNS& lns, const std::vector<int>& initialAgentsToCompute,
    int /*alnsHeuristicForIter*/, ConflictMap& potentialNeighborhood,
    MovingMetrics& metrics) {
  CandidatePhaseResult result;

  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };

  std::vector<int> agentsToCompute = initialAgentsToCompute;
  ValidationOrchestrator::refreshAgentsForJoin(
      lns.solution_, lns.previousSolution_, lns.instance_.getTasksNum(),
      agentsToCompute);

  const Time::time_point joinStart = Time::now();
  if (!lns.solution_.joinPaths(agentsToCompute)) {
    result.timeJoinPathsSec += elapsedSecSince(joinStart);
    result.status = CandidatePhaseStatus::join_failed;
    return result;
  }
  result.timeJoinPathsSec += elapsedSecSince(joinStart);

  if (lns.isGoalOccupationRepositionTrue()) {
    const std::vector<int> terminalReplanAgents =
        lns.selectTerminalReplanAgents(agentsToCompute);
    if (!terminalReplanAgents.empty()) {
      const Time::time_point terminalStart = Time::now();
      const bool terminalOk =
          lns.planTerminalReposition(terminalReplanAgents, false);
      result.timeTerminalReplanSec += elapsedSecSince(terminalStart);
      if (!terminalOk) {
        result.status = CandidatePhaseStatus::terminal_failed;
        return result;
      }
    }
  }

  const Time::time_point recomputeSocStart = Time::now();
  long long recomputedSoc = 0;
  for (int agent = 0; agent < lns.instance_.getAgentNum(); agent++) {
    recomputedSoc +=
        static_cast<long long>(lns.solution_.agents[agent].path.endTimeOrZero());
  }
  if (recomputedSoc > std::numeric_limits<int>::max()) {
    PLOGW << "LNS::run: sum of costs overflowed int during recomputation;"
             " clamping to INT_MAX\n";
    lns.solution_.sumOfCosts = std::numeric_limits<int>::max();
  } else if (recomputedSoc < std::numeric_limits<int>::min()) {
    PLOGW << "LNS::run: sum of costs underflowed int during recomputation;"
             " clamping to INT_MIN\n";
    lns.solution_.sumOfCosts = std::numeric_limits<int>::min();
  } else {
    lns.solution_.sumOfCosts = static_cast<int>(recomputedSoc);
  }
  PLOGD << "Old sum of costs = " << lns.previousSolution_.sumOfCosts << "\n";
  PLOGD << "New sum of costs = " << lns.solution_.sumOfCosts << "\n";
  PLOGD << "Old objective(" << lns.optimizationObjective_ << ") = "
        << lns.previousObjectiveValue() << "\n";
  PLOGD << "New objective(" << lns.optimizationObjective_ << ") = "
        << lns.currentObjectiveValue() << "\n";
  result.timeRecomputeSocSec += elapsedSecSince(recomputeSocStart);

  const RemovedTaskChangeStats removedTaskChanges =
      ValidationOrchestrator::computeRemovedTaskChangeStats(
          lns.previousSolution_, lns.solution_,
          lns.lnsNeighborhood_.immutableRemovedTasks, lns.instance_.getTasksNum());
  result.removedTasksChangedAgent = removedTaskChanges.changedAgent;
  result.removedTasksChangedOrder = removedTaskChanges.changedOrder;
  result.removedTasksUnchanged = removedTaskChanges.unchanged;

  const Time::time_point validationStart = Time::now();
  potentialNeighborhood.clear();
  LNS::ValidationStats candidateValidationStats;
  std::vector<std::pair<int, int>> candidateCollisionPairs;
  lns.useTerminalPathsInValidation_ = lns.isGoalOccupationRepositionTrue();
  result.candidateValid =
      lns.validateSolution(&potentialNeighborhood, &candidateValidationStats,
                           &candidateCollisionPairs);
  lns.useTerminalPathsInValidation_ = false;
  std::sort(candidateCollisionPairs.begin(), candidateCollisionPairs.end());
  candidateCollisionPairs.erase(
      std::unique(candidateCollisionPairs.begin(), candidateCollisionPairs.end()),
      candidateCollisionPairs.end());
  lns.lastValidationCollisionPairs_ = candidateCollisionPairs;
  lns.lastValidationConflictTasks_.clear();
  lns.lastValidationConflictTasks_.reserve(potentialNeighborhood.size());
  std::unordered_set<int> conflictAgentsSet;
  for (const auto& [task, conflict] : potentialNeighborhood) {
    lns.lastValidationConflictTasks_.push_back(task);
    if (conflict.agent >= 0 && conflict.agent < lns.instance_.getAgentNum()) {
      conflictAgentsSet.insert(conflict.agent);
    } else if (task >= 0 && task < (int)lns.solution_.taskAgentMap.size()) {
      const int owner = lns.solution_.taskAgentMap[task];
      if (owner >= 0 && owner < lns.instance_.getAgentNum()) {
        conflictAgentsSet.insert(owner);
      }
    }
  }
  for (const auto& [a, b] : candidateCollisionPairs) {
    if (a >= 0 && a < lns.instance_.getAgentNum()) {
      conflictAgentsSet.insert(a);
    }
    if (b >= 0 && b < lns.instance_.getAgentNum()) {
      conflictAgentsSet.insert(b);
    }
  }
  lns.lastValidationConflictAgents_.assign(conflictAgentsSet.begin(),
                                           conflictAgentsSet.end());
  std::sort(lns.lastValidationConflictAgents_.begin(),
            lns.lastValidationConflictAgents_.end());
  result.precedenceViolations = candidateValidationStats.precedenceViolations;
  result.vertexCollisions = candidateValidationStats.vertexCollisions;
  result.edgeSwapCollisions = candidateValidationStats.edgeSwapCollisions;
  result.structuralViolations = candidateValidationStats.structuralViolations;
  result.precedenceDebt = candidateValidationStats.precedenceDebt;
  result.precedencePairsChecked = candidateValidationStats.precedencePairsChecked;
  result.candidateConflictSignal = candidateValidationStats.totalConflictEvents();
  PLOGD << "Conflict signal in new solution: " << result.candidateConflictSignal
        << "\n";

  lns.solution_.utility = metrics.computeMovingMetrics(
      result.candidateConflictSignal, lns.currentObjectiveValue());
  if (result.candidateValid) {
    result.feasibleBestUpdate = lns.extractFeasibleSolution();
  } else {
    result.feasibleBestUpdate = false;
    PLOGE << "The solution was not valid!\n";
  }
  result.timeValidationSec += elapsedSecSince(validationStart);
  result.proposedSoc = lns.currentObjectiveValue();

  return result;
}
