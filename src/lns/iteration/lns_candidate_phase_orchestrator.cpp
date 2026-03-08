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
  return lns.runCandidatePhase(initialAgentsToCompute, potentialNeighborhood,
                               metrics);
}

CandidatePhaseResult LNS::runCandidatePhase(
    const std::vector<int>& initialAgentsToCompute,
    ConflictMap& potentialNeighborhood, MovingMetrics& metrics) {
  CandidatePhaseResult result;
  const int agentCount = instance_.getAgentNum();
  std::vector<char> touchedByAgent(agentCount, 0);
  auto markTouched = [&](int agent) {
    if (agent >= 0 && agent < agentCount) {
      touchedByAgent[agent] = 1;
    }
  };

  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };

  std::vector<int> agentsToCompute = initialAgentsToCompute;
  ValidationOrchestrator::refreshAgentsForJoin(
      solution_, previousSolution_, instance_.getTasksNum(),
      agentsToCompute);
  for (int agent : agentsToCompute) {
    markTouched(agent);
  }

  const Time::time_point joinStart = Time::now();
  if (!solution_.joinPaths(agentsToCompute)) {
    result.timeJoinPathsSec += elapsedSecSince(joinStart);
    result.status = CandidatePhaseStatus::join_failed;
    return result;
  }
  result.timeJoinPathsSec += elapsedSecSince(joinStart);

  if (isGoalOccupationRepositionTrue()) {
    const std::vector<int> terminalReplanAgents =
        selectTerminalReplanAgents(agentsToCompute);
    if (!terminalReplanAgents.empty()) {
      for (int agent : terminalReplanAgents) {
        markTouched(agent);
      }
      const Time::time_point terminalStart = Time::now();
      const bool terminalOk =
          planTerminalReposition(terminalReplanAgents, false);
      result.timeTerminalReplanSec += elapsedSecSince(terminalStart);
      if (!terminalOk) {
        result.status = CandidatePhaseStatus::terminal_failed;
        return result;
      }
    }
  }

  const Time::time_point recomputeSocStart = Time::now();
  long long recomputedSoc = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    recomputedSoc +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  if (recomputedSoc > std::numeric_limits<int>::max()) {
    PLOGW << "LNS::run: sum of costs overflowed int during recomputation;"
             " clamping to INT_MAX\n";
    solution_.sumOfCosts = std::numeric_limits<int>::max();
  } else if (recomputedSoc < std::numeric_limits<int>::min()) {
    PLOGW << "LNS::run: sum of costs underflowed int during recomputation;"
             " clamping to INT_MIN\n";
    solution_.sumOfCosts = std::numeric_limits<int>::min();
  } else {
    solution_.sumOfCosts = static_cast<int>(recomputedSoc);
  }
  PLOGD << "Old sum of costs = " << previousSolution_.sumOfCosts << "\n";
  PLOGD << "New sum of costs = " << solution_.sumOfCosts << "\n";
  PLOGD << "Old objective(" << optimizationObjective_ << ") = "
        << previousObjectiveValue() << "\n";
  PLOGD << "New objective(" << optimizationObjective_ << ") = "
        << currentObjectiveValue() << "\n";
  result.timeRecomputeSocSec += elapsedSecSince(recomputeSocStart);

  const RemovedTaskChangeStats removedTaskChanges =
      ValidationOrchestrator::computeRemovedTaskChangeStats(
          previousSolution_, solution_, lnsNeighborhood_.immutableRemovedTasks,
          instance_.getTasksNum());
  result.removedTasksChangedAgent = removedTaskChanges.changedAgent;
  result.removedTasksChangedOrder = removedTaskChanges.changedOrder;
  result.removedTasksUnchanged = removedTaskChanges.unchanged;

  const Time::time_point validationStart = Time::now();
  potentialNeighborhood.clear();
  LNS::ValidationStats candidateValidationStats;
  std::vector<std::pair<int, int>> candidateCollisionPairs;
  useTerminalPathsInValidation_ = isGoalOccupationRepositionTrue();
  result.candidateValid =
      validateSolution(&potentialNeighborhood, &candidateValidationStats,
                           &candidateCollisionPairs);
  useTerminalPathsInValidation_ = false;
  std::sort(candidateCollisionPairs.begin(), candidateCollisionPairs.end());
  candidateCollisionPairs.erase(
      std::unique(candidateCollisionPairs.begin(), candidateCollisionPairs.end()),
      candidateCollisionPairs.end());
  softRecoveryState_.lastValidationCollisionPairs = candidateCollisionPairs;
  softRecoveryState_.lastValidationConflictTasks.clear();
  softRecoveryState_.lastValidationConflictTasks.reserve(potentialNeighborhood.size());
  std::unordered_set<int> conflictAgentsSet;
  for (const auto& [task, conflict] : potentialNeighborhood) {
    softRecoveryState_.lastValidationConflictTasks.push_back(task);
    if (conflict.agent >= 0 && conflict.agent < instance_.getAgentNum()) {
      conflictAgentsSet.insert(conflict.agent);
    } else if (task >= 0 && task < (int)solution_.taskAgentMap.size()) {
      const int owner = solution_.taskAgentMap[task];
      if (owner >= 0 && owner < instance_.getAgentNum()) {
        conflictAgentsSet.insert(owner);
      }
    }
  }
  for (const auto& [a, b] : candidateCollisionPairs) {
    if (a >= 0 && a < instance_.getAgentNum()) {
      conflictAgentsSet.insert(a);
    }
    if (b >= 0 && b < instance_.getAgentNum()) {
      conflictAgentsSet.insert(b);
    }
  }
  softRecoveryState_.lastValidationConflictAgents.assign(conflictAgentsSet.begin(),
                                       conflictAgentsSet.end());
  std::sort(softRecoveryState_.lastValidationConflictAgents.begin(),
            softRecoveryState_.lastValidationConflictAgents.end());
  result.precedenceViolations = candidateValidationStats.precedenceViolations;
  result.vertexCollisions = candidateValidationStats.vertexCollisions;
  result.edgeSwapCollisions = candidateValidationStats.edgeSwapCollisions;
  result.structuralViolations = candidateValidationStats.structuralViolations;
  result.precedenceDebt = candidateValidationStats.precedenceDebt;
  result.precedencePairsChecked = candidateValidationStats.precedencePairsChecked;
  result.candidateConflictSignal = candidateValidationStats.totalConflictEvents();
  PLOGD << "Conflict signal in new solution: " << result.candidateConflictSignal
        << "\n";

  solution_.utility = metrics.computeMovingMetrics(
      result.candidateConflictSignal, currentObjectiveValue());
  if (result.candidateValid) {
    result.feasibleBestUpdate = extractFeasibleSolution();
  } else {
    result.feasibleBestUpdate = false;
    PLOGE << "The solution was not valid!\n";
  }
  result.timeValidationSec += elapsedSecSince(validationStart);
  result.proposedSoc = currentObjectiveValue();
  result.candidateTouchedAgents.reserve(agentsToCompute.size());
  for (int agent = 0; agent < agentCount; agent++) {
    if (touchedByAgent[agent]) {
      result.candidateTouchedAgents.push_back(agent);
    }
  }

  return result;
}
