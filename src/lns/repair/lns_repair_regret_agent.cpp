#include "lns.hpp"
#include "lns_repair_internal.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

void LNS::computeRegretForTaskWithAgent(
    TaskRegretPacket regretPacket, RegretWorkspace& workspace,
    vector<pair<int, int>>* precedenceConstraints,
    const TaskBaselineMetrics& baselineMetrics,
    pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
    const vector<int>* assignmentOwnerLookup,
    const vector<int>* assignmentPosLookup,
    const vector<int>* previousAssignmentOwnerLookup,
    const vector<int>* previousAssignmentPosLookup) {
  if (runtimeBudgetExhausted()) {
    return;
  }

  regretEvalStatsCurrent_.agentEvaluations++;
  regretEvalStatsTotal_.agentEvaluations++;

  const auto& assignmentsForAgent = workspace.assignments(regretPacket.agent);
  const auto& pathsForAgent = workspace.taskPaths(regretPacket.agent);
  if (assignmentsForAgent.size() != pathsForAgent.size()) {
    PLOGE << "computeRegretForTaskWithAgent: assignment/path size mismatch for "
          << "agent " << regretPacket.agent << "\n";
    return;
  }

  // Compute the first position along the agent's task assignments where we can insert this task
  int firstValidPosition = 0;
  for (int j = (int)assignmentsForAgent.size() - 1; j >= 0; j--) {
    int beginTime = pathsForAgent[j].beginTime;
    int endTime = pathsForAgent[j].endTime();
    if ((regretPacket.earliestTimestep > endTime) ||
        (regretPacket.earliestTimestep <= endTime &&
         regretPacket.earliestTimestep >= beginTime)) {
      firstValidPosition = j + 1;
      break;
    }
  }

  SingleAgentSolver& candidatePlanner =
      getReusableLocalPlanner(regretPacket.agent);

  const auto originalConflictIt =
      lnsNeighborhood_.removedTasks.find(regretPacket.task);
  const bool hasOriginalConflict =
      (originalConflictIt != end(lnsNeighborhood_.removedTasks));

  vector<int> allCandidatePositions;
  allCandidatePositions.reserve((int)assignmentsForAgent.size() -
                                firstValidPosition + 1);
  for (int pos = firstValidPosition; pos <= (int)assignmentsForAgent.size();
       pos++) {
    allCandidatePositions.push_back(pos);
  }

  vector<int> candidatePositions = allCandidatePositions;

  auto evaluateCandidatePositions = [&](const vector<int>& positions) {
    for (int pos : positions) {
      if (runtimeBudgetExhausted()) {
        return;
      }

      regretEvalStatsCurrent_.candidateInsertionsTried++;
      regretEvalStatsTotal_.candidateInsertionsTried++;

      if (hasOriginalConflict &&
          originalConflictIt->second.agent == regretPacket.agent &&
          originalConflictIt->second.taskPosition == pos) {
        // We dont want to compute regret for the same agent, task positions
        // that led to the original conflict!
        continue;
      }
      regretPacket.taskPosition = pos;

      std::variant<bool, Utility> insertCulmination =
          insertTask(regretPacket, workspace, precedenceConstraints,
                     &baselineMetrics, &candidatePlanner, true,
                     assignmentOwnerLookup, assignmentPosLookup,
                     previousAssignmentOwnerLookup,
                     previousAssignmentPosLookup);
      if (std::holds_alternative<Utility>(insertCulmination)) {
        regretEvalStatsCurrent_.candidateInsertionsFeasible++;
        regretEvalStatsTotal_.candidateInsertionsFeasible++;
        serviceTimes->push(std::get<Utility>(insertCulmination));
      }
    }
  };

  evaluateCandidatePositions(candidatePositions);

}
