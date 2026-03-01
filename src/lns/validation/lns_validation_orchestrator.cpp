#include "lns_validation_orchestrator.hpp"

#include <algorithm>
#include <numeric>

#include "internal/task_position_index.hpp"

void ValidationOrchestrator::refreshAgentsForJoin(
    const Solution& candidate, const Solution& previous, int taskCount,
    std::vector<int>& agentsToCompute) {
  const int agentCount = candidate.numOfAgents;
  std::vector<char> joinMarked(agentCount, 0);
  std::vector<int> refreshedAgents;
  refreshedAgents.reserve(agentCount);
  auto markForJoin = [&](int agent) {
    if (agent < 0 || agent >= agentCount || joinMarked[agent]) {
      return;
    }
    joinMarked[agent] = 1;
    refreshedAgents.push_back(agent);
  };

  for (int agent : agentsToCompute) {
    markForJoin(agent);
  }

  for (int task = 0; task < taskCount; task++) {
    const int prevOwner =
        (task >= 0 && task < (int)previous.taskAgentMap.size())
            ? previous.taskAgentMap[task]
            : UNASSIGNED;
    const int curOwner =
        (task >= 0 && task < (int)candidate.taskAgentMap.size())
            ? candidate.taskAgentMap[task]
            : UNASSIGNED;
    if (prevOwner != curOwner) {
      markForJoin(prevOwner);
      markForJoin(curOwner);
    }
  }

  for (int agent = 0; agent < agentCount; agent++) {
    if (candidate.agents[agent].taskAssignments !=
            previous.agents[agent].taskAssignments ||
        candidate.agents[agent].path.empty()) {
      markForJoin(agent);
    }
  }

  if (refreshedAgents.empty()) {
    refreshedAgents.resize(agentCount);
    std::iota(refreshedAgents.begin(), refreshedAgents.end(), 0);
  }
  agentsToCompute.swap(refreshedAgents);
}

RemovedTaskChangeStats ValidationOrchestrator::computeRemovedTaskChangeStats(
    const Solution& previous, const Solution& candidate,
    const ConflictMap& immutableRemovedTasks, int taskCount) {
  RemovedTaskChangeStats stats;
  const std::vector<int> previousTaskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(previous,
                                                                 taskCount);
  const std::vector<int> candidateTaskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(candidate,
                                                                 taskCount);

  for (const auto& [_, conflict] : immutableRemovedTasks) {
    const int task = conflict.task;
    if (task < 0 || task >= taskCount) {
      continue;
    }
    const int prevAgent =
        (task < (int)previous.taskAgentMap.size()) ? previous.taskAgentMap[task]
                                                   : UNASSIGNED;
    const int candAgent =
        (task < (int)candidate.taskAgentMap.size()) ? candidate.taskAgentMap[task]
                                                    : UNASSIGNED;
    if (prevAgent != candAgent) {
      stats.changedAgent++;
      continue;
    }
    const int prevPos = (task < (int)previousTaskPosByTask.size())
                            ? previousTaskPosByTask[task]
                            : UNASSIGNED;
    const int candPos = (task < (int)candidateTaskPosByTask.size())
                            ? candidateTaskPosByTask[task]
                            : UNASSIGNED;
    if (prevPos != candPos) {
      stats.changedOrder++;
    } else {
      stats.unchanged++;
    }
  }
  return stats;
}
