#pragma once

#include <vector>
#include "lns.hpp"

namespace mapf_pc_lns {
namespace internal {

inline std::vector<int> buildTaskPositionIndexByMappedAgent(
    const Solution& solution, int taskCount) {
  std::vector<int> taskToPosition(taskCount, UNASSIGNED);
  for (int agent = 0; agent < solution.numOfAgents; agent++) {
    const auto& assignments = solution.agents[agent].taskAssignments;
    for (int pos = 0; pos < (int)assignments.size(); pos++) {
      const int task = assignments[pos];
      if (task < 0 || task >= taskCount) {
        continue;
      }
      if (task >= (int)solution.taskAgentMap.size() ||
          solution.taskAgentMap[task] != agent) {
        continue;
      }
      if (taskToPosition[task] == UNASSIGNED) {
        taskToPosition[task] = pos;
      }
    }
  }
  return taskToPosition;
}

}  // namespace internal
}  // namespace mapf_pc_lns
