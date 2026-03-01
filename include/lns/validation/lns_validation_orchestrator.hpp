#pragma once

#include <vector>

#include "lns_types.hpp"

struct RemovedTaskChangeStats {
  int changedAgent = 0;
  int changedOrder = 0;
  int unchanged = 0;
};

class ValidationOrchestrator {
 public:
  // Refresh join scope to include all agents whose realized schedules can
  // differ after repair/commit.
  static void refreshAgentsForJoin(const Solution& candidate,
                                   const Solution& previous, int taskCount,
                                   std::vector<int>& agentsToCompute);

  // Summarize whether removed tasks were reassigned or re-ordered.
  static RemovedTaskChangeStats computeRemovedTaskChangeStats(
      const Solution& previous, const Solution& candidate,
      const ConflictMap& immutableRemovedTasks, int taskCount);
};
