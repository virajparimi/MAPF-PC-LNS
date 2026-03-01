#pragma once

#include <vector>

#include "instance.hpp"
#include "lns_iteration_phase_types.hpp"
#include "lns_types.hpp"

class LNS;
class MovingMetrics;

class CandidatePhaseOrchestrator {
 public:
  static CandidatePhaseResult run(LNS& lns,
                                  const std::vector<int>& initialAgentsToCompute,
                                  int alnsHeuristicForIter,
                                  ConflictMap& potentialNeighborhood,
                                  MovingMetrics& metrics);

  // Compute impacted agents for candidate join/revalidation based on removed
  // tasks, precedence closure, ownership changes, and dirty-path filtering.
  static std::vector<int> collectImpactedAgents(const Instance& instance,
                                                const Solution& candidate,
                                                const Solution& previous,
                                                const ConflictMap& removedTasks);
};
