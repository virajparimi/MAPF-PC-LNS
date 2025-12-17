#include "astar.hpp"

void SingleAgentSolver::computeHeuristics() {
  const size_t numGoals = goalLocations.size();

  heuristic.assign(numGoals, nullptr);
  heuristicLandmarks.assign(numGoals, 0);

  unordered_map<int, int> locationToGlobalTask;
  locationToGlobalTask.reserve(instance.taskLocations_.size());
  for (int globalTask = 0; globalTask < (int)instance.taskLocations_.size();
       globalTask++) {
    locationToGlobalTask[instance.taskLocations_[globalTask]] = globalTask;
  }

  for (size_t stage = 0; stage < numGoals; stage++) {
    auto it = locationToGlobalTask.find(goalLocations[stage]);
    assert(it != locationToGlobalTask.end());
    heuristic[stage] = &instance.heuristics_[it->second];
  }

  // Landmark heuristic is a suffix-sum of distances between consecutive goals.
  for (size_t stage = numGoals; stage-- > 1;) {
    const size_t prevStage = stage - 1;
    heuristicLandmarks[prevStage] =
        heuristicLandmarks[stage] + (*heuristic[stage])[goalLocations[prevStage]];
  }
}
