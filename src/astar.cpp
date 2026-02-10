#include "astar.hpp"

void SingleAgentSolver::computeHeuristics() {
  const size_t numGoals = goalLocations.size();

  heuristic.assign(numGoals, nullptr);
  heuristicLandmarks.assign(numGoals, 0);

  const auto& locationToGlobalTask = instance.taskLocationToGlobalTask_;
  assert(locationToGlobalTask.size() <= instance.taskLocations_.size());

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
