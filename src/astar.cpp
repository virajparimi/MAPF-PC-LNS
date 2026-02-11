#include "astar.hpp"

void SingleAgentSolver::computeHeuristics() {
  const size_t numGoals = goalLocations.size();

  heuristicTaskIdx.assign(numGoals, -1);
  heuristicLandmarks.assign(numGoals, 0);

  const auto& locationToGlobalTask = instance.taskLocationToGlobalTask_;
  assert(locationToGlobalTask.size() <= instance.taskLocations_.size());

  for (size_t stage = 0; stage < numGoals; stage++) {
    auto it = locationToGlobalTask.find(goalLocations[stage]);
    if (it == locationToGlobalTask.end()) {
      PLOGE << "computeHeuristics: goal location " << goalLocations[stage]
            << " is not present in task-location index\n";
      throw std::runtime_error(
          "SingleAgentSolver::computeHeuristics: missing goal location");
    }
    heuristicTaskIdx[stage] = it->second;
  }

  // Landmark heuristic is a suffix-sum of distances between consecutive goals.
  for (size_t stage = numGoals; stage-- > 1;) {
    const size_t prevStage = stage - 1;
    const long long suffix =
        (long long)heuristicLandmarks[stage] +
        (long long)instance.heuristics_[heuristicTaskIdx[stage]]
                                   [goalLocations[prevStage]];
    if (suffix > std::numeric_limits<int>::max()) {
      heuristicLandmarks[prevStage] = std::numeric_limits<int>::max();
    } else if (suffix < std::numeric_limits<int>::min()) {
      heuristicLandmarks[prevStage] = std::numeric_limits<int>::min();
    } else {
      heuristicLandmarks[prevStage] = (int)suffix;
    }
  }
}
