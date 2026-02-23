#include "instance.hpp"
#include <stdexcept>
#include <unordered_map>
#include "utils.hpp"

Instance::Instance(const string& mapFname, const string& agentTaskFname,
                   int numOfAgents, int numOfTasks)
    : mapFname_(mapFname),
      agentTaskFname_(agentTaskFname),
      numOfAgents_(numOfAgents),
      numOfTasks_(numOfTasks) {
  bool succ = loadMap();
  if (!succ) {
    throw std::runtime_error("Failed to load map '" + mapFname_ +
                             "'. See preceding log messages for details.");
  }

  succ = loadAgentsAndTasks();
  if (!succ) {
    throw std::runtime_error(
        "Failed to load agent/task data '" + agentTaskFname_ +
        "'. See preceding log messages for details.");
  }
  buildTaskLocationIndex();
  preComputeNeighbors();
  preComputeHeuristics();
}

void Instance::buildTaskLocationIndex() {
  taskLocationToGlobalTask_.clear();
  taskLocationToGlobalTask_.reserve(taskLocations_.size());
  for (int globalTask = 0; globalTask < (int)taskLocations_.size();
       ++globalTask) {
    const int taskLocation = taskLocations_[globalTask];
    // Multiple tasks may share a location; one representative id is sufficient
    // because heuristics are location-based and thus identical for duplicates.
    taskLocationToGlobalTask_.try_emplace(taskLocation, globalTask);
  }
}

void Instance::preComputeNeighbors() {
  neighborsCache_.clear();
  neighborsCache_.resize(mapSize);
  for (int current = 0; current < mapSize; current++) {
    auto& neighbors = neighborsCache_[current];
    neighbors.clear();
    if (map_[current]) {
      continue;
    }
    neighbors.reserve(4);
    int candidates[4] = {current + 1, current - 1, current + numOfCols,
                         current - numOfCols};
    for (int next : candidates) {
      if (validMove(current, next)) {
        neighbors.push_back(next);
      }
    }
  }
}

void Instance::preComputeHeuristics() {
  heuristics_.clear();
  heuristics_.resize(numOfTasks_);
  extraGoalHeuristicsCache_.clear();
  std::unordered_map<int, int> rootToFirstTask;
  rootToFirstTask.reserve(numOfTasks_);

  for (int i = 0; i < numOfTasks_; i++) {
    const int root = taskLocations_[i];
    const auto reused = rootToFirstTask.find(root);
    if (reused != rootToFirstTask.end()) {
      heuristics_[i] = heuristics_[reused->second];
      continue;
    }
    rootToFirstTask.emplace(root, i);

    heuristics_[i].resize(mapSize, MAX_TIMESTEP);
    heuristics_[i][taskLocations_[i]] = 0;
    deque<int> frontier;
    frontier.push_back(root);

    // Unit-cost graph: BFS computes exact shortest-path distances.
    while (!frontier.empty()) {
      const int current = frontier.front();
      frontier.pop_front();
      const int nextDistance = heuristics_[i][current] + 1;
      for (int nextLocation : getNeighbors(current)) {
        if (heuristics_[i][nextLocation] > nextDistance) {
          heuristics_[i][nextLocation] = nextDistance;
          frontier.push_back(nextLocation);
        }
      }
    }
  }
}

const vector<int>& Instance::getGoalDistanceTable(int goalLocation) const {
  if (goalLocation < 0 || goalLocation >= mapSize || isObstacle(goalLocation)) {
    throw std::out_of_range("Instance::getGoalDistanceTable: invalid goal");
  }

  const auto taskIt = taskLocationToGlobalTask_.find(goalLocation);
  if (taskIt != taskLocationToGlobalTask_.end()) {
    return heuristics_.at(taskIt->second);
  }

  const auto cacheIt = extraGoalHeuristicsCache_.find(goalLocation);
  if (cacheIt != extraGoalHeuristicsCache_.end()) {
    return cacheIt->second;
  }

  vector<int> distances(mapSize, MAX_TIMESTEP);
  distances[goalLocation] = 0;
  deque<int> frontier;
  frontier.push_back(goalLocation);
  while (!frontier.empty()) {
    const int current = frontier.front();
    frontier.pop_front();
    const int nextDistance = distances[current] + 1;
    for (int nextLocation : getNeighbors(current)) {
      if (distances[nextLocation] > nextDistance) {
        distances[nextLocation] = nextDistance;
        frontier.push_back(nextLocation);
      }
    }
  }

  auto inserted =
      extraGoalHeuristicsCache_.emplace(goalLocation, std::move(distances));
  return inserted.first->second;
}

int Instance::getDistanceToGoal(int goalLocation, int location) const {
  if (location < 0 || location >= mapSize) {
    return MAX_TIMESTEP;
  }
  const auto& distances = getGoalDistanceTable(goalLocation);
  return distances[location];
}


void Instance::printMap() const {
  std::string row;
  row.reserve(numOfCols);
  for (int i = 0; i < numOfRows; i++) {
    row.clear();
    for (int j = 0; j < numOfCols; j++) {
      row += map_[linearizeCoordinate(i, j)] ? '@' : '.';
    }
    PLOGI << row;
  }
}
