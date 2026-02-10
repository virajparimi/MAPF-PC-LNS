#include "instance.hpp"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "internal/parse_helpers.hpp"
#include "utils.hpp"

namespace {

bool detectKivaMapFormat(const std::string& mapPath) {
  using namespace boost;
  std::ifstream file(mapPath.c_str());
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  if (!std::getline(file, line)) {
    return false;
  }

  char_separator<char> sep(",");
  tokenizer<char_separator<char>> tokenizer(line, sep);
  auto it = tokenizer.begin();
  const auto end = tokenizer.end();
  int rows = 0, cols = 0;
  if (!parse_helpers::parseNextInt(it, end, rows) ||
      !parse_helpers::parseNextInt(it, end, cols) ||
      rows <= 0 || cols <= 0) {
    return false;
  }

  // Kiva map files contain three integer metadata lines after the first header
  // line: workpoint count, agent count, and max time.
  int metadata = 0;
  for (int i = 0; i < 3; i++) {
    if (!std::getline(file, line) ||
        !parse_helpers::parseIntStrict(line, metadata)) {
      return false;
    }
  }
  return true;
}

}  // namespace

Instance::Instance(const string& mapFname, const string& agentTaskFname,
                   int numOfAgents, int numOfTasks)
    : mapFname_(mapFname),
      agentTaskFname_(agentTaskFname),
      numOfAgents_(numOfAgents),
      numOfTasks_(numOfTasks) {
  const bool kivaFormat = detectKivaMapFormat(mapFname_);
  bool succ = false;
  if (kivaFormat) {
    // We are going to work with KIVA instances
    succ = loadKivaMap();
  } else {
    succ = loadMap();
  }
  if (!succ) {
    throw std::runtime_error("Failed to load map '" + mapFname_ +
                             "'. See preceding log messages for details.");
  }

  ancestors_.resize(numOfTasks_);
  successors_.resize(numOfTasks_);

  if (kivaFormat) {
    // We are going to load KIVA tasks with implicit precedence constraints
    succ = loadKivaTasks();
  } else {
    succ = loadAgentsAndTasks();
  }
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

  for (int i = 0; i < numOfTasks_; i++) {
    heuristics_[i].resize(mapSize, MAX_TIMESTEP);
    const int root = taskLocations_[i];
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
