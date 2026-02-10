#include "instance.hpp"
#include <boost/tokenizer.hpp>
#include <limits>
#include <fstream>
#include <sstream>
#include "internal/parse_helpers.hpp"
#include "utils.hpp"

bool Instance::loadKivaMap() {
  using namespace std;
  using namespace boost;

  ifstream file(mapFname_.c_str());
  if (!file.is_open()) {
    return false;
  }

  string line;
  tokenizer<char_separator<char>>::iterator begin;

  if (!getline(file, line)) {
    return false;
  }
  char_separator<char> sep(",");
  tokenizer<char_separator<char>> tokenizer(line, sep);
  begin = tokenizer.begin();
  auto end = tokenizer.end();
  int rows = 0, cols = 0;
  if (!parse_helpers::parseNextInt(begin, end, rows) ||
      !parse_helpers::parseNextInt(begin, end, cols)) {
    PLOGE << "Malformed Kiva map header (expected rows,cols): " << line << "\n";
    return false;
  }
  numOfRows = rows + 2;  // Read the number of rows
  numOfCols = cols + 2;  // Read the number of columns

  if (!getline(file, line)) {  // Workpoint number
    return false;
  }
  if (!getline(file, line)) {  // Number of agents
    return false;
  }
  int mapDeclaredAgents = 0;
  if (!parse_helpers::parseIntStrict(line, mapDeclaredAgents) ||
      mapDeclaredAgents <= 0) {
    PLOGE << "Invalid Kiva map agent count line: " << line << "\n";
    return false;
  }
  if (numOfAgents_ == 0) {
    numOfAgents_ = mapDeclaredAgents;
  } else if (numOfAgents_ != mapDeclaredAgents) {
    PLOGE << "Mismatch between configured agents and map metadata: numOfAgents="
          << numOfAgents_ << ", map declares " << mapDeclaredAgents << ".\n";
    return false;
  }
  if (!getline(file, line)) {  // Maximum time
    return false;
  }

  if (numOfRows <= 0 || numOfCols <= 0) {
    PLOGE << "Invalid Kiva map dimensions: rows=" << numOfRows
          << ", cols=" << numOfCols << "\n";
    return false;
  }
  if (numOfRows > std::numeric_limits<int>::max() / numOfCols) {
    PLOGE << "Kiva map size overflow for dimensions: rows=" << numOfRows
          << ", cols=" << numOfCols << "\n";
    return false;
  }

  // Initialize the agent start locations
  int agentNum = 0;
  startLocations_.resize(numOfAgents_);

  mapSize = numOfCols * numOfRows;
  map_.resize(mapSize, false);
  for (int i = 1; i < numOfRows - 1; i++) {
    if (!getline(file, line)) {
      PLOGE << "Unexpected EOF while reading Kiva map row " << (i - 1) << ".\n";
      return false;
    }
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if ((int)line.size() < numOfCols - 2) {
      PLOGE << "Malformed Kiva map row " << (i - 1) << ": expected at least "
            << (numOfCols - 2) << " columns, got " << line.size() << ".\n";
      return false;
    }
    for (int j = 1; j < numOfCols - 1; j++) {
      const char cell = line[j - 1];
      map_[linearizeCoordinate(i, j)] = (cell == '@');
      if (cell == 'r') {
        // This is a robot spawn location
        if (agentNum >= numOfAgents_) {
          PLOGE << "More robot spawns in map than expected: found at least "
                << (agentNum + 1) << ", configured numOfAgents="
                << numOfAgents_ << ".\n";
          return false;
        }
        startLocations_[agentNum] = linearizeCoordinate(i, j);
        if (isObstacle(startLocations_[agentNum])) {
          PLOGE << "Robot spawn location is blocked at linearized index "
                << startLocations_[agentNum] << ".\n";
          return false;
        }
        agentNum++;
      }
      if (cell == 'e') {
        // This is a task spawn location
        endPoints_.push_back(linearizeCoordinate(i, j));
        if (isObstacle(endPoints_.back())) {
          PLOGE << "Endpoint location is blocked at linearized index "
                << endPoints_.back() << ".\n";
          return false;
        }
      }
    }
  }

  for (int i = 0; i < numOfRows; i++) {
    map_[i * numOfCols] = true;
    map_[i * numOfCols + numOfCols - 1] = true;
  }
  for (int j = 1; j < numOfCols - 1; j++) {
    map_[j] = true;
    map_[mapSize - numOfCols + j] = true;
  }

  if (agentNum != numOfAgents_) {
    PLOGE << "Mismatch between configured agents and map spawns: numOfAgents="
          << numOfAgents_ << ", robot spawns found=" << agentNum << ".\n";
    return false;
  }
  file.close();
  return true;
}

bool Instance::loadKivaTasks() {
  using namespace std;
  using namespace boost;

  ifstream file(agentTaskFname_.c_str());
  if (!file.is_open()) {
    return false;
  }

  string line;
  int taskNum;
  if (!getline(file, line)) {
    return false;
  }
  if (!parse_helpers::parseIntStrict(line, taskNum)) {
    PLOGE << "Invalid Kiva task count line: " << line << "\n";
    return false;
  }

  if (taskNum < 0) {
    PLOGE << "Invalid Kiva task count (negative): " << taskNum << ".\n";
    return false;
  }
  const long long expectedExpandedTasks = static_cast<long long>(taskNum) * 2LL;
  if (numOfTasks_ == 0) {
    if (expectedExpandedTasks > static_cast<long long>(std::numeric_limits<int>::max())) {
      PLOGE << "Expanded Kiva task count overflows int: " << expectedExpandedTasks
            << ".\n";
      return false;
    }
    numOfTasks_ = static_cast<int>(expectedExpandedTasks);
  } else if (expectedExpandedTasks != static_cast<long long>(numOfTasks_)) {
    PLOGE << "Kiva task count mismatch: file contains " << taskNum
          << " tasks (expects " << taskNum * 2
          << " expanded pickup+delivery tasks), but --taskNum is "
          << numOfTasks_ << ".\n";
    return false;
  }
  ancestors_.assign(numOfTasks_, {});
  successors_.assign(numOfTasks_, {});
  // Initialize the task locations
  taskLocations_.resize(numOfTasks_);
  vector<pair<int, int>> temporalDependencies;

  if (endPoints_.empty()) {
    PLOGE << "Kiva map has no endpoints ('e' cells), cannot map tasks.\n";
    return false;
  }

  bool warnedWrappedTaskIndex = false;
  for (int i = 0; i < numOfTasks_; i += 2) {
    int releaseTime, startTask, goalTask, timeOfStartTask, timeOfGoalTask;
    if (!getline(file, line)) {
      return false;
    }
    std::istringstream stringLine(line);
    if (!(stringLine >> releaseTime >> startTask >> goalTask >>
          timeOfStartTask >> timeOfGoalTask)) {
      return false;
    }
    (void)releaseTime;
    (void)timeOfStartTask;
    (void)timeOfGoalTask;

    if ((startTask < 0 || startTask >= (int)endPoints_.size() ||
         goalTask < 0 || goalTask >= (int)endPoints_.size()) &&
        !warnedWrappedTaskIndex) {
      PLOGW << "Kiva task endpoint indices exceed endpoint count. Applying modulo"
               " mapping to preserve legacy behavior.\n";
      warnedWrappedTaskIndex = true;
    }
    const int endpointCount = (int)endPoints_.size();
    startTask = ((startTask % endpointCount) + endpointCount) % endpointCount;
    goalTask = ((goalTask % endpointCount) + endpointCount) % endpointCount;
    assert(startTask < (int)endPoints_.size());
    assert(goalTask < (int)endPoints_.size());

    taskLocations_[i] = endPoints_[startTask];
    taskLocations_[i + 1] = endPoints_[goalTask];
    assert(!isObstacle(taskLocations_[i]));
    assert(!isObstacle(taskLocations_[i + 1]));

    temporalDependencies.emplace_back(i, i + 1);
  }

  for (const auto& dependency : temporalDependencies) {
    int i, j;
    tie(i, j) = dependency;
    ancestors_[j].push_back(i);
    successors_[i].push_back(j);
    inputPrecedenceConstraints_.emplace_back(i, j);
  }

  PLOGD << "# Agents: " << numOfAgents_ << "\t # Tasks: " << numOfTasks_
        << "\t # Dependencies: " << (int)temporalDependencies.size() << "\n";

  file.close();

  if (!topologicalSort(this, inputPrecedenceConstraints_, inputPlanningOrder_)) {
    PLOGE << "Input precedence constraints contain a cycle or are invalid.\n";
    return false;
  }

  return true;
}
