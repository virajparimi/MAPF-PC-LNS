#include "instance.hpp"
#include <boost/tokenizer.hpp>
#include <limits>
#include <fstream>
#include <sstream>
#include "internal/parse_helpers.hpp"
#include "utils.hpp"

bool Instance::loadMap() {
  using namespace std;
  using namespace boost;

  ifstream file(mapFname_.c_str());
  if (!file.is_open()) {
    return false;
  }

  string line;
  tokenizer<char_separator<char>>::iterator begin;

  if (!getline(file, line) || line.empty()) {
    return false;
  }

  if (line[0] == 't') {
    // Original MAPF benchmarks
    char_separator<char> sep(" ");
    if (!getline(file, line)) {
      PLOGE << "Failed to read MAPF benchmark height line.\n";
      return false;
    }
    tokenizer<char_separator<char>> tokenizer(line, sep);
    begin = tokenizer.begin();
    auto end = tokenizer.end();
    if (begin == end) {
      PLOGE << "Malformed MAPF height line: " << line << "\n";
      return false;
    }
    ++begin;  // Skip label token.
    if (!parse_helpers::parseNextInt(begin, end, numOfRows)) {
      PLOGE << "Invalid MAPF height value in line: " << line << "\n";
      return false;
    }
    if (!getline(file, line)) {
      PLOGE << "Failed to read MAPF benchmark width line.\n";
      return false;
    }
    tokenizer.assign(line, sep);
    begin = tokenizer.begin();
    end = tokenizer.end();
    if (begin == end) {
      PLOGE << "Malformed MAPF width line: " << line << "\n";
      return false;
    }
    ++begin;  // Skip label token.
    if (!parse_helpers::parseNextInt(begin, end, numOfCols)) {
      PLOGE << "Invalid MAPF width value in line: " << line << "\n";
      return false;
    }
    if (!getline(file, line)) {  // Skip the "map" marker line.
      PLOGE << "Failed to read MAPF map marker line.\n";
      return false;
    }
  } else {
    // Custom empty benchmark
    char_separator<char> sep(",");
    tokenizer<char_separator<char>> tokenizer(line, sep);
    begin = tokenizer.begin();
    auto end = tokenizer.end();
    if (!parse_helpers::parseNextInt(begin, end, numOfRows) ||
        !parse_helpers::parseNextInt(begin, end, numOfCols)) {
      PLOGE << "Malformed custom map header (expected rows,cols): " << line
            << "\n";
      return false;
    }
  }

  if (numOfRows <= 0 || numOfCols <= 0) {
    PLOGE << "Invalid map dimensions: rows=" << numOfRows
          << ", cols=" << numOfCols << "\n";
    return false;
  }
  if (numOfRows > std::numeric_limits<int>::max() / numOfCols) {
    PLOGE << "Map size overflow for dimensions: rows=" << numOfRows
          << ", cols=" << numOfCols << "\n";
    return false;
  }
  mapSize = numOfCols * numOfRows;
  map_.resize(mapSize, false);
  for (int i = 0; i < numOfRows; i++) {
    if (!getline(file, line)) {
      PLOGE << "Failed to read map row " << i << ".\n";
      return false;
    }
    if ((int)line.size() < numOfCols) {
      PLOGE << "Map row " << i << " is too short. Expected at least "
            << numOfCols << " cells, got " << line.size() << ".\n";
      return false;
    }
    for (int j = 0; j < numOfCols; j++) {
      map_[linearizeCoordinate(i, j)] = (line[j] != '.');
    }
  }
  file.close();
  return true;
}

bool Instance::loadAgentsAndTasks() {
  using namespace std;
  using namespace boost;

  ifstream file(agentTaskFname_.c_str());
  if (!file.is_open()) {
    return false;
  }

  string line;
  char_separator<char> sep(",");
  tokenizer<char_separator<char>>::iterator begin;

  if (!getline(file, line)) {
    return false;
  }
  int inputNumAgents = 0;
  if (!parse_helpers::parseIntStrict(line, inputNumAgents)) {
    PLOGE << "Invalid number of agents in agent/task file: " << line << "\n";
    return false;
  }
  if (numOfAgents_ != inputNumAgents) {
    PLOGE << "The number of robots passed in command line and the agent file "
             "do not match.\n";
    return false;
  }

  if (numOfAgents_ == 0) {
    PLOGE << "The number of agents should be larger than 0.\n";
    return false;
  }

  if (numOfTasks_ == 0) {
    PLOGE << "The number of tasks should be larger than 0.\n";
    return false;
  }

  // Reading the agent start locations
  startLocations_.resize(numOfAgents_);

  for (int i = 0; i < numOfAgents_; i++) {
    if (!getline(file, line)) {
      return false;
    }
    tokenizer<char_separator<char>> tokenizer(line, sep);
    begin = tokenizer.begin();
    auto end = tokenizer.end();
    int col = 0, row = 0;
    if (!parse_helpers::parseNextInt(begin, end, col) ||
        !parse_helpers::parseNextInt(begin, end, row)) {
      PLOGE << "Invalid agent start location line (expected col,row): " << line
            << "\n";
      return false;
    }
    startLocations_[i] = linearizeCoordinate(row, col);
    assert(!isObstacle(startLocations_[i]));
  }

  auto skipUntilSection = [&file](string& l) {
    while (getline(file, l)) {
      if (!l.empty() && l[0] == 't') {
        return true;
      }
    }
    return false;
  };

  // Skipping the extra white lines / header lines
  if (!skipUntilSection(line)) {
    return false;
  }

  if (!getline(file, line)) {
    return false;
  }
  int inputNumTasks = 0;
  if (!parse_helpers::parseIntStrict(line, inputNumTasks)) {
    PLOGE << "Invalid number of tasks in agent/task file: " << line << "\n";
    return false;
  }
  if (numOfTasks_ != inputNumTasks) {
    PLOGE << "The number of tasks passed in the command line and the agent "
             "file do not match.\n";
    return false;
  }

  // Reading the task goal locations
  taskLocations_.resize(numOfTasks_);

  for (int i = 0; i < numOfTasks_; i++) {
    if (!getline(file, line)) {
      return false;
    }
    tokenizer<char_separator<char>> tokenizer(line, sep);
    begin = tokenizer.begin();
    auto end = tokenizer.end();
    int col = 0, row = 0;
    if (!parse_helpers::parseNextInt(begin, end, col) ||
        !parse_helpers::parseNextInt(begin, end, row)) {
      PLOGE << "Invalid task location line (expected col,row): " << line
            << "\n";
      return false;
    }
    taskLocations_[i] = linearizeCoordinate(row, col);
    assert(!isObstacle(taskLocations_[i]));
  }

  // Skipping the extra white lines / header lines
  if (!skipUntilSection(line)) {
    return false;
  }

  if (!getline(file, line)) {
    return false;
  }
  int numDependencies = 0;
  if (!parse_helpers::parseIntStrict(line, numDependencies)) {
    PLOGE << "Invalid number of dependencies in input: " << line << "\n";
    return false;
  }
  if (numDependencies < 0) {
    PLOGE << "Invalid number of dependencies in input: " << numDependencies
          << "\n";
    return false;
  }
  vector<pair<int, int>> temporalDependencies;
  temporalDependencies.reserve((size_t)numDependencies);

  for (int i = 0; i < numDependencies; i++) {
    if (!getline(file, line)) {
      return false;
    }
    tokenizer<char_separator<char>> tokenizer(line, sep);
    if (std::distance(tokenizer.begin(), tokenizer.end()) < 2) {
      PLOGE << "Invalid dependency line (expected two integers): " << line
            << "\n";
      return false;
    }
    begin = tokenizer.begin();
    auto end = tokenizer.end();
    int predecessor = 0, successor = 0;
    if (!parse_helpers::parseNextInt(begin, end, predecessor) ||
        !parse_helpers::parseNextInt(begin, end, successor)) {
      PLOGE << "Invalid dependency line (expected two integers): " << line
            << "\n";
      return false;
    }
    if (predecessor < 0 || predecessor >= numOfTasks_ || successor < 0 ||
        successor >= numOfTasks_) {
      PLOGE << "Dependency index out of bounds: " << predecessor << " -> "
            << successor << " with numOfTasks = " << numOfTasks_ << "\n";
      return false;
    }
    temporalDependencies.emplace_back(predecessor, successor);
  }

  for (const auto& dependency : temporalDependencies) {
    int i, j;
    tie(i, j) = dependency;
    ancestors_[j].push_back(i);
    successors_[i].push_back(j);
    inputPrecedenceConstraints_.emplace_back(i, j);
  }

  PLOGD << "# Agents: " << numOfAgents_ << "\t # Tasks: " << numOfTasks_
        << "\t # Dependencies: " << numDependencies << "\n";

  file.close();

  if (!topologicalSort(this, inputPrecedenceConstraints_, inputPlanningOrder_)) {
    PLOGE << "Input precedence constraints contain a cycle or are invalid.\n";
    return false;
  }

  return true;
}
