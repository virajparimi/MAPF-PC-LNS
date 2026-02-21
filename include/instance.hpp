#pragma once

#include <plog/Log.h>
#include "common.hpp"

class Instance {

 protected:
  vector<bool> map_;
  string mapFname_;
  string agentTaskFname_;

  vector<vector<int>> heuristics_;
  vector<vector<int>> neighborsCache_;
  int numOfAgents_{}, numOfTasks_{};
  vector<int> endPoints_, taskLocations_, startLocations_, inputPlanningOrder_;
  unordered_map<int, int> taskLocationToGlobalTask_;
  vector<vector<int>> ancestors_, successors_;
  vector<pair<int, int>> inputPrecedenceConstraints_;

  bool loadMap();
  bool loadAgentsAndTasks();
  void buildTaskLocationIndex();
  void printMap() const;
  void preComputeNeighbors();
  inline void validateTaskIndex(int globalTask, const char* caller) const {
    if (globalTask < 0 || globalTask >= numOfTasks_) {
      throw std::out_of_range(string(caller) + ": task index out of range: " +
                              std::to_string(globalTask));
    }
  }
  friend class Solution;
  friend class SingleAgentSolver;

 public:
  int mapSize{}, numOfCols{}, numOfRows{};

  Instance() = default;
  Instance(const string& mapFname, const string& agentTaskFname,
           int numOfAgents = 0, int numOfTasks = 0);

  inline int getTaskLocations(int task) const {
    validateTaskIndex(task, "Instance::getTaskLocations");
    return taskLocations_[task];
  }
  // Prefer ref-returning accessors in performance-sensitive code.
  inline const vector<int>& getTaskLocationsRef() const { return taskLocations_; }
  vector<int> getTaskLocations(const vector<int>& tasks) const {
    vector<int> taskLocs(tasks.size(), 0);
    for (int i = 0; i < (int)tasks.size(); i++) {
      validateTaskIndex(tasks[i], "Instance::getTaskLocations(vector)");
      taskLocs[i] = taskLocations_[tasks[i]];
    }
    return taskLocs;
  }
  [[deprecated("Use getHeuristicsRef()")]] vector<vector<int>> getHeuristics() {
    return heuristics_;
  }
  inline const vector<vector<int>>& getHeuristicsRef() const { return heuristics_; }
  [[deprecated("Use getHeuristicsRef(int)")]] vector<int> getHeuristics(
      int globalTask) {
    validateTaskIndex(globalTask, "Instance::getHeuristics");
    return heuristics_[globalTask];
  }
  inline const vector<int>& getHeuristicsRef(int globalTask) const {
    validateTaskIndex(globalTask, "Instance::getHeuristicsRef");
    return heuristics_[globalTask];
  }
  inline const vector<int>& getNeighbors(int current) const {
    assert(current >= 0 && current < mapSize);
    return neighborsCache_[current];
  }
  inline bool isObstacle(int loc) const {
    if (loc < 0 || loc >= mapSize) {
      return true;
    }
    return map_[loc];
  }
  inline bool validMove(int curr, int next) const {
    if (curr < 0 || curr >= mapSize || map_[curr] || next < 0 ||
        next >= mapSize || map_[next]) {
      return false;
    }
    return getManhattanDistance(curr, next) < 2;
  };
  inline int linearizeCoordinate(int row, int col) const {
    return (this->numOfCols * row + col);
  }
  inline int getRowCoordinate(int id) const {
    if (this->numOfCols <= 0) {
      throw std::logic_error(
          "Instance::getRowCoordinate called with invalid numOfCols");
    }
    return id / this->numOfCols;
  }
  inline int getColCoordinate(int id) const {
    if (this->numOfCols <= 0) {
      throw std::logic_error(
          "Instance::getColCoordinate called with invalid numOfCols");
    }
    return id % this->numOfCols;
  }
  inline pair<int, int> getCoordinate(int id) const {
    return make_pair(getRowCoordinate(id), getColCoordinate(id));
  }
  inline int getCols() const { return numOfCols; }
  inline int getAgentNum() const { return numOfAgents_; }
  inline int getTasksNum() const { return numOfTasks_; }
  [[deprecated("Use getTaskLocationsRef()")]] inline vector<int>
  getTaskLocations() const {
    return taskLocations_;
  }
  inline const vector<int>& getStartLocationsRef() const { return startLocations_; }
  [[deprecated("Use getStartLocationsRef()")]] inline vector<int>
  getStartLocations() const {
    return startLocations_;
  }
  inline const vector<pair<int, int>>& getInputPrecedenceConstraintsRef() const {
    return inputPrecedenceConstraints_;
  }
  [[deprecated("Use getInputPrecedenceConstraintsRef()")]] inline vector<
      pair<int, int>>
  getInputPrecedenceConstraints() const {
    return inputPrecedenceConstraints_;
  }
  inline const vector<vector<int>>& getAncestorsRef() const { return ancestors_; }
  [[deprecated("Use getAncestorsRef()")]] inline vector<vector<int>>
  getAncestors() const {
    return ancestors_;
  }
  inline const vector<int>& getAncestorsRef(int globalTask) const {
    validateTaskIndex(globalTask, "Instance::getAncestorsRef");
    return ancestors_[globalTask];
  }
  [[deprecated("Use getAncestorsRef(int)")]] inline vector<int> getAncestors(
      int globalTask) const {
    validateTaskIndex(globalTask, "Instance::getAncestors");
    return ancestors_[globalTask];
  }
  inline const vector<vector<int>>& getSuccessorsRef() const {
    return successors_;
  }
  [[deprecated("Use getSuccessorsRef()")]] inline vector<vector<int>>
  getSuccessors() const {
    return successors_;
  }
  inline const vector<int>& getSuccessorsRef(int globalTask) const {
    validateTaskIndex(globalTask, "Instance::getSuccessorsRef");
    return successors_[globalTask];
  }
  [[deprecated("Use getSuccessorsRef(int)")]] inline vector<int> getSuccessors(
      int globalTask) const {
    validateTaskIndex(globalTask, "Instance::getSuccessors");
    return successors_[globalTask];
  }
  inline const vector<int>& getInputPlanningOrderRef() const {
    return inputPlanningOrder_;
  }
  [[deprecated("Use getInputPlanningOrderRef()")]] inline vector<int>
  getInputPlanningOrder() const {
    return inputPlanningOrder_;
  }
  inline int getManhattanDistance(int loc1, int loc2) const {
    int loc1X = getRowCoordinate(loc1);
    int loc1Y = getColCoordinate(loc1);
    int loc2X = getRowCoordinate(loc2);
    int loc2Y = getColCoordinate(loc2);

    return abs(loc1X - loc2X) + abs(loc1Y - loc2Y);
  }
  inline int getManhattanDistance(const pair<int, int>& loc1,
                                  const pair<int, int>& loc2) const {
    return abs(loc1.first - loc2.first) + abs(loc1.second - loc2.second);
  }
  int getDefaultNumberOfTasks() const { return numOfTasks_; }
  int getDefaultNumberOfAgents() const { return numOfAgents_; }
  string getAgentTaskFName() const { return agentTaskFname_; }
  string getMapName() const { return mapFname_; }

  void preComputeHeuristics();
};
