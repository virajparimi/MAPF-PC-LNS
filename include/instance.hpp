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
  // Maps given task to all its predecessors as given in the input
  map<int, vector<int>> taskDependencies_;
  vector<vector<int>> ancestors_, successors_;
  vector<pair<int, int>> inputPrecedenceConstraints_;

  bool loadMap();
  bool loadKivaMap();
  bool loadKivaTasks();
  bool loadAgentsAndTasks();
  void printMap() const;
  void preComputeNeighbors();
  friend class Solution;
  friend class SingleAgentSolver;

 public:
  int mapSize{}, numOfCols{}, numOfRows{};

  Instance() = default;
  Instance(const string& mapFname, const string& agentTaskFname,
           int numOfAgents = 0, int numOfTasks = 0);

  inline int getTaskLocations(int task) const { return taskLocations_[task]; }
  // Prefer ref-returning accessors in performance-sensitive code.
  inline const vector<int>& getTaskLocationsRef() const { return taskLocations_; }
  vector<int> getTaskLocations(vector<int> tasks) const {
    vector<int> taskLocs(tasks.size(), 0);
    for (int i = 0; i < (int)tasks.size(); i++) {
      taskLocs[i] = taskLocations_[tasks[i]];
    }
    return taskLocs;
  }
  vector<vector<int>> getHeuristics() { return heuristics_; }
  inline const vector<vector<int>>& getHeuristicsRef() const { return heuristics_; }
  vector<int> getHeuristics(int globalTask) {
    assert(globalTask < numOfTasks_);
    return heuristics_[globalTask];
  }
  inline const vector<int>& getHeuristicsRef(int globalTask) const {
    assert(globalTask < numOfTasks_);
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
  inline int getRowCoordinate(int id) const { return id / this->numOfCols; }
  inline int getColCoordinate(int id) const { return id % this->numOfCols; }
  inline pair<int, int> getCoordinate(int id) const {
    return make_pair(getRowCoordinate(id), getColCoordinate(id));
  }
  inline int getCols() const { return numOfCols; }
  inline int getAgentNum() const { return numOfAgents_; }
  inline int getTasksNum() const { return numOfTasks_; }
  inline vector<int> getTaskLocations() const { return taskLocations_; }
  inline const vector<int>& getStartLocationsRef() const { return startLocations_; }
  inline vector<int> getStartLocations() const { return startLocations_; }
  inline map<int, vector<int>> getTaskDependencies() const {
    return taskDependencies_;
  }
  inline const map<int, vector<int>>& getTaskDependenciesRef() const {
    return taskDependencies_;
  }
  inline const vector<pair<int, int>>& getInputPrecedenceConstraintsRef() const {
    return inputPrecedenceConstraints_;
  }
  inline vector<pair<int, int>> getInputPrecedenceConstraints() const {
    return inputPrecedenceConstraints_;
  }
  inline const vector<vector<int>>& getAncestorsRef() const { return ancestors_; }
  inline vector<vector<int>> getAncestors() const { return ancestors_; }
  inline const vector<int>& getAncestorsRef(int globalTask) const {
    assert(globalTask < numOfTasks_);
    return ancestors_[globalTask];
  }
  inline vector<int> getAncestors(int globalTask) const {
    assert(globalTask < numOfTasks_);
    return ancestors_[globalTask];
  }
  inline const vector<vector<int>>& getSuccessorsRef() const {
    return successors_;
  }
  inline vector<vector<int>> getSuccessors() const { return successors_; }
  inline const vector<int>& getSuccessorsRef(int globalTask) const {
    assert(globalTask < numOfTasks_);
    return successors_[globalTask];
  }
  inline vector<int> getSuccessors(int globalTask) const {
    assert(globalTask < numOfTasks_);
    return successors_[globalTask];
  }
  inline const vector<int>& getInputPlanningOrderRef() const {
    return inputPlanningOrder_;
  }
  inline vector<int> getInputPlanningOrder() const {
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

  void preComputeHeuristics() {
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
};
