#pragma once

#include <plog/Log.h>
#include <cstdint>
#include <limits>
#include <memory>
#include "common.hpp"
#include "constrainttable.hpp"
#include "instance.hpp"

#include <utility>

class LLNode {

 public:
  int secondaryKey = 0;

  LLNode* parent = nullptr;
  int location{}, gVal{}, hVal = 0, timestep = 0, numOfConflicts = 0;
  bool inOpenlist = false, waitAtGoal = false;
  unsigned int stage = 0, distanceToNext = 0;
  uint64_t tieBreaker = 0;

  static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  void refreshTieBreaker() {
    // Deterministic pseudo-random tie-breaker derived from the node's identity.
    // This avoids using randomness inside heap comparators (which breaks strict
    // weak ordering), while still providing a "random-looking" ordering.
    uint64_t x = 0;
    x ^= (uint64_t)(uint32_t)location;
    x ^= (uint64_t)(uint32_t)timestep << 32;
    x ^= (uint64_t)stage * 0x632BE59BD9B4E019ULL;
    x ^= waitAtGoal ? 0xD1B54A32D192ED03ULL : 0;
    tieBreaker = mix64(x);
  }

  struct OpenCompareNode {
    bool operator()(const LLNode* lhs, const LLNode* rhs) const {
      if (lhs->gVal + lhs->hVal == rhs->gVal + rhs->hVal) {
        if (lhs->hVal == rhs->hVal) {
          if (lhs->tieBreaker == rhs->tieBreaker) {
            return false;
          }
          return lhs->tieBreaker > rhs->tieBreaker;
        }
        return lhs->hVal > rhs->hVal;
      }
      return lhs->gVal + lhs->hVal > rhs->gVal + rhs->hVal;
    }
  };

  struct FocalCompareNode {
    bool operator()(const LLNode* lhs, const LLNode* rhs) const {
      if (lhs->secondaryKey != rhs->secondaryKey) {
        return lhs->secondaryKey > rhs->secondaryKey;
      }
      if (lhs->numOfConflicts == rhs->numOfConflicts) {
        if (lhs->gVal + lhs->hVal == rhs->gVal + rhs->hVal) {
          if (lhs->hVal == rhs->hVal) {
            if (lhs->tieBreaker == rhs->tieBreaker) {
              return false;
            }
            return lhs->tieBreaker > rhs->tieBreaker;
          }
          return lhs->hVal > rhs->hVal;
        }
        return lhs->gVal + lhs->hVal > rhs->gVal + rhs->hVal;
      }
      return lhs->numOfConflicts > rhs->numOfConflicts;
    }
  };

  LLNode() { refreshTieBreaker(); }
  LLNode(LLNode* parent, int location, int gVal, int hVal, int timestep,
         int numOfConflicts, unsigned int stage)
      : parent(parent),
        location(location),
        gVal(gVal),
        hVal(hVal),
        timestep(timestep),
        numOfConflicts(numOfConflicts),
        stage(stage) {
    refreshTieBreaker();
  }
  // Disallow copy construction: copying nodes can create detached objects with
  // parent pointers into another search's storage.
  LLNode(const LLNode& old) = delete;

  // Disallow general assignment. In-place node updates during search should use
  // overwriteSearchStateFrom(...) to make intent explicit.
  LLNode& operator=(const LLNode& old) = delete;

  void overwriteSearchStateFrom(const LLNode& old) {
    secondaryKey = old.secondaryKey;
    inOpenlist = old.inOpenlist;
    parent = old.parent;
    location = old.location;
    gVal = old.gVal;
    hVal = old.hVal;
    timestep = old.timestep;
    numOfConflicts = old.numOfConflicts;
    waitAtGoal = old.waitAtGoal;
    stage = old.stage;
    distanceToNext = old.distanceToNext;
    tieBreaker = old.tieBreaker;
  }

  inline int getFVal() const {
    const long long fVal = (long long)gVal + (long long)hVal;
    if (fVal > std::numeric_limits<int>::max()) {
      return std::numeric_limits<int>::max();
    }
    if (fVal < std::numeric_limits<int>::min()) {
      return std::numeric_limits<int>::min();
    }
    return (int)fVal;
  }
};

class SingleAgentSolver {
 public:
  uint64_t numExpanded = 0, numGenerated = 0;
  double segmentTimeoutSec = 600.0;

  const Instance& instance;

  int startLocation;
  vector<int> goalLocations;
  vector<int> heuristicLandmarks;

  // For each stage, stores the global task index used to access
  // instance.heuristics_. This avoids storing raw pointers into Instance-owned
  // vectors.
  vector<int> heuristicTaskIdx;

  void computeHeuristics();
  int getStageGoalDistance(int stage, int location) const {
    if (stage < 0 || stage >= (int)heuristicTaskIdx.size()) {
      assert(false);
      return MAX_TIMESTEP;
    }
    const int globalTask = heuristicTaskIdx[stage];
    if (globalTask < 0 || globalTask >= (int)instance.heuristics_.size()) {
      assert(false);
      return MAX_TIMESTEP;
    }
    if (location < 0 || location >= (int)instance.heuristics_[globalTask].size()) {
      assert(false);
      return MAX_TIMESTEP;
    }
    return instance.heuristics_[globalTask][location];
  }
  int getHeuristic(int stage, int location) const {
    if (stage < 0 || stage >= (int)heuristicLandmarks.size()) {
      assert(false);
      return MAX_TIMESTEP;
    }
    return getStageGoalDistance(stage, location) + heuristicLandmarks[stage];
  }
  inline void setGoalLocations(vector<int> goals) {
    goalLocations = std::move(goals);
    computeHeuristics();
  }
  inline void setSegmentTimeout(double timeoutSec) {
    segmentTimeoutSec = timeoutSec;
  }
  inline double getSegmentTimeout() const { return segmentTimeoutSec; }

  virtual std::shared_ptr<SingleAgentSolver> cloneForAgent(int agent) const = 0;
  virtual void copyStateFrom(const SingleAgentSolver& other) {
    if (this == &other) {
      return;
    }
    numExpanded = other.numExpanded;
    numGenerated = other.numGenerated;
    segmentTimeoutSec = other.segmentTimeoutSec;
    goalLocations = other.goalLocations;
    heuristicLandmarks = other.heuristicLandmarks;
    heuristicTaskIdx = other.heuristicTaskIdx;
  }

  virtual string getName() const = 0;
  virtual AgentTaskPath findPathSegment(ConstraintTable& constraintTable,
                                        int startTime, int stage,
                                        int lowerBound) = 0;
  const vector<int>& getNeighbors(int curr) const {
    return instance.getNeighbors(curr);
  }

  SingleAgentSolver(const Instance& instance, int agent,
                    bool initializeHeuristics = true)
      : instance(instance),
        startLocation(instance.startLocations_[agent]),
        // Default to all tasks so shared helpers (e.g., greedy assignment)
        // can query any task-distance immediately; replanning paths should
        // narrow goals via setGoalLocations(...).
        goalLocations(instance.taskLocations_) {
    if (initializeHeuristics) {
      computeHeuristics();
    }
  }
  virtual ~SingleAgentSolver() = default;

 protected:
  void copyPlannerStateTo(SingleAgentSolver& target) const {
    target.numExpanded = numExpanded;
    target.numGenerated = numGenerated;
    target.segmentTimeoutSec = segmentTimeoutSec;
    target.goalLocations = goalLocations;
    target.heuristicLandmarks = heuristicLandmarks;
    target.heuristicTaskIdx = heuristicTaskIdx;
  }
};
