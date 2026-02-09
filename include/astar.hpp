#pragma once

#include <plog/Log.h>
#include <cstdint>
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
          return lhs->tieBreaker >= rhs->tieBreaker;
        }
        return lhs->hVal >= rhs->hVal;
      }
      return lhs->gVal + lhs->hVal >= rhs->gVal + rhs->hVal;
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
            return lhs->tieBreaker >= rhs->tieBreaker;
          }
          return lhs->hVal >= rhs->hVal;
        }
        return lhs->gVal + lhs->hVal >= rhs->gVal + rhs->hVal;
      }
      return lhs->numOfConflicts >= rhs->numOfConflicts;
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
  LLNode(const LLNode& old)
      : secondaryKey(old.secondaryKey),
        parent(old.parent),
        location(old.location),
        gVal(old.gVal),
        hVal(old.hVal),
        timestep(old.timestep),
        numOfConflicts(old.numOfConflicts),
        waitAtGoal(old.waitAtGoal),
        stage(old.stage),
        distanceToNext(old.distanceToNext),
        tieBreaker(old.tieBreaker) {}

  inline int getFVal() const { return gVal + hVal; }

  void copy(const LLNode& old) {
    secondaryKey = old.secondaryKey;
    location = old.location;
    gVal = old.gVal;
    hVal = old.hVal;
    parent = old.parent;
    timestep = old.timestep;
    numOfConflicts = old.numOfConflicts;
    waitAtGoal = old.waitAtGoal;
    stage = old.stage;
    distanceToNext = old.distanceToNext;
    tieBreaker = old.tieBreaker;
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

  // For each stage, points to the precomputed heuristic vector for that global task.
  // Owned by Instance; valid for the lifetime of `instance`.
  vector<const vector<int>*> heuristic;

  void computeHeuristics();
  int getHeuristic(int stage, int location) const {
    return (*heuristic[stage])[location] + heuristicLandmarks[stage];
  }
  int computeHeuristic(int from, int to) const {
    return instance.getManhattanDistance(from, to);
  }
  inline void setGoalLocations(vector<int> goals) {
    goalLocations = std::move(goals);
  }
  inline void setSegmentTimeout(double timeoutSec) {
    segmentTimeoutSec = timeoutSec;
  }
  inline double getSegmentTimeout() const { return segmentTimeoutSec; }

  virtual string getName() const = 0;
  virtual AgentTaskPath findPathSegment(ConstraintTable& constraintTable,
                                        int startTime, int stage,
                                        int lowerBound) = 0;
  const vector<int>& getNeighbors(int curr) const {
    return instance.getNeighbors(curr);
  }

  SingleAgentSolver(const Instance& instance, int agent)
      : instance(instance),
        startLocation(instance.startLocations_[agent]),
        goalLocations(instance.taskLocations_) {
    computeHeuristics();
  }
  virtual ~SingleAgentSolver() = default;
};
