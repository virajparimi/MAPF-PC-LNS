#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "sipps.hpp"

namespace sipps_internal {

struct TimeInterval {
  int start = 0;
  int end = 0;  // Exclusive
};

struct SIPPSNode;

struct SIPPSOpenCompare {
  bool operator()(const SIPPSNode* lhs, const SIPPSNode* rhs) const;
};

struct SIPPSFocalCompare {
  bool operator()(const SIPPSNode* lhs, const SIPPSNode* rhs) const;
};

struct SIPPSNode {
  pairing_heap<SIPPSNode*, compare<SIPPSOpenCompare>>::handle_type openHandle;
  pairing_heap<SIPPSNode*, compare<SIPPSFocalCompare>>::handle_type
      focalHandle;
  bool inOpenlist = false;
  bool inFocal = false;

  SIPPSNode* parent = nullptr;
  int location = -1;
  int intervalId = -1;
  int highGeneration = 0;
  int highExpansion = 0;
  int timestep = 0;  // Absolute time of arrival.
  int gVal = 0;      // Cost from start in timesteps.
  int hVal = 0;
  int numConflicts = 0;
  bool collisionV = false;
  bool waitAtGoal = false;
  bool isGoalTerminal = false;
  int secondaryKey = 0;
  uint64_t tieBreaker = 0;

  int getFVal() const {
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

inline bool SIPPSOpenCompare::operator()(const SIPPSNode* lhs,
                                         const SIPPSNode* rhs) const {
  if (lhs->getFVal() != rhs->getFVal()) {
    return lhs->getFVal() > rhs->getFVal();
  }
  if (lhs->hVal != rhs->hVal) {
    return lhs->hVal > rhs->hVal;
  }
  return lhs->tieBreaker > rhs->tieBreaker;
}

inline bool SIPPSFocalCompare::operator()(const SIPPSNode* lhs,
                                          const SIPPSNode* rhs) const {
  if (lhs->secondaryKey != rhs->secondaryKey) {
    return lhs->secondaryKey > rhs->secondaryKey;
  }
  if (lhs->getFVal() != rhs->getFVal()) {
    return lhs->getFVal() > rhs->getFVal();
  }
  if (lhs->hVal != rhs->hVal) {
    return lhs->hVal > rhs->hVal;
  }
  return lhs->tieBreaker > rhs->tieBreaker;
}

struct StateKey {
  int location = -1;
  int intervalId = -1;
};

struct StateKeyHash {
  size_t operator()(const StateKey& key) const {
    const uint64_t packed = (uint64_t)(uint32_t)key.location ^
                            ((uint64_t)(uint32_t)key.intervalId << 32);
    return (size_t)LLNode::mix64(packed);
  }
};

struct StateKeyEqual {
  bool operator()(const StateKey& lhs, const StateKey& rhs) const {
    return lhs.location == rhs.location && lhs.intervalId == rhs.intervalId;
  }
};

uint64_t makeNodeTieBreaker(int location, int intervalId, int timestep);
std::vector<TimeInterval> mergeIntervals(const vector<pair<int, int>>* sourceIntervals,
                                         int upperExclusive);
std::vector<TimeInterval> computeSafeIntervalsForLocation(
    const ConstraintTable& constraintTable, int location, int upperExclusive);
int findIntervalContainingTime(const std::vector<TimeInterval>& intervals,
                               int time);
int findEarliestEdgeFeasibleArrival(
    int lowerBound, int upperBound,
    const std::vector<TimeInterval>& blockedEdgeIntervals);
void reconstructPath(const SIPPSNode* goal, int goalArrivalTime, int startTime,
                     int goalLocation, AgentTaskPath& outPath);
bool samePathSignature(const AgentTaskPath& lhs, const AgentTaskPath& rhs);
int firstPathDifferenceIndex(const AgentTaskPath& lhs,
                             const AgentTaskPath& rhs);
bool isNeighborOrWait(const Instance& instance, int from, int to);
bool validatePathAgainstConstraints(const AgentTaskPath& path,
                                    const ConstraintTable& constraintTable,
                                    const Instance& instance, int startTime,
                                    int startLocation, int goalLocation,
                                    int holdingTime, string* reason,
                                    bool allowConstrainedStart = false);
string compactPathSummary(const AgentTaskPath& path);

}  // namespace sipps_internal
