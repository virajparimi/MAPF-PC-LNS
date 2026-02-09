#include "sipps.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <sstream>
#include "mlastar.hpp"

namespace {

struct TimeInterval {
  int start = 0;
  int end = 0;  // Exclusive
};

struct SIPPSNode {
  SIPPSNode* parent = nullptr;
  int location = -1;
  int intervalId = -1;
  int timestep = 0;  // Absolute time of arrival.
  int gVal = 0;      // Cost from start in timesteps.
  int hVal = 0;
  uint64_t tieBreaker = 0;

  int getFVal() const { return gVal + hVal; }
};

struct SIPPSNodeCompare {
  bool operator()(const SIPPSNode* lhs, const SIPPSNode* rhs) const {
    if (lhs->getFVal() != rhs->getFVal()) {
      return lhs->getFVal() > rhs->getFVal();
    }
    if (lhs->hVal != rhs->hVal) {
      return lhs->hVal > rhs->hVal;
    }
    return lhs->tieBreaker >= rhs->tieBreaker;
  }
};

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

inline uint64_t makeNodeTieBreaker(int location, int intervalId, int timestep) {
  uint64_t x = 0;
  x ^= (uint64_t)(uint32_t)location;
  x ^= ((uint64_t)(uint32_t)intervalId) << 21;
  x ^= ((uint64_t)(uint32_t)timestep) << 42;
  return LLNode::mix64(x);
}

std::vector<TimeInterval> mergeIntervals(
    const vector<pair<int, int>>* sourceIntervals, int upperExclusive) {
  std::vector<TimeInterval> merged;
  if (sourceIntervals == nullptr || upperExclusive <= 0) {
    return merged;
  }

  merged.reserve(sourceIntervals->size());
  for (const auto& it : *sourceIntervals) {
    int start = max(0, it.first);
    int end = min(upperExclusive, it.second);
    if (end > start) {
      if (!merged.empty() && start <= merged.back().end) {
        merged.back().end = max(merged.back().end, end);
      } else {
        merged.push_back({start, end});
      }
    }
  }
  return merged;
}

std::vector<TimeInterval> computeSafeIntervalsForLocation(
    const ConstraintTable& constraintTable, int location, int upperExclusive) {
  std::vector<TimeInterval> blocked = mergeIntervals(
      constraintTable.getConstraintIntervals(location), upperExclusive);

  // Positive landmarks force the agent to be at a specific location and are
  // vertex constraints for all other locations at those timesteps.
  for (const auto& landmark : constraintTable.getLandmarksRef()) {
    const int t = (int)landmark.first;
    const int requiredLocation = (int)landmark.second;
    if (requiredLocation == location || t < 0 || t >= upperExclusive) {
      continue;
    }
    blocked.push_back({t, t + 1});
  }

  if (!blocked.empty()) {
    std::sort(blocked.begin(), blocked.end(),
              [](const TimeInterval& lhs, const TimeInterval& rhs) {
                if (lhs.start == rhs.start) {
                  return lhs.end < rhs.end;
                }
                return lhs.start < rhs.start;
              });
    std::vector<TimeInterval> mergedBlocked;
    mergedBlocked.reserve(blocked.size());
    mergedBlocked.push_back(blocked[0]);
    for (int i = 1; i < (int)blocked.size(); i++) {
      TimeInterval& back = mergedBlocked.back();
      if (blocked[i].start <= back.end) {
        back.end = max(back.end, blocked[i].end);
      } else {
        mergedBlocked.push_back(blocked[i]);
      }
    }
    blocked.swap(mergedBlocked);
  }

  std::vector<TimeInterval> safe;
  int currentStart = 0;
  for (const auto& it : blocked) {
    if (currentStart < it.start) {
      safe.push_back({currentStart, it.start});
    }
    currentStart = max(currentStart, it.end);
    if (currentStart >= upperExclusive) {
      break;
    }
  }
  if (currentStart < upperExclusive) {
    safe.push_back({currentStart, upperExclusive});
  }

  return safe;
}

int findIntervalContainingTime(const std::vector<TimeInterval>& intervals,
                               int time) {
  if (intervals.empty()) {
    return -1;
  }
  int low = 0, high = (int)intervals.size() - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (time < intervals[mid].start) {
      high = mid - 1;
    } else if (time >= intervals[mid].end) {
      low = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}

int findEarliestEdgeFeasibleArrival(
    int lowerBound, int upperBound,
    const std::vector<TimeInterval>& blockedEdgeIntervals) {
  int candidate = lowerBound;
  for (const auto& blocked : blockedEdgeIntervals) {
    if (blocked.end <= candidate) {
      continue;
    }
    if (blocked.start > upperBound) {
      break;
    }
    if (candidate < blocked.start) {
      return candidate;
    }
    candidate = blocked.end;
    if (candidate > upperBound) {
      return std::numeric_limits<int>::max();
    }
  }
  return candidate;
}

void reconstructPath(const SIPPSNode* goal, int goalArrivalTime,
                     int startTime, AgentTaskPath& outPath) {
  std::vector<const SIPPSNode*> nodesReversed;
  const SIPPSNode* current = goal;
  while (current != nullptr) {
    nodesReversed.push_back(current);
    current = current->parent;
  }
  std::reverse(nodesReversed.begin(), nodesReversed.end());

  outPath.beginTime = startTime;
  if (nodesReversed.empty()) {
    return;
  }

  const int finalG = goal->gVal + (goalArrivalTime - goal->timestep);
  outPath.path.resize(finalG + 1);
  int pathIndex = 0;
  outPath[pathIndex].location = nodesReversed[0]->location;

  for (int i = 1; i < (int)nodesReversed.size(); i++) {
    const SIPPSNode* parent = nodesReversed[i - 1];
    const SIPPSNode* child = nodesReversed[i];
    int delta = child->timestep - parent->timestep;
    assert(delta >= 1);

    for (int wait = 1; wait < delta; wait++) {
      pathIndex++;
      outPath[pathIndex].location = parent->location;
    }

    pathIndex++;
    outPath[pathIndex].location = child->location;
  }

  while (pathIndex < finalG) {
    pathIndex++;
    outPath[pathIndex].location = goal->location;
  }
}

bool samePathSignature(const AgentTaskPath& lhs, const AgentTaskPath& rhs) {
  if (lhs.empty() != rhs.empty()) {
    return false;
  }
  if (lhs.empty()) {
    return true;
  }
  if (lhs.beginTime != rhs.beginTime || lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < (int)lhs.size(); i++) {
    if (lhs[i].location != rhs[i].location) {
      return false;
    }
  }
  return true;
}

int firstPathDifferenceIndex(const AgentTaskPath& lhs,
                             const AgentTaskPath& rhs) {
  if (lhs.beginTime != rhs.beginTime) {
    return 0;
  }
  const int common = min((int)lhs.size(), (int)rhs.size());
  for (int i = 0; i < common; i++) {
    if (lhs[i].location != rhs[i].location) {
      return i;
    }
  }
  if (lhs.size() != rhs.size()) {
    return common;
  }
  return -1;
}

bool isNeighborOrWait(const Instance& instance, int from, int to) {
  if (from == to) {
    return true;
  }
  const vector<int>& neighbors = instance.getNeighbors(from);
  return std::find(neighbors.begin(), neighbors.end(), to) != neighbors.end();
}

bool validatePathAgainstConstraints(const AgentTaskPath& path,
                                    const ConstraintTable& constraintTable,
                                    const Instance& instance, int startTime,
                                    int startLocation, int goalLocation,
                                    int holdingTime, string* reason) {
  if (path.empty()) {
    if (reason != nullptr) {
      *reason = "empty";
    }
    return false;
  }
  if (path.beginTime != startTime) {
    if (reason != nullptr) {
      *reason = "beginTime mismatch";
    }
    return false;
  }
  if (path.front().location != startLocation) {
    if (reason != nullptr) {
      *reason = "start location mismatch";
    }
    return false;
  }
  if (path.back().location != goalLocation) {
    if (reason != nullptr) {
      *reason = "goal location mismatch";
    }
    return false;
  }

  for (int i = 0; i < (int)path.size(); i++) {
    const int timestep = path.beginTime + i;
    const int location = path[i].location;
    if (constraintTable.constrained(location, timestep)) {
      if (reason != nullptr) {
        *reason = "vertex constraint violated";
      }
      return false;
    }

    if (i > 0) {
      const int prevLocation = path[i - 1].location;
      if (!isNeighborOrWait(instance, prevLocation, location)) {
        if (reason != nullptr) {
          *reason = "non-adjacent move";
        }
        return false;
      }
      if (constraintTable.constrained(prevLocation, location, timestep)) {
        if (reason != nullptr) {
          *reason = "edge constraint violated";
        }
        return false;
      }
    }
  }

  if (path.endTimeChecked() < holdingTime) {
    if (reason != nullptr) {
      *reason = "holding time violated";
    }
    return false;
  }
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

bool shouldEmitParityLog(int maxLogs) {
  static std::atomic<int> emitted{0};
  int current = emitted.load();
  while (current < maxLogs) {
    if (emitted.compare_exchange_weak(current, current + 1)) {
      return true;
    }
  }
  return false;
}

string compactPathSummary(const AgentTaskPath& path) {
  if (path.empty()) {
    return "empty";
  }
  return "len=" + std::to_string((int)path.size()) + ", begin=" +
         std::to_string(path.beginTime) + ", end=" +
         std::to_string(path.endTime()) + ", first=" +
         std::to_string(path.front().location) + ", last=" +
         std::to_string(path.back().location);
}

}  // namespace

AgentTaskPath MultiLabelSIPPS::findPathSegment(ConstraintTable& constraintTable,
                                               int startTime, int stage,
                                               int lb) {
  high_resolution_clock::time_point timeStart = Time::now();
  AgentTaskPath path;
  path.beginTime = startTime;
  int start = -1;
  int goal = -1;
  int holdingTime = -1;

  auto finalizeAndReturn = [&](const AgentTaskPath& sippsPath,
                               const char* exitReason) -> AgentTaskPath {
    if (!plannerParityCheck_ || stage < 0 || stage >= (int)goalLocations.size()) {
      return sippsPath;
    }

    ConstraintTable mlaConstraintTable(constraintTable);
    MultiLabelSpaceTimeAStar mlaSolver(instance, agent_);
    mlaSolver.setGoalLocations(goalLocations);
    mlaSolver.computeHeuristics();
    AgentTaskPath mlaPath =
        mlaSolver.findPathSegment(mlaConstraintTable, startTime, stage, lb);

    string sippsValidityReason;
    string mlaValidityReason;
    const bool sippsValid =
        sippsPath.empty()
            ? false
            : validatePathAgainstConstraints(sippsPath, constraintTable, instance,
                                             startTime, start, goal, holdingTime,
                                             &sippsValidityReason);
    const bool mlaValid =
        mlaPath.empty()
            ? false
            : validatePathAgainstConstraints(mlaPath, constraintTable, instance,
                                             startTime, start, goal, holdingTime,
                                             &mlaValidityReason);

    const bool successMismatch = sippsPath.empty() != mlaPath.empty();
    const bool signatureMismatch =
        !successMismatch && !sippsPath.empty() &&
        !samePathSignature(sippsPath, mlaPath);
    const bool validityMismatch = sippsValid != mlaValid;
    const bool hasMismatch =
        successMismatch || signatureMismatch || validityMismatch;

    if (hasMismatch && shouldEmitParityLog(plannerParityMaxLogs_)) {
      string mismatchType;
      if (successMismatch) {
        mismatchType = "success_mismatch";
      } else if (validityMismatch) {
        mismatchType = "validity_mismatch";
      } else {
        mismatchType = "path_signature_mismatch";
      }
      const int diffIndex =
          signatureMismatch ? firstPathDifferenceIndex(sippsPath, mlaPath) : -1;

      std::ostringstream msg;
      msg << "[PlannerParity] SIPPS vs MLA* mismatch (agent=" << agent_
          << ", stage=" << stage << ", startTime=" << startTime
          << ", lb=" << lb << ", start=" << start << ", goal=" << goal
          << ", holdingTime=" << holdingTime
          << ", exitReason=" << exitReason
          << ", mismatchType=" << mismatchType
          << ", diffIndex=" << diffIndex << ")"
          << " | SIPPS(" << compactPathSummary(sippsPath)
          << ", valid=" << sippsValid
          << (sippsValidityReason.empty()
                  ? string("")
                  : string(", reason=") + sippsValidityReason)
          << (diffIndex >= 0 && diffIndex < (int)sippsPath.size()
                  ? string(", loc@diff=") +
                        std::to_string(sippsPath[diffIndex].location)
                  : string(""))
          << ")"
          << " | MLA*(" << compactPathSummary(mlaPath)
          << ", valid=" << mlaValid
          << (mlaValidityReason.empty()
                  ? string("")
                  : string(", reason=") + mlaValidityReason)
          << (diffIndex >= 0 && diffIndex < (int)mlaPath.size()
                  ? string(", loc@diff=") +
                        std::to_string(mlaPath[diffIndex].location)
                  : string(""))
          << ")";
      if (mismatchType != "path_signature_mismatch") {
        PLOGW << msg.str();
      } else {
        PLOGD << msg.str();
      }
    }
    return sippsPath;
  };

  if (stage < 0 || stage >= (int)goalLocations.size()) {
    return finalizeAndReturn(path, "invalid_stage");
  }

  start = startLocation;
  if (stage > 0) {
    start = goalLocations[stage - 1];
  }
  goal = goalLocations[stage];

  holdingTime = constraintTable.lengthMin;
  if (stage == (int)goalLocations.size() - 1) {
    holdingTime = constraintTable.getHoldingTime();
  }

  if (constraintTable.lengthMax < startTime || holdingTime > constraintTable.lengthMax) {
    return finalizeAndReturn(path, "length_bounds");
  }

  // Upper bound for time-indexed safety reasoning.
  const int upperExclusive = constraintTable.lengthMax + 1;
  if (upperExclusive <= 0) {
    return finalizeAndReturn(path, "invalid_upper_exclusive");
  }

  unordered_map<int, std::vector<TimeInterval>> safeIntervalsCache;
  unordered_map<uint64_t, std::vector<TimeInterval>> edgeBlockedIntervalsCache;

  auto getSafeIntervals = [&](int location) -> const std::vector<TimeInterval>& {
    auto it = safeIntervalsCache.find(location);
    if (it == safeIntervalsCache.end()) {
      auto inserted = safeIntervalsCache.emplace(
          location,
          computeSafeIntervalsForLocation(constraintTable, location, upperExclusive));
      return inserted.first->second;
    }
    return it->second;
  };

  auto getEdgeBlockedIntervals = [&](int from, int to) -> const std::vector<TimeInterval>& {
    const uint64_t key =
        ((uint64_t)(uint32_t)from << 32) ^ (uint64_t)(uint32_t)to;
    auto it = edgeBlockedIntervalsCache.find(key);
    if (it == edgeBlockedIntervalsCache.end()) {
      auto inserted = edgeBlockedIntervalsCache.emplace(
          key, mergeIntervals(constraintTable.getEdgeConstraintIntervals(from, to),
                              upperExclusive));
      return inserted.first->second;
    }
    return it->second;
  };

  const auto& startSafeIntervals = getSafeIntervals(start);
  const int startInterval = findIntervalContainingTime(startSafeIntervals, startTime);
  const bool useVirtualStart = (startInterval < 0);
  if (useVirtualStart && start == goal && startTime >= holdingTime) {
    // Match MLA* semantics: start node can terminate immediately at goal even
    // if the start vertex is constrained at startTime.
    path.path.resize(1);
    path.path[0].location = start;
    return finalizeAndReturn(path, "goal_at_start_virtual");
  }

  std::priority_queue<SIPPSNode*, std::vector<SIPPSNode*>, SIPPSNodeCompare> open;
  std::vector<std::unique_ptr<SIPPSNode>> allNodes;
  allNodes.reserve(1024);

  unordered_map<StateKey, int, StateKeyHash, StateKeyEqual> bestArrivalTime;
  bestArrivalTime.reserve(1024);

  auto emplaceNode = [&](SIPPSNode* parent, int location, int intervalId,
                         int timestep, int gVal) -> SIPPSNode* {
    int hVal = max((*heuristic[stage])[location], holdingTime - timestep);
    auto node = std::make_unique<SIPPSNode>();
    node->parent = parent;
    node->location = location;
    node->intervalId = intervalId;
    node->timestep = timestep;
    node->gVal = gVal;
    node->hVal = hVal;
    node->tieBreaker = makeNodeTieBreaker(location, intervalId, timestep);
    SIPPSNode* raw = node.get();
    allNodes.push_back(std::move(node));
    open.push(raw);
    numGenerated++;
    return raw;
  };

  const int initialIntervalId = useVirtualStart ? -1 : startInterval;
  emplaceNode(nullptr, start, initialIntervalId, startTime, 0);
  bestArrivalTime[{start, initialIntervalId}] = startTime;

  while (!open.empty()) {
    SIPPSNode* current = open.top();
    open.pop();
    numExpanded++;

    const StateKey currentKey{current->location, current->intervalId};
    auto bestIt = bestArrivalTime.find(currentKey);
    if (bestIt == bestArrivalTime.end() || bestIt->second != current->timestep) {
      continue;
    }

    const bool isVirtualStartNode =
        (current->intervalId == -1 && current->location == start &&
         current->timestep == startTime);
    TimeInterval currentInterval;
    if (!isVirtualStartNode) {
      const auto& currentSafeIntervals = getSafeIntervals(current->location);
      if (current->intervalId < 0 ||
          current->intervalId >= (int)currentSafeIntervals.size()) {
        continue;
      }
      currentInterval = currentSafeIntervals[current->intervalId];
    }

    if (current->location == goal) {
      if (current->timestep >= holdingTime) {
        reconstructPath(current, current->timestep, startTime, path);
        return finalizeAndReturn(path, "goal_found");
      }
      if (!isVirtualStartNode && holdingTime < currentInterval.end) {
        reconstructPath(current, holdingTime, startTime, path);
        return finalizeAndReturn(path, "goal_wait");
      }
    }

    if (current->timestep >= constraintTable.lengthMax) {
      continue;
    }

    int latestDeparture = constraintTable.lengthMax - 1;
    if (isVirtualStartNode) {
      // If startTime is constrained at the start location, emulate MLA* by
      // allowing only immediate departure (no virtual waiting).
      latestDeparture = min(latestDeparture, current->timestep);
    } else {
      latestDeparture = min(currentInterval.end - 1, latestDeparture);
    }
    if (latestDeparture < current->timestep) {
      continue;
    }

    const vector<int>& successors = instance.getNeighbors(current->location);
    for (int successor : successors) {
      const auto& successorSafeIntervals = getSafeIntervals(successor);
      if (successorSafeIntervals.empty()) {
        continue;
      }
      const auto& edgeBlockedIntervals =
          getEdgeBlockedIntervals(current->location, successor);

      for (int succIntervalId = 0;
           succIntervalId < (int)successorSafeIntervals.size();
           succIntervalId++) {
        const TimeInterval succInterval = successorSafeIntervals[succIntervalId];

        int earliestArrival = max(current->timestep + 1, succInterval.start);
        int latestArrival =
            min(latestDeparture + 1, succInterval.end - 1);
        if (earliestArrival > latestArrival) {
          continue;
        }

        int feasibleArrival = findEarliestEdgeFeasibleArrival(
            earliestArrival, latestArrival, edgeBlockedIntervals);
        if (feasibleArrival == std::numeric_limits<int>::max() ||
            feasibleArrival > latestArrival) {
          continue;
        }

        const int transitionDuration = feasibleArrival - current->timestep;
        if (transitionDuration <= 0) {
          continue;
        }
        const int gVal = current->gVal + transitionDuration;

        const StateKey childKey{successor, succIntervalId};
        auto childBestIt = bestArrivalTime.find(childKey);
        if (childBestIt != bestArrivalTime.end() &&
            childBestIt->second <= feasibleArrival) {
          continue;
        }

        bestArrivalTime[childKey] = feasibleArrival;
        emplaceNode(current, successor, succIntervalId, feasibleArrival, gVal);
      }
    }

    const auto elapsed = ((fsec)(Time::now() - timeStart)).count();
    if (elapsed > 600) {
      return finalizeAndReturn(path, "timeout");
    }
  }

  return finalizeAndReturn(path, "search_exhausted");
}
