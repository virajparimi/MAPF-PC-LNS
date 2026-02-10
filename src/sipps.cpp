#include "internal/sipps_internal.hpp"
#include "mlastar.hpp"

using namespace sipps_internal;

AgentTaskPath MultiLabelSIPPS::findPathSegment(ConstraintTable& constraintTable,
                                               int startTime, int stage,
                                               int lb) {
  Time::time_point timeStart = Time::now();
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
    mlaSolver.setSegmentTimeout(segmentTimeoutSec);
    AgentTaskPath mlaPath =
        mlaSolver.findPathSegment(mlaConstraintTable, startTime, stage, lb);

    string sippsValidityReason;
    string mlaValidityReason;
    // Match low-level planner semantics: start node is a root state and is not
    // vertex-constraint checked at startTime.
    constexpr bool kAllowConstrainedStartForValidation = true;
    const bool sippsValid =
        sippsPath.empty()
            ? false
            : validatePathAgainstConstraints(sippsPath, constraintTable, instance,
                                             startTime, start, goal, holdingTime,
                                             &sippsValidityReason,
                                             kAllowConstrainedStartForValidation);
    const bool mlaValid =
        mlaPath.empty()
            ? false
            : validatePathAgainstConstraints(mlaPath, constraintTable, instance,
                                             startTime, start, goal, holdingTime,
                                             &mlaValidityReason,
                                             kAllowConstrainedStartForValidation);

    const bool successMismatch = sippsPath.empty() != mlaPath.empty();
    const bool signatureMismatch =
        !successMismatch && !sippsPath.empty() &&
        !samePathSignature(sippsPath, mlaPath);
    const bool validityMismatch = sippsValid != mlaValid;
    const bool hasMismatch =
        successMismatch || signatureMismatch || validityMismatch;

    const bool canEmitParityLog =
        plannerParityLogsEmitted_ < plannerParityMaxLogs_;
    if (hasMismatch && canEmitParityLog) {
      plannerParityLogsEmitted_++;
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
    // if the start vertex is constrained at startTime, because the root state
    // is not vertex-constraint checked.
    path.path.resize(1);
    path.path[0].location = start;
    return finalizeAndReturn(path, "goal_at_start_virtual");
  }

  pairing_heap<SIPPSNode*, compare<SIPPSOpenCompare>> openList;
  pairing_heap<SIPPSNode*, compare<SIPPSFocalCompare>> focalList;
  int minFVal = 0;
  int lowerBound = 0;
  std::vector<std::unique_ptr<SIPPSNode>> allNodes;
  allNodes.reserve(1024);
  constexpr uint32_t kTimeoutCheckStride = 64;
  uint32_t intervalChecksSinceTimeoutProbe = 0;
  const auto timedOut = [&]() -> bool {
    return ((fsec)(Time::now() - timeStart)).count() > segmentTimeoutSec;
  };

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
    node->secondaryKey = -gVal;
    node->tieBreaker = makeNodeTieBreaker(location, intervalId, timestep);
    SIPPSNode* raw = node.get();
    allNodes.push_back(std::move(node));
    return raw;
  };

  auto pushNode = [&](SIPPSNode* node) {
    numGenerated++;
    node->inOpenlist = true;
    node->openHandle = openList.push(node);
    if (node->getFVal() <= lowerBound) {
      node->focalHandle = focalList.push(node);
      node->inFocal = true;
    }
  };

  auto updateFocalList = [&]() {
    if (openList.empty()) {
      return;
    }
    SIPPSNode* openHead = openList.top();
    if (openHead->getFVal() > minFVal) {
      const int newMinFVal = openHead->getFVal();
      const int newLowerBound = max(lowerBound, newMinFVal);
      for (SIPPSNode* node : openList) {
        if (!node->inFocal && node->inOpenlist &&
            node->getFVal() > lowerBound &&
            node->getFVal() <= newLowerBound) {
          node->focalHandle = focalList.push(node);
          node->inFocal = true;
        }
      }
      minFVal = newMinFVal;
      lowerBound = newLowerBound;
    }
  };

  auto popNode = [&]() -> SIPPSNode* {
    while (!focalList.empty()) {
      SIPPSNode* node = focalList.top();
      focalList.pop();
      node->inFocal = false;
      if (!node->inOpenlist) {
        continue;
      }
      numExpanded++;
      node->inOpenlist = false;
      openList.erase(node->openHandle);
      return node;
    }
    return nullptr;
  };

  const int initialIntervalId = useVirtualStart ? -1 : startInterval;
  SIPPSNode* startNode =
      emplaceNode(nullptr, start, initialIntervalId, startTime, 0);
  bestArrivalTime[{start, initialIntervalId}] = startTime;
  minFVal = startNode->getFVal();
  // Mirror MLA* bounded search behavior: the focal bound is initialized from
  // both the current minimum f and the provided lower bound `lb`.
  lowerBound = max(holdingTime - startTime, max(minFVal, lb));
  pushNode(startNode);

  while (!openList.empty()) {
    if (timedOut()) {
      return finalizeAndReturn(path, "timeout");
    }
    updateFocalList();
    SIPPSNode* current = popNode();
    if (current == nullptr) {
      // No focal-eligible nodes left (can happen with stale entries).
      continue;
    }

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
        reconstructPath(current, current->timestep, startTime, goal, path);
        return finalizeAndReturn(path, "goal_found");
      }
      if (!isVirtualStartNode && holdingTime < currentInterval.end) {
        reconstructPath(current, holdingTime, startTime, goal, path);
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
        intervalChecksSinceTimeoutProbe++;
        if (intervalChecksSinceTimeoutProbe >= kTimeoutCheckStride) {
          intervalChecksSinceTimeoutProbe = 0;
          if (timedOut()) {
            return finalizeAndReturn(path, "timeout");
          }
        }

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
        SIPPSNode* next =
            emplaceNode(current, successor, succIntervalId, feasibleArrival,
                        gVal);
        pushNode(next);
      }
    }

  }

  return finalizeAndReturn(path, "search_exhausted");
}
