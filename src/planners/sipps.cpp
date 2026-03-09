#include "internal/sipps_internal.hpp"
#include "mlastar.hpp"
#include <cstring>
#include <deque>

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
    SearchOutcome outcome = SearchOutcome::unknown;
    if (exitReason == nullptr) {
      outcome = sippsPath.empty() ? SearchOutcome::search_exhausted
                                  : SearchOutcome::found;
    } else if (strcmp(exitReason, "timeout") == 0) {
      outcome = SearchOutcome::timeout;
    } else if (strcmp(exitReason, "goal_found") == 0 ||
               strcmp(exitReason, "goal_wait") == 0 ||
               strcmp(exitReason, "goal_at_start_virtual") == 0) {
      outcome = SearchOutcome::found;
    } else if (strcmp(exitReason, "invalid_stage") == 0 ||
               strcmp(exitReason, "length_bounds") == 0 ||
               strcmp(exitReason, "invalid_upper_exclusive") == 0) {
      outcome = SearchOutcome::invalid_input;
    } else if (strcmp(exitReason, "search_exhausted") == 0) {
      outcome = SearchOutcome::search_exhausted;
    } else {
      outcome = sippsPath.empty() ? SearchOutcome::search_exhausted
                                  : SearchOutcome::found;
    }
    setLastSearchOutcome(outcome);

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
        plannerParityLogsEmitted_ < kPlannerParityMaxLogs;
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

  boost::unordered_map<int, std::vector<TimeInterval>> safeIntervalsCache;
  boost::unordered_map<uint64_t, std::vector<TimeInterval>> edgeBlockedIntervalsCache;

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

  std::deque<SIPPSNode> allNodes;
  pairing_heap<SIPPSNode*, compare<SIPPSOpenCompare>> openList;
  pairing_heap<SIPPSNode*, compare<SIPPSFocalCompare>> focalList;
  int minFVal = 0;
  int lowerBound = 0;
  constexpr uint32_t kTimeoutCheckStride = 64;
  uint32_t intervalChecksSinceTimeoutProbe = 0;
  const auto timedOut = [&]() -> bool {
    return ((fsec)(Time::now() - timeStart)).count() > segmentTimeoutSec;
  };

  struct DominanceKey {
    int location = -1;
    int highGeneration = 0;
    bool waitAtGoal = false;
    bool isGoalTerminal = false;
  };
  struct DominanceKeyHash {
    size_t operator()(const DominanceKey& key) const {
      const uint64_t packed =
          ((uint64_t)(uint32_t)key.location) ^
          (((uint64_t)(uint32_t)key.highGeneration) << 32) ^
          (key.waitAtGoal ? 0x9E3779B97F4A7C15ULL : 0ULL) ^
          (key.isGoalTerminal ? 0xC2B2AE3D27D4EB4FULL : 0ULL);
      return (size_t)LLNode::mix64(packed);
    }
  };
  struct DominanceKeyEq {
    bool operator()(const DominanceKey& lhs, const DominanceKey& rhs) const {
      return lhs.location == rhs.location &&
             lhs.highGeneration == rhs.highGeneration &&
             lhs.waitAtGoal == rhs.waitAtGoal &&
             lhs.isGoalTerminal == rhs.isGoalTerminal;
    }
  };
  boost::unordered_map<DominanceKey, list<SIPPSNode*>, DominanceKeyHash, DominanceKeyEq>
      dominanceTable;
  dominanceTable.reserve(1024);

  auto emplaceNode = [&](SIPPSNode* parent, int location, int intervalId,
                         int timestep, int gVal, int numConflicts,
                         int highGeneration, int highExpansion,
                         bool collisionV = false, int forcedHVal = -1,
                         bool waitAtGoal = false,
                         bool isGoalTerminal = false) -> SIPPSNode* {
    int hVal = forcedHVal;
    if (hVal < 0) {
      hVal = max(getStageGoalDistance(stage, location), holdingTime - timestep);
    }
    allNodes.emplace_back();
    SIPPSNode* node = &allNodes.back();
    node->parent = parent;
    node->location = location;
    node->intervalId = intervalId;
    node->highGeneration = highGeneration;
    node->highExpansion = highExpansion;
    node->timestep = timestep;
    node->gVal = gVal;
    node->hVal = hVal;
    node->numConflicts = numConflicts;
    node->collisionV = collisionV;
    node->waitAtGoal = waitAtGoal;
    node->isGoalTerminal = isGoalTerminal;
    node->secondaryKey = numConflicts;
    node->tieBreaker = makeNodeTieBreaker(location, intervalId, timestep);
    return node;
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
      if (newLowerBound == lowerBound) {
        // No expansion of the focal bound; no node can become newly focal.
        minFVal = newMinFVal;
        return;
      }
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

  auto dominanceCheck = [&](SIPPSNode* newNode) -> bool {
    if (newNode->highExpansion <= newNode->timestep) {
      return false;
    }

    const DominanceKey key{newNode->location, newNode->highGeneration,
                           newNode->waitAtGoal,
                           newNode->isGoalTerminal};
    auto& bucket = dominanceTable[key];
    for (auto it = bucket.begin(); it != bucket.end();) {
      SIPPSNode* oldNode = *it;
      if (oldNode == nullptr) {
        it = bucket.erase(it);
        continue;
      }

      if (oldNode->timestep <= newNode->timestep &&
          oldNode->numConflicts <= newNode->numConflicts) {
        return false;
      }
      if (oldNode->timestep >= newNode->timestep &&
          oldNode->numConflicts >= newNode->numConflicts) {
        if (oldNode->inOpenlist) {
          if (oldNode->inFocal) {
            focalList.erase(oldNode->focalHandle);
            oldNode->inFocal = false;
          }
          openList.erase(oldNode->openHandle);
          oldNode->inOpenlist = false;
        }
        it = bucket.erase(it);
        // Keep scanning so newNode is registered and overlap clipping is
        // applied against all remaining entries in the bucket.
        continue;
      }

      if (oldNode->timestep < newNode->highExpansion &&
          newNode->timestep < oldNode->highExpansion) {
        if (oldNode->timestep <= newNode->timestep) {
          assert(oldNode->numConflicts > newNode->numConflicts);
          oldNode->highExpansion = min(oldNode->highExpansion, newNode->timestep);
        } else {
          assert(oldNode->numConflicts <= newNode->numConflicts);
          newNode->highExpansion = min(newNode->highExpansion, oldNode->timestep);
          if (newNode->highExpansion <= newNode->timestep) {
            return false;
          }
        }
      }
      ++it;
    }
    bucket.push_back(newNode);
    return true;
  };

  const int initialIntervalId = useVirtualStart ? -1 : startInterval;
  const int initialHighGeneration =
      useVirtualStart ? (startTime + 1) : startSafeIntervals[startInterval].end;
  SIPPSNode* startNode =
      emplaceNode(nullptr, start, initialIntervalId, startTime, 0, 0,
                  initialHighGeneration, initialHighGeneration);
  if (!dominanceCheck(startNode)) {
    return finalizeAndReturn(path, "search_exhausted");
  }
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

    if (current->isGoalTerminal) {
      reconstructPath(current, current->timestep, startTime, goal, path);
      return finalizeAndReturn(path, "goal_found");
    }

    const bool isVirtualStartNode =
        (current->intervalId == -1 && current->location == start &&
         current->timestep == startTime);
    const std::vector<TimeInterval>* currentSafeIntervals = nullptr;
    TimeInterval currentInterval;
    if (!isVirtualStartNode) {
      currentSafeIntervals = &getSafeIntervals(current->location);
      if (current->intervalId < 0 ||
          current->intervalId >= (int)currentSafeIntervals->size()) {
        continue;
      }
      currentInterval = (*currentSafeIntervals)[current->intervalId];
    }

    if (current->location == goal) {
      int goalArrival = -1;
      if (current->timestep >= holdingTime) {
        goalArrival = current->timestep;
      } else if (!isVirtualStartNode && holdingTime < currentInterval.end) {
        goalArrival = holdingTime;
      }
      if (goalArrival >= 0) {
        const int goalGVal =
            current->gVal + max(0, goalArrival - current->timestep);
        const int futureConflicts =
            constraintTable.getFutureSoftConflicts(goal, goalArrival);
        const long long goalConflictsLL =
            (long long)current->numConflicts + (long long)futureConflicts;
        const int goalConflicts =
            goalConflictsLL > std::numeric_limits<int>::max()
                ? std::numeric_limits<int>::max()
                : (int)goalConflictsLL;
        SIPPSNode* goalParent = current;
        SIPPSNode* goalNode = emplaceNode(
            goalParent, goal, current->intervalId, goalArrival, goalGVal,
            goalConflicts, current->highGeneration, current->highExpansion,
            false, 0, false, true);
        if (dominanceCheck(goalNode)) {
          pushNode(goalNode);
        }
      }
    }

    if (current->timestep >= constraintTable.lengthMax) {
      continue;
    }

    int latestDeparture =
        min(constraintTable.lengthMax - 1, current->highExpansion - 1);
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

        const auto stepConflictsAt = [&](int arrivalTimestep) -> int {
          return constraintTable.getSoftNumOfConflictsForStep(
              current->location, successor, arrivalTimestep);
        };

        const int initialStepConflicts = stepConflictsAt(feasibleArrival);
        int firstZeroConflictArrival = -1;
        if (initialStepConflicts > 0) {
          firstZeroConflictArrival =
              constraintTable.getEarliestSoftConflictFreeTimestep(
                  current->location, successor, feasibleArrival, latestArrival);
        }

        auto tryCreateChild = [&](int childArrival, int childHighExpansion,
                                  int stepConflicts, bool collisionV) {
          if (childHighExpansion <= childArrival) {
            return;
          }
          const int transitionDuration = childArrival - current->timestep;
          const int gVal = current->gVal + transitionDuration;
          const long long childConflictsLL =
              (long long)current->numConflicts + (long long)stepConflicts;
          const int childConflicts =
              childConflictsLL > std::numeric_limits<int>::max()
                  ? std::numeric_limits<int>::max()
                  : (int)childConflictsLL;
          const int pathMaxTarget =
              (childConflicts > 0) ? holdingTime : current->getFVal();
          const int childHVal = max(getStageGoalDistance(stage, successor),
                                    pathMaxTarget - childArrival);
          SIPPSNode* child = emplaceNode(current, successor, succIntervalId,
                                         childArrival, gVal, childConflicts,
                                         succInterval.end, childHighExpansion,
                                         collisionV, childHVal);
          if (dominanceCheck(child)) {
            pushNode(child);
          }
        };

        if (firstZeroConflictArrival > feasibleArrival) {
          tryCreateChild(feasibleArrival, firstZeroConflictArrival,
                         initialStepConflicts, true);
          if (firstZeroConflictArrival < succInterval.end) {
            tryCreateChild(firstZeroConflictArrival, succInterval.end, 0, false);
          }
        } else {
          tryCreateChild(feasibleArrival, succInterval.end, initialStepConflicts,
                         initialStepConflicts > 0);
        }
      }
    }

    if (isVirtualStartNode) {
      const auto& virtualSafeIntervals = getSafeIntervals(current->location);
      int nextIntervalId = -1;
      for (int idx = 0; idx < (int)virtualSafeIntervals.size(); idx++) {
        if (virtualSafeIntervals[idx].start > current->timestep) {
          nextIntervalId = idx;
          break;
        }
      }
      if (nextIntervalId >= 0) {
        const TimeInterval nextInterval = virtualSafeIntervals[nextIntervalId];
        const int waitArrival = nextInterval.start;
        if (waitArrival <= constraintTable.lengthMax) {
          const int waitTransitionDuration = waitArrival - current->timestep;
          if (waitTransitionDuration > 0) {
            const int waitStepConflicts =
                constraintTable.getSoftNumOfConflictsForStep(
                    current->location, current->location, waitArrival);
            const long long waitConflictsLL =
                (long long)current->numConflicts +
                (long long)waitStepConflicts;
            const int waitConflicts =
                waitConflictsLL > std::numeric_limits<int>::max()
                    ? std::numeric_limits<int>::max()
                    : (int)waitConflictsLL;
            const int pathMaxTarget =
                (waitConflicts > 0) ? holdingTime : current->getFVal();
            const int nextHVal = max(getStageGoalDistance(stage, current->location),
                                     pathMaxTarget - waitArrival);
            if (waitArrival + nextHVal <= constraintTable.lengthMax) {
              SIPPSNode* waitNode = emplaceNode(
                  current, current->location, nextIntervalId, waitArrival,
                  current->gVal + waitTransitionDuration, waitConflicts,
                  nextInterval.end, nextInterval.end, waitStepConflicts > 0,
                  nextHVal, current->location == goal, false);
              if (dominanceCheck(waitNode)) {
                pushNode(waitNode);
              }
            }
          }
        }
      }
    } else if (currentSafeIntervals != nullptr &&
        current->highExpansion == current->highGeneration) {
      const int waitArrival = current->highExpansion;
      if (waitArrival <= constraintTable.lengthMax) {
        const int nextIntervalId =
            findIntervalContainingTime(*currentSafeIntervals, waitArrival);
        if (nextIntervalId >= 0 &&
            nextIntervalId < (int)currentSafeIntervals->size() &&
            (*currentSafeIntervals)[nextIntervalId].start == waitArrival) {
          const TimeInterval nextInterval = (*currentSafeIntervals)[nextIntervalId];
          const int waitTransitionDuration = waitArrival - current->timestep;
          if (waitTransitionDuration > 0) {
            const int waitStepConflicts =
                constraintTable.getSoftNumOfConflictsForStep(
                    current->location, current->location, waitArrival);
            const long long waitConflictsLL =
                (long long)current->numConflicts +
                (long long)waitStepConflicts;
            const int waitConflicts =
                waitConflictsLL > std::numeric_limits<int>::max()
                    ? std::numeric_limits<int>::max()
                    : (int)waitConflictsLL;
            const int pathMaxTarget =
                (waitConflicts > 0) ? holdingTime : current->getFVal();
            const int nextHVal =
                max(getStageGoalDistance(stage, current->location),
                    pathMaxTarget - waitArrival);
            if (waitArrival + nextHVal <= constraintTable.lengthMax) {
              SIPPSNode* waitNode = emplaceNode(
                  current, current->location, nextIntervalId, waitArrival,
                  current->gVal + waitTransitionDuration, waitConflicts,
                  nextInterval.end, nextInterval.end, waitStepConflicts > 0,
                  nextHVal, current->location == goal, false);
              if (dominanceCheck(waitNode)) {
                pushNode(waitNode);
              }
            }
          }
        }
      }
    }

  }

  return finalizeAndReturn(path, "search_exhausted");
}
