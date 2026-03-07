#include "internal/sipps_internal.hpp"

#include <cstdint>
#include <limits>
#include <sstream>

namespace sipps_internal {
uint64_t makeNodeTieBreaker(int location, int intervalId, int timestep) {
  const uint64_t hLocation =
      LLNode::mix64((uint64_t)(uint32_t)location ^ 0x9E3779B97F4A7C15ULL);
  const uint64_t hInterval =
      LLNode::mix64((uint64_t)(uint32_t)intervalId ^ 0xC2B2AE3D27D4EB4FULL);
  const uint64_t hTimestep =
      LLNode::mix64((uint64_t)(uint32_t)timestep ^ 0x165667B19E3779F9ULL);
  uint64_t combined = hLocation;
  combined ^= hInterval + 0x9E3779B97F4A7C15ULL + (combined << 6) +
              (combined >> 2);
  combined ^= hTimestep + 0x9E3779B97F4A7C15ULL + (combined << 6) +
              (combined >> 2);
  return LLNode::mix64(combined);
}

std::vector<TimeInterval> mergeIntervals(
    const vector<pair<int, int>>* sourceIntervals, int upperExclusive) {
  std::vector<TimeInterval> clipped;
  if (sourceIntervals == nullptr || upperExclusive <= 0) {
    return clipped;
  }

  clipped.reserve(sourceIntervals->size());
  for (const auto& it : *sourceIntervals) {
    int start = max(0, it.first);
    int end = min(upperExclusive, it.second);
    if (end > start) {
      clipped.push_back({start, end});
    }
  }

  if (clipped.empty()) {
    return clipped;
  }

  // ConstraintTable currently provides sorted/merged intervals, but keep this
  // helper robust for future callers that may pass unsorted vectors.
  if (!std::is_sorted(clipped.begin(), clipped.end(),
                      [](const TimeInterval& lhs, const TimeInterval& rhs) {
                        if (lhs.start == rhs.start) {
                          return lhs.end < rhs.end;
                        }
                        return lhs.start < rhs.start;
                      })) {
    std::sort(clipped.begin(), clipped.end(),
              [](const TimeInterval& lhs, const TimeInterval& rhs) {
                if (lhs.start == rhs.start) {
                  return lhs.end < rhs.end;
                }
                return lhs.start < rhs.start;
              });
  }

  std::vector<TimeInterval> merged;
  merged.reserve(clipped.size());
  merged.push_back(clipped.front());
  for (int i = 1; i < (int)clipped.size(); i++) {
    if (clipped[i].start <= merged.back().end) {
      merged.back().end = max(merged.back().end, clipped[i].end);
    } else {
      merged.push_back(clipped[i]);
    }
  }

  return merged;
}

std::vector<TimeInterval> computeSafeIntervalsForLocation(
    const ConstraintTable& constraintTable, int location, int upperExclusive) {
  // `mergeIntervals` already returns sorted, disjoint blocked intervals.
  std::vector<TimeInterval> blocked = mergeIntervals(
      constraintTable.getConstraintIntervals(location), upperExclusive);

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
                     int startTime, int /*goalLocation*/,
                     AgentTaskPath& outPath) {
  if (goal == nullptr) {
    PLOGE << "reconstructPath: goal node is null\n";
    outPath.path.clear();
    outPath.beginTime = startTime;
    return;
  }
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

  if (goalArrivalTime < goal->timestep) {
    PLOGE << "reconstructPath: goalArrivalTime (" << goalArrivalTime
          << ") < goal timestep (" << goal->timestep << ")\n";
    outPath.path.clear();
    return;
  }
  assert(goalArrivalTime >= goal->timestep);
  const int finalG = goal->gVal + (goalArrivalTime - goal->timestep);
  if (finalG < 0) {
    PLOGE << "reconstructPath: computed negative finalG (" << finalG << ")\n";
    outPath.path.clear();
    return;
  }
  assert(finalG >= goal->gVal);
  assert(finalG >= 0);
  outPath.path.resize(finalG + 1);
  int pathIndex = 0;
  if (pathIndex >= (int)outPath.size()) {
    PLOGE << "reconstructPath: invalid initial path index " << pathIndex
          << " for path size " << outPath.size() << "\n";
    outPath.path.clear();
    return;
  }
  outPath[pathIndex].location = nodesReversed[0]->location;

  for (int i = 1; i < (int)nodesReversed.size(); i++) {
    const SIPPSNode* parent = nodesReversed[i - 1];
    const SIPPSNode* child = nodesReversed[i];
    int delta = child->timestep - parent->timestep;
    if (delta < 0) {
      PLOGE << "reconstructPath: non-positive timestep delta " << delta
            << " between parent t=" << parent->timestep
            << " and child t=" << child->timestep << "\n";
      outPath.path.clear();
      return;
    }
    if (delta == 0) {
      // Allow zero-duration identity links used by terminal goal wrappers.
      // These links carry conflict metadata but do not add path transitions.
      if (parent->location != child->location ||
          parent->gVal != child->gVal) {
        PLOGE << "reconstructPath: invalid zero-duration transition between "
              << "parent(loc=" << parent->location << ", g=" << parent->gVal
              << ") and child(loc=" << child->location
              << ", g=" << child->gVal << ")\n";
        outPath.path.clear();
        return;
      }
      continue;
    }
    assert(delta >= 1);

    for (int wait = 1; wait < delta; wait++) {
      pathIndex++;
      if (pathIndex >= (int)outPath.size()) {
        PLOGE << "reconstructPath: wait write index " << pathIndex
              << " out of bounds for path size " << outPath.size() << "\n";
        outPath.path.clear();
        return;
      }
      outPath[pathIndex].location = parent->location;
    }

    pathIndex++;
    if (pathIndex >= (int)outPath.size()) {
      PLOGE << "reconstructPath: move write index " << pathIndex
            << " out of bounds for path size " << outPath.size() << "\n";
      outPath.path.clear();
      return;
    }
    outPath[pathIndex].location = child->location;
  }

  while (pathIndex < finalG) {
    pathIndex++;
    if (pathIndex >= (int)outPath.size()) {
      PLOGE << "reconstructPath: goal-fill write index " << pathIndex
            << " out of bounds for path size " << outPath.size() << "\n";
      outPath.path.clear();
      return;
    }
    outPath[pathIndex].location = goal->location;
  }

  // Mark the terminal arrival: after goal-fill, the last element is the
  // effective stage completion point. Marking first hit is incorrect when the
  // path transits through goalLocation before final holding completion.
  if (!outPath.path.empty()) {
    outPath.path.back().isGoal = true;
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
                                    int holdingTime, string* reason,
                                    bool allowConstrainedStart) {
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
    const bool skipStartVertexCheck = allowConstrainedStart && i == 0;
    if (!skipStartVertexCheck && constraintTable.constrained(location, timestep)) {
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

}  // namespace sipps_internal
