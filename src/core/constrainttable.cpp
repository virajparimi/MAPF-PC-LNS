#include "constrainttable.hpp"

void ConstraintTable::normalizeIntervals(IntervalBucket& bucket) {
  auto& intervals = bucket.intervals;
  if (intervals.size() <= 1) {
    return;
  }

  std::sort(intervals.begin(), intervals.end(),
            [](const pair<int, int>& lhs, const pair<int, int>& rhs) {
              if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });

  int write = 0;
  for (int read = 1; read < (int)intervals.size(); read++) {
    if (intervals[read].first <= intervals[write].second) {
      intervals[write].second =
          max(intervals[write].second, intervals[read].second);
    } else {
      write++;
      intervals[write] = intervals[read];
    }
  }
  intervals.resize(write + 1);
}

void ConstraintTable::insertMergedInterval(IntervalBucket& bucket, int tMin,
                                           int tMax) {
  auto& intervals = bucket.intervals;
  if (intervals.empty()) {
    intervals.emplace_back(tMin, tMax);
    return;
  }

  // In-place merge with a single insertion/erase range.
  // Intervals remain sorted and merged under [start, end) semantics.
  auto firstOverlapOrAdjacent =
      std::lower_bound(intervals.begin(), intervals.end(), tMin,
                       [](const pair<int, int>& interval, int value) {
                         return interval.second < value;
                       });

  int newStart = tMin;
  int newEnd = tMax;
  auto mergeEnd = firstOverlapOrAdjacent;
  while (mergeEnd != intervals.end() && !(newEnd < mergeEnd->first)) {
    newStart = min(newStart, mergeEnd->first);
    newEnd = max(newEnd, mergeEnd->second);
    ++mergeEnd;
  }

  if (firstOverlapOrAdjacent == mergeEnd) {
    intervals.insert(firstOverlapOrAdjacent, {newStart, newEnd});
    return;
  }

  firstOverlapOrAdjacent->first = newStart;
  firstOverlapOrAdjacent->second = newEnd;
  if (std::next(firstOverlapOrAdjacent) != mergeEnd) {
    intervals.erase(std::next(firstOverlapOrAdjacent), mergeEnd);
  }
}

void ConstraintTable::incrementSoftConflictCount(uint64_t key, int timestep) {
  if (timestep < 0) {
    return;
  }
  auto& bucket = softConflictTable_[key];
  if ((int)bucket.counts.size() <= timestep) {
    bucket.counts.resize(timestep + 1, 0);
  }
  if (bucket.counts[timestep] < INT_MAX) {
    bucket.counts[timestep]++;
  }
}

int ConstraintTable::lookupSoftConflictCount(uint64_t key, int timestep) const {
  if (timestep < 0) {
    return 0;
  }
  const auto it = softConflictTable_.find(key);
  if (it == softConflictTable_.end()) {
    return 0;
  }
  if (timestep >= (int)it->second.counts.size()) {
    return 0;
  }
  return it->second.counts[timestep];
}

void ConstraintTable::insertSoftGoalStart(int location, int startTime) {
  if (location < 0 || startTime < 0) {
    return;
  }
  auto& starts = softGoalOccupancyStarts_[location];
  auto pos = std::upper_bound(starts.begin(), starts.end(), startTime);
  starts.insert(pos, startTime);
}

int ConstraintTable::getHoldingTime() const {
  int holdingTime = lengthMin;
  if (goalLocation >= 0) {
    auto it = constraintTable_.find((uint64_t)goalLocation);
    if (it != constraintTable_.end()) {
      // No normalization is required here: holding time depends only on the
      // maximum interval end, which is unchanged by sorting/merging.
      for (const auto& timeRange : it->second.intervals) {
        holdingTime = max(holdingTime, timeRange.second);
      }
    }
  }
  return holdingTime;
}

bool ConstraintTable::constrained(size_t location, int timestep) const {
  assert(timestep >= 0);
  const auto it = constraintTable_.find((uint64_t)location);
  if (it == constraintTable_.end()) {
    return false;
  }
  const auto& intervals = it->second.intervals;
  if (intervals.empty()) {
    return false;
  }
  const auto upper =
      std::upper_bound(intervals.begin(), intervals.end(), timestep,
                       [](int t, const pair<int, int>& interval) {
                         return t < interval.first;
                       });
  if (upper == intervals.begin()) {
    return false;
  }
  const auto& candidate = *(upper - 1);
  return candidate.first <= timestep && timestep < candidate.second;
}

bool ConstraintTable::constrained(size_t currentLocation, size_t nextLocation,
                                  int nextTimestep) const {
  return constrained(getEdgeIndex(currentLocation, nextLocation), nextTimestep);
}

void ConstraintTable::insert2CT(size_t location, int tMin, int tMax) {
  assert(tMin >= 0 && tMax > tMin);
  auto& bucket = constraintTable_[(uint64_t)location];
  insertMergedInterval(bucket, tMin, tMax);
  if (tMax < MAX_TIMESTEP && tMax > latestTimestep) {
    latestTimestep = tMax;
  } else if (tMax == MAX_TIMESTEP && tMin > latestTimestep) {
    latestTimestep = tMin;
  }
  temporalExtent = max(temporalExtent, latestTimestep);
}

void ConstraintTable::insert2CT(size_t from, size_t to, int tMin, int tMax) {
  insert2CT(getEdgeIndex(from, to), tMin, tMax);
}

void ConstraintTable::addPath(const Path& path, bool waitAtGoal) {
  if (path.empty()) {
    return;
  }
  int offset = path.beginTime;
  for (int i = 0; i < (int)path.size() - 1; i++) {
    int timestep = i + offset;
    insert2CT(path[i].location, timestep, timestep + 1);
    if (path[i].location != path[i + 1].location) {
      insert2CT(path[i + 1].location, path[i].location, timestep + 1,
                timestep + 2);
    }
  }
  int last = (int)path.size() - 1;
  int timestep = last + offset;
  if (waitAtGoal) {
    insert2CT(path[last].location, timestep, MAX_TIMESTEP);
  } else {
    insert2CT(path[last].location, timestep, timestep + 1);
  }
}

void ConstraintTable::addSoftPath(const Path& path, bool waitAtGoal) {
  if (path.empty()) {
    return;
  }

  const int offset = path.beginTime;
  for (int i = 0; i < (int)path.size(); i++) {
    const int timestep = offset + i;
    incrementSoftConflictCount((uint64_t)path[i].location, timestep);
  }

  for (int i = 0; i < (int)path.size() - 1; i++) {
    const int timestep = offset + i;
    if (path[i].location != path[i + 1].location) {
      incrementSoftConflictCount(
          getEdgeIndex((size_t)path[i + 1].location, (size_t)path[i].location),
          timestep + 1);
    }
  }

  if (waitAtGoal) {
    const int startTime = offset + (int)path.size() - 1;
    insertSoftGoalStart(path.back().location, startTime);
  }
}

int ConstraintTable::getSoftNumOfConflictsForStep(size_t currentLocation,
                                                  size_t nextLocation,
                                                  int nextTimestep) const {
  if (nextTimestep < 0) {
    return 0;
  }

  long long conflicts =
      lookupSoftConflictCount((uint64_t)nextLocation, nextTimestep);
  if (currentLocation != nextLocation) {
    conflicts += lookupSoftConflictCount(getEdgeIndex(currentLocation, nextLocation),
                                         nextTimestep);
  }

  const auto goalIt =
      softGoalOccupancyStarts_.find((int)nextLocation);
  if (goalIt != softGoalOccupancyStarts_.end()) {
    // Target occupancy starts strictly after the path's final timestep.
    const auto activeGoalsEnd =
        std::lower_bound(goalIt->second.begin(), goalIt->second.end(),
                         nextTimestep);
    conflicts += std::distance(goalIt->second.begin(), activeGoalsEnd);
  }

  if (conflicts > INT_MAX) {
    return INT_MAX;
  }
  return (int)conflicts;
}

int ConstraintTable::getEarliestSoftConflictFreeTimestep(size_t currentLocation,
                                                         size_t nextLocation,
                                                         int tMin,
                                                         int tMax) const {
  if (tMin > tMax || tMax < 0) {
    return -1;
  }
  tMin = max(tMin, 0);
  if (tMin > tMax) {
    return -1;
  }

  const vector<int>* vertexCounts = nullptr;
  const auto vertexIt = softConflictTable_.find((uint64_t)nextLocation);
  if (vertexIt != softConflictTable_.end()) {
    vertexCounts = &vertexIt->second.counts;
  }

  const vector<int>* edgeCounts = nullptr;
  if (currentLocation != nextLocation) {
    const auto edgeIt =
        softConflictTable_.find(getEdgeIndex(currentLocation, nextLocation));
    if (edgeIt != softConflictTable_.end()) {
      edgeCounts = &edgeIt->second.counts;
    }
  }

  int maxGoalConflictFreeTime = tMax;
  const auto goalIt = softGoalOccupancyStarts_.find((int)nextLocation);
  if (goalIt != softGoalOccupancyStarts_.end() && !goalIt->second.empty()) {
    const int firstGoalStart = goalIt->second.front();
    if (firstGoalStart < tMin) {
      // A persistent target occupancy already exists at tMin.
      return -1;
    }
    maxGoalConflictFreeTime = min(maxGoalConflictFreeTime, firstGoalStart);
  }
  if (tMin > maxGoalConflictFreeTime) {
    return -1;
  }

  const int vertexHorizon =
      vertexCounts == nullptr ? -1 : (int)vertexCounts->size() - 1;
  const int edgeHorizon =
      edgeCounts == nullptr ? -1 : (int)edgeCounts->size() - 1;
  const int finiteConflictHorizon = max(vertexHorizon, edgeHorizon);

  if (finiteConflictHorizon < tMin) {
    return tMin;
  }

  const int scanEnd = min(maxGoalConflictFreeTime, finiteConflictHorizon);
  for (int t = tMin; t <= scanEnd; t++) {
    const int vertexConflicts =
        (vertexCounts != nullptr && t <= vertexHorizon) ? (*vertexCounts)[t] : 0;
    const int edgeConflicts =
        (edgeCounts != nullptr && t <= edgeHorizon) ? (*edgeCounts)[t] : 0;
    if (vertexConflicts == 0 && edgeConflicts == 0) {
      return t;
    }
  }

  if (scanEnd < maxGoalConflictFreeTime) {
    return scanEnd + 1;
  }
  return -1;
}

int ConstraintTable::getFutureSoftConflicts(size_t location, int timestep) const {
  if (timestep >= lengthMax) {
    return 0;
  }
  const int horizon = lengthMax;
  long long conflicts = 0;

  const auto vertexIt = softConflictTable_.find((uint64_t)location);
  if (vertexIt != softConflictTable_.end()) {
    const int begin = max(0, timestep + 1);
    const int end = min(horizon, (int)vertexIt->second.counts.size() - 1);
    for (int t = begin; t <= end; t++) {
      conflicts += vertexIt->second.counts[t];
    }
  }

  const auto goalIt = softGoalOccupancyStarts_.find((int)location);
  if (goalIt != softGoalOccupancyStarts_.end()) {
    for (int startTime : goalIt->second) {
      // Permanent occupancy contributes strictly after startTime.
      const int begin = max(timestep + 1, startTime + 1);
      if (begin > horizon) {
        continue;
      }
      conflicts += (long long)(horizon - begin + 1);
    }
  }

  if (conflicts > INT_MAX) {
    return INT_MAX;
  }
  return (int)conflicts;
}
