#include "constrainttable.hpp"

void ConstraintTable::normalizeIntervals(size_t key) const {
  const auto it = constraintTable_.find(key);
  if (it == constraintTable_.end()) {
    return;
  }
  auto& bucket = it->second;
  auto& intervals = bucket.intervals;
  if (bucket.normalized || intervals.size() <= 1) {
    bucket.normalized = true;
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
  bucket.normalized = true;
}

int ConstraintTable::getHoldingTime() const {
  int holdingTime = lengthMin;
  if (goalLocation >= 0) {
    auto it = constraintTable_.find((size_t)goalLocation);
    if (it != constraintTable_.end()) {
      for (const auto& timeRange : it->second.intervals) {
        holdingTime = max(holdingTime, timeRange.second);
      }
    }
  }
  for (const auto& landmark : landmarks_) {
    if ((int)landmark.second != goalLocation) {
      holdingTime = max(holdingTime, (int)landmark.first + 1);
    }
  }
  return holdingTime;
}

bool ConstraintTable::constrained(size_t location, int timestep) const {
  assert(timestep >= 0);
  if (location < mapSize) {
    const auto it = landmarks_.find((size_t)timestep);
    if (it != landmarks_.end() && it->second != location) {
      return true;  // Violate the positive vertex constraint
    }
  }

  const auto it = constraintTable_.find(location);
  if (it == constraintTable_.end()) {
    return false;
  }
  normalizeIntervals(location);
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

void ConstraintTable::insertLandmark(size_t location, int timestep) {
  assert(timestep >= 0);
  const auto it = landmarks_.find((size_t)timestep);
  if (it == landmarks_.end()) {
    landmarks_[(size_t)timestep] = location;
    latestTimestep = max(latestTimestep, timestep);
    size = max(size, latestTimestep);
  } else {
    assert(it->second == location);
  }
}

void ConstraintTable::insert2CT(size_t location, int tMin, int tMax) {
  assert(tMin >= 0 && tMax > tMin);
  auto& bucket = constraintTable_[location];
  bucket.intervals.emplace_back(tMin, tMax);
  bucket.normalized = false;
  if (tMax < MAX_TIMESTEP && tMax > latestTimestep) {
    latestTimestep = tMax;
  } else if (tMax == MAX_TIMESTEP && tMin > latestTimestep) {
    latestTimestep = tMin;
  }
  size = max(size, latestTimestep);
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
    insert2CT(path[i + 1].location, path[i].location, timestep + 1,
              timestep + 2);
  }
  int last = (int)path.size() - 1;
  int timestep = last + offset;
  if (waitAtGoal) {
    insert2CT(path[last].location, timestep, MAX_TIMESTEP);
  } else {
    insert2CT(path[last].location, timestep, timestep + 1);
  }
}
