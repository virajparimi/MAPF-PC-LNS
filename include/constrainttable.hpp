#pragma once

#include <plog/Log.h>
#include "common.hpp"

class ConstraintTable {

 protected:
  struct IntervalBucket {
    // Stored as sorted, merged [start, end) intervals.
    vector<pair<int, int>> intervals;
  };
  struct SoftTimelineBucket {
    // Stored as per-timestep collision counts.
    vector<int> counts;
  };

  unordered_map<uint64_t, IntervalBucket>
      constraintTable_;  // (key, value) - (location/edge key, occupied time intervals)
  unordered_map<uint64_t, SoftTimelineBucket> softConflictTable_;
  unordered_map<int, vector<int>> softGoalOccupancyStarts_;

  void normalizeIntervals(IntervalBucket& bucket);
  void insertMergedInterval(IntervalBucket& bucket, int tMin, int tMax);
  void incrementSoftConflictCount(uint64_t key, int timestep);
  int lookupSoftConflictCount(uint64_t key, int timestep) const;
  void insertSoftGoalStart(int location, int startTime);
  inline uint64_t getEdgeIndex(size_t from, size_t to) const {
    // Key-space invariant:
    // vertex keys are [0, mapSize), edge keys are [mapSize, ...].
    // Requires from/to to be valid vertex ids.
    assert(from < mapSize && to < mapSize);
    return (uint64_t)(1 + from) * (uint64_t)mapSize + (uint64_t)to;
  }

 public:
  size_t numCol{}, mapSize{};
  // Maximum finite timestep represented by current constraints.
  int temporalExtent = 0, lengthMin = 0, lengthMax = MAX_TIMESTEP,
      goalLocation = -1;
  // Latest recorded timestep in the occupied interval table. Cannot be the
  // MAX_TIMESTEP
  int latestTimestep = 0;

  ConstraintTable() = default;
  ConstraintTable(size_t numCol, size_t mapSize)
      : numCol(numCol), mapSize(mapSize) {}
  ConstraintTable(const ConstraintTable&) = default;
  ConstraintTable& operator=(const ConstraintTable&) = default;
  ConstraintTable(ConstraintTable&&) noexcept = default;
  ConstraintTable& operator=(ConstraintTable&&) noexcept = default;

  int getHoldingTime() const;
  bool constrained(size_t location, int timestep) const;
  bool constrained(size_t currentLocation, size_t nextLocation,
                   int nextTimestep) const;

  // Safer overloads for callers that use signed vertex IDs.
  inline bool constrained(int location, int timestep) const {
    if (location < 0) {
      return true;
    }
    return constrained((size_t)location, timestep);
  }
  inline bool constrained(int currentLocation, int nextLocation,
                          int nextTimestep) const {
    if (currentLocation < 0 || nextLocation < 0) {
      return true;
    }
    return constrained((size_t)currentLocation, (size_t)nextLocation,
                       nextTimestep);
  }

  void insert2CT(size_t location, int tMin, int tMax);
  void insert2CT(size_t from, size_t to, int tMin, int tMax);

  inline void insert2CT(int location, int tMin, int tMax) {
    if (location < 0) {
      return;
    }
    insert2CT((size_t)location, tMin, tMax);
  }
  inline void insert2CT(int from, int to, int tMin, int tMax) {
    if (from < 0 || to < 0) {
      return;
    }
    insert2CT((size_t)from, (size_t)to, tMin, tMax);
  }

  void addPath(const Path& path, bool waitAtGoal);
  void addSoftPath(const Path& path, bool waitAtGoal);
  int getSoftNumOfConflictsForStep(size_t currentLocation, size_t nextLocation,
                                   int nextTimestep) const;
  int getEarliestSoftConflictFreeTimestep(size_t currentLocation,
                                          size_t nextLocation, int tMin,
                                          int tMax) const;
  int getFutureSoftConflicts(size_t location, int timestep) const;

  inline int getSoftNumOfConflictsForStep(int currentLocation, int nextLocation,
                                          int nextTimestep) const {
    if (currentLocation < 0 || nextLocation < 0 || nextTimestep < 0) {
      return 0;
    }
    return getSoftNumOfConflictsForStep((size_t)currentLocation,
                                        (size_t)nextLocation, nextTimestep);
  }
  inline int getFutureSoftConflicts(int location, int timestep) const {
    if (location < 0 || timestep < 0) {
      return 0;
    }
    return getFutureSoftConflicts((size_t)location, timestep);
  }
  inline int getEarliestSoftConflictFreeTimestep(int currentLocation,
                                                 int nextLocation, int tMin,
                                                 int tMax) const {
    if (currentLocation < 0 || nextLocation < 0 || tMin > tMax) {
      return -1;
    }
    return getEarliestSoftConflictFreeTimestep(
        (size_t)currentLocation, (size_t)nextLocation, tMin, tMax);
  }
  const vector<pair<int, int>>* getConstraintIntervals(size_t key) const {
    const auto it = constraintTable_.find((uint64_t)key);
    if (it == constraintTable_.end()) {
      return nullptr;
    }
    return &it->second.intervals;
  }
  const vector<pair<int, int>>* getConstraintIntervals(int key) const {
    if (key < 0) {
      return nullptr;
    }
    return getConstraintIntervals((size_t)key);
  }
  const vector<pair<int, int>>* getEdgeConstraintIntervals(size_t from,
                                                           size_t to) const {
    return getConstraintIntervals(getEdgeIndex(from, to));
  }
  const vector<pair<int, int>>* getEdgeConstraintIntervals(int from,
                                                           int to) const {
    assert(from >= 0 && to >= 0);
    return getEdgeConstraintIntervals((size_t)from, (size_t)to);
  }
};
