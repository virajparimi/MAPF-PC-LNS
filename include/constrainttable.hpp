#pragma once

#include <plog/Log.h>
#include "common.hpp"

class ConstraintTable {

 protected:
  struct IntervalBucket {
    // Stored as sorted, merged [start, end) intervals after normalization.
    mutable vector<pair<int, int>> intervals;
    mutable bool normalized = true;
  };

  unordered_map<size_t, IntervalBucket>
      constraintTable_;  // (key, value) - (location/edge key, occupied time intervals)

  void normalizeIntervals(size_t key) const;
  inline size_t getEdgeIndex(size_t from, size_t to) const {
    return (1 + from) * mapSize + to;
  }

 public:
  size_t numCol{}, mapSize{};
  int size = 0, lengthMin = 0, lengthMax = MAX_TIMESTEP, goalLocation = -1;
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
    assert(location >= 0);
    return constrained((size_t)location, timestep);
  }
  inline bool constrained(int currentLocation, int nextLocation,
                          int nextTimestep) const {
    assert(currentLocation >= 0 && nextLocation >= 0);
    return constrained((size_t)currentLocation, (size_t)nextLocation,
                       nextTimestep);
  }

  void insert2CT(size_t location, int tMin, int tMax);
  void insert2CT(size_t from, size_t to, int tMin, int tMax);

  inline void insert2CT(int location, int tMin, int tMax) {
    assert(location >= 0);
    insert2CT((size_t)location, tMin, tMax);
  }
  inline void insert2CT(int from, int to, int tMin, int tMax) {
    assert(from >= 0 && to >= 0);
    insert2CT((size_t)from, (size_t)to, tMin, tMax);
  }

  void addPath(const Path& path, bool waitAtGoal);
  const vector<pair<int, int>>* getConstraintIntervals(size_t key) const {
    const auto it = constraintTable_.find(key);
    if (it == constraintTable_.end()) {
      return nullptr;
    }
    normalizeIntervals(key);
    return &it->second.intervals;
  }
  const vector<pair<int, int>>* getConstraintIntervals(int key) const {
    assert(key >= 0);
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
