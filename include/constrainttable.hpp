#pragma once

#include <plog/Log.h>
#include "common.hpp"

class ConstraintTable {

 protected:
  unordered_map<size_t, size_t> landmarks_;  // (key, value) - (time, location)
  unordered_map<size_t, list<pair<int, int>>>
      constraintTable_;  // (key, value) - (location, occupied time intervals)

  void insertLandmark(size_t location, int timestep);
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
  ConstraintTable(const ConstraintTable& old) { copy(old); }

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

  void copy(const ConstraintTable& old);
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
  unordered_map<size_t, size_t> getLandmarks() const { return landmarks_; }
};
