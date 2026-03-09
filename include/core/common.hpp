#pragma once
#include <algorithm>
// Work around a GCC 13 ICE triggered by boost::heap concept-check
// instantiation when d_ary_heap<mutable_<true>> is used.
#ifndef BOOST_NO_CONCEPT_CHECKS
#define BOOST_NO_CONCEPT_CHECKS
#endif
#include <boost/heap/d_ary_heap.hpp>
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <cassert>
#include <climits>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using boost::heap::compare;
using boost::heap::pairing_heap;
using std::deque;
using std::distance;
using std::get;
using std::hash;
using std::list;
using std::make_pair;
using std::make_shared;
using std::make_tuple;
using std::make_unique;
using std::map;
using std::max;
using std::min;
using std::ofstream;
using std::pair;
using std::reverse;
using std::set;
using std::shared_ptr;
using std::stack;
using std::string;
using std::tie;
using std::tuple;
using std::unique_ptr;
using std::vector;
using Time = std::chrono::high_resolution_clock;
using fsec = std::chrono::duration<float>;
using ppq = std::priority_queue<pair<int, int>, vector<pair<int, int>>>;
using ppqg =
    std::priority_queue<pair<int, int>, vector<pair<int, int>>, std::greater<>>;

namespace mapf_pc_lns {
inline constexpr int kMaxTimestep = (INT_MAX / 2);
inline constexpr int kMaxCost = (INT_MAX / 2);
inline constexpr int kMaxNodes = (INT_MAX / 2);
inline constexpr int kUnassigned = -1;
inline constexpr int kUndefined = -2;

inline std::pair<int, int> toCoordinate(int id, int numCols) {
  assert(numCols > 0);
  return std::make_pair(id / numCols, id % numCols);
}
}  // namespace mapf_pc_lns

// Transitional aliases to avoid a broad edit in one change. These are typed
// constants (not macros) and can be removed after call sites migrate to the
// scoped names.
inline constexpr int MAX_TIMESTEP = mapf_pc_lns::kMaxTimestep;
inline constexpr int MAX_COST = mapf_pc_lns::kMaxCost;
inline constexpr int MAX_NODES = mapf_pc_lns::kMaxNodes;
inline constexpr int UNASSIGNED = mapf_pc_lns::kUnassigned;
inline constexpr int UNDEFINED = mapf_pc_lns::kUndefined;

struct PathEntry {
  bool isGoal{};
  int location = -1;
};

enum class IterationQuality {
  bestSolutionYet = 1,
  improvedSolution = 2,
  downgradedButAccepted = 3,
  couldNotFind = 4,
  none = 5
};

struct Path {
  int beginTime = 0;
  int endTimeOrZero() const { return empty() ? 0 : beginTime + (int)size() - 1; }
  int endTimeChecked() const {
    assert(!empty());
    return beginTime + (int)size() - 1;
  }
  int endTime() const { return endTimeOrZero(); }

  vector<PathEntry> path;

  bool empty() const { return path.empty(); }
  size_t size() const { return path.size(); }
  PathEntry& back() { return path.back(); }
  PathEntry& front() { return path.front(); }
  const PathEntry& back() const { return path.back(); }
  const PathEntry& front() const { return path.front(); }
  const PathEntry& at(int idx) const {
    if (idx < 0 || idx >= (int)path.size()) {
      throw std::out_of_range("Path::at index out of range");
    }
    return path[idx];
  }

  PathEntry& operator[](int idx) {
    assert(idx >= 0 && idx < (int)path.size());
    return path[idx];
  }
  const PathEntry& operator[](int idx) const {
    assert(idx >= 0 && idx < (int)path.size());
    return path[idx];
  }

  Path() = default;
  explicit Path(int size) {
    if (size < 0) {
      throw std::invalid_argument("Path size must be non-negative");
    }
    path = vector<PathEntry>(static_cast<size_t>(size));
  }
  ~Path() = default;
};

struct AgentTaskPath : public Path {
  vector<int> timeStamps;
};

std::ostream& operator<<(std::ostream& os, const Path& path);

struct IterationStats {
  double runtime;
  string algorithm;
  bool feasibleSolutionFound;
  int numOfAgents, numOfTasks, sumOfCosts, sumOfCostsLowerBound,
      numOfConflictingPairs;
  int makespan;
  double precedenceWait;
  IterationQuality quality;
  IterationStats(double runtimeIn, string algorithmIn, int numOfAgentsIn,
                 int numOfTasksIn, int sumOfCostsIn,
                 bool feasibleSolutionFoundIn, IterationQuality qualityIn,
                 int sumOfCostsLowerBoundIn = 0,
                 int numOfConflictingPairsIn = 0, int makespanIn = -1,
                 double precedenceWaitIn =
                     std::numeric_limits<double>::quiet_NaN())
      : runtime(runtimeIn),
        algorithm(std::move(algorithmIn)),
        feasibleSolutionFound(feasibleSolutionFoundIn),
        numOfAgents(numOfAgentsIn),
        numOfTasks(numOfTasksIn),
        sumOfCosts(sumOfCostsIn),
        sumOfCostsLowerBound(sumOfCostsLowerBoundIn),
        numOfConflictingPairs(numOfConflictingPairsIn),
        makespan(makespanIn),
        precedenceWait(precedenceWaitIn),
        quality(qualityIn) {}
};
