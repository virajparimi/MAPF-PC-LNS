#pragma once
#include <algorithm>
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
#include <tuple>
#include <utility>
#include <vector>

using boost::unordered_map;
using boost::unordered_set;
using boost::heap::compare;
using boost::heap::pairing_heap;
using std::cerr;
using std::clock;
using std::cout;
using std::deque;
using std::distance;
using std::endl;
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
using namespace std::chrono;
using Time = std::chrono::high_resolution_clock;
using fsec = std::chrono::duration<float>;
using pq = std::priority_queue<int, vector<int>>;
using ppq = std::priority_queue<pair<int, int>, vector<pair<int, int>>>;
using ppqg =
    std::priority_queue<pair<int, int>, vector<pair<int, int>>, std::greater<>>;

#define MAX_TIMESTEP INT_MAX / 2
#define MAX_COST INT_MAX / 2
#define MAX_NODES INT_MAX / 2
#define UNASSIGNED -1
#define UNDEFINED -1

struct PathEntry {
  bool isGoal{};
  int location = -1;
};

enum IterationQuality {
  bestSolutionYet = 1,
  improvedSolution = 2,
  dowgradedButAccepted = 3,
  couldNotFind = 4,
  none = 5
};

struct Path {
  int beginTime = 0;
  int endTime() const { return beginTime + (int)size() - 1; }

  vector<PathEntry> path;

  bool empty() const { return path.empty(); }
  size_t size() const { return path.size(); }
  PathEntry& back() { return path.back(); }
  PathEntry& front() { return path.front(); }
  const PathEntry& back() const { return path.back(); }
  const PathEntry& front() const { return path.front(); }
  const PathEntry& at(int idx) const { return path[idx]; }

  PathEntry& operator[](int idx) { return path[idx]; }
  const PathEntry& operator[](int idx) const { return path[idx]; }

  Path() = default;
  Path(int size) : path(vector<PathEntry>(size)) {}
};

struct AgentTaskPath : public Path {
  vector<int> timeStamps;
};

std::ostream& operator<<(std::ostream& os, const Path& path);
bool isSamePath(const Path& p1, const Path& p2);

struct IterationStats {
  double runtime;
  string algorithm;
  bool feasibleSolutionFound;
  int numOfAgents, numOfTasks, sumOfCosts, sumOfCostsLowerBound,
      numOfConflictingPairs;
  IterationQuality quality;
  IterationStats(double runtime, string algorithm, int numOfAgents,
                 int numOfTasks, int sumOfCosts, bool feasibleSolutionFound,
                 IterationQuality quality, int sumOfCostsLowerBound = 0,
                 int numOfConflictingPairs = 0)
      : runtime(runtime),
        algorithm(std::move(algorithm)),
        feasibleSolutionFound(feasibleSolutionFound),
        numOfAgents(numOfAgents),
        numOfTasks(numOfTasks),
        sumOfCosts(sumOfCosts),
        sumOfCostsLowerBound(sumOfCostsLowerBound),
        numOfConflictingPairs(numOfConflictingPairs),
        quality(quality) {}
};
