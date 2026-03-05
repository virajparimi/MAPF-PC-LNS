#pragma once

#include <limits>
#include <string>
#include <vector>
#include "lns.hpp"

struct FeasibleTrajectoryStats {
  // Trajectory of strictly improving feasible iterations.
  std::vector<int> improvingIterations;
  std::vector<double> improvingRuntimes;
  std::vector<int> improvingValues;
  std::vector<int> improvingMakespans;
  std::vector<double> improvingPrecedenceWaits;
  double firstFeasibleRuntime = std::numeric_limits<double>::infinity();
  int numFeasibleIterations = 0;
  int numImprovingFeasibleUpdates = 0;
  int couldNotFindIterations = 0;
};

FeasibleTrajectoryStats collectFeasibleTrajectoryStats(const LNS& lns,
                                                       bool success);
void printAdaptiveLNSPerformance(const LNS& lns,
                                 const std::string& destroyHeuristic);
void printFeasibleTrajectoryReport(const FeasibleTrajectoryStats& stats);
void printRunSummaryReport(const LNS& lns, const FeasibleSolution& solution,
                           bool success, bool marketHeuristics,
                           bool incrementalRegret,
                           const FeasibleTrajectoryStats& stats);
