#include "run_reporting.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace {

string formatDouble(double value, int precision = 4) {
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? "-inf" : "inf";
  }
  if (!std::isfinite(value)) {
    return "inf";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

void printMetric(const string& key, const string& value) {
  std::cout << "  " << std::left << std::setw(34) << key << ": " << value
            << '\n';
}

void printMetric(const string& key, double value, int precision = 4) {
  printMetric(key, formatDouble(value, precision));
}

void printMetric(const string& key, int64_t value) {
  printMetric(key, std::to_string(value));
}

void printMetric(const string& key, int value) {
  printMetric(key, std::to_string(value));
}

void printIntListLine(const string& label, const vector<int>& values) {
  std::cout << "  " << std::left << std::setw(34) << label << ": ";
  if (values.empty()) {
    std::cout << "(none)\n";
    return;
  }
  for (size_t i = 0; i < values.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << '\n';
}

void printDoubleListLine(const string& label, const vector<double>& values,
                         int precision = 4) {
  std::cout << "  " << std::left << std::setw(34) << label << ": ";
  if (values.empty()) {
    std::cout << "(none)\n";
    return;
  }
  for (size_t i = 0; i < values.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << formatDouble(values[i], precision);
  }
  std::cout << '\n';
}

double percentileValue(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  percentile = std::max(0.0, std::min(1.0, percentile));
  std::sort(values.begin(), values.end());
  const double raw =
      std::ceil(percentile * static_cast<double>(values.size()));
  const size_t idx =
      (raw < 1.0) ? 0u : static_cast<size_t>(raw - 1.0);
  return values[std::min(idx, values.size() - 1)];
}

vector<double> computeIterationDurations(const vector<IterationStats>& stats) {
  vector<double> durations;
  durations.reserve(stats.size());
  double previousRuntime = 0.0;
  for (const IterationStats& iter : stats) {
    const double delta = std::max(0.0, iter.runtime - previousRuntime);
    durations.push_back(delta);
    previousRuntime = iter.runtime;
  }
  return durations;
}

double computeTimeToBestFeasible(const vector<IterationStats>& stats) {
  int bestSoC = std::numeric_limits<int>::max();
  double timeToBest = std::numeric_limits<double>::infinity();
  for (const IterationStats& iter : stats) {
    if (!iter.feasibleSolutionFound) {
      continue;
    }
    if (iter.sumOfCosts < bestSoC) {
      bestSoC = iter.sumOfCosts;
      timeToBest = iter.runtime;
    }
  }
  return timeToBest;
}

}  // namespace

FeasibleTrajectoryStats collectFeasibleTrajectoryStats(const LNS& lns,
                                                       bool success) {
  FeasibleTrajectoryStats stats;
  if (!success) {
    return stats;
  }

  for (const IterationStats& iter : lns.iterationStats) {
    if (iter.feasibleSolutionFound) {
      stats.firstFeasibleRuntime = iter.runtime;
      break;
    }
  }

  int counter = 0;
  for (const IterationStats& iter : lns.iterationStats) {
    if (iter.feasibleSolutionFound) {
      stats.numFeasibleIterations += 1;
      const bool isImproving =
          iter.quality == IterationQuality::bestSolutionYet ||
          iter.quality == IterationQuality::improvedSolution;
      if (isImproving) {
        stats.numImprovingFeasibleUpdates += 1;
        stats.improvingIterations.push_back(counter);
        stats.improvingRuntimes.push_back(iter.runtime);
        stats.improvingValues.push_back(iter.sumOfCosts);
      }
    }
    if (iter.quality == IterationQuality::couldNotFind) {
      stats.couldNotFindIterations += 1;
    }
    counter += 1;
  }
  return stats;
}

void printAdaptiveLNSPerformance(const LNS& lns,
                                 const std::string& destroyHeuristic) {
  if (destroyHeuristic != "alns") {
    return;
  }

  const ALNS& adaptiveLNS = lns.getAdaptiveLNSRef();
  constexpr int kExpectedDestroyHeuristicCount = 7;
  static_assert((int)DestroyHeuristic::destroyHeuristicCount ==
                    kExpectedDestroyHeuristicCount,
                "DestroyHeuristic enum changed; update heuristicName() in "
                "printAdaptiveLNSPerformance().");
  constexpr int kNumDestroyHeuristics =
      (int)DestroyHeuristic::destroyHeuristicCount;
  vector<int> destroyHeuristicFrequency(kNumDestroyHeuristics, 0);
  std::cout << "Size of destroy heuristic history -> "
            << adaptiveLNS.destroyHeuristicHistory.size() << '\n';
  for (int destroyHeuristicUsed : adaptiveLNS.destroyHeuristicHistory) {
    if (destroyHeuristicUsed >= 0 &&
        destroyHeuristicUsed < kNumDestroyHeuristics) {
      destroyHeuristicFrequency[destroyHeuristicUsed] += 1;
    }
  }
  auto heuristicName = [](DestroyHeuristic h) -> const char* {
    switch (h) {
    case DestroyHeuristic::randomRemoval:
      return "Random";
    case DestroyHeuristic::worstRemoval:
      return "Worst";
    case DestroyHeuristic::conflictRemoval:
      return "Conflict";
    case DestroyHeuristic::shawRemoval:
      return "Shaw";
    case DestroyHeuristic::precedenceWaitRemoval:
      return "PrecedenceWait";
    case DestroyHeuristic::lowSlackRemoval:
      return "LowSlack";
    case DestroyHeuristic::marketTatonnementRemoval:
      return "MarketTatonnement";
    default:
      return "Unknown";
    }
  };

  if (adaptiveLNS.destroyHeuristicHistory.empty()) {
    std::cout << "Adaptive LNS performance: no heuristic samples collected.\n";
    return;
  }

  const std::ios::fmtflags oldFlags = std::cout.flags();
  const std::streamsize oldPrecision = std::cout.precision();

  std::cout << "Adaptive LNS performance:\n";
  std::cout << std::left << std::setw(18) << "Heuristic" << std::right
            << std::setw(9) << "Share%" << std::setw(10) << "Selected"
            << std::setw(10) << "Accept%" << std::setw(12) << "BestUpd%"
            << std::setw(12) << "FailFind%" << std::setw(15)
            << "AvgDelta(all)" << std::setw(15) << "AvgDelta(acc)" << '\n';
  std::cout << std::string(101, '-') << '\n';

  std::cout << std::fixed << std::setprecision(2);
  const double historySize =
      static_cast<double>(adaptiveLNS.destroyHeuristicHistory.size());
  for (int i = 0; i < kNumDestroyHeuristics; i++) {
    const double share = static_cast<double>(destroyHeuristicFrequency[i]) /
                         historySize;
    const int64_t selected = adaptiveLNS.selections[i];
    const int64_t accepted = adaptiveLNS.accepted[i];
    const int64_t bestUpdates = adaptiveLNS.bestUpdates[i];
    const int64_t couldNotFind = adaptiveLNS.couldNotFind[i];
    const double acceptRate =
        selected > 0 ? (double)accepted / (double)selected : 0.0;
    const double bestRate =
        selected > 0 ? (double)bestUpdates / (double)selected : 0.0;
    const double couldNotFindRate =
        selected > 0 ? (double)couldNotFind / (double)selected : 0.0;
    const double avgDeltaSocAll =
        selected > 0 ? adaptiveLNS.deltaSocAll[i] / (double)selected : 0.0;
    const double avgDeltaSocAccepted =
        accepted > 0 ? adaptiveLNS.deltaSocAccepted[i] / (double)accepted
                     : 0.0;
    std::cout << std::left << std::setw(18)
              << heuristicName((DestroyHeuristic)i) << std::right
              << std::setw(9) << (share * 100.0) << std::setw(10) << selected
              << std::setw(10) << (acceptRate * 100.0) << std::setw(12)
              << (bestRate * 100.0) << std::setw(12)
              << (couldNotFindRate * 100.0) << std::setw(15) << avgDeltaSocAll
              << std::setw(15) << avgDeltaSocAccepted << '\n';
  }
  std::cout.flags(oldFlags);
  std::cout.precision(oldPrecision);
}

void printFeasibleTrajectoryReport(const FeasibleTrajectoryStats& stats) {
  std::cout << "\n=== Feasible Trajectory ===\n";
  printMetric("Could-not-find iterations", stats.couldNotFindIterations);
  printMetric("Total feasible iterations", stats.numFeasibleIterations);
  printMetric("Improving feasible updates", stats.numImprovingFeasibleUpdates);
  printIntListLine("Improving iterations", stats.improvingIterations);
  printDoubleListLine("Improving runtimes (s)", stats.improvingRuntimes);
  printIntListLine("Improving solution costs", stats.improvingValues);
}

void printRunSummaryReport(const LNS& lns, const FeasibleSolution& solution,
                           bool success, bool marketHeuristics,
                           bool incrementalRegret,
                           const FeasibleTrajectoryStats& stats) {
  const vector<double> iterDurations = computeIterationDurations(lns.iterationStats);
  const double avgIterSec = iterDurations.empty()
                                ? 0.0
                                : std::accumulate(iterDurations.begin(),
                                                  iterDurations.end(),
                                                  0.0) /
                                      static_cast<double>(iterDurations.size());
  const double p95IterSec = percentileValue(iterDurations, 0.95);
  const double iterPerSec = lns.runtime > 0.0
                                ? static_cast<double>(lns.iterationStats.size()) /
                                      lns.runtime
                                : 0.0;
  const double timeToBestFeasible = computeTimeToBestFeasible(lns.iterationStats);
  const LNS::LowLevelSearchStats lowLevelStats = lns.getLowLevelSearchStats();
  const double lowLevelCallsPerSec =
      lns.runtime > 0.0 ? static_cast<double>(lowLevelStats.calls) / lns.runtime
                        : 0.0;
  const double lowLevelExpandedPerSec =
      lns.runtime > 0.0
          ? static_cast<double>(lowLevelStats.expanded) / lns.runtime
          : 0.0;
  const double lowLevelGeneratedPerSec =
      lns.runtime > 0.0
          ? static_cast<double>(lowLevelStats.generated) / lns.runtime
          : 0.0;
  const double expandedPerCall =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.expanded) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double generatedPerCall =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.generated) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;

  std::cout << "\n=== Run Summary ===\n";
  printMetric("Runtime (s)", lns.runtime);
  printMetric("Iterations", (int)lns.iterationStats.size());
  printMetric("Iterations/sec", iterPerSec);
  printMetric("Avg iteration runtime (ms)", avgIterSec * 1000.0);
  printMetric("P95 iteration runtime (ms)", p95IterSec * 1000.0);
  printMetric("First feasible runtime (s)", stats.firstFeasibleRuntime);
  printMetric("Time-to-best feasible (s)", timeToBestFeasible);
  printMetric("Improving feasible updates", stats.numImprovingFeasibleUpdates);
  printMetric("Total feasible iterations", stats.numFeasibleIterations);
  printMetric("Solution cost", solution.sumOfCosts);
  printMetric("Number of failures", lns.numOfFailures);
  printMetric("Success", success ? "true" : "false");

  std::cout << "\n=== Low-Level Planner Throughput ===\n";
  printMetric("Low-level calls",
              static_cast<int64_t>(lowLevelStats.calls));
  printMetric("Low-level calls/sec", lowLevelCallsPerSec);
  printMetric("Low-level nodes expanded",
              static_cast<int64_t>(lowLevelStats.expanded));
  printMetric("Low-level nodes generated",
              static_cast<int64_t>(lowLevelStats.generated));
  printMetric("Expanded/sec", lowLevelExpandedPerSec);
  printMetric("Generated/sec", lowLevelGeneratedPerSec);
  printMetric("Expanded/call", expandedPerCall);
  printMetric("Generated/call", generatedPerCall);

  if (marketHeuristics) {
    const MarketStats marketStats = lns.getMarketStats();
    std::cout << "\n=== Market Stats ===\n";
    printMetric("Updates", marketStats.updates);
    printMetric("Contended resources", marketStats.contendedResources);
    printMetric("Mean price (contended)", marketStats.meanPriceContended);
    printMetric("Max price", marketStats.maxPrice);
    printMetric("Top price-mass fraction", marketStats.topPriceMassFrac);
    printMetric("Total precedence wait", marketStats.totalPrecedenceWait);
    printMetric("Max precedence wait", marketStats.maxPrecedenceWait);
  }

  const auto& regretStats = lns.getRegretEvalStatsRef();
  const double feasibleRate =
      regretStats.candidateInsertionsTried > 0
          ? (double)regretStats.candidateInsertionsFeasible /
                (double)regretStats.candidateInsertionsTried
          : 0.0;
  const double avgRemovedTasks =
      regretStats.neighborhoods > 0
          ? (double)regretStats.removedTasksSum /
                (double)regretStats.neighborhoods
          : 0.0;
  std::cout << "\n=== Regret Evaluation Stats ===\n";
  printMetric("Recompute calls", regretStats.recomputeCalls);
  printMetric("Tasks evaluated", regretStats.tasksEvaluated);
  printMetric("Task-agent evaluations", regretStats.agentEvaluations);
  printMetric("Candidate insertions tried",
              regretStats.candidateInsertionsTried);
  printMetric("Candidate insertions feasible",
              regretStats.candidateInsertionsFeasible);
  printMetric("Feasible rate", feasibleRate);
  printMetric("Repair neighborhoods", regretStats.neighborhoods);
  printMetric("Avg removed tasks/neighborhood", avgRemovedTasks);
  printMetric("Max removed tasks/neighborhood", regretStats.removedTasksMax);

  if (!incrementalRegret) {
    return;
  }

  const auto statsOpt = lns.getIncrementalRegretStats();
  if (!statsOpt.has_value()) {
    return;
  }
  const auto& incrementalStats = statsOpt.value();
  const double avgDirty =
      incrementalStats.commits > 0
          ? (double)incrementalStats.dirtySum /
                (double)incrementalStats.commits
          : 0.0;
  const double avgChanged =
      incrementalStats.commits > 0
          ? (double)incrementalStats.changedSum /
                (double)incrementalStats.commits
          : 0.0;
  const double avgRecomputedTasksPerCommit =
      incrementalStats.commits > 0
          ? (double)incrementalStats.recomputedTasks /
                (double)incrementalStats.commits
          : 0.0;

  std::cout << "\n=== Incremental Regret Stats ===\n";
  printMetric("Mode", lns.getIncrementalRegretMode());
  printMetric("Commits", incrementalStats.commits);
  printMetric("Recompute calls", incrementalStats.recomputeCalls);
  printMetric("Recomputed tasks", incrementalStats.recomputedTasks);
  printMetric("Recomputed tasks/commit", avgRecomputedTasksPerCommit);
  printMetric("Stale heap pops", incrementalStats.stalePops);
  printMetric("Heap rebuilds", incrementalStats.heapRebuilds);
  printMetric("Full refreshes", incrementalStats.fullRefreshes);
  printMetric("Avg dirty tasks/commit", avgDirty);
  printMetric("Max dirty tasks", incrementalStats.dirtyMax);
  printMetric("Avg changed end-times/commit", avgChanged);
  printMetric("Max changed end-times", incrementalStats.changedMax);
}
