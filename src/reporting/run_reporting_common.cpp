#include "run_reporting_internal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace run_reporting_internal {

string formatDouble(double value, int precision) {
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? "-inf" : "inf";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

void printMetric(const string& key, const string& value) {
  std::cout << "  " << std::left << std::setw(34) << key << ": " << value
            << '\n';
}

void printMetric(const string& key, double value, int precision) {
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
                         int precision) {
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

int countLnsIterations(const vector<IterationStats>& stats) {
  int count = 0;
  for (const IterationStats& iter : stats) {
    if (iter.algorithm == "LNS") {
      count++;
    }
  }
  return count;
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

FinalSolutionScheduleMetrics computeFinalSolutionScheduleMetrics(
    const FeasibleSolution& solution, const Instance& instance) {
  FinalSolutionScheduleMetrics metrics;
  const int agentCount = static_cast<int>(solution.agentPaths.size());
  for (const auto& path : solution.agentPaths) {
    const int cost = path.endTimeOrZero();
    metrics.makespan = max(metrics.makespan, cost);
    metrics.maxIndividualCost = max(metrics.maxIndividualCost, cost);
  }

  const int taskCount = instance.getTasksNum();
  if (taskCount <= 0) {
    return metrics;
  }
  if ((int)solution.taskAgentMap.size() != taskCount ||
      (int)solution.agentTaskAssignments.size() != agentCount ||
      (int)solution.agentTaskPaths.size() != agentCount) {
    return metrics;
  }

  vector<char> taskValid(taskCount, 0);
  vector<int> taskArrive(taskCount, 0);
  vector<int> taskEnd(taskCount, 0);
  bool anyTaskValid = false;
  for (int agent = 0; agent < agentCount; agent++) {
    const auto& assignments = solution.agentTaskAssignments[agent];
    const auto& taskPaths = solution.agentTaskPaths[agent];
    if (assignments.size() != taskPaths.size()) {
      continue;
    }
    for (int localTask = 0; localTask < (int)assignments.size(); localTask++) {
      const int task = assignments[localTask];
      if (task < 0 || task >= taskCount) {
        continue;
      }
      const AgentTaskPath& taskPath = taskPaths[localTask];
      if (taskPath.empty()) {
        continue;
      }
      int arrive = taskPath.endTime();
      const int taskLocation = instance.getTaskLocations(task);
      for (int step = 0; step < (int)taskPath.size(); step++) {
        if (taskPath[step].location == taskLocation) {
          arrive = taskPath.beginTime + step;
          break;
        }
      }
      taskValid[task] = 1;
      taskArrive[task] = arrive;
      taskEnd[task] = taskPath.endTime();
      anyTaskValid = true;
    }
  }
  if (!anyTaskValid) {
    return metrics;
  }
  metrics.hasTaskScheduleData = true;

  const auto& predecessors = instance.getAncestorsRef();
  for (int task = 0; task < taskCount; task++) {
    if (!taskValid[task]) {
      continue;
    }
    int release = 0;
    for (int pred : predecessors[task]) {
      if (pred >= 0 && pred < taskCount && taskValid[pred]) {
        release = max(release, taskEnd[pred]);
      }
    }
    const int waitPrec = max(0, release - taskArrive[task]);
    metrics.totalPrecedenceWait += waitPrec;
    metrics.maxPrecedenceWait = max(metrics.maxPrecedenceWait, waitPrec);
  }

  const auto& successors = instance.getSuccessorsRef();
  for (int task = 0; task < taskCount; task++) {
    if (!taskValid[task]) {
      continue;
    }
    for (int succ : successors[task]) {
      if (succ < 0 || succ >= taskCount || !taskValid[succ]) {
        continue;
      }
      metrics.precedenceSlacks.push_back(
          static_cast<double>(taskArrive[succ] - taskEnd[task]));
    }
  }

  return metrics;
}

}  // namespace run_reporting_internal

using run_reporting_internal::printDoubleListLine;
using run_reporting_internal::printIntListLine;
using run_reporting_internal::printMetric;

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
  int bestFeasibleSoC = std::numeric_limits<int>::max();
  for (const IterationStats& iter : lns.iterationStats) {
    if (iter.feasibleSolutionFound) {
      stats.numFeasibleIterations += 1;
      const bool isImproving = iter.sumOfCosts < bestFeasibleSoC;
      if (isImproving) {
        bestFeasibleSoC = iter.sumOfCosts;
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
            << std::setw(11) << "PropBetter" << std::setw(10) << "PropEq"
            << std::setw(11) << "PropWorse" << std::setw(11) << "AccWorse"
            << std::setw(10) << "Accept%" << std::setw(12) << "BestUpd%"
            << std::setw(12) << "FailFind%" << std::setw(12) << "Cascade%"
            << std::setw(15) << "AvgDelta(all)" << std::setw(15)
            << "AvgDelta(acc)" << '\n';
  std::cout << std::string(156, '-') << '\n';

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
    const int64_t cascadeAborted = adaptiveLNS.cascadeAborted[i];
    const int64_t proposedBetter = adaptiveLNS.proposedBetter[i];
    const int64_t proposedEqual = adaptiveLNS.proposedEqual[i];
    const int64_t proposedWorse = adaptiveLNS.proposedWorse[i];
    const int64_t acceptedWorse = adaptiveLNS.acceptedWorse[i];
    const double acceptRate =
        selected > 0 ? (double)accepted / (double)selected : 0.0;
    const double bestRate =
        selected > 0 ? (double)bestUpdates / (double)selected : 0.0;
    const double couldNotFindRate =
        selected > 0 ? (double)couldNotFind / (double)selected : 0.0;
    const double cascadeAbortRate =
        selected > 0 ? (double)cascadeAborted / (double)selected : 0.0;
    const double avgDeltaSocAll =
        selected > 0 ? adaptiveLNS.deltaSocAll[i] / (double)selected : 0.0;
    const double avgDeltaSocAccepted =
        accepted > 0 ? adaptiveLNS.deltaSocAccepted[i] / (double)accepted
                     : 0.0;
    std::cout << std::left << std::setw(18)
              << heuristicName((DestroyHeuristic)i) << std::right
              << std::setw(9) << (share * 100.0) << std::setw(10) << selected
              << std::setw(11) << proposedBetter << std::setw(10)
              << proposedEqual << std::setw(11) << proposedWorse
              << std::setw(11) << acceptedWorse
              << std::setw(10) << (acceptRate * 100.0) << std::setw(12)
              << (bestRate * 100.0) << std::setw(12)
              << (couldNotFindRate * 100.0) << std::setw(12)
              << (cascadeAbortRate * 100.0) << std::setw(15) << avgDeltaSocAll
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
