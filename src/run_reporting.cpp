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
  const double llFoundRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.found) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llTimeoutRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.timeout) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llExhaustedRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.searchExhausted) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llBudgetExhaustedRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.budgetExhausted) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const LNS::AcceptanceDiagnostics acceptanceDiag =
      lns.getAcceptanceDiagnostics();
  auto average = [](double sum, int64_t count) {
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
  };

  std::cout << "\n=== Run Summary ===\n";
  printMetric("Requested initial solution",
              lns.getInitialSolutionRequested());
  printMetric("Effective initial solution", lns.getInitialSolutionEffective());
  printMetric("Initial fallback used",
              lns.wasInitialSolutionFallbackUsed() ? "true" : "false");
  printMetric("Initial fallback reason",
              lns.getInitialSolutionFallbackReason());
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
  printMetric("Reject invalid candidates",
              lns.rejectInvalidCandidatesEnabled() ? "true" : "false");
  printMetric("Utility uses conflict events",
              lns.utilityUsesConflictEventCount() ? "true" : "false");
  printMetric("Acceptance feasibility-first debt",
              lns.acceptanceUsesFeasibilityFirstPrecedenceDebt() ? "true"
                                                                 : "false");
  printMetric("Acceptance dedicated invalid temperature",
              lns.acceptanceUsesDedicatedInvalidTemperature() ? "true"
                                                              : "false");
  printMetric("MLA* incremental focal refresh",
              lns.mlastarIncrementalFocalRefreshEnabled() ? "true" : "false");
  printMetric("Invalid candidate rejections",
              lns.invalidCandidateRejections);
  printMetric("Market guard rejections", lns.marketGuardRejections);
  printMetric("FF decisions", acceptanceDiag.feasibilityFirstDecisions);
  printMetric("FF invalid->valid accepted",
              acceptanceDiag.invalidToValidAccepted);
  printMetric("FF valid->invalid compared",
              acceptanceDiag.validToInvalidCompared);
  printMetric("FF valid->invalid accepted",
              acceptanceDiag.validToInvalidAccepted);
  printMetric("FF valid->invalid rejected",
              acceptanceDiag.validToInvalidRejected);
  printMetric("FF invalid-vs-invalid compared",
              acceptanceDiag.invalidVsInvalidComparisons);
  printMetric("FF invalid-vs-invalid accepted",
              acceptanceDiag.invalidVsInvalidAccepted);
  printMetric("FF invalid-vs-invalid rejected",
              acceptanceDiag.invalidVsInvalidRejected);
  printMetric("FF avg previous invalid score",
              average(acceptanceDiag.previousInvalidScoreSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg candidate invalid score",
              average(acceptanceDiag.candidateInvalidScoreSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg previous spatial(norm)",
              average(acceptanceDiag.previousSpatialNormSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg candidate spatial(norm)",
              average(acceptanceDiag.candidateSpatialNormSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg previous precedenceDebt(norm)",
              average(acceptanceDiag.previousPrecedenceDebtNormSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg candidate precedenceDebt(norm)",
              average(acceptanceDiag.candidatePrecedenceDebtNormSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg previous soc(norm)",
              average(acceptanceDiag.previousSocNormSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF avg candidate soc(norm)",
              average(acceptanceDiag.candidateSocNormSum,
                      acceptanceDiag.invalidVsInvalidComparisons));
  printMetric("FF invalid-score compared",
              acceptanceDiag.invalidScoreComparisons);
  printMetric("FF invalid-score accepted",
              acceptanceDiag.invalidScoreAccepted);
  printMetric("FF invalid-score rejected",
              acceptanceDiag.invalidScoreRejected);
  printMetric("FF invalid-score worse compared",
              acceptanceDiag.invalidScoreWorseComparisons);
  printMetric("FF invalid-score worse accepted",
              acceptanceDiag.invalidScoreWorseAccepted);
  printMetric("FF avg invalid-score delta",
              average(acceptanceDiag.invalidScoreDeltaSum,
                      acceptanceDiag.invalidScoreComparisons));
  printMetric("FF avg invalid-score abs delta",
              average(acceptanceDiag.invalidScoreAbsDeltaSum,
                      acceptanceDiag.invalidScoreComparisons));
  printMetric("FF avg invalid-temp before",
              average(acceptanceDiag.invalidAcceptanceTempBeforeSum,
                      acceptanceDiag.invalidScoreComparisons));
  printMetric("FF avg invalid-temp after",
              average(acceptanceDiag.invalidAcceptanceTempAfterSum,
                      acceptanceDiag.invalidScoreComparisons));
  printMetric("FF dedicated invalid-temp init count",
              acceptanceDiag.invalidDedicatedTempInitCount);
  printMetric("FF avg dedicated invalid-temp init",
              average(acceptanceDiag.invalidDedicatedInitTempSum,
                      acceptanceDiag.invalidDedicatedTempInitCount));
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
  printMetric("LL outcome found", static_cast<int64_t>(lowLevelStats.found));
  printMetric("LL outcome timeout", static_cast<int64_t>(lowLevelStats.timeout));
  printMetric("LL outcome exhausted",
              static_cast<int64_t>(lowLevelStats.searchExhausted));
  printMetric("LL outcome invalid-input",
              static_cast<int64_t>(lowLevelStats.invalidInput));
  printMetric("LL outcome budget-exhausted",
              static_cast<int64_t>(lowLevelStats.budgetExhausted));
  printMetric("LL outcome unknown", static_cast<int64_t>(lowLevelStats.unknown));
  printMetric("LL found rate", llFoundRate);
  printMetric("LL timeout rate", llTimeoutRate);
  printMetric("LL exhausted rate", llExhaustedRate);
  printMetric("LL budget-exhausted rate", llBudgetExhaustedRate);

  const auto& cascadeStats = lns.getCascadeStatsRef();
  const int totalTasks = lns.getInstance().getTasksNum();
  const double cascadeAbortRate =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.budgetAborts /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgSeedTasks =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.seedTasksSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgClosureTasks =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.closureTasksSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgClosureAdded =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.closureAddedSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgCascadeBudgetUsed =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.budgetUsedSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgClosureFracOfTasks =
      (cascadeStats.prepareCalls > 0 && totalTasks > 0)
          ? (double)cascadeStats.closureAddedSum /
                ((double)cascadeStats.prepareCalls * (double)totalTasks)
          : 0.0;

  std::cout << "\n=== Cascade Stats ===\n";
  printMetric("Cascade adaptive budget enabled",
              lns.isAdaptiveCascadeBudgetEnabled() ? "true" : "false");
  printMetric("Cascade budget baseline (added tasks)", lns.getCascadeTaskBudget());
  printMetric("Cascade budget current (added tasks)",
              lns.getAdaptiveCascadeBudgetCurrent());
  printMetric("Cascade budget avg used", avgCascadeBudgetUsed);
  printMetric("Cascade budget min used",
              cascadeStats.prepareCalls > 0 ? cascadeStats.budgetUsedMin : 0);
  printMetric("Cascade budget max used",
              cascadeStats.prepareCalls > 0 ? cascadeStats.budgetUsedMax : 0);
  printMetric("Cascade budget increases",
              cascadeStats.adaptiveBudgetIncreases);
  printMetric("Cascade budget decreases",
              cascadeStats.adaptiveBudgetDecreases);
  printMetric("prepareNextIteration calls", cascadeStats.prepareCalls);
  printMetric("Cascade budget aborts", cascadeStats.budgetAborts);
  printMetric("Cascade abort rate", cascadeAbortRate);
  printMetric("Avg seed removed tasks", avgSeedTasks);
  printMetric("Avg closure removed tasks", avgClosureTasks);
  printMetric("Avg closure added tasks", avgClosureAdded);
  printMetric("Max closure removed tasks", cascadeStats.closureTasksMax);
  printMetric("Max closure added tasks", cascadeStats.closureAddedMax);
  printMetric("Avg closure_added/total_tasks", avgClosureFracOfTasks);

  const auto& restoreStats = lns.getSolutionRestoreStats();
  const double partialRestoreRate =
      restoreStats.restoreCalls > 0
          ? (double)restoreStats.partialRestores /
                (double)restoreStats.restoreCalls
          : 0.0;
  const double avgAgentsPerPartialRestore =
      restoreStats.partialRestores > 0
          ? (double)restoreStats.partialAgentsRestored /
                (double)restoreStats.partialRestores
          : 0.0;
  std::cout << "\n=== Solution Restore Stats ===\n";
  printMetric("Partial restore enabled",
              lns.isPartialSolutionRestoreEnabled() ? "true" : "false");
  printMetric("Restore calls", restoreStats.restoreCalls);
  printMetric("Full restores", restoreStats.fullRestores);
  printMetric("Partial restores", restoreStats.partialRestores);
  printMetric("Partial restore fallbacks",
              restoreStats.partialRestoreFallbacks);
  printMetric("Partial restore rate", partialRestoreRate);
  printMetric("Avg agents/partial restore", avgAgentsPerPartialRestore);

  const auto& terminalStats = lns.getTerminalRepositionStats();
  if (terminalStats.replansRequested > 0) {
    const double plannedRate =
        terminalStats.agentsEvaluated > 0
            ? (double)terminalStats.agentsPlanned /
                  (double)terminalStats.agentsEvaluated
            : 0.0;
    const double noDemandRate =
        terminalStats.agentsEvaluated > 0
            ? (double)terminalStats.skippedNoDemand /
                  (double)terminalStats.agentsEvaluated
            : 0.0;
    std::cout << "\n=== Terminal Reposition Stats ===\n";
    printMetric("Replan calls", terminalStats.replansRequested);
    printMetric("Agents evaluated", terminalStats.agentsEvaluated);
    printMetric("Agents planned", terminalStats.agentsPlanned);
    printMetric("Skipped (no demand)", terminalStats.skippedNoDemand);
    printMetric("Planning failures", terminalStats.planningFailures);
    printMetric("Candidate cache hits", terminalStats.candidateCacheHits);
    printMetric("Candidate cache misses", terminalStats.candidateCacheMisses);
    printMetric("Planned/evaluated", plannedRate);
    printMetric("No-demand/evaluated", noDemandRate);
  }

  if (marketHeuristics) {
    const MarketStats marketStats = lns.getMarketStats();
    std::cout << "\n=== Market Stats ===\n";
    printMetric("Updates", marketStats.updates);
    printMetric("Destroy warmup skipped", marketStats.destroyWarmupSkipped);
    printMetric("Destroy unstable skipped", marketStats.destroyUnstableSkipped);
    printMetric("Contended resources", marketStats.contendedResources);
    printMetric("Mean price (contended)", marketStats.meanPriceContended);
    printMetric("Max price", marketStats.maxPrice);
    printMetric("Top price-mass fraction", marketStats.topPriceMassFrac);
    printMetric("Price rel-L1 delta", marketStats.priceRelL1Delta);
    printMetric("Price rel-L1 delta EMA", marketStats.priceRelL1DeltaEma);
    printMetric("Top price-mass delta", marketStats.topPriceMassDelta);
    printMetric("Top price-mass delta EMA", marketStats.topPriceMassDeltaEma);
    printMetric("Contended Jaccard", marketStats.contendedJaccard);
    printMetric("Contended Jaccard EMA", marketStats.contendedJaccardEma);
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
  printMetric("Shortlist agent evals", regretStats.shortlistAgentEvaluations);
  printMetric("Shortlist fallback evals",
              regretStats.shortlistFallbackEvaluations);
  printMetric("Shortlist fallback recovered",
              regretStats.shortlistFallbackRecovered);
  printMetric("Adaptive regret Top-K enabled",
              lns.isAdaptiveRegretTopKEnabled() ? "true" : "false");
  printMetric("Adaptive regret Top-K current", lns.getAdaptiveRegretTopKCurrent());
  if (regretStats.adaptiveTopKEvaluations > 0) {
    const double adaptiveTopKAvg =
        (double)regretStats.adaptiveTopKUsedSum /
        (double)regretStats.adaptiveTopKEvaluations;
    printMetric("Adaptive regret Top-K evals",
                regretStats.adaptiveTopKEvaluations);
    printMetric("Adaptive regret Top-K avg used", adaptiveTopKAvg);
    printMetric("Adaptive regret Top-K min used",
                regretStats.adaptiveTopKUsedMin);
    printMetric("Adaptive regret Top-K max used",
                regretStats.adaptiveTopKUsedMax);
    printMetric("Adaptive regret Top-K increases",
                regretStats.adaptiveTopKIncreases);
    printMetric("Adaptive regret Top-K decreases",
                regretStats.adaptiveTopKDecreases);
    printMetric("Adaptive regret Top-K no-feasible signals",
                regretStats.adaptiveTopKNoFeasibleSignals);
    printMetric("Adaptive regret Top-K fallback-recovered signals",
                regretStats.adaptiveTopKFallbackRecoverySignals);
  }
  printMetric("Workspace agents cloned", regretStats.workspaceAgentsCloned);
  printMetric("Max cloned agents/task",
              regretStats.workspaceMaxClonedPerTask);
  printMetric("Feasible rate", feasibleRate);
  printMetric("Repair neighborhoods", regretStats.neighborhoods);
  printMetric("Avg removed tasks/neighborhood", avgRemovedTasks);
  printMetric("Max removed tasks/neighborhood", regretStats.removedTasksMax);
  if (regretStats.waitProxyDiagEvaluations > 0) {
    const double waitPositiveEvalRate =
        (double)regretStats.waitProxyDiagPositiveEvals /
        (double)regretStats.waitProxyDiagEvaluations;
    const double waitVaryingEvalRate =
        (double)regretStats.waitProxyDiagVaryingEvals /
        (double)regretStats.waitProxyDiagEvaluations;
    const double normalizedActiveEvalRate =
        (double)regretStats.waitProxyDiagNormalizedActiveEvals /
        (double)regretStats.waitProxyDiagEvaluations;
    const double waitNonZeroCandidateRate =
        regretStats.waitProxyDiagFiniteCandidates > 0
            ? (double)regretStats.waitProxyDiagNonZeroCandidates /
                  (double)regretStats.waitProxyDiagFiniteCandidates
            : 0.0;
    const double top1ChangedRate =
        (double)regretStats.waitProxyDiagTop1Changed /
        (double)regretStats.waitProxyDiagEvaluations;
    const double topKChangedRate =
        (double)regretStats.waitProxyDiagTopKChanged /
        (double)regretStats.waitProxyDiagEvaluations;
    const double avgTopKOverlap =
        regretStats.waitProxyDiagTopKOverlapFracSum /
        (double)regretStats.waitProxyDiagEvaluations;
    const double avgAbsWaitZ =
        regretStats.waitProxyDiagMeanAbsWaitZSum /
        (double)regretStats.waitProxyDiagEvaluations;
    const double avgAbsDistanceZ =
        regretStats.waitProxyDiagMeanAbsDistanceZSum /
        (double)regretStats.waitProxyDiagEvaluations;
    const double waitToDistanceZRatio =
        avgAbsDistanceZ > 1e-12 ? avgAbsWaitZ / avgAbsDistanceZ : 0.0;
    printMetric("Wait-proxy diag evals", regretStats.waitProxyDiagEvaluations);
    printMetric("Wait-proxy finite candidates",
                regretStats.waitProxyDiagFiniteCandidates);
    printMetric("Wait-proxy nonzero candidates",
                regretStats.waitProxyDiagNonZeroCandidates);
    printMetric("Wait-proxy positive eval rate", waitPositiveEvalRate);
    printMetric("Wait-proxy varying eval rate", waitVaryingEvalRate);
    printMetric("Wait-proxy normalized-active eval rate",
                normalizedActiveEvalRate);
    printMetric("Wait-proxy nonzero candidate rate",
                waitNonZeroCandidateRate);
    printMetric("Wait-proxy top1 changed rate", top1ChangedRate);
    printMetric("Wait-proxy topK changed rate", topKChangedRate);
    printMetric("Wait-proxy avg topK overlap", avgTopKOverlap);
    printMetric("Wait-proxy avg |z_wait|", avgAbsWaitZ);
    printMetric("Wait-proxy avg |z_dist|", avgAbsDistanceZ);
    printMetric("Wait-proxy |z_wait|/|z_dist|", waitToDistanceZRatio);
  }
  if (regretStats.successorPressureDiagEvaluations > 0) {
    const double successorPositiveEvalRate =
        (double)regretStats.successorPressureDiagPositiveEvals /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorVaryingEvalRate =
        (double)regretStats.successorPressureDiagVaryingEvals /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorNormalizedActiveEvalRate =
        (double)regretStats.successorPressureDiagNormalizedActiveEvals /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorNonZeroCandidateRate =
        regretStats.successorPressureDiagFiniteCandidates > 0
            ? (double)regretStats.successorPressureDiagNonZeroCandidates /
                  (double)regretStats.successorPressureDiagFiniteCandidates
            : 0.0;
    const double successorTop1ChangedRate =
        (double)regretStats.successorPressureDiagTop1Changed /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorTopKChangedRate =
        (double)regretStats.successorPressureDiagTopKChanged /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorAvgTopKOverlap =
        regretStats.successorPressureDiagTopKOverlapFracSum /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorAvgAbsPressureZ =
        regretStats.successorPressureDiagMeanAbsPressureZSum /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorAvgAbsDistanceZ =
        regretStats.successorPressureDiagMeanAbsDistanceZSum /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorToDistanceZRatio =
        successorAvgAbsDistanceZ > 1e-12
            ? successorAvgAbsPressureZ / successorAvgAbsDistanceZ
            : 0.0;
    const int64_t successorSignalCount =
        regretStats.successorPressureDiagDepth1Signals +
        regretStats.successorPressureDiagDepthGt1Signals;
    const double successorPrevFallbackPerEval =
        (double)regretStats.successorPressureDiagPrevFallbackCount /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorPrevFallbackPerSignal =
        successorSignalCount > 0
            ? (double)regretStats.successorPressureDiagPrevFallbackCount /
                  (double)successorSignalCount
            : 0.0;
    const double successorPrecedenceClampPerEval =
        (double)regretStats.successorPressureDiagPrecedenceClampCount /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorPrecedenceClampPerSignal =
        successorSignalCount > 0
            ? (double)regretStats.successorPressureDiagPrecedenceClampCount /
                  (double)successorSignalCount
            : 0.0;
    const double successorPrecedenceClampAvgDelta =
        regretStats.successorPressureDiagPrecedenceClampCount > 0
            ? regretStats.successorPressureDiagPrecedenceClampDeltaSum /
                  (double)regretStats.successorPressureDiagPrecedenceClampCount
            : 0.0;
    const double successorDepth1SignalsPerEval =
        (double)regretStats.successorPressureDiagDepth1Signals /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorDepthGt1SignalsPerEval =
        (double)regretStats.successorPressureDiagDepthGt1Signals /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorDescendantActiveEvalRate =
        (double)regretStats.successorPressureDiagDescendantActiveEvals /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorAvgDepth1Contribution =
        regretStats.successorPressureDiagMeanDepth1ContributionSum /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorAvgDepthGt1Contribution =
        regretStats.successorPressureDiagMeanDepthGt1ContributionSum /
        (double)regretStats.successorPressureDiagEvaluations;
    const double successorDescendantContributionShare =
        (successorAvgDepth1Contribution + successorAvgDepthGt1Contribution) > 1e-12
            ? successorAvgDepthGt1Contribution /
                  (successorAvgDepth1Contribution +
                   successorAvgDepthGt1Contribution)
            : 0.0;
    printMetric("Successor-pressure diag evals",
                regretStats.successorPressureDiagEvaluations);
    printMetric("Successor-pressure finite candidates",
                regretStats.successorPressureDiagFiniteCandidates);
    printMetric("Successor-pressure nonzero candidates",
                regretStats.successorPressureDiagNonZeroCandidates);
    printMetric("Successor-pressure positive eval rate",
                successorPositiveEvalRate);
    printMetric("Successor-pressure varying eval rate",
                successorVaryingEvalRate);
    printMetric("Successor-pressure normalized-active eval rate",
                successorNormalizedActiveEvalRate);
    printMetric("Successor-pressure nonzero candidate rate",
                successorNonZeroCandidateRate);
    printMetric("Successor-pressure top1 changed rate",
                successorTop1ChangedRate);
    printMetric("Successor-pressure topK changed rate",
                successorTopKChangedRate);
    printMetric("Successor-pressure avg topK overlap",
                successorAvgTopKOverlap);
    printMetric("Successor-pressure avg |z_succ|",
                successorAvgAbsPressureZ);
    printMetric("Successor-pressure avg |z_dist|",
                successorAvgAbsDistanceZ);
    printMetric("Successor-pressure |z_succ|/|z_dist|",
                successorToDistanceZRatio);
    printMetric("Successor-pressure prev fallback count",
                regretStats.successorPressureDiagPrevFallbackCount);
    printMetric("Successor-pressure prev fallback per eval",
                successorPrevFallbackPerEval);
    printMetric("Successor-pressure prev fallback per signal",
                successorPrevFallbackPerSignal);
    printMetric("Successor-pressure precedence clamp count",
                regretStats.successorPressureDiagPrecedenceClampCount);
    printMetric("Successor-pressure precedence clamp per eval",
                successorPrecedenceClampPerEval);
    printMetric("Successor-pressure precedence clamp per signal",
                successorPrecedenceClampPerSignal);
    printMetric("Successor-pressure precedence clamp avg delta",
                successorPrecedenceClampAvgDelta);
    printMetric("Successor-pressure depth1 signals/eval",
                successorDepth1SignalsPerEval);
    printMetric("Successor-pressure depth>1 signals/eval",
                successorDepthGt1SignalsPerEval);
    printMetric("Successor-pressure descendant-active eval rate",
                successorDescendantActiveEvalRate);
    printMetric("Successor-pressure avg depth1 contribution",
                successorAvgDepth1Contribution);
    printMetric("Successor-pressure avg depth>1 contribution",
                successorAvgDepthGt1Contribution);
    printMetric("Successor-pressure descendant contribution share",
                successorDescendantContributionShare);
  }

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
