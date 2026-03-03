#include "run_reporting.hpp"
#include "run_reporting_internal.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

using run_reporting_internal::FinalSolutionScheduleMetrics;
using run_reporting_internal::computeFinalSolutionScheduleMetrics;
using run_reporting_internal::computeIterationDurations;
using run_reporting_internal::computeTimeToBestFeasible;
using run_reporting_internal::countLnsIterations;
using run_reporting_internal::percentileValue;
using run_reporting_internal::printMetric;

void printRunSummaryReport(const LNS& lns, const FeasibleSolution& solution,
                           bool success, bool marketHeuristics,
                           bool incrementalRegret,
                           const FeasibleTrajectoryStats& stats) {
  const int lnsIterations = countLnsIterations(lns.iterationStats);
  const vector<double> iterDurations = computeIterationDurations(lns.iterationStats);
  const double avgIterSec = iterDurations.empty()
                                ? 0.0
                                : std::accumulate(iterDurations.begin(),
                                                  iterDurations.end(),
                                                  0.0) /
                                      static_cast<double>(iterDurations.size());
  const double p95IterSec = percentileValue(iterDurations, 0.95);
  const double iterPerSec =
      lns.runtime > 0.0 ? static_cast<double>(lnsIterations) / lns.runtime
                        : 0.0;
  const double timeToBestFeasible = computeTimeToBestFeasible(lns.iterationStats);
  const FinalSolutionScheduleMetrics finalSolutionMetrics =
      computeFinalSolutionScheduleMetrics(solution, lns.getInstance());
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

  std::cout << "\n=== Run Summary ===\n";
  printMetric("Requested initial solution",
              lns.getInitialSolutionRequested());
  printMetric("Effective initial solution", lns.getInitialSolutionEffective());
  printMetric("Initial fallback used",
              lns.wasInitialSolutionFallbackUsed() ? "true" : "false");
  printMetric("Initial fallback reason",
              lns.getInitialSolutionFallbackReason());
  printMetric("Runtime (s)", lns.runtime);
  printMetric("Iterations", lnsIterations);
  printMetric("Iterations/sec", iterPerSec);
  printMetric("Avg iteration runtime (ms)", avgIterSec * 1000.0);
  printMetric("P95 iteration runtime (ms)", p95IterSec * 1000.0);
  printMetric("First feasible runtime (s)", stats.firstFeasibleRuntime);
  printMetric("Time-to-best feasible (s)", timeToBestFeasible);
  printMetric("Improving feasible updates", stats.numImprovingFeasibleUpdates);
  printMetric("Total feasible iterations", stats.numFeasibleIterations);
  printMetric("Solution cost", solution.sumOfCosts);
  printMetric("Makespan", finalSolutionMetrics.makespan);
  printMetric("Max individual cost", finalSolutionMetrics.maxIndividualCost);
  if (finalSolutionMetrics.hasTaskScheduleData) {
    printMetric("Total precedence wait",
                finalSolutionMetrics.totalPrecedenceWait);
    printMetric("Max precedence wait", finalSolutionMetrics.maxPrecedenceWait);
    printMetric("Precedence slack edges",
                static_cast<int64_t>(finalSolutionMetrics.precedenceSlacks.size()));
    if (!finalSolutionMetrics.precedenceSlacks.empty()) {
      const double minSlack =
          *std::min_element(finalSolutionMetrics.precedenceSlacks.begin(),
                            finalSolutionMetrics.precedenceSlacks.end());
      printMetric("Min precedence slack", minSlack);
      printMetric("Precedence slack p25",
                  percentileValue(finalSolutionMetrics.precedenceSlacks, 0.25));
      printMetric("Precedence slack p50",
                  percentileValue(finalSolutionMetrics.precedenceSlacks, 0.50));
      printMetric("Precedence slack p75",
                  percentileValue(finalSolutionMetrics.precedenceSlacks, 0.75));
    } else {
      printMetric("Min precedence slack", "n/a");
      printMetric("Precedence slack p25", "n/a");
      printMetric("Precedence slack p50", "n/a");
      printMetric("Precedence slack p75", "n/a");
    }
  } else {
    printMetric("Total precedence wait", "n/a");
    printMetric("Max precedence wait", "n/a");
    printMetric("Precedence slack edges", "n/a");
    printMetric("Min precedence slack", "n/a");
    printMetric("Precedence slack p25", "n/a");
    printMetric("Precedence slack p50", "n/a");
    printMetric("Precedence slack p75", "n/a");
  }
  printMetric("Number of failures", lns.numOfFailures);
  printMetric("Invalid candidate rejections",
              lns.invalidCandidateRejections);
  printMetric("Market guard rejections", lns.marketGuardRejections);
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
  printMetric("LL timeout reason goal-permanent-before-lb",
              static_cast<int64_t>(
                  lowLevelStats.timeoutGoalPermanentBeforeArrivalLb));
  printMetric("LL timeout reason start-trapped@t+1",
              static_cast<int64_t>(lowLevelStats.timeoutStartTrappedAtTPlus1));
  printMetric("LL timeout reason static-disconnected",
              static_cast<int64_t>(
                  lowLevelStats.timeoutStaticDisconnectedPermanent));
  printMetric("LL timeout reason other",
              static_cast<int64_t>(lowLevelStats.timeoutOther));
  printMetric("LL timeout reduced-by-global-budget",
              static_cast<int64_t>(lowLevelStats.timeoutReducedByGlobalBudget));
  printMetric("LL timeout multi-certificate",
              static_cast<int64_t>(lowLevelStats.timeoutMultiCertificate));
  printMetric("LL structural pre-pruned",
              static_cast<int64_t>(lowLevelStats.structuralPrePruned));
  printMetric("LL structural pre-pruned goal-permanent-before-lb",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedGoalPermanentBeforeArrivalLb));
  printMetric("LL structural pre-pruned start-trapped@t+1",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedStartTrappedAtTPlus1));
  printMetric("LL structural pre-pruned static-disconnected",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedStaticDisconnectedPermanent));
  printMetric("LL structural pre-pruned multi-certificate",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedMultiCertificate));
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
  std::cout << "\n=== Solution Restore Stats ===\n";
  printMetric("Restore calls", restoreStats.restoreCalls);
  printMetric("Full restores", restoreStats.fullRestores);

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
    printMetric("Market total precedence wait", marketStats.totalPrecedenceWait);
    printMetric("Market max precedence wait", marketStats.maxPrecedenceWait);
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

  const auto& nrrStats = lns.getNrrStats();
  if (nrrStats.attempts > 0) {
    const auto formatHistogram = [](const std::map<int, int64_t>& hist) {
      if (hist.empty()) {
        return string("(none)");
      }
      std::ostringstream oss;
      bool first = true;
      for (const auto& [key, count] : hist) {
        if (!first) {
          oss << ", ";
        }
        first = false;
        oss << key << "->" << count;
      }
      return oss.str();
    };
    const auto formatReasonHistogram =
        [](const std::map<std::string, int64_t>& hist) {
          if (hist.empty()) {
            return string("(none)");
          }
          std::ostringstream oss;
          bool first = true;
          for (const auto& [key, count] : hist) {
            if (!first) {
              oss << ", ";
            }
            first = false;
            oss << key << "->" << count;
          }
          return oss.str();
        };
    const double successRate =
        nrrStats.attempts > 0
            ? (double)nrrStats.success / (double)nrrStats.attempts
            : 0.0;
    const double avgAttemptTimeSec =
        nrrStats.attempts > 0 ? nrrStats.runtimeSecSum / (double)nrrStats.attempts
                              : 0.0;
    const double avgDestroyed =
        nrrStats.attempts > 0
            ? (double)nrrStats.removedTasksSum / (double)nrrStats.attempts
            : 0.0;
    const double avgNeighborhoodAgents =
        nrrStats.attempts > 0
            ? (double)nrrStats.neighborhoodAgentsSum / (double)nrrStats.attempts
            : 0.0;
    const double avgAcceptedDeltaSoc =
        nrrStats.acceptedDeltaSocCount > 0
            ? nrrStats.acceptedDeltaSocSum /
                  (double)nrrStats.acceptedDeltaSocCount
            : 0.0;
    const double avgSoftCandidateConflicts =
        nrrStats.softCandidateConflictSamples > 0
            ? nrrStats.softCandidateConflictSum /
                  (double)nrrStats.softCandidateConflictSamples
            : 0.0;
    const double avgSoftConflictDescentPerResolvedEntry =
        nrrStats.softModeResolvedEntries > 0
            ? (double)(nrrStats.softModeEntryConflictSum -
                       nrrStats.softModeExitConflictSum) /
                  (double)nrrStats.softModeResolvedEntries
            : 0.0;
    auto formatSoftHeuristicHistogram = [](const std::vector<int64_t>& values) {
      if (values.empty()) {
        return string("(none)");
      }
      auto nameForId = [](int id) -> const char* {
        switch (id) {
          case DestroyHeuristic::randomRemoval:
            return "random";
          case DestroyHeuristic::worstRemoval:
            return "worst";
          case DestroyHeuristic::conflictRemoval:
            return "conflict";
          case DestroyHeuristic::shawRemoval:
            return "shaw";
          case DestroyHeuristic::precedenceWaitRemoval:
            return "precedence_wait";
          case DestroyHeuristic::lowSlackRemoval:
            return "low_slack";
          case DestroyHeuristic::marketTatonnementRemoval:
            return "market_tatonnement";
          case DestroyHeuristic::collisionSoftRemoval:
            return "collision_soft";
          case DestroyHeuristic::failureSoftRemoval:
            return "failure_soft";
          default:
            return "unknown";
        }
      };
      std::ostringstream oss;
      bool first = true;
      for (int i = 0; i < (int)values.size(); i++) {
        if (values[i] <= 0) {
          continue;
        }
        if (!first) {
          oss << ", ";
        }
        first = false;
        oss << nameForId(i) << "->" << values[i];
      }
      if (first) {
        return string("(all_zero)");
      }
      return oss.str();
    };
    std::cout << "\n=== NRR Stats ===\n";
    printMetric("Attempts", nrrStats.attempts);
    printMetric("Success", nrrStats.success);
    printMetric("Fallback to standard", nrrStats.fallbackToStandard);
    printMetric("Success rate", successRate);
    printMetric("Avg attempt time (s)", avgAttemptTimeSec);
    printMetric("Solver PBS count", nrrStats.solverPbs);
    printMetric("Solver CBS count", nrrStats.solverCbs);
    printMetric("Avg |D|", avgDestroyed);
    printMetric("Max |D|", nrrStats.removedTasksMax);
    printMetric("Avg |A'|", avgNeighborhoodAgents);
    printMetric("Max |A'|", nrrStats.neighborhoodAgentsMax);
    printMetric("|D| distribution",
                formatHistogram(nrrStats.removedTasksHistogram));
    printMetric("|A'| distribution",
                formatHistogram(nrrStats.neighborhoodAgentsHistogram));
    printMetric("Failure reasons",
                formatReasonHistogram(nrrStats.failureReasonHistogram));
    printMetric("Soft candidates produced", nrrStats.softCandidatesProduced);
    printMetric("Soft candidates accepted", nrrStats.softCandidatesAccepted);
    printMetric("Soft candidates rejected", nrrStats.softCandidatesRejected);
    printMetric("Soft-recovery entries", nrrStats.softRecoveryEntries);
    printMetric("Soft-recovery exits", nrrStats.softRecoveryExits);
    printMetric("Soft-recovery accepted entry",
                nrrStats.softRecoveryAcceptedEntry);
    printMetric("Soft-recovery accepted descent",
                nrrStats.softRecoveryAcceptedDescent);
    printMetric("Soft-recovery rejected", nrrStats.softRecoveryRejected);
    printMetric("Soft-mode entry conflict sum",
                nrrStats.softModeEntryConflictSum);
    printMetric("Soft-mode exit conflict sum",
                nrrStats.softModeExitConflictSum);
    printMetric("Soft-mode resolved entries",
                nrrStats.softModeResolvedEntries);
    printMetric("Soft-mode avg conflict descent/resolved-entry",
                avgSoftConflictDescentPerResolvedEntry);
    printMetric("Soft-mode heuristic selections",
                formatSoftHeuristicHistogram(
                    nrrStats.softModeSelectionsByDestroy));
    printMetric("Soft-mode heuristic accepted",
                formatSoftHeuristicHistogram(
                    nrrStats.softModeAcceptedByDestroy));
    printMetric("Soft-mode heuristic best updates",
                formatSoftHeuristicHistogram(
                    nrrStats.softModeBestUpdatesByDestroy));
    printMetric("Soft-candidate avg conflicts", avgSoftCandidateConflicts);
    printMetric("Stitched-invalid attempts", nrrStats.stitchedInvalidAttempts);
    printMetric("Stitched-invalid precedence violations",
                nrrStats.stitchedInvalidPrecedenceViolations);
    printMetric("Stitched-invalid vertex collisions",
                nrrStats.stitchedInvalidVertexCollisions);
    printMetric("Stitched-invalid edge-swap collisions",
                nrrStats.stitchedInvalidEdgeSwapCollisions);
    printMetric("Stitched-invalid structural violations",
                nrrStats.stitchedInvalidStructuralViolations);
    printMetric("Accepted avg DeltaSoC", avgAcceptedDeltaSoc);
  }

  if (lns.isDebugImprovementDiagnosticsEnabled()) {
    const auto& debugStats = lns.getImprovementDiagnosticsStats();
    const int64_t started = debugStats.iterationsStarted;
    const int64_t candidateEvaluated =
        debugStats.candidateValid + debugStats.candidateInvalid;
    const int64_t accepted = debugStats.accepted;
    const int64_t rejected = debugStats.rejected;
    const int64_t decisions = accepted + rejected;
    const double acceptanceRate =
        decisions > 0 ? (double)accepted / (double)decisions : 0.0;
    const double candidateValidRate =
        candidateEvaluated > 0
            ? (double)debugStats.candidateValid / (double)candidateEvaluated
            : 0.0;
    const double acceptedFeasibleNoBestRate =
        accepted > 0
            ? (double)debugStats.acceptedFeasibleNoBestUpdate /
                  (double)accepted
            : 0.0;
    const double acceptedFingerprintRepeatRate =
        accepted > 0
            ? (double)debugStats.acceptedFingerprintRepeat / (double)accepted
            : 0.0;
    const int64_t neighborhoodFingerprintSamples =
        debugStats.neighborhoodFingerprintUnique +
        debugStats.neighborhoodFingerprintRepeat;
    const double neighborhoodFingerprintRepeatRate =
        neighborhoodFingerprintSamples > 0
            ? (double)debugStats.neighborhoodFingerprintRepeat /
                  (double)neighborhoodFingerprintSamples
            : 0.0;
    const double neighborhoodMeanJaccardPrev =
        debugStats.neighborhoodJaccardPrevSamples > 0
            ? debugStats.neighborhoodJaccardPrevSum /
                  (double)debugStats.neighborhoodJaccardPrevSamples
            : 0.0;
    const double neighborhoodMeanJaccardPrevWhenRepeat =
        debugStats.neighborhoodJaccardPrevRepeatSamples > 0
            ? debugStats.neighborhoodJaccardPrevRepeatSum /
                  (double)debugStats.neighborhoodJaccardPrevRepeatSamples
            : 0.0;
    const double avgRemovedTasksPerNeighborhood =
        debugStats.neighborhoodsCount > 0
            ? (double)debugStats.removedTasksTotal /
                  (double)debugStats.neighborhoodsCount
            : 0.0;
    const double avgChangedRemovedTasksPerNeighborhood =
        debugStats.neighborhoodsCount > 0
            ? (double)(debugStats.removedTasksChangedAgentTotal +
                       debugStats.removedTasksChangedOrderTotal) /
                  (double)debugStats.neighborhoodsCount
            : 0.0;
    const double changedNeighborhoodRate =
        debugStats.neighborhoodsCount > 0
            ? (double)debugStats.neighborhoodsWithChanges /
                  (double)debugStats.neighborhoodsCount
            : 0.0;
    const double avgAcceptedRemovedTasks =
        debugStats.accepted > 0
            ? (double)debugStats.acceptedRemovedTasksTotal /
                  (double)debugStats.accepted
            : 0.0;
    const double avgAcceptedChangedRemovedTasks =
        debugStats.accepted > 0
            ? (double)(debugStats.acceptedRemovedTasksChangedAgentTotal +
                       debugStats.acceptedRemovedTasksChangedOrderTotal) /
                  (double)debugStats.accepted
            : 0.0;
    const double meanPreviousSoc =
        started > 0 ? debugStats.sumPreviousSoc / (double)started : 0.0;
    const double meanCandidateSoc =
        candidateEvaluated > 0
            ? debugStats.sumCandidateSoc / (double)candidateEvaluated
            : 0.0;
    const double meanIncumbentSoc =
        debugStats.incumbentSocSamples > 0
            ? debugStats.sumIncumbentSoc /
                  (double)debugStats.incumbentSocSamples
            : 0.0;
    const double meanPreviousConflictSignal =
        candidateEvaluated > 0
            ? debugStats.sumPreviousConflictSignal /
                  (double)candidateEvaluated
            : 0.0;
    const double meanCandidateConflictSignal =
        candidateEvaluated > 0
            ? debugStats.sumCandidateConflictSignal /
                  (double)candidateEvaluated
            : 0.0;
    const double avgSoftCandidateConflict =
        debugStats.softCandidateConflictSamples > 0
            ? debugStats.softCandidateConflictSum /
                  (double)debugStats.softCandidateConflictSamples
            : 0.0;
    const double avgSoftConflictDescentPerResolvedEntry =
        debugStats.softModeResolvedEntries > 0
            ? (double)(debugStats.softModeEntryConflictSum -
                       debugStats.softModeExitConflictSum) /
                  (double)debugStats.softModeResolvedEntries
            : 0.0;

    const double measuredDebugSec =
        debugStats.timeDestroyAndPrepareSec +
        debugStats.timeRepairAndCommitSec + debugStats.timeJoinPathsSec +
        debugStats.timeTerminalReplanSec + debugStats.timeRecomputeSocSec +
        debugStats.timeValidationSec + debugStats.timeAcceptanceSec +
        debugStats.timeBookkeepingSec;
    const double measuredShareOfRuntime =
        lns.runtime > 0.0 ? measuredDebugSec / lns.runtime : 0.0;
    const auto phaseShare = [&](double phaseSec) -> double {
      return measuredDebugSec > 0.0 ? phaseSec / measuredDebugSec : 0.0;
    };

    std::cout << "\n=== Improvement Diagnostics ===\n";
    printMetric("Iterations started", started);
    printMetric("Iterations with candidate eval", candidateEvaluated);
    printMetric("Candidate valid", debugStats.candidateValid);
    printMetric("Candidate invalid", debugStats.candidateInvalid);
    printMetric("Candidate valid rate", candidateValidRate);
    printMetric("Accepted", accepted);
    printMetric("Rejected", rejected);
    printMetric("Guard rejected", debugStats.guardRejected);
    printMetric("Acceptance rate", acceptanceRate);
    printMetric("Accepted utility better/equal",
                debugStats.acceptedAsBetterOrEqualUtility);
    printMetric("Accepted utility worse", debugStats.acceptedAsWorseUtility);
    printMetric("Accepted SoC better vs previous",
                debugStats.acceptedSocBetterVsPrevious);
    printMetric("Accepted SoC equal vs previous",
                debugStats.acceptedSocEqualVsPrevious);
    printMetric("Accepted SoC worse vs previous",
                debugStats.acceptedSocWorseVsPrevious);
    printMetric("Accepted SoC better vs incumbent",
                debugStats.acceptedSocBetterVsIncumbent);
    printMetric("Accepted SoC equal vs incumbent",
                debugStats.acceptedSocEqualVsIncumbent);
    printMetric("Accepted SoC worse vs incumbent",
                debugStats.acceptedSocWorseVsIncumbent);
    printMetric("Feasible best updates", debugStats.feasibleBestUpdates);
    printMetric("Feasible no-best updates", debugStats.feasibleNoBestUpdate);
    printMetric("Accepted feasible no-best updates",
                debugStats.acceptedFeasibleNoBestUpdate);
    printMetric("Accepted feasible-no-best rate",
                acceptedFeasibleNoBestRate);
    printMetric("Accepted invalid", debugStats.acceptedInvalid);
    printMetric("Soft candidates produced",
                debugStats.softCandidateProduced);
    printMetric("Soft candidates accepted",
                debugStats.softCandidateAccepted);
    printMetric("Soft candidates rejected",
                debugStats.softCandidateRejected);
    printMetric("Soft-recovery entries",
                debugStats.softRecoveryEntries);
    printMetric("Soft-recovery exits",
                debugStats.softRecoveryExits);
    printMetric("Soft-recovery accepted entry",
                debugStats.softRecoveryAcceptedEntry);
    printMetric("Soft-recovery accepted descent",
                debugStats.softRecoveryAcceptedDescent);
    printMetric("Soft-recovery rejected",
                debugStats.softRecoveryRejected);
    printMetric("Soft-mode entry conflict sum",
                debugStats.softModeEntryConflictSum);
    printMetric("Soft-mode exit conflict sum",
                debugStats.softModeExitConflictSum);
    printMetric("Soft-mode resolved entries",
                debugStats.softModeResolvedEntries);
    printMetric("Soft-mode avg conflict descent/resolved-entry",
                avgSoftConflictDescentPerResolvedEntry);
    printMetric("Avg soft-candidate conflicts",
                avgSoftCandidateConflict);
    printMetric("Conflict signal better", debugStats.conflictSignalBetter);
    printMetric("Conflict signal equal", debugStats.conflictSignalEqual);
    printMetric("Conflict signal worse", debugStats.conflictSignalWorse);
    printMetric("Accepted conflict better",
                debugStats.acceptedConflictSignalBetter);
    printMetric("Accepted conflict equal",
                debugStats.acceptedConflictSignalEqual);
    printMetric("Accepted conflict worse",
                debugStats.acceptedConflictSignalWorse);
    printMetric("Accepted fingerprint unique",
                debugStats.acceptedFingerprintUnique);
    printMetric("Accepted fingerprint repeat",
                debugStats.acceptedFingerprintRepeat);
    printMetric("Accepted fingerprint repeat rate",
                acceptedFingerprintRepeatRate);
    printMetric("Neighborhoods analyzed", debugStats.neighborhoodsCount);
    printMetric("Neighborhoods with changes",
                debugStats.neighborhoodsWithChanges);
    printMetric("Neighborhoods without changes",
                debugStats.neighborhoodsWithoutChanges);
    printMetric("Neighborhood fingerprint unique",
                debugStats.neighborhoodFingerprintUnique);
    printMetric("Neighborhood fingerprint repeat",
                debugStats.neighborhoodFingerprintRepeat);
    printMetric("Neighborhood fingerprint repeat rate",
                neighborhoodFingerprintRepeatRate);
    printMetric("Neighborhood repeat streak max",
                debugStats.neighborhoodRepeatStreakMax);
    printMetric("Neighborhood mean Jaccard vs previous",
                neighborhoodMeanJaccardPrev);
    printMetric("Neighborhood mean Jaccard vs previous (repeat only)",
                neighborhoodMeanJaccardPrevWhenRepeat);
    printMetric("Neighborhoods with changes rate",
                changedNeighborhoodRate);
    printMetric("Avg removed tasks/neighborhood",
                avgRemovedTasksPerNeighborhood);
    printMetric("Avg changed removed tasks/neighborhood",
                avgChangedRemovedTasksPerNeighborhood);
    printMetric("Max removed tasks/neighborhood",
                debugStats.removedTasksMax);
    printMetric("Max changed removed tasks/neighborhood",
                debugStats.changedTasksPerNeighborhoodMax);
    printMetric("Avg accepted removed tasks",
                avgAcceptedRemovedTasks);
    printMetric("Avg accepted changed removed tasks",
                avgAcceptedChangedRemovedTasks);
    printMetric("Max accepted removed tasks",
                debugStats.acceptedRemovedTasksMax);
    printMetric("Max accepted changed removed tasks",
                debugStats.acceptedChangedTasksPerNeighborhoodMax);
    printMetric("Early aborts (prepare)", debugStats.earlyAbortPrepare);
    printMetric("Early aborts (repair)", debugStats.earlyAbortRepair);
    printMetric("Early aborts (join)", debugStats.earlyAbortJoin);
    printMetric("Early aborts (terminal)", debugStats.earlyAbortTerminal);
    printMetric("Mean previous SoC", meanPreviousSoc);
    printMetric("Mean candidate SoC", meanCandidateSoc);
    printMetric("Mean incumbent SoC", meanIncumbentSoc);
    printMetric("Mean previous conflict signal",
                meanPreviousConflictSignal);
    printMetric("Mean candidate conflict signal",
                meanCandidateConflictSignal);

    std::cout << "\n=== Iteration Time Split (Debug) ===\n";
    printMetric("Measured debug time (s)", measuredDebugSec);
    printMetric("Measured time / runtime", measuredShareOfRuntime);
    printMetric("Destroy+prepare time (s)",
                debugStats.timeDestroyAndPrepareSec);
    printMetric("Repair+commit time (s)",
                debugStats.timeRepairAndCommitSec);
    printMetric("Join paths time (s)", debugStats.timeJoinPathsSec);
    printMetric("Terminal replan time (s)",
                debugStats.timeTerminalReplanSec);
    printMetric("Recompute SoC time (s)", debugStats.timeRecomputeSocSec);
    printMetric("Validation time (s)", debugStats.timeValidationSec);
    printMetric("Acceptance time (s)", debugStats.timeAcceptanceSec);
    printMetric("Bookkeeping time (s)", debugStats.timeBookkeepingSec);
    printMetric("Destroy+prepare share", phaseShare(debugStats.timeDestroyAndPrepareSec));
    printMetric("Repair+commit share", phaseShare(debugStats.timeRepairAndCommitSec));
    printMetric("Join paths share", phaseShare(debugStats.timeJoinPathsSec));
    printMetric("Terminal replan share", phaseShare(debugStats.timeTerminalReplanSec));
    printMetric("Recompute SoC share", phaseShare(debugStats.timeRecomputeSocSec));
    printMetric("Validation share", phaseShare(debugStats.timeValidationSec));
    printMetric("Acceptance share", phaseShare(debugStats.timeAcceptanceSec));
    printMetric("Bookkeeping share", phaseShare(debugStats.timeBookkeepingSec));
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
  const double avgChangedAgents =
      incrementalStats.commits > 0
          ? (double)incrementalStats.changedAgentsSum /
                (double)incrementalStats.commits
          : 0.0;
  const double avgRecomputedTasksPerCommit =
      incrementalStats.commits > 0
          ? (double)incrementalStats.recomputedTasks /
                (double)incrementalStats.commits
          : 0.0;
  const double avgDirtyByDescendants =
      incrementalStats.commits > 0
          ? (double)incrementalStats.dirtyByDescendants /
                (double)incrementalStats.commits
          : 0.0;
  const double avgDirtyByCandidateAgent =
      incrementalStats.commits > 0
          ? (double)incrementalStats.dirtyByCandidateAgent /
                (double)incrementalStats.commits
          : 0.0;
  const double avgDirtyByAncestors =
      incrementalStats.commits > 0
          ? (double)incrementalStats.dirtyByAncestors /
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
  printMetric("Avg changed agents/commit", avgChangedAgents);
  printMetric("Max changed agents", incrementalStats.changedAgentsMax);
  printMetric("Avg dirty-by-descendants/commit", avgDirtyByDescendants);
  printMetric("Avg dirty-by-candidate-agent/commit", avgDirtyByCandidateAgent);
  printMetric("Avg dirty-by-ancestors/commit", avgDirtyByAncestors);
  printMetric("Refreshes by high stale load",
              incrementalStats.refreshByHighStale);
  printMetric("Refreshes by stale-growth stall",
              incrementalStats.refreshByStaleGrowth);
  printMetric("Refreshes by periodic safety",
              incrementalStats.refreshByPeriodic);
  printMetric("Endgame full recomputes",
              incrementalStats.endgameFullRecomputes);
}
