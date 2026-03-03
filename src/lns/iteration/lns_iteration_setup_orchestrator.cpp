#include "lns_iteration_setup_orchestrator.hpp"

#include "lns_market_iteration_orchestrator.hpp"

#include <limits>

void IterationSetupOrchestrator::initialize(
    LNS& lns, const LNS::ValidationStats& currentValidationStats,
    IterationExecutionContext& context) {
  const IterationMarketContext marketContext =
      MarketIterationOrchestrator::begin(lns);

  context.previousSocForIter = lns.previousSolution_.sumOfCosts;
  context.previousValidationStatsForIter = currentValidationStats;
  context.previousPressureForIter = marketContext.previousPressure;
  context.previousWaitForIter = marketContext.previousWait;
  const int64_t iterationIndex =
      static_cast<int64_t>(lns.iterationDebugRecords_.size());
  context.incumbentSocBeforeIter =
      lns.incumbentSolution_.agentPaths.empty()
          ? std::numeric_limits<int>::max()
          : lns.incumbentSolution_.sumOfCosts;
  context.collectIterationDebug =
      lns.debugImprovementDiagnostics_ || !lns.debugIterationTsvPath_.empty();
  context.debugRow.iteration = iterationIndex;
  context.debugRow.previousSoc = context.previousSocForIter;
  context.debugRow.candidateSoc = context.previousSocForIter;
  context.debugRow.incumbentSocBefore = context.incumbentSocBeforeIter;
  context.softRecoveryModeBefore = lns.softRecoveryActive_;
  context.softRecoveryModeAfter = lns.softRecoveryActive_;
  context.debugRow.softRecoveryModeBefore = lns.softRecoveryActive_;
  context.debugRow.softRecoveryModeAfter = lns.softRecoveryActive_;

  lns.improvementDiagnosticsStats_.iterationsStarted++;
  lns.improvementDiagnosticsStats_.sumPreviousSoc +=
      static_cast<double>(context.previousSocForIter);
  if (context.incumbentSocBeforeIter != std::numeric_limits<int>::max()) {
    lns.improvementDiagnosticsStats_.sumIncumbentSoc +=
        static_cast<double>(context.incumbentSocBeforeIter);
    lns.improvementDiagnosticsStats_.incumbentSocSamples++;
  }
}
