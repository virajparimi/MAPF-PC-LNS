#include "lns_iteration_setup_orchestrator.hpp"

#include "lns_market_iteration_orchestrator.hpp"

void IterationSetupOrchestrator::initialize(
    LNS& lns, const LNS::ValidationStats& currentValidationStats,
    IterationExecutionContext& context) {
  lns.initializeIterationContext(currentValidationStats, context);
}

void LNS::initializeIterationContext(
    const ValidationStats& currentValidationStats,
    IterationExecutionContext& context) {
  const IterationMarketContext marketContext = beginIterationMarket();

  context.previousSocForIter = previousObjectiveValue();
  context.previousValidationStatsForIter = currentValidationStats;
  context.previousPressureForIter = marketContext.previousPressure;
  context.previousWaitForIter = marketContext.previousWait;
  const int64_t iterationIndex =
      static_cast<int64_t>(iterationDebugRecords_.size());
  context.incumbentSocBeforeIter = incumbentObjectiveValueOrMax();
  context.collectIterationDebug =
      debugImprovementDiagnostics_ || !debugIterationTsvPath_.empty();
  context.debugRow.iteration = iterationIndex;
  context.debugRow.previousSoc = context.previousSocForIter;
  context.debugRow.candidateSoc = context.previousSocForIter;
  context.debugRow.incumbentSocBefore = context.incumbentSocBeforeIter;
  context.softRecoveryModeBefore = acceptanceState_.softRecoveryActive;
  context.softRecoveryModeAfter = acceptanceState_.softRecoveryActive;
  context.debugRow.softRecoveryModeBefore = acceptanceState_.softRecoveryActive;
  context.debugRow.softRecoveryModeAfter = acceptanceState_.softRecoveryActive;

  improvementDiagnosticsStats_.iterationsStarted++;
  improvementDiagnosticsStats_.sumPreviousSoc +=
      static_cast<double>(context.previousSocForIter);
  if (context.incumbentSocBeforeIter != std::numeric_limits<int>::max()) {
    improvementDiagnosticsStats_.sumIncumbentSoc +=
        static_cast<double>(context.incumbentSocBeforeIter);
    improvementDiagnosticsStats_.incumbentSocSamples++;
  }
}
