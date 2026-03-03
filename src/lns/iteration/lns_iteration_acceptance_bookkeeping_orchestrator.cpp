#include "lns_iteration_acceptance_bookkeeping_orchestrator.hpp"

#include "lns_acceptance_orchestrator.hpp"
#include "lns_market_iteration_orchestrator.hpp"

bool IterationAcceptanceBookkeepingOrchestrator::run(
    LNS& lns, bool nrrRepairSucceeded, ConflictMap& potentialNeighborhood,
    const ConflictMap& oldNeighborhood, bool& currentSolutionValid,
    LNS::ValidationStats& currentValidationStats,
    IterationExecutionContext& context) {
  const AcceptanceDecisionResult acceptanceDecision =
      AcceptanceOrchestrator::runDecision(
          lns, context.candidateValid, context.previousPressureForIter,
          context.previousWaitForIter, context.previousConflictSignalForIter,
          context.debugRow);
  context.softRecoveryModeBefore = acceptanceDecision.softRecoveryModeBefore;
  context.softRecoveryModeAfter = acceptanceDecision.softRecoveryModeAfter;
  context.softRecoveryDecisionReason =
      acceptanceDecision.softRecoveryDecisionReason;
  context.timeAcceptanceSec += acceptanceDecision.timeAcceptanceSec;
  if (!acceptanceDecision.ok) {
    lns.runtime = ((fsec)(Time::now() - lns.plannerStartTime_)).count();
    return false;
  }
  AcceptanceOrchestrator::updateAlnsStats(
      lns, context.alnsHeuristicForIter, context.previousSocForIter,
      context.proposedSocForIter, context.candidateValid,
      acceptanceDecision.accepted, acceptanceDecision.acceptedAsWorse,
      context.feasibleSolutionUpdated);
  AcceptanceOrchestrator::applyOutcome(
      lns, acceptanceDecision, context.candidateValid,
      context.feasibleSolutionUpdated, nrrRepairSucceeded,
      context.previousSocForIter, context.proposedSocForIter,
      context.incumbentSocBeforeIter, context.previousConflictSignalForIter,
      context.candidateConflictSignal, context.candidateValidationStats,
      potentialNeighborhood, oldNeighborhood, currentSolutionValid,
      currentValidationStats, context.quality, context.debugRow);

  MarketIterationOrchestrator::finalize(lns, acceptanceDecision.accepted);

  if (context.debugRow.earlyAbortReason.empty()) {
    context.debugRow.earlyAbortReason = "none";
  }
  const Time::time_point bookkeepingStart = Time::now();
  lns.runtime = ((fsec)(Time::now() - lns.plannerStartTime_)).count();
  double costToLog = (context.feasibleSolutionUpdated)
                         ? lns.incumbentSolution_.sumOfCosts
                         : lns.solution_.sumOfCosts;
  lns.appendIterationStatBounded(IterationStats(
      lns.runtime, "LNS", lns.instance_.getAgentNum(), lns.instance_.getTasksNum(),
      costToLog, context.feasibleSolutionUpdated, context.quality));
  context.timeBookkeepingSec +=
      ((fsec)(Time::now() - bookkeepingStart)).count();
  return true;
}
