#include "lns_iteration_acceptance_bookkeeping_orchestrator.hpp"

#include "lns_acceptance_orchestrator.hpp"
#include "lns_market_iteration_orchestrator.hpp"
#include <algorithm>
#include <limits>

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
  double costToLog = context.feasibleSolutionUpdated
                         ? lns.incumbentObjectiveValueOrMax()
                         : lns.currentObjectiveValue();
  int makespanToLog = -1;
  if (context.feasibleSolutionUpdated) {
    long long makespanLl = 0;
    for (const auto& agent : lns.solution_.agents) {
      makespanLl = std::max(
          makespanLl, static_cast<long long>(agent.path.endTimeOrZero()));
    }
    makespanToLog =
        makespanLl > std::numeric_limits<int>::max()
            ? std::numeric_limits<int>::max()
            : static_cast<int>(makespanLl);
  }
  const double precedenceWaitToLog =
      context.feasibleSolutionUpdated ? lns.computeSolutionPrecedenceWait()
                                      : std::numeric_limits<double>::quiet_NaN();
  lns.appendIterationStatBounded(IterationStats(
      lns.runtime, "LNS", lns.instance_.getAgentNum(), lns.instance_.getTasksNum(),
      costToLog, context.feasibleSolutionUpdated, context.quality, 0, 0,
      makespanToLog, precedenceWaitToLog));
  context.timeBookkeepingSec +=
      ((fsec)(Time::now() - bookkeepingStart)).count();
  return true;
}
