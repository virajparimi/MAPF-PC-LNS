#include "lns_iteration_outcome_orchestrator.hpp"

#include "lns.hpp"

bool IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
    LNS& lns, const std::string& reason, bool restorePrevious,
    ConflictMap& potentialNeighborhood, IterationExecutionContext& context) {
  return lns.finalizeCouldNotFindAbort(
      reason, restorePrevious, potentialNeighborhood, context);
}

bool LNS::finalizeCouldNotFindAbort(
    const std::string& reason, bool restorePrevious,
    ConflictMap& potentialNeighborhood, IterationExecutionContext& context) {
  if (restorePrevious) {
    restoreSolutionFromPrevious();
  }
  context.debugRow.earlyAbortReason = reason;
  context.feasibleSolutionUpdated = false;
  context.quality = IterationQuality::couldNotFind;
  const Time::time_point bookkeepingStart = Time::now();
  runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
  appendIterationStatBounded(IterationStats(
      runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
      currentObjectiveValue(), context.feasibleSolutionUpdated,
      context.quality));
  context.timeBookkeepingSec += ((fsec)(Time::now() - bookkeepingStart)).count();
  if (acceptanceState_.forceNeighborhoodChangeOnReject) {
    potentialNeighborhood.clear();
  }
  return true;
}
