#include "lns_iteration_outcome_orchestrator.hpp"

#include "lns.hpp"
#include "lns_market_iteration_orchestrator.hpp"

bool IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
    LNS& lns, const std::string& reason, bool restorePrevious,
    ConflictMap& potentialNeighborhood, IterationExecutionContext& context) {
  if (restorePrevious) {
    lns.restoreSolutionFromPrevious();
  }
  context.debugRow.earlyAbortReason = reason;
  context.feasibleSolutionUpdated = false;
  context.quality = IterationQuality::couldNotFind;
  const Time::time_point bookkeepingStart = Time::now();
  lns.runtime = ((fsec)(Time::now() - lns.plannerStartTime_)).count();
  lns.appendIterationStatBounded(IterationStats(
      lns.runtime, "LNS", lns.instance_.getAgentNum(), lns.instance_.getTasksNum(),
      lns.solution_.sumOfCosts, context.feasibleSolutionUpdated, context.quality));
  context.timeBookkeepingSec += ((fsec)(Time::now() - bookkeepingStart)).count();
  if (lns.forceNeighborhoodChangeOnReject_) {
    potentialNeighborhood.clear();
  }
  MarketIterationOrchestrator::finalize(lns, false);
  return true;
}
