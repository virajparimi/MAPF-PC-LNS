#include "lns_market_iteration_orchestrator.hpp"

#include <algorithm>

IterationMarketContext MarketIterationOrchestrator::begin(LNS& lns) {
  lns.market_.candidateUpdateConsumed = false;
  IterationMarketContext ctx;
  ctx.previousPressure =
      lns.market_.heuristics ? lns.computeSolutionMarketPressure() : 0.0;
  ctx.previousWait =
      lns.market_.heuristics ? lns.computeSolutionPrecedenceWait() : 0.0;
  return ctx;
}

MarketGuardDecision MarketIterationOrchestrator::evaluateGuards(
    LNS& lns, double previousPressureForIter, double previousWaitForIter) {
  MarketGuardDecision decision;
  decision.candidatePressure =
      lns.market_.heuristics ? lns.computeSolutionMarketPressure() : 0.0;
  decision.candidateWait =
      lns.market_.heuristics ? lns.computeSolutionPrecedenceWait() : 0.0;

  if (lns.market_.heuristics && !lns.market_.updateOnAcceptedOnly &&
      lns.market_.updateFromCandidate) {
    lns.maybeUpdateMarketState(false, true);
  }
  if (lns.market_.heuristics && lns.market_.acceptanceGuards &&
      !lns.passMarketAcceptanceGuards(previousPressureForIter,
                                      decision.candidatePressure,
                                      previousWaitForIter,
                                      decision.candidateWait,
                                      lns.previousSolution_.utility <
                                          lns.solution_.utility)) {
    lns.marketGuardRejections++;
    decision.allowed = false;
    decision.guardRejected = true;
  }
  return decision;
}

void MarketIterationOrchestrator::updateBestOnAccepted(
    LNS& lns, double candidatePressure, double candidateWait) {
  if (!lns.market_.heuristics) {
    return;
  }
  lns.market_.bestPressure = std::min(lns.market_.bestPressure, candidatePressure);
  lns.market_.bestWait = std::min(lns.market_.bestWait, candidateWait);
}

void MarketIterationOrchestrator::finalize(LNS& lns, bool accepted) {
  lns.maybeUpdateMarketState(accepted);
}
