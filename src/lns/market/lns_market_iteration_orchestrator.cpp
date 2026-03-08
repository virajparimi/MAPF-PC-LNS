#include "lns_market_iteration_orchestrator.hpp"

#include <algorithm>

IterationMarketContext MarketIterationOrchestrator::begin(LNS& lns) {
  return lns.beginIterationMarket();
}

IterationMarketContext LNS::beginIterationMarket() {
  market_.candidateUpdateConsumed = false;
  IterationMarketContext ctx;
  ctx.previousPressure = market_.heuristics ? computeSolutionMarketPressure() : 0.0;
  ctx.previousWait = market_.heuristics ? computeSolutionPrecedenceWait() : 0.0;
  return ctx;
}

MarketGuardDecision MarketIterationOrchestrator::evaluateGuards(
    LNS& lns, double previousPressureForIter, double previousWaitForIter) {
  return lns.evaluateIterationMarketGuards(previousPressureForIter,
                                           previousWaitForIter);
}

MarketGuardDecision LNS::evaluateIterationMarketGuards(
    double previousPressureForIter, double previousWaitForIter) {
  MarketGuardDecision decision;
  decision.candidatePressure =
      market_.heuristics ? computeSolutionMarketPressure() : 0.0;
  decision.candidateWait = market_.heuristics ? computeSolutionPrecedenceWait() : 0.0;

  if (market_.heuristics && !market_.updateOnAcceptedOnly &&
      market_.updateFromCandidate) {
    maybeUpdateMarketState(false, true);
  }
  if (market_.heuristics && market_.acceptanceGuards &&
      !passMarketAcceptanceGuards(previousPressureForIter,
                                  decision.candidatePressure,
                                  previousWaitForIter, decision.candidateWait,
                                  previousSolution_.utility < solution_.utility)) {
    marketGuardRejections++;
    decision.allowed = false;
    decision.guardRejected = true;
  }
  return decision;
}

void MarketIterationOrchestrator::updateBestOnAccepted(
    LNS& lns, double candidatePressure, double candidateWait) {
  lns.updateIterationMarketBestOnAccepted(candidatePressure, candidateWait);
}

void LNS::updateIterationMarketBestOnAccepted(double candidatePressure,
                                              double candidateWait) {
  if (!market_.heuristics) {
    return;
  }
  market_.bestPressure = std::min(market_.bestPressure, candidatePressure);
  market_.bestWait = std::min(market_.bestWait, candidateWait);
}

void MarketIterationOrchestrator::finalize(LNS& lns, bool accepted) {
  lns.finalizeIterationMarket(accepted);
}

void LNS::finalizeIterationMarket(bool accepted) {
  maybeUpdateMarketState(accepted);
}
