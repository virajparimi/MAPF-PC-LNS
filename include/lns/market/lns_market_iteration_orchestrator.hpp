#pragma once

#include "lns.hpp"
#include "lns_iteration_phase_types.hpp"

class MarketIterationOrchestrator {
 public:
  static IterationMarketContext begin(LNS& lns);

  static MarketGuardDecision evaluateGuards(LNS& lns,
                                            double previousPressureForIter,
                                            double previousWaitForIter);

  static void updateBestOnAccepted(LNS& lns, double candidatePressure,
                                   double candidateWait);

  static void finalize(LNS& lns, bool accepted);
};
