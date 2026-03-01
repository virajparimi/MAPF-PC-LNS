#pragma once

#include "lns.hpp"
#include "lns_iteration_context.hpp"
#include "lns_iteration_phase_types.hpp"
#include "lns_types.hpp"

class IterationAcceptanceBookkeepingOrchestrator {
 public:
  // Runs acceptance decision + outcome application + end-of-iteration
  // bookkeeping. Returns false only for fatal acceptance-decision errors.
  static bool run(LNS& lns, bool nrrRepairSucceeded,
                  ConflictMap& potentialNeighborhood,
                  const ConflictMap& oldNeighborhood,
                  bool& currentSolutionValid,
                  LNS::ValidationStats& currentValidationStats,
                  IterationExecutionContext& context);
};
