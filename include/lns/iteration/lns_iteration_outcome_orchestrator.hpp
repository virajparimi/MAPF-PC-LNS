#pragma once

#include <string>

#include "lns.hpp"
#include "lns_iteration_context.hpp"
#include "lns_iteration_phase_types.hpp"
#include "lns_types.hpp"

class IterationOutcomeOrchestrator {
 public:
  // Finalize a could-not-find abort path (restore policy + bookkeeping + market
  // finalize). Returns true to mirror runOneIteration abort-return convention.
  static bool finalizeCouldNotFindAbort(
      LNS& lns, const std::string& reason, bool restorePrevious,
      ConflictMap& potentialNeighborhood, IterationExecutionContext& context);
};
