#pragma once

#include "lns.hpp"
#include "lns_iteration_context.hpp"
#include "lns_iteration_phase_types.hpp"

class IterationLifecycleOrchestrator {
 public:
  // Commit per-iteration phase timing aggregates and optionally append the
  // debug row. Idempotent via iterationTimingCommitted flag.
  static void commitTimingAndMaybeRecord(LNS& lns,
                                         IterationExecutionContext& context);
};
