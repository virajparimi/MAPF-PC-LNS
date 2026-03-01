#pragma once

#include "lns.hpp"
#include "lns_iteration_context.hpp"
#include "lns_iteration_phase_types.hpp"

class IterationSetupOrchestrator {
 public:
  static void initialize(LNS& lns,
                         const LNS::ValidationStats& currentValidationStats,
                         IterationExecutionContext& context);
};
