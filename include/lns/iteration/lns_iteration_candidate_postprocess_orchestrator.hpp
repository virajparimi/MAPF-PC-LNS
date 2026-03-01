#pragma once

#include "lns.hpp"
#include "lns_iteration_context.hpp"
#include "lns_iteration_phase_types.hpp"

class IterationCandidatePostprocessOrchestrator {
 public:
  static void process(LNS& lns, const CandidatePhaseResult& candidatePhase,
                      IterationExecutionContext& context);
};
