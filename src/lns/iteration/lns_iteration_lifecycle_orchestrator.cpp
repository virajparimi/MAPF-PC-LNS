#include "lns_iteration_lifecycle_orchestrator.hpp"

void IterationLifecycleOrchestrator::commitTimingAndMaybeRecord(
    LNS& lns, IterationExecutionContext& context) {
  lns.commitIterationTimingAndMaybeRecord(context);
}

void LNS::commitIterationTimingAndMaybeRecord(
    IterationExecutionContext& context) {
  if (context.iterationTimingCommitted) {
    return;
  }
  improvementDiagnosticsStats_.timeDestroyAndPrepareSec +=
      context.timeDestroyAndPrepareSec;
  improvementDiagnosticsStats_.timeRepairAndCommitSec +=
      context.timeRepairAndCommitSec;
  improvementDiagnosticsStats_.timeRegretCandidateEvalSec +=
      context.timeRegretCandidateEvalSec;
  improvementDiagnosticsStats_.timeRegretCommitSec +=
      context.timeRegretCommitSec;
  improvementDiagnosticsStats_.timeRegretLowLevelSec +=
      context.timeRegretLowLevelSec;
  improvementDiagnosticsStats_.timeJoinPathsSec += context.timeJoinPathsSec;
  improvementDiagnosticsStats_.timeRecomputeSocSec +=
      context.timeRecomputeSocSec;
  improvementDiagnosticsStats_.timeValidationSec +=
      context.timeValidationSec;
  improvementDiagnosticsStats_.timeAcceptanceSec +=
      context.timeAcceptanceSec;
  improvementDiagnosticsStats_.timeBookkeepingSec +=
      context.timeBookkeepingSec;
  context.iterationTimingCommitted = true;
  if (context.collectIterationDebug && !context.iterationRowCommitted) {
    context.debugRow.runtimeSec = runtime;
    context.debugRow.quality = iterationQualityName(context.quality);
    context.debugRow.timeDestroyAndPrepareSec = context.timeDestroyAndPrepareSec;
    context.debugRow.timeRepairAndCommitSec = context.timeRepairAndCommitSec;
    context.debugRow.timeRegretCandidateEvalSec =
        context.timeRegretCandidateEvalSec;
    context.debugRow.timeRegretCommitSec = context.timeRegretCommitSec;
    context.debugRow.timeRegretLowLevelSec = context.timeRegretLowLevelSec;
    context.debugRow.timeJoinPathsSec = context.timeJoinPathsSec;
    context.debugRow.timeRecomputeSocSec = context.timeRecomputeSocSec;
    context.debugRow.timeValidationSec = context.timeValidationSec;
    context.debugRow.timeAcceptanceSec = context.timeAcceptanceSec;
    context.debugRow.timeBookkeepingSec = context.timeBookkeepingSec;
    iterationDebugRecords_.push_back(context.debugRow);
    context.iterationRowCommitted = true;
  }
}
