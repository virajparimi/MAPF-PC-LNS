#include "lns_iteration_lifecycle_orchestrator.hpp"

void IterationLifecycleOrchestrator::commitTimingAndMaybeRecord(
    LNS& lns, IterationExecutionContext& context) {
  if (context.iterationTimingCommitted) {
    return;
  }
  lns.improvementDiagnosticsStats_.timeDestroyAndPrepareSec +=
      context.timeDestroyAndPrepareSec;
  lns.improvementDiagnosticsStats_.timeRepairAndCommitSec +=
      context.timeRepairAndCommitSec;
  lns.improvementDiagnosticsStats_.timeJoinPathsSec += context.timeJoinPathsSec;
  lns.improvementDiagnosticsStats_.timeTerminalReplanSec +=
      context.timeTerminalReplanSec;
  lns.improvementDiagnosticsStats_.timeRecomputeSocSec +=
      context.timeRecomputeSocSec;
  lns.improvementDiagnosticsStats_.timeValidationSec +=
      context.timeValidationSec;
  lns.improvementDiagnosticsStats_.timeAcceptanceSec +=
      context.timeAcceptanceSec;
  lns.improvementDiagnosticsStats_.timeBookkeepingSec +=
      context.timeBookkeepingSec;
  context.iterationTimingCommitted = true;
  if (context.collectIterationDebug && !context.iterationRowCommitted) {
    context.debugRow.runtimeSec = lns.runtime;
    context.debugRow.quality = lns.iterationQualityName(context.quality);
    context.debugRow.timeDestroyAndPrepareSec = context.timeDestroyAndPrepareSec;
    context.debugRow.timeRepairAndCommitSec = context.timeRepairAndCommitSec;
    context.debugRow.timeJoinPathsSec = context.timeJoinPathsSec;
    context.debugRow.timeTerminalReplanSec = context.timeTerminalReplanSec;
    context.debugRow.timeRecomputeSocSec = context.timeRecomputeSocSec;
    context.debugRow.timeValidationSec = context.timeValidationSec;
    context.debugRow.timeAcceptanceSec = context.timeAcceptanceSec;
    context.debugRow.timeBookkeepingSec = context.timeBookkeepingSec;
    lns.iterationDebugRecords_.push_back(context.debugRow);
    context.iterationRowCommitted = true;
  }
}
