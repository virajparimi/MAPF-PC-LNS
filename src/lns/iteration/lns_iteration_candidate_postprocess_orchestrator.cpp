#include "lns_iteration_candidate_postprocess_orchestrator.hpp"

#include <algorithm>

void IterationCandidatePostprocessOrchestrator::process(
    LNS& lns, const CandidatePhaseResult& candidatePhase,
    IterationExecutionContext& context) {
  const int removedTasksChangedAgent = candidatePhase.removedTasksChangedAgent;
  const int removedTasksChangedOrder = candidatePhase.removedTasksChangedOrder;
  const int removedTasksUnchanged = candidatePhase.removedTasksUnchanged;
  context.debugRow.removedTasksChangedAgent = removedTasksChangedAgent;
  context.debugRow.removedTasksChangedOrder = removedTasksChangedOrder;
  context.debugRow.removedTasksUnchanged = removedTasksUnchanged;
  const int changedRemovedTasks = removedTasksChangedAgent + removedTasksChangedOrder;
  lns.improvementDiagnosticsStats_.removedTasksChangedAgentTotal +=
      removedTasksChangedAgent;
  lns.improvementDiagnosticsStats_.removedTasksChangedOrderTotal +=
      removedTasksChangedOrder;
  lns.improvementDiagnosticsStats_.removedTasksUnchangedTotal +=
      removedTasksUnchanged;
  lns.improvementDiagnosticsStats_.changedTasksPerNeighborhoodMax = std::max(
      lns.improvementDiagnosticsStats_.changedTasksPerNeighborhoodMax,
      (int64_t)changedRemovedTasks);
  if (changedRemovedTasks > 0) {
    lns.improvementDiagnosticsStats_.neighborhoodsWithChanges++;
  } else {
    lns.improvementDiagnosticsStats_.neighborhoodsWithoutChanges++;
  }

  context.previousConflictSignalForIter =
      context.previousValidationStatsForIter.totalConflictEvents();
  context.debugRow.previousConflictSignal = context.previousConflictSignalForIter;
  PLOGD << "Conflict signal in old solution: "
        << context.previousConflictSignalForIter << "\n";

  context.candidateValidationStats.precedenceViolations =
      candidatePhase.precedenceViolations;
  context.candidateValidationStats.vertexCollisions =
      candidatePhase.vertexCollisions;
  context.candidateValidationStats.edgeSwapCollisions =
      candidatePhase.edgeSwapCollisions;
  context.candidateValidationStats.structuralViolations =
      candidatePhase.structuralViolations;
  context.candidateValidationStats.precedenceDebt = candidatePhase.precedenceDebt;
  context.candidateValidationStats.precedencePairsChecked =
      candidatePhase.precedencePairsChecked;
  context.candidateValid = candidatePhase.candidateValid;
  context.candidateConflictSignal = candidatePhase.candidateConflictSignal;
  context.candidateNrrSoftCandidate = lns.lastNrrSoftCandidate_;
  context.candidateNrrSoftOnlyInvalid = lns.lastNrrSoftOnlyInvalid_;
  context.candidateNrrSoftConflictCount = lns.lastNrrSoftConflictCount_;
  context.softRecoveryModeBefore = lns.softRecoveryActive_;
  context.softRecoveryModeAfter = lns.softRecoveryActive_;
  context.softRecoveryDecisionReason = "none";
  context.debugRow.candidateConflictSignal = context.candidateConflictSignal;
  context.debugRow.nrrSoftCandidate = context.candidateNrrSoftCandidate;
  context.debugRow.nrrSoftOnlyInvalid = context.candidateNrrSoftOnlyInvalid;
  context.debugRow.nrrSoftConflictCount = context.candidateNrrSoftConflictCount;
  context.debugRow.softRecoveryModeBefore = context.softRecoveryModeBefore;
  context.debugRow.softRecoveryModeAfter = context.softRecoveryModeAfter;
  context.debugRow.softRecoveryDecisionReason = context.softRecoveryDecisionReason;
  if (context.candidateNrrSoftCandidate &&
      context.candidateNrrSoftConflictCount >= 0) {
    lns.improvementDiagnosticsStats_.softCandidateProduced++;
    lns.improvementDiagnosticsStats_.softCandidateConflictSum +=
        static_cast<double>(context.candidateNrrSoftConflictCount);
    lns.improvementDiagnosticsStats_.softCandidateConflictSamples++;
  }

  if (!context.candidateValid) {
    context.feasibleSolutionUpdated = false;
    context.debugRow.feasibleBestUpdate = false;
  } else {
    if (candidatePhase.feasibleBestUpdate) {
      context.quality = IterationQuality::bestSolutionYet;
      context.feasibleSolutionUpdated = true;
      context.debugRow.feasibleBestUpdate = true;
    } else {
      context.feasibleSolutionUpdated = false;
      context.debugRow.feasibleBestUpdate = false;
    }
  }

  context.proposedSocForIter = candidatePhase.proposedSoc;
  context.debugRow.candidateSoc = context.proposedSocForIter;
  lns.improvementDiagnosticsStats_.sumCandidateSoc +=
      static_cast<double>(context.proposedSocForIter);
  lns.improvementDiagnosticsStats_.sumPreviousConflictSignal +=
      static_cast<double>(context.previousConflictSignalForIter);
  lns.improvementDiagnosticsStats_.sumCandidateConflictSignal +=
      static_cast<double>(context.candidateConflictSignal);
  if (context.candidateConflictSignal < context.previousConflictSignalForIter) {
    lns.improvementDiagnosticsStats_.conflictSignalBetter++;
  } else if (context.candidateConflictSignal >
             context.previousConflictSignalForIter) {
    lns.improvementDiagnosticsStats_.conflictSignalWorse++;
  } else {
    lns.improvementDiagnosticsStats_.conflictSignalEqual++;
  }
  if (context.candidateValid) {
    context.debugRow.candidateValid = true;
    lns.improvementDiagnosticsStats_.candidateValid++;
    if (context.feasibleSolutionUpdated) {
      lns.improvementDiagnosticsStats_.feasibleBestUpdates++;
    } else {
      lns.improvementDiagnosticsStats_.feasibleNoBestUpdate++;
    }
  } else {
    context.debugRow.candidateValid = false;
    lns.improvementDiagnosticsStats_.candidateInvalid++;
  }
}
