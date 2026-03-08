#include "lns_iteration_candidate_postprocess_orchestrator.hpp"

#include <algorithm>

void IterationCandidatePostprocessOrchestrator::process(
    LNS& lns, const CandidatePhaseResult& candidatePhase,
    IterationExecutionContext& context) {
  lns.processCandidatePhasePost(candidatePhase, context);
}

void LNS::processCandidatePhasePost(
    const CandidatePhaseResult& candidatePhase,
    IterationExecutionContext& context) {
  const int removedTasksChangedAgent = candidatePhase.removedTasksChangedAgent;
  const int removedTasksChangedOrder = candidatePhase.removedTasksChangedOrder;
  const int removedTasksUnchanged = candidatePhase.removedTasksUnchanged;
  context.debugRow.removedTasksChangedAgent = removedTasksChangedAgent;
  context.debugRow.removedTasksChangedOrder = removedTasksChangedOrder;
  context.debugRow.removedTasksUnchanged = removedTasksUnchanged;
  const int changedRemovedTasks = removedTasksChangedAgent + removedTasksChangedOrder;
  improvementDiagnosticsStats_.removedTasksChangedAgentTotal +=
      removedTasksChangedAgent;
  improvementDiagnosticsStats_.removedTasksChangedOrderTotal +=
      removedTasksChangedOrder;
  improvementDiagnosticsStats_.removedTasksUnchangedTotal +=
      removedTasksUnchanged;
  improvementDiagnosticsStats_.changedTasksPerNeighborhoodMax = std::max(
      improvementDiagnosticsStats_.changedTasksPerNeighborhoodMax,
      (int64_t)changedRemovedTasks);
  if (changedRemovedTasks > 0) {
    improvementDiagnosticsStats_.neighborhoodsWithChanges++;
  } else {
    improvementDiagnosticsStats_.neighborhoodsWithoutChanges++;
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
  context.candidateNrrSoftCandidate = lastNrrSoftCandidate_;
  context.candidateNrrSoftOnlyInvalid = lastNrrSoftOnlyInvalid_;
  context.candidateNrrSoftConflictCount = lastNrrSoftConflictCount_;
  context.softRecoveryModeBefore = acceptanceState_.softRecoveryActive;
  context.softRecoveryModeAfter = acceptanceState_.softRecoveryActive;
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
    improvementDiagnosticsStats_.softCandidateProduced++;
    improvementDiagnosticsStats_.softCandidateConflictSum +=
        static_cast<double>(context.candidateNrrSoftConflictCount);
    improvementDiagnosticsStats_.softCandidateConflictSamples++;
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
  improvementDiagnosticsStats_.sumCandidateSoc +=
      static_cast<double>(context.proposedSocForIter);
  improvementDiagnosticsStats_.sumPreviousConflictSignal +=
      static_cast<double>(context.previousConflictSignalForIter);
  improvementDiagnosticsStats_.sumCandidateConflictSignal +=
      static_cast<double>(context.candidateConflictSignal);
  if (context.candidateConflictSignal < context.previousConflictSignalForIter) {
    improvementDiagnosticsStats_.conflictSignalBetter++;
  } else if (context.candidateConflictSignal >
             context.previousConflictSignalForIter) {
    improvementDiagnosticsStats_.conflictSignalWorse++;
  } else {
    improvementDiagnosticsStats_.conflictSignalEqual++;
  }
  if (context.candidateValid) {
    context.debugRow.candidateValid = true;
    improvementDiagnosticsStats_.candidateValid++;
    if (context.feasibleSolutionUpdated) {
      improvementDiagnosticsStats_.feasibleBestUpdates++;
    } else {
      improvementDiagnosticsStats_.feasibleNoBestUpdate++;
    }
  } else {
    context.debugRow.candidateValid = false;
    improvementDiagnosticsStats_.candidateInvalid++;
  }
}
