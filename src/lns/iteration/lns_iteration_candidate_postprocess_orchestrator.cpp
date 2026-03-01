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
  context.debugRow.candidateConflictSignal = context.candidateConflictSignal;

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
