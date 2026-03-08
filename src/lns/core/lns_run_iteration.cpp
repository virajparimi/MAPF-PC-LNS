#include "lns.hpp"
#include "lns_acceptance_orchestrator.hpp"
#include "lns_candidate_phase_orchestrator.hpp"
#include "lns_destroy_orchestrator.hpp"
#include "lns_iteration_diagnostics_orchestrator.hpp"
#include "lns_iteration_acceptance_bookkeeping_orchestrator.hpp"
#include "lns_iteration_candidate_postprocess_orchestrator.hpp"
#include "lns_iteration_context.hpp"
#include "lns_iteration_lifecycle_orchestrator.hpp"
#include "lns_iteration_outcome_orchestrator.hpp"
#include "lns_iteration_setup_orchestrator.hpp"
#include "lns_market_iteration_orchestrator.hpp"
#include "lns_repair_engine.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

bool LNS::runOneIteration(ConflictMap& potentialNeighborhood,
                          ConflictMap& oldNeighborhood,
                          MovingMetrics& metrics,
                          bool& currentSolutionValid,
                          ValidationStats& currentValidationStats,
                          bool& feasibleSolutionUpdated) {
  IterationExecutionContext context;
  IterationSetupOrchestrator::initialize(*this, currentValidationStats, context);
  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };
  auto commitIterationTiming = [&]() {
    IterationLifecycleOrchestrator::commitTimingAndMaybeRecord(*this, context);
  };
  Time::time_point phaseStart = Time::now();

  // Destroy phase populates lnsNeighborhood_.removedTasks.
  string destroyError;
  if (!DestroyOrchestrator::runPhase(*this, &potentialNeighborhood,
                                     context.alnsHeuristicForIter,
                                     destroyError)) {
    PLOGE << destroyError << "\n";
    context.debugRow.earlyAbortReason = "destroy_heuristic_error";
    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    commitIterationTiming();
    feasibleSolutionUpdated = context.feasibleSolutionUpdated;
    return false;
  }
  context.debugRow.destroyHeuristicId = context.alnsHeuristicForIter;
  context.debugRow.destroyHeuristicName =
      DestroyOrchestrator::heuristicNameFromId(context.alnsHeuristicForIter);
  context.debugRow.destroySelectedInSoftMode = softRecoveryState_.lastDestroySampledInSoftMode;
  if (context.debugRow.destroySelectedInSoftMode &&
      context.alnsHeuristicForIter >= 0 &&
      context.alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
    nrrStats_.softModeSelectionsByDestroy[context.alnsHeuristicForIter]++;
    improvementDiagnosticsStats_
        .softModeSelectionsByDestroy[context.alnsHeuristicForIter]++;
  }

  oldNeighborhood = lnsNeighborhood_.removedTasks;

  PLOGD << "Printing neighborhood conflict tasks\n";
  PLOGD << "Size: " << lnsNeighborhood_.removedTasks.size() << "\n";
  for (const auto& [_, conflictTask] : lnsNeighborhood_.removedTasks) {
    PLOGD << "Conflicted Task : " << conflictTask.task << "\n";
  }

  if (!prepareNextIteration()) {
    context.timeDestroyAndPrepareSec += elapsedSecSince(phaseStart);
    improvementDiagnosticsStats_.earlyAbortPrepare++;
    const bool cascadeAbort = cascadeState_.lastPrepareAborted;
    if (context.alnsHeuristicForIter >= 0 &&
        context.alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      if (cascadeAbort) {
        adaptiveLNS_.cascadeAborted[context.alnsHeuristicForIter]++;
      } else {
        adaptiveLNS_.couldNotFind[context.alnsHeuristicForIter]++;
      }
    }
    if (cascadeAbort) {
      PLOGD << "prepareNextIteration aborted due to cascade budget\n";
      const bool finalized = IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
          *this, "cascade_budget_abort", false, potentialNeighborhood,
          context);
      commitIterationTiming();
      feasibleSolutionUpdated = context.feasibleSolutionUpdated;
      return finalized;
    } else {
      const bool finalized = IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
          *this, "prepare_failed", true, potentialNeighborhood, context);
      commitIterationTiming();
      feasibleSolutionUpdated = context.feasibleSolutionUpdated;
      return finalized;
    }
  }
  context.timeDestroyAndPrepareSec += elapsedSecSince(phaseStart);
  phaseStart = Time::now();

  // This needs to happen after prepare iteration since we updated the conflictedTasks variable in the prepare next iteration function
  for (const auto& [_, conflictedTask] : lnsNeighborhood_.removedTasks) {
    if (conflictedTask.task >= 0 &&
        conflictedTask.task < (int)lnsNeighborhood_.committedTasks.size()) {
      lnsNeighborhood_.committedTasks[conflictedTask.task] = 0;
    }
  }
  lnsNeighborhood_.immutableRemovedTasks = lnsNeighborhood_.removedTasks;
  const NeighborhoodDiagnosticsResult neighborhoodDiagnostics =
      IterationDiagnosticsOrchestrator::analyzeNeighborhood(*this);
  context.debugRow.removedTasks = neighborhoodDiagnostics.removedTasks;
  context.debugRow.neighborhoodFingerprint =
      neighborhoodDiagnostics.neighborhoodFingerprint;
  context.debugRow.neighborhoodFingerprintSeenBefore =
      neighborhoodDiagnostics.neighborhoodFingerprintSeenBefore;
  context.debugRow.neighborhoodRepeatStreak =
      neighborhoodDiagnostics.neighborhoodRepeatStreak;
  context.debugRow.removedTaskIdsCsv = neighborhoodDiagnostics.removedTaskIdsCsv;
  context.debugRow.neighborhoodJaccardPrev =
      neighborhoodDiagnostics.neighborhoodJaccardPrev;
  // Collect agents impacted by this neighborhood once and reuse across:
  // 1) terminal-path invalidation, 2) path-join scope, 3) terminal replanning.
  vector<int> agentsToCompute = CandidatePhaseOrchestrator::collectImpactedAgents(
      instance_, solution_, previousSolution_, lnsNeighborhood_.immutableRemovedTasks);

  bool repairFailed = false;
  bool nrrRepairSucceeded = false;
  const double regretEvalBefore = cumulativeRegretCandidateEvalSec_;
  const double regretCommitBefore = cumulativeRegretCommitSec_;
  const double regretLowLevelBefore = cumulativeLowLevelSearchSec_;
  RepairEngine::run(*this, repairFailed, nrrRepairSucceeded);
  context.timeRegretCandidateEvalSec +=
      std::max(0.0, cumulativeRegretCandidateEvalSec_ - regretEvalBefore);
  context.timeRegretCommitSec +=
      std::max(0.0, cumulativeRegretCommitSec_ - regretCommitBefore);
  context.timeRegretLowLevelSec +=
      std::max(0.0, cumulativeLowLevelSearchSec_ - regretLowLevelBefore);

  context.timeRepairAndCommitSec += elapsedSecSince(phaseStart);

  // If we could not successfully compute the regrets and commit to all the tasks in the neighborhood then we need to reset this neighborhood!
  if (repairFailed || !lnsNeighborhood_.removedTasks.empty()) {
    improvementDiagnosticsStats_.earlyAbortRepair++;
    if (context.alnsHeuristicForIter >= 0 &&
        context.alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      adaptiveLNS_.couldNotFind[context.alnsHeuristicForIter]++;
    }
    PLOGD << "Could not find paths for the neighborhood! Attempting a new "
             "neighborhood computation\n";
    const bool finalized = IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
        *this, "repair_failed_or_incomplete", true, potentialNeighborhood,
        context);
    commitIterationTiming();
    feasibleSolutionUpdated = context.feasibleSolutionUpdated;
    return finalized;
  }

  const CandidatePhaseResult candidatePhase = CandidatePhaseOrchestrator::run(
      *this, agentsToCompute, context.alnsHeuristicForIter, potentialNeighborhood,
      metrics);
  context.timeJoinPathsSec += candidatePhase.timeJoinPathsSec;
  context.timeTerminalReplanSec += candidatePhase.timeTerminalReplanSec;
  context.timeRecomputeSocSec += candidatePhase.timeRecomputeSocSec;
  context.timeValidationSec += candidatePhase.timeValidationSec;
  context.candidateTouchedAgents = candidatePhase.candidateTouchedAgents;
  if (candidatePhase.status == CandidatePhaseStatus::join_failed) {
    improvementDiagnosticsStats_.earlyAbortJoin++;
    PLOGE << "run: failed to join agent paths for candidate solution\n";
    if (context.alnsHeuristicForIter >= 0 &&
        context.alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      adaptiveLNS_.couldNotFind[context.alnsHeuristicForIter]++;
    }
    const bool finalized = IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
        *this, "join_failed", true, potentialNeighborhood, context);
    commitIterationTiming();
    feasibleSolutionUpdated = context.feasibleSolutionUpdated;
    return finalized;
  }
  if (candidatePhase.status == CandidatePhaseStatus::terminal_failed) {
    improvementDiagnosticsStats_.earlyAbortTerminal++;
    PLOGE << "run: failed to replan terminal reposition paths for "
             "candidate solution\n";
    if (context.alnsHeuristicForIter >= 0 &&
        context.alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      adaptiveLNS_.couldNotFind[context.alnsHeuristicForIter]++;
    }
    const bool finalized = IterationOutcomeOrchestrator::finalizeCouldNotFindAbort(
        *this, "terminal_replan_failed", true, potentialNeighborhood, context);
    commitIterationTiming();
    feasibleSolutionUpdated = context.feasibleSolutionUpdated;
    return finalized;
  }

  IterationCandidatePostprocessOrchestrator::process(*this, candidatePhase,
                                                     context);
  const bool acceptanceAndBookkeepingOk =
      IterationAcceptanceBookkeepingOrchestrator::run(
          *this, nrrRepairSucceeded, potentialNeighborhood, oldNeighborhood,
          currentSolutionValid, currentValidationStats, context);
  if (!acceptanceAndBookkeepingOk) {
    commitIterationTiming();
    feasibleSolutionUpdated = context.feasibleSolutionUpdated;
    return false;
  }
  commitIterationTiming();
  feasibleSolutionUpdated = context.feasibleSolutionUpdated;
  return true;
}
