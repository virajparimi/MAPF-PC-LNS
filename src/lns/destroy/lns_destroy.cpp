#include "lns.hpp"
#include "lns_destroy_orchestrator.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"

void LNS::clearNeighborhood() {
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.committedTasks.assign(instance_.getTasksNum(), -1);
  lnsNeighborhood_.removedTasksPathSize.assign(instance_.getTasksNum(), -1);
  lnsNeighborhood_.removedTasks.clear();
  lnsNeighborhood_.immutableRemovedTasks.clear();
}

bool DestroyOrchestrator::executeHeuristic(
    LNS& lns, int destroyHeuristicId,
    const ConflictMap* potentialNeighborhood) {
  return lns.executeDestroyHeuristic(destroyHeuristicId, potentialNeighborhood);
}

bool LNS::executeDestroyHeuristic(
    int destroyHeuristicId, const ConflictMap* potentialNeighborhood) {
  switch (destroyHeuristicId) {
    case DestroyHeuristic::randomRemoval:
      randomRemoval();
      return true;
    case DestroyHeuristic::worstRemoval:
      worstRemoval();
      return true;
    case DestroyHeuristic::conflictRemoval:
      conflictRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::shawRemoval:
      shawRemoval(neighborSize_ * 3);
      return true;
    case DestroyHeuristic::precedenceWaitRemoval:
      precedenceWaitRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::lowSlackRemoval:
      lowSlackRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::marketTatonnementRemoval:
      marketTatonnementRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::collisionSoftRemoval:
      collisionSoftRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::failureSoftRemoval:
      failureSoftRemoval(potentialNeighborhood);
      return true;
    default:
      PLOGE << "Sampled a non-existent destroy heuristic: "
            << destroyHeuristicId << "\n";
      return false;
  }
}

bool DestroyOrchestrator::runPhase(LNS& lns,
                                   const ConflictMap* potentialNeighborhood,
                                   int& alnsHeuristicForIter,
                                   std::string& errorMessage) {
  return lns.runDestroyPhase(potentialNeighborhood, alnsHeuristicForIter,
                             errorMessage);
}

bool LNS::runDestroyPhase(const ConflictMap* potentialNeighborhood,
                          int& alnsHeuristicForIter,
                          std::string& errorMessage) {
  alnsHeuristicForIter = -1;
  errorMessage.clear();
  if (destroyHeuristic == "alns") {
    if (!alnsRemoval(potentialNeighborhood)) {
      errorMessage = "ALNS sampling/execution failed";
      return false;
    }
    alnsHeuristicForIter = adaptiveLNS_.recentDestroyHeuristic;
    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      adaptiveLNS_.selections[alnsHeuristicForIter]++;
    }
    return true;
  }

  const std::optional<int> heuristicId =
      DestroyOrchestrator::heuristicIdFromName(destroyHeuristic);
  if (!heuristicId.has_value()) {
    errorMessage = "Unknown destroy heuristic: " + destroyHeuristic;
    return false;
  }
  if (*heuristicId == DestroyHeuristic::marketTatonnementRemoval &&
      !market_.heuristics) {
    errorMessage =
        "destroyHeuristic='market_tatonnement' requires market heuristics to "
        "be enabled";
    return false;
  }
  if (!executeDestroyHeuristic(*heuristicId, potentialNeighborhood)) {
    errorMessage = "Failed to execute destroy heuristic id " +
                   std::to_string(*heuristicId);
    return false;
  }
  softRecoveryState_.lastDestroySampledInSoftMode = false;
  return true;
}

bool LNS::alnsRemoval(const ConflictMap* potentialNeighborhood) {

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  adaptiveLNS_.alnsCounter++;

  // Cannot update ALNS scores before a destroy heuristic has been sampled at
  // least once in this run.
  if (iterationStats.size() > 1 &&
      !adaptiveLNS_.destroyHeuristicHistory.empty()) {
    // Incorporate the results of the heuristic performance in the last iteration
    switch (iterationStats.back().quality) {
      case IterationQuality::bestSolutionYet:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta1;
        break;
      case IterationQuality::improvedSolution:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta2;
        break;
      case IterationQuality::downgradedButAccepted:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta3;
        break;
      default:
        break;
    }
  }

  if (adaptiveLNS_.alnsCounter >= adaptiveLNS_.alnsCounterThreshold) {
    // Need to update the weights here!
    for (int i = 0; i < adaptiveLNS_.numDestroyHeuristics; i++) {
      if (adaptiveLNS_.used[i] > 0) {
        adaptiveLNS_.weights[i] =
            (1.0 - adaptiveLNS_.reactionFactor) * adaptiveLNS_.weights[i] +
            adaptiveLNS_.reactionFactor *
                (adaptiveLNS_.success[i] / adaptiveLNS_.used[i]);
      } else {
        adaptiveLNS_.weights[i] =
            (1.0 - adaptiveLNS_.reactionFactor) * adaptiveLNS_.weights[i];
      }
      adaptiveLNS_.used[i] = 0;
      adaptiveLNS_.success[i] = 0;
    }
    adaptiveLNS_.alnsCounter = 0;
  }
  softRecoveryState_.lastDestroySampledInSoftMode = false;
  const bool marketWarmupReady =
      !market_.heuristics || market_.destroyWarmupUpdates <= 0 ||
      market_.stats.updates >=
          static_cast<int64_t>(market_.destroyWarmupUpdates);
  const bool marketStableReady = marketDestroyStabilityReady();
  DestroySamplingContext ctx{
      adaptiveLNS_,
      rng_,
      alnsEnablePrecedenceAwareDestroy_,
      softRecoveryState_.destroyMode,
      acceptanceState_.softRecoveryActive,
      market_.heuristics,
      market_.destroySoftGate,
      marketWarmupReady,
      marketStableReady,
      market_.destroyWarmupWeightScale,
      market_.destroyUnstableWeightScale,
      market_.destroyMinAlnsWeight,
      market_.stats.destroyWarmupSkipped,
      market_.stats.destroyUnstableSkipped};
  const std::optional<int> sampledDestroyHeuristicOpt =
      DestroyOrchestrator::sampleAlnsHeuristic(ctx);
  if (!sampledDestroyHeuristicOpt.has_value()) {
    PLOGE << "ALNS has no eligible destroy heuristics to sample\n";
    return false;
  }
  const int sampledDestroyHeuristic = *sampledDestroyHeuristicOpt;
  adaptiveLNS_.recentDestroyHeuristic = sampledDestroyHeuristic;
  softRecoveryState_.lastDestroySampledInSoftMode =
      softRecoveryState_.destroyMode && acceptanceState_.softRecoveryActive;
  if (!executeDestroyHeuristic(sampledDestroyHeuristic, potentialNeighborhood)) {
    return false;
  }

  adaptiveLNS_.used[sampledDestroyHeuristic] += 1;
  adaptiveLNS_.destroyHeuristicHistory.push_back(sampledDestroyHeuristic);
  return true;
}
