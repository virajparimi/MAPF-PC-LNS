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
  switch (destroyHeuristicId) {
    case DestroyHeuristic::randomRemoval:
      lns.randomRemoval();
      return true;
    case DestroyHeuristic::worstRemoval:
      lns.worstRemoval();
      return true;
    case DestroyHeuristic::conflictRemoval:
      lns.conflictRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::shawRemoval:
      lns.shawRemoval(lns.neighborSize_ * 3);
      return true;
    case DestroyHeuristic::precedenceWaitRemoval:
      lns.precedenceWaitRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::lowSlackRemoval:
      lns.lowSlackRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::marketTatonnementRemoval:
      lns.marketTatonnementRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::collisionSoftRemoval:
      lns.collisionSoftRemoval(potentialNeighborhood);
      return true;
    case DestroyHeuristic::failureSoftRemoval:
      lns.failureSoftRemoval(potentialNeighborhood);
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
  alnsHeuristicForIter = -1;
  errorMessage.clear();
  if (lns.destroyHeuristic == "alns") {
    if (!lns.alnsRemoval(potentialNeighborhood)) {
      errorMessage = "ALNS sampling/execution failed";
      return false;
    }
    alnsHeuristicForIter = lns.adaptiveLNS_.recentDestroyHeuristic;
    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < lns.adaptiveLNS_.numDestroyHeuristics) {
      lns.adaptiveLNS_.selections[alnsHeuristicForIter]++;
    }
    return true;
  }

  const std::optional<int> heuristicId =
      DestroyOrchestrator::heuristicIdFromName(lns.destroyHeuristic);
  if (!heuristicId.has_value()) {
    errorMessage = "Unknown destroy heuristic: " + lns.destroyHeuristic;
    return false;
  }
  if (*heuristicId == DestroyHeuristic::marketTatonnementRemoval &&
      !lns.market_.heuristics) {
    errorMessage =
        "destroyHeuristic='market_tatonnement' requires market heuristics to "
        "be enabled";
    return false;
  }
  if (!executeHeuristic(lns, *heuristicId, potentialNeighborhood)) {
    errorMessage = "Failed to execute destroy heuristic id " +
                   std::to_string(*heuristicId);
    return false;
  }
  lns.lastDestroySampledInSoftMode_ = false;
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
  lastDestroySampledInSoftMode_ = false;
  const bool marketWarmupReady =
      !market_.heuristics || market_.destroyWarmupUpdates <= 0 ||
      market_.stats.updates >=
          static_cast<int64_t>(market_.destroyWarmupUpdates);
  const bool marketStableReady = marketDestroyStabilityReady();
  DestroySamplingContext ctx{
      adaptiveLNS_,
      rng_,
      alnsEnablePrecedenceAwareDestroy_,
      softRecoveryDestroyMode_,
      softRecoveryActive_,
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
  lastDestroySampledInSoftMode_ =
      softRecoveryDestroyMode_ && softRecoveryActive_;
  if (!DestroyOrchestrator::executeHeuristic(*this, sampledDestroyHeuristic,
                                             potentialNeighborhood)) {
    return false;
  }

  adaptiveLNS_.used[sampledDestroyHeuristic] += 1;
  adaptiveLNS_.destroyHeuristicHistory.push_back(sampledDestroyHeuristic);
  return true;
}
