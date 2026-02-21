#include "lns.hpp"
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


void LNS::alnsRemoval(const ConflictMap* potentialNeighborhood) {

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
      case bestSolutionYet:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta1;
        break;
      case improvedSolution:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta2;
        break;
      case downgradedButAccepted:
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
  // Sample the destroy heuristic and extract the neighborhood.
  // When market heuristics are disabled, exclude marketTatonnementRemoval from
  // ALNS sampling entirely.
  const bool marketWarmupReady =
      !market_.heuristics || market_.destroyWarmupUpdates <= 0 ||
      market_.stats.updates >=
          static_cast<int64_t>(market_.destroyWarmupUpdates);
  const bool marketStableReady = marketDestroyStabilityReady();
  const bool marketNotReady = (!marketWarmupReady || !marketStableReady);
  if (market_.heuristics && marketNotReady) {
    if (!marketWarmupReady) {
      market_.stats.destroyWarmupSkipped++;
    } else {
      market_.stats.destroyUnstableSkipped++;
    }
  }

  vector<int> eligibleHeuristics;
  eligibleHeuristics.reserve(adaptiveLNS_.numDestroyHeuristics);
  for (int i = 0; i < adaptiveLNS_.numDestroyHeuristics; i++) {
    if (!alnsEnablePrecedenceAwareDestroy_ &&
        (i == DestroyHeuristic::precedenceWaitRemoval ||
         i == DestroyHeuristic::lowSlackRemoval)) {
      continue;
    }
    if (i == DestroyHeuristic::marketTatonnementRemoval) {
      if (!market_.heuristics) {
        continue;
      }
      if (!market_.destroySoftGate &&
          (!marketWarmupReady || !marketStableReady)) {
        continue;
      }
    }
    eligibleHeuristics.push_back(i);
  }
  if (eligibleHeuristics.empty()) {
    PLOGE << "ALNS has no eligible destroy heuristics to sample\n";
    assert(false);
    return;
  }

  constexpr double kSuccessEpsilon = 1e-12;
  constexpr double kMinUsedBeforeSuppression = 8.0;
  bool hasPositiveRecentSuccess = false;
  for (int i : eligibleHeuristics) {
    if (adaptiveLNS_.success[i] > kSuccessEpsilon) {
      hasPositiveRecentSuccess = true;
      break;
    }
  }

  vector<double> eligibleWeights;
  eligibleWeights.reserve(eligibleHeuristics.size());
  for (int i : eligibleHeuristics) {
    double effectiveWeight = max(0.0, adaptiveLNS_.weights[i]);
    const bool suppressZeroSuccessHeuristic =
        hasPositiveRecentSuccess &&
        adaptiveLNS_.used[i] >= kMinUsedBeforeSuppression &&
        adaptiveLNS_.success[i] <= kSuccessEpsilon &&
        !(i == DestroyHeuristic::marketTatonnementRemoval &&
          market_.destroySoftGate);
    if (suppressZeroSuccessHeuristic) {
      effectiveWeight = 0.0;
    }
    if (i == DestroyHeuristic::marketTatonnementRemoval &&
        market_.destroySoftGate && marketNotReady) {
      if (!marketWarmupReady) {
        effectiveWeight *= market_.destroyWarmupWeightScale;
      }
      if (!marketStableReady) {
        effectiveWeight *= market_.destroyUnstableWeightScale;
      }
      effectiveWeight = max(effectiveWeight, market_.destroyMinAlnsWeight);
    }
    eligibleWeights.push_back(effectiveWeight);
  }

  int sampledDestroyHeuristic = eligibleHeuristics.front();
  double weightSum = 0.0;
  for (double w : eligibleWeights) {
    weightSum += w;
  }
  if (weightSum <= std::numeric_limits<double>::epsilon()) {
    vector<int> fallbackHeuristics;
    if (hasPositiveRecentSuccess) {
      for (int i : eligibleHeuristics) {
        if (adaptiveLNS_.success[i] > kSuccessEpsilon) {
          fallbackHeuristics.push_back(i);
        }
      }
    }
    if (fallbackHeuristics.empty()) {
      fallbackHeuristics = eligibleHeuristics;
    }
    std::uniform_int_distribution<int> distribution(
        0, (int)fallbackHeuristics.size() - 1);
    sampledDestroyHeuristic = fallbackHeuristics[distribution(rng_)];
  } else {
    std::discrete_distribution<> distribution(eligibleWeights.begin(),
                                              eligibleWeights.end());
    sampledDestroyHeuristic =
        eligibleHeuristics[distribution(rng_)];
  }
  adaptiveLNS_.recentDestroyHeuristic = sampledDestroyHeuristic;
  switch (sampledDestroyHeuristic) {
    case DestroyHeuristic::randomRemoval:  // RANDOM
      randomRemoval();
      break;
    case DestroyHeuristic::worstRemoval:  // WORST
      worstRemoval();
      break;
    case DestroyHeuristic::conflictRemoval:  // CONFLICT
      conflictRemoval(potentialNeighborhood);
      break;
    case DestroyHeuristic::shawRemoval:  // SHAW
      shawRemoval(neighborSize_ * 3);
      break;
    case DestroyHeuristic::precedenceWaitRemoval:  // PRECEDENCE WAIT
      precedenceWaitRemoval(potentialNeighborhood);
      break;
    case DestroyHeuristic::lowSlackRemoval:  // LOW SLACK
      lowSlackRemoval(potentialNeighborhood);
      break;
    case DestroyHeuristic::marketTatonnementRemoval:  // MARKET TATONNEMENT
      marketTatonnementRemoval(potentialNeighborhood);
      break;
    default:
      PLOGE << "Sampled a non-existent destroy heuristic: "
            << sampledDestroyHeuristic << "\n";
      assert(false);
      return;
  }

  adaptiveLNS_.used[sampledDestroyHeuristic] += 1;
  adaptiveLNS_.destroyHeuristicHistory.push_back(sampledDestroyHeuristic);
}
