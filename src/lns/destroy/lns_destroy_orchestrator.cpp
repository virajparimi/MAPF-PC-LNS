#include "lns_destroy_orchestrator.hpp"

#include <algorithm>
#include <limits>
#include <vector>

std::optional<int> DestroyOrchestrator::sampleAlnsHeuristic(
    DestroySamplingContext& ctx) {
  const bool marketNotReady = (!ctx.marketWarmupReady || !ctx.marketStableReady);
  if (ctx.marketHeuristicsEnabled && marketNotReady) {
    if (!ctx.marketWarmupReady) {
      ctx.marketDestroyWarmupSkipped++;
    } else {
      ctx.marketDestroyUnstableSkipped++;
    }
  }

  std::vector<int> eligibleHeuristics;
  eligibleHeuristics.reserve(ctx.adaptiveLns.numDestroyHeuristics);
  for (int i = 0; i < ctx.adaptiveLns.numDestroyHeuristics; i++) {
    if (!ctx.alnsEnablePrecedenceAwareDestroy &&
        (i == DestroyHeuristic::precedenceWaitRemoval ||
         i == DestroyHeuristic::lowSlackRemoval)) {
      continue;
    }
    if (i == DestroyHeuristic::marketTatonnementRemoval) {
      if (!ctx.marketHeuristicsEnabled) {
        continue;
      }
      if (!ctx.marketDestroySoftGate &&
          (!ctx.marketWarmupReady || !ctx.marketStableReady)) {
        continue;
      }
    }
    eligibleHeuristics.push_back(i);
  }
  if (eligibleHeuristics.empty()) {
    return std::nullopt;
  }

  constexpr double kSuccessEpsilon = 1e-12;
  constexpr double kMinUsedBeforeSuppression = 8.0;
  bool hasPositiveRecentSuccess = false;
  for (int i : eligibleHeuristics) {
    if (ctx.adaptiveLns.success[i] > kSuccessEpsilon) {
      hasPositiveRecentSuccess = true;
      break;
    }
  }

  std::vector<double> eligibleWeights;
  eligibleWeights.reserve(eligibleHeuristics.size());
  for (int i : eligibleHeuristics) {
    double effectiveWeight = std::max(0.0, ctx.adaptiveLns.weights[i]);
    const bool suppressZeroSuccessHeuristic =
        hasPositiveRecentSuccess &&
        ctx.adaptiveLns.used[i] >= kMinUsedBeforeSuppression &&
        ctx.adaptiveLns.success[i] <= kSuccessEpsilon &&
        !(i == DestroyHeuristic::marketTatonnementRemoval &&
          ctx.marketDestroySoftGate);
    if (suppressZeroSuccessHeuristic) {
      effectiveWeight = 0.0;
    }
    if (i == DestroyHeuristic::marketTatonnementRemoval &&
        ctx.marketDestroySoftGate && marketNotReady) {
      if (!ctx.marketWarmupReady) {
        effectiveWeight *= ctx.marketDestroyWarmupWeightScale;
      }
      if (!ctx.marketStableReady) {
        effectiveWeight *= ctx.marketDestroyUnstableWeightScale;
      }
      effectiveWeight = std::max(effectiveWeight, ctx.marketDestroyMinAlnsWeight);
    }
    eligibleWeights.push_back(effectiveWeight);
  }

  int sampledDestroyHeuristic = eligibleHeuristics.front();
  double weightSum = 0.0;
  for (double w : eligibleWeights) {
    weightSum += w;
  }
  if (weightSum <= std::numeric_limits<double>::epsilon()) {
    std::vector<int> fallbackHeuristics;
    if (hasPositiveRecentSuccess) {
      for (int i : eligibleHeuristics) {
        if (ctx.adaptiveLns.success[i] > kSuccessEpsilon) {
          fallbackHeuristics.push_back(i);
        }
      }
    }
    if (fallbackHeuristics.empty()) {
      fallbackHeuristics = eligibleHeuristics;
    }
    std::uniform_int_distribution<int> distribution(
        0, static_cast<int>(fallbackHeuristics.size()) - 1);
    sampledDestroyHeuristic = fallbackHeuristics[distribution(ctx.rng)];
  } else {
    std::discrete_distribution<> distribution(eligibleWeights.begin(),
                                              eligibleWeights.end());
    sampledDestroyHeuristic =
        eligibleHeuristics[distribution(ctx.rng)];
  }
  return sampledDestroyHeuristic;
}

std::optional<int> DestroyOrchestrator::heuristicIdFromName(
    const std::string& name) {
  if (name == "random") {
    return DestroyHeuristic::randomRemoval;
  }
  if (name == "worst") {
    return DestroyHeuristic::worstRemoval;
  }
  if (name == "conflict") {
    return DestroyHeuristic::conflictRemoval;
  }
  if (name == "shaw") {
    return DestroyHeuristic::shawRemoval;
  }
  if (name == "precedence_wait") {
    return DestroyHeuristic::precedenceWaitRemoval;
  }
  if (name == "low_slack") {
    return DestroyHeuristic::lowSlackRemoval;
  }
  if (name == "market_tatonnement") {
    return DestroyHeuristic::marketTatonnementRemoval;
  }
  return std::nullopt;
}
