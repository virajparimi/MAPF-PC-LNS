#include "lns_acceptance_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

bool AcceptancePolicy::simulatedAnnealing(AcceptancePolicyContext& ctx,
                                          const RejectFn& onReject) {
  bool accepted = false;
  if (!std::isfinite(ctx.temperature) ||
      ctx.temperature <= std::numeric_limits<double>::epsilon()) {
    constexpr double kMinTemperature = 1e-9;
    const double fallbackTemperature =
        std::max(kMinTemperature, std::max(ctx.initialTemperature, 1.0));
    ctx.temperature = fallbackTemperature;
    accepted = ctx.currentUtility <= ctx.previousUtility;
    if (!accepted && onReject) {
      onReject();
    }
    ctx.temperature = std::max(kMinTemperature, ctx.temperature * ctx.coolingCoefficient);
    return accepted;
  }

  const double utilityDelta = ctx.currentUtility - ctx.previousUtility;
  if (utilityDelta <= 0.0) {
    accepted = true;
    ctx.temperature *= ctx.coolingCoefficient;
    return accepted;
  }

  const double exponent =
      (ctx.previousUtility - ctx.currentUtility) / ctx.temperature;
  constexpr double kMinExpArg = -700.0;
  const double acceptanceProb = std::exp(std::max(exponent, kMinExpArg));
  std::uniform_real_distribution<double> unit01(0.0, 1.0);
  if (unit01(ctx.rng) < acceptanceProb) {
    accepted = true;
  } else if (onReject) {
    onReject();
  }
  ctx.temperature *= ctx.coolingCoefficient;
  return accepted;
}

bool AcceptancePolicy::thresholdAcceptance(AcceptancePolicyContext& ctx,
                                           const RejectFn& onReject) {
  const bool accepted =
      (ctx.currentUtility - ctx.previousUtility <= ctx.temperature);
  if (!accepted && onReject) {
    onReject();
  }
  ctx.temperature *= ctx.coolingCoefficient;
  return accepted;
}

bool AcceptancePolicy::oldBachelorsAcceptance(AcceptancePolicyContext& ctx,
                                              const RejectFn& onReject) {
  const bool accepted =
      (ctx.currentUtility - ctx.previousUtility < ctx.temperature);
  if (accepted) {
    ctx.temperature *= ctx.coolingCoefficient;
    return true;
  }
  if (onReject) {
    onReject();
  }
  const double reheated = ctx.temperature * ctx.heatingCoefficient;
  ctx.temperature = std::min(reheated, ctx.maxTemperature);
  return false;
}

bool AcceptancePolicy::greatDelugeAlgorithm(AcceptancePolicyContext& ctx,
                                            const RejectFn& onReject) {
  const bool accepted =
      (ctx.currentUtility - ctx.previousUtility < ctx.temperature);
  if (accepted) {
    ctx.temperature = std::max(0.0, ctx.temperature - ctx.greatDelugeDecay);
    return true;
  }
  if (onReject) {
    onReject();
  }
  return false;
}
