#pragma once

#include <functional>
#include <random>

struct AcceptancePolicyContext {
  std::mt19937& rng;
  double& temperature;
  double initialTemperature;
  double currentUtility;
  double previousUtility;
  double coolingCoefficient;
  double heatingCoefficient;
  double maxTemperature;
  double greatDelugeDecay;
};

class AcceptancePolicy {
 public:
  using RejectFn = std::function<void()>;

  static bool simulatedAnnealing(AcceptancePolicyContext& ctx,
                                 const RejectFn& onReject);
  static bool thresholdAcceptance(AcceptancePolicyContext& ctx,
                                  const RejectFn& onReject);
  static bool oldBachelorsAcceptance(AcceptancePolicyContext& ctx,
                                     const RejectFn& onReject);
  static bool greatDelugeAlgorithm(AcceptancePolicyContext& ctx,
                                   const RejectFn& onReject);
};
