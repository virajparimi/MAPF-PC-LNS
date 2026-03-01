#include "lns.hpp"
#include "lns_acceptance_policy.hpp"

bool LNS::simulatedAnnealing() {
  AcceptancePolicyContext ctx{rng_,
                              temperature_,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              coolingCoefficient_,
                              heatingCoefficient_,
                              maxTemperature_,
                              greatDelugeDecay_};
  return AcceptancePolicy::simulatedAnnealing(ctx, [this]() {
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution!\n";
  });
}

bool LNS::thresholdAcceptance() {
  AcceptancePolicyContext ctx{rng_,
                              temperature_,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              coolingCoefficient_,
                              heatingCoefficient_,
                              maxTemperature_,
                              greatDelugeDecay_};
  return AcceptancePolicy::thresholdAcceptance(ctx, [this]() {
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution!\n";
  });
}

bool LNS::oldBachelorsAcceptance() {
  AcceptancePolicyContext ctx{rng_,
                              temperature_,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              coolingCoefficient_,
                              heatingCoefficient_,
                              maxTemperature_,
                              greatDelugeDecay_};
  return AcceptancePolicy::oldBachelorsAcceptance(ctx, [this]() {
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution\n";
  });
}

bool LNS::greatDelugeAlgorithm() {
  AcceptancePolicyContext ctx{rng_,
                              temperature_,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              coolingCoefficient_,
                              heatingCoefficient_,
                              maxTemperature_,
                              greatDelugeDecay_};
  return AcceptancePolicy::greatDelugeAlgorithm(ctx, [this]() {
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution\n";
  });
}
