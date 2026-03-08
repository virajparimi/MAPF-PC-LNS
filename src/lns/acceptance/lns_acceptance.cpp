#include "lns.hpp"
#include "lns_acceptance_policy.hpp"

bool LNS::simulatedAnnealing(
    const std::vector<int>& candidateTouchedAgents) {
  AcceptancePolicyContext ctx{rng_,
                              acceptanceState_.temperature,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              acceptanceState_.coolingCoefficient,
                              acceptanceState_.heatingCoefficient,
                              acceptanceState_.maxTemperature,
                              acceptanceState_.greatDelugeDecay};
  return AcceptancePolicy::simulatedAnnealing(
      ctx, [this, &candidateTouchedAgents]() {
    restoreSolutionFromPrevious(candidateTouchedAgents);
    PLOGD << "Rejecting this solution!\n";
  });
}

bool LNS::thresholdAcceptance(
    const std::vector<int>& candidateTouchedAgents) {
  AcceptancePolicyContext ctx{rng_,
                              acceptanceState_.temperature,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              acceptanceState_.coolingCoefficient,
                              acceptanceState_.heatingCoefficient,
                              acceptanceState_.maxTemperature,
                              acceptanceState_.greatDelugeDecay};
  return AcceptancePolicy::thresholdAcceptance(
      ctx, [this, &candidateTouchedAgents]() {
    restoreSolutionFromPrevious(candidateTouchedAgents);
    PLOGD << "Rejecting this solution!\n";
  });
}

bool LNS::oldBachelorsAcceptance(
    const std::vector<int>& candidateTouchedAgents) {
  AcceptancePolicyContext ctx{rng_,
                              acceptanceState_.temperature,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              acceptanceState_.coolingCoefficient,
                              acceptanceState_.heatingCoefficient,
                              acceptanceState_.maxTemperature,
                              acceptanceState_.greatDelugeDecay};
  return AcceptancePolicy::oldBachelorsAcceptance(
      ctx, [this, &candidateTouchedAgents]() {
    restoreSolutionFromPrevious(candidateTouchedAgents);
    PLOGD << "Rejecting this solution\n";
  });
}

bool LNS::greatDelugeAlgorithm(
    const std::vector<int>& candidateTouchedAgents) {
  AcceptancePolicyContext ctx{rng_,
                              acceptanceState_.temperature,
                              initialTemperature_,
                              solution_.utility,
                              previousSolution_.utility,
                              acceptanceState_.coolingCoefficient,
                              acceptanceState_.heatingCoefficient,
                              acceptanceState_.maxTemperature,
                              acceptanceState_.greatDelugeDecay};
  return AcceptancePolicy::greatDelugeAlgorithm(
      ctx, [this, &candidateTouchedAgents]() {
    restoreSolutionFromPrevious(candidateTouchedAgents);
    PLOGD << "Rejecting this solution\n";
  });
}
