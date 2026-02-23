#include "lns.hpp"
#include <cmath>
#include <limits>

bool LNS::simulatedAnnealing() {

  bool accepted = false;
  // Guard against degenerate temperatures to avoid NaN/inf behavior.
  if (!std::isfinite(temperature_) ||
      temperature_ <= std::numeric_limits<double>::epsilon()) {
    constexpr double kMinTemperature = 1e-9;
    const double fallbackTemperature =
        max(kMinTemperature, max(initialTemperature_, 1.0));
    temperature_ = fallbackTemperature;
    accepted = solution_.utility <= previousSolution_.utility;
    if (!accepted) {
      restoreSolutionFromPrevious();
      PLOGD << "Rejecting this solution!\n";
    }
    temperature_ = max(kMinTemperature, temperature_ * coolingCoefficient_);
    return accepted;
  }

  // Better (or equal) utility is always accepted.
  const double utilityDelta = solution_.utility - previousSolution_.utility;
  if (utilityDelta <= 0.0) {
    accepted = true;
    temperature_ *= coolingCoefficient_;
    return accepted;
  }

  // For worse moves, compute exp(delta/T) with a clamp to avoid under/overflow.
  const double exponent = (previousSolution_.utility - solution_.utility) /
                          temperature_;  // strictly negative here
  constexpr double kMinExpArg = -700.0;
  const double acceptanceProb = std::exp(std::max(exponent, kMinExpArg));
  std::uniform_real_distribution<double> unit01(0.0, 1.0);
  if (unit01(rng_) < acceptanceProb) {
    // Use simulated annealing to potentially accept worse solutions!
    accepted = true;
  } else {
    // Reject this solution
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution!\n";
  }
  temperature_ *= coolingCoefficient_;
  return accepted;
}

bool LNS::thresholdAcceptance() {

  bool accepted = false;
  // In this case we are worse than the previous solution but within some threshold so we can accept this one
  if (solution_.utility - previousSolution_.utility <= temperature_) {
    accepted = true;
  } else {
    // Reject this solution
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution!\n";
  }
  temperature_ *= coolingCoefficient_;
  return accepted;
}

bool LNS::oldBachelorsAcceptance() {

  bool accepted = false;
  if (solution_.utility - previousSolution_.utility < temperature_) {
    // Accept this solution and reduce the temperature
    temperature_ *= coolingCoefficient_;
    accepted = true;
  } else {
    // Reject this solution and increase the temperature
    restoreSolutionFromPrevious();
    const double reheated = temperature_ * heatingCoefficient_;
    temperature_ = std::min(reheated, maxTemperature_);
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}

bool LNS::greatDelugeAlgorithm() {

  bool accepted = false;
  if (solution_.utility - previousSolution_.utility < temperature_) {
    // Additive water-level decay (canonical Great Deluge shape) keeps cooling
    // progression stable across runtime and avoids multiplicative stalls.
    temperature_ = max(0.0, temperature_ - greatDelugeDecay_);
    accepted = true;
  } else {
    // Reject this solution but dont change the temperature'
    restoreSolutionFromPrevious();
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}
