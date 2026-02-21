#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"
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
      restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
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
    restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
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
    restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
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
    restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
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
    restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}

bool LNS::acceptScoreWithCurrentCriterion(
    double candidateScore, double previousScore, double* temperatureOverride,
    double* initialTemperatureOverride, double* maxTemperatureOverride,
    double* greatDelugeDecayOverride) {
  constexpr double kMinTemperature = 1e-9;
  double& activeTemperature =
      (temperatureOverride != nullptr) ? *temperatureOverride : temperature_;
  const double activeInitialTemperature =
      (initialTemperatureOverride != nullptr) ? *initialTemperatureOverride
                                              : initialTemperature_;
  double& activeMaxTemperature =
      (maxTemperatureOverride != nullptr) ? *maxTemperatureOverride
                                          : maxTemperature_;
  double& activeGreatDelugeDecay =
      (greatDelugeDecayOverride != nullptr) ? *greatDelugeDecayOverride
                                            : greatDelugeDecay_;

  if (acceptanceCriteria == "SA") {
    if (!std::isfinite(activeTemperature) ||
        activeTemperature <= std::numeric_limits<double>::epsilon()) {
      const double fallbackTemperature =
          max(kMinTemperature, max(activeInitialTemperature, 1.0));
      activeTemperature = fallbackTemperature;
      const bool accepted = candidateScore <= previousScore;
      activeTemperature =
          max(kMinTemperature, activeTemperature * coolingCoefficient_);
      return accepted;
    }

    if (candidateScore <= previousScore) {
      activeTemperature *= coolingCoefficient_;
      return true;
    }

    const double exponent = (previousScore - candidateScore) / activeTemperature;
    constexpr double kMinExpArg = -700.0;
    const double acceptanceProb = std::exp(std::max(exponent, kMinExpArg));
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    const bool accepted = unit01(rng_) < acceptanceProb;
    activeTemperature *= coolingCoefficient_;
    return accepted;
  }

  if (acceptanceCriteria == "TA") {
    const bool accepted = (candidateScore - previousScore) <= activeTemperature;
    activeTemperature *= coolingCoefficient_;
    return accepted;
  }

  if (acceptanceCriteria == "OBA") {
    if ((candidateScore - previousScore) < activeTemperature) {
      activeTemperature *= coolingCoefficient_;
      return true;
    }
    const double reheated = activeTemperature * heatingCoefficient_;
    activeTemperature = std::min(reheated, activeMaxTemperature);
    return false;
  }

  if (acceptanceCriteria == "GDA") {
    if ((candidateScore - previousScore) < activeTemperature) {
      activeTemperature = max(0.0, activeTemperature - activeGreatDelugeDecay);
      return true;
    }
    return false;
  }

  PLOGE << "Unknown acceptance criteria: " << acceptanceCriteria << "\n";
  return false;
}

void LNS::ensureInvalidAcceptanceTemperatureInitialized(double previousScore,
                                                        double candidateScore) {
  constexpr double kMinTemperature = 1e-9;
  const bool initialized =
      std::isfinite(invalidTemperature_) &&
      invalidTemperature_ > std::numeric_limits<double>::epsilon();
  if (invalidTemperatureInitialized_ && initialized) {
    return;
  }

  const double absPrevious = std::abs(previousScore);
  const double absCandidate = std::abs(candidateScore);
  const double absDelta = std::abs(candidateScore - previousScore);
  const double scoreScale = max(1.0, max(absPrevious, max(absCandidate, absDelta)));
  const double minTemperature =
      max(kMinTemperature, acceptanceInvalidTemperatureFloor_);
  double initTemperature = acceptanceInvalidTemperatureScale_ * scoreScale;
  if (!std::isfinite(initTemperature) || initTemperature < minTemperature) {
    initTemperature = minTemperature;
  }
  if (acceptanceCriteria == "SA") {
    initTemperature /= log(2);
  }
  invalidTemperature_ = max(minTemperature, initTemperature);
  invalidInitialTemperature_ = invalidTemperature_;
  invalidMaxTemperature_ = max(invalidInitialTemperature_, 1.0) * 1000.0;
  if (numOfIterations_ > 0) {
    invalidGreatDelugeDecay_ = invalidInitialTemperature_ / max(1, numOfIterations_);
  } else {
    invalidGreatDelugeDecay_ = invalidInitialTemperature_ / 1000.0;
  }
  if (!std::isfinite(invalidGreatDelugeDecay_) ||
      invalidGreatDelugeDecay_ < 0.0) {
    invalidGreatDelugeDecay_ = 0.0;
  }
  invalidTemperatureInitialized_ = true;
  acceptanceDiagnostics_.invalidDedicatedTempInitCount++;
  acceptanceDiagnostics_.invalidDedicatedInitTempSum += invalidInitialTemperature_;
}

double LNS::computeInvalidAcceptanceScore(
    const ValidationStats& selfStats, int selfSoc,
    const ValidationStats& peerStats, int peerSoc, double* spatialNorm,
    double* precedenceDebtNorm, double* socNorm) const {
  const double selfSpatial =
      static_cast<double>(selfStats.spatialConflictEvents());
  const double peerSpatial =
      static_cast<double>(peerStats.spatialConflictEvents());
  const double spatialScale = max(1.0, max(selfSpatial, peerSpatial));
  const double selfSpatialNorm = selfSpatial / spatialScale;

  const double selfDebt = static_cast<double>(selfStats.precedenceDebt);
  const double peerDebt = static_cast<double>(peerStats.precedenceDebt);
  const double debtScale = max(1.0, max(selfDebt, peerDebt));
  const double selfDebtNorm = selfDebt / debtScale;

  const double selfSocAbs = std::abs(static_cast<double>(selfSoc));
  const double peerSocAbs = std::abs(static_cast<double>(peerSoc));
  const double socScale = max(1.0, max(selfSocAbs, peerSocAbs));
  const double selfSocNorm = selfSocAbs / socScale;

  if (spatialNorm != nullptr) {
    *spatialNorm = selfSpatialNorm;
  }
  if (precedenceDebtNorm != nullptr) {
    *precedenceDebtNorm = selfDebtNorm;
  }
  if (socNorm != nullptr) {
    *socNorm = selfSocNorm;
  }

  return acceptanceInvalidSpatialWeight_ * selfSpatialNorm +
         acceptanceInvalidPrecedenceDebtWeight_ * selfDebtNorm +
         acceptanceInvalidSocTieBreakWeight_ * selfSocNorm;
}

