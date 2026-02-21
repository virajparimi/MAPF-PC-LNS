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

bool LNS::run() {
  invalidCandidateRejections = 0;
  marketGuardRejections = 0;
  acceptanceDiagnostics_ = AcceptanceDiagnostics{};
  invalidTemperatureInitialized_ = false;
  invalidTemperature_ = 0.0;
  invalidInitialTemperature_ = 0.0;
  invalidMaxTemperature_ = std::numeric_limits<double>::infinity();
  invalidGreatDelugeDecay_ = 0.0;

  auto runInitialSolutionStrategy =
      [&](const string& strategy,
          std::optional<double> armBudgetSec = std::nullopt) -> bool {
    if (strategy == "greedy") {
      // Run the greedy task assignment and subsequent path finding algorithm.
      return buildGreedySolution();
    }
    if (strategy == "prioritized") {
      // Plan tasks in global topological order with reservations from already
      // planned task segments.
      return buildPrioritizedInitialSolution();
    }
    if (strategy == "greedy_precedence_only") {
      // Precedence-feasible, collision-infeasible warm start.
      return buildGreedySolutionPrecedenceOnly();
    }
    if (strategy.find("sota") != string::npos) {
      // Run the greedy task assignment and use CBS-PC for finding agent paths.
      int mapfPcTimeoutSec = 120;
      if (armBudgetSec.has_value() && std::isfinite(*armBudgetSec) &&
          *armBudgetSec > 0.0) {
        mapfPcTimeoutSec = max(1, (int)std::ceil(*armBudgetSec));
      }
      return buildGreedySolutionWithMAPFPC(strategy, mapfPcTimeoutSec);
    }
    PLOGE << "Unknown initial solution strategy '" << strategy << "'\n";
    return false;
  };

  auto runInitialSolutionStrategyWithBudget =
      [&](const string& strategy,
          std::optional<double> armBudgetSec = std::nullopt) -> bool {
    const double savedTimeLimit = timeLimit_;
    if (armBudgetSec.has_value() && std::isfinite(*armBudgetSec) &&
        *armBudgetSec > 0.0) {
      const double elapsedBeforeArm = elapsedRuntimeSec();
      const double armDeadline = elapsedBeforeArm + *armBudgetSec;
      timeLimit_ = min(savedTimeLimit, armDeadline);
    }
    const bool success = runInitialSolutionStrategy(strategy, armBudgetSec);
    timeLimit_ = savedTimeLimit;
    return success;
  };

  struct InitialCheckpoint {
    double runtimeSec = 0.0;
    string label;
    int soc = 0;
    bool feasible = false;
    IterationQuality quality = IterationQuality::none;
  };
  vector<InitialCheckpoint> initialCheckpoints;

  initialSolutionRequested_ = initialSolutionStrategy;
  initialSolutionEffective_ = initialSolutionStrategy;
  initialSolutionFallbackUsed_ = false;
  initialSolutionFallbackReason_ = "none";

  bool success = false;
  if (initialSolutionStrategy == "portfolio") {
    // Anytime-safe portfolio: fixed arm order, fixed budget from cutoff.
    const vector<string> portfolioArms = {"prioritized", "sota_pbs",
                                          "sota_cbs"};
    const double remainingBudget = remainingRuntimeBudgetSec();
    double portfolioBudgetSec =
        std::min(remainingBudget,
                 max(0.0, timeLimit_ * initialPortfolioTimeFraction_));
    if (portfolioBudgetSec <= 0.0 && remainingBudget > 0.0) {
      // Ensure at least one short arm when portfolio is explicitly requested.
      portfolioBudgetSec =
          std::min(remainingBudget, initialPortfolioMinArmTimeSec_);
    }

    bool haveBestPortfolioSolution = false;
    int bestPortfolioSoc = std::numeric_limits<int>::max();
    string bestPortfolioArm;
    Solution bestPortfolioSolution(instance_);

    for (int i = 0; i < (int)portfolioArms.size(); i++) {
      if (runtimeBudgetExhausted() || portfolioBudgetSec <= 0.0) {
        break;
      }
      const int armsLeft = (int)portfolioArms.size() - i;
      const double fairShare = portfolioBudgetSec / max(1, armsLeft);
      double armBudgetSec =
          std::max(initialPortfolioMinArmTimeSec_, fairShare);
      armBudgetSec = std::min(armBudgetSec, portfolioBudgetSec);
      armBudgetSec = std::min(armBudgetSec, remainingRuntimeBudgetSec());
      if (armBudgetSec <= 0.0) {
        break;
      }

      const string& arm = portfolioArms[i];
      const double armStartSec = elapsedRuntimeSec();
      const bool armSuccess =
          runInitialSolutionStrategyWithBudget(arm, armBudgetSec);
      const double armEndSec = elapsedRuntimeSec();
      const double consumedBudget = max(0.0, armEndSec - armStartSec);
      portfolioBudgetSec = max(0.0, portfolioBudgetSec - consumedBudget);

      bool armFeasible = false;
      int armSoc = 0;
      IterationQuality armQuality = IterationQuality::none;
      if (armSuccess) {
        armSoc = solution_.sumOfCosts;
        ValidationStats armValidationStats;
        ConflictMap armPotentialNeighborhood;
        const bool previousTerminalValidationFlag =
            useTerminalPathsInValidation_;
        useTerminalPathsInValidation_ = false;
        armFeasible = validateSolution(&armPotentialNeighborhood,
                                       &armValidationStats);
        useTerminalPathsInValidation_ = previousTerminalValidationFlag;
        if (armFeasible && armSoc < bestPortfolioSoc) {
          bestPortfolioSoc = armSoc;
          bestPortfolioArm = arm;
          bestPortfolioSolution = solution_;
          haveBestPortfolioSolution = true;
          armQuality = IterationQuality::bestSolutionYet;
        }
      }

      InitialCheckpoint checkpoint;
      checkpoint.runtimeSec = armEndSec;
      checkpoint.label = "InitPortfolio:" + arm;
      checkpoint.soc = armSoc;
      checkpoint.feasible = armFeasible;
      checkpoint.quality = armQuality;
      initialCheckpoints.push_back(std::move(checkpoint));

      if (initialPortfolioStopOnFirstFeasible_ && haveBestPortfolioSolution) {
        break;
      }
    }

    if (haveBestPortfolioSolution) {
      solution_ = bestPortfolioSolution;
      initialSolutionEffective_ = "portfolio(" + bestPortfolioArm + ")";
      initialSolutionFallbackReason_ = "none";
      success = true;
    } else {
      initialSolutionFallbackReason_ = "portfolio_no_feasible_arm";
      success = false;
    }
  } else {
    success = runInitialSolutionStrategy(initialSolutionStrategy);
  }

  if (!success && initialSolutionStrategy != "greedy") {
    if (initialSolutionFallback == "greedy") {
      PLOGW << "Initial solution strategy '" << initialSolutionStrategy
            << "' failed; falling back to 'greedy'\n";
      initialSolutionFallbackUsed_ = true;
      initialSolutionEffective_ = "greedy";
      initialSolutionFallbackReason_ = "requested_strategy_failed";
      success = runInitialSolutionStrategyWithBudget("greedy");
      if (!success) {
        initialSolutionFallbackReason_ = "fallback_greedy_failed";
      }
    } else if (initialSolutionFallback == "none") {
      PLOGW << "Initial solution strategy '" << initialSolutionStrategy
            << "' failed; fallback disabled\n";
      initialSolutionFallbackReason_ = "fallback_disabled";
    } else {
      PLOGW << "Unknown initialSolutionFallback '" << initialSolutionFallback
            << "'; defaulting to 'greedy'\n";
      initialSolutionFallbackUsed_ = true;
      initialSolutionEffective_ = "greedy";
      initialSolutionFallbackReason_ = "unknown_fallback_defaulted_to_greedy";
      success = runInitialSolutionStrategyWithBudget("greedy");
      if (!success) {
        initialSolutionFallbackReason_ = "fallback_greedy_failed";
      }
    }
  }

  // If the initial solution strategy failed then we cannot do anything!
  if (!success) {
    if (initialSolutionFallbackReason_ == "none") {
      initialSolutionFallbackReason_ = "initializer_failed";
    }
    return success;
  }

  initialSolutionRuntime_ = ((fsec)(Time::now() - plannerStartTime_)).count();
  runtime = initialSolutionRuntime_;

  PLOGD << "Initial solution cost = " << solution_.sumOfCosts
        << ", Runtime = " << initialSolutionRuntime_ << "\n";

  if (goalOccupationMode_ == "reposition_true") {
    vector<int> allAgents(instance_.getAgentNum());
    std::iota(allAgents.begin(), allAgents.end(), 0);
    if (!planTerminalReposition(allAgents, true)) {
      PLOGE << "run: true terminal reposition planning failed during "
               "initialization\n";
      return false;
    }
  }

  ConflictMap potentialNeighborhood;  // Need for the conflict removal case
  ValidationStats currentValidationStats;
  useTerminalPathsInValidation_ = (goalOccupationMode_ == "reposition_true");
  bool currentSolutionValid =
      validateSolution(&potentialNeighborhood, &currentValidationStats);
  useTerminalPathsInValidation_ = false;

  bool feasibleSolutionUpdated = false;
  if (currentSolutionValid) {
    feasibleSolutionUpdated = true;
    extractFeasibleSolution();
  }

  if (market_.heuristics) {
    updateMarketStateFromCurrentSolution();
    if (currentSolutionValid) {
      market_.bestPressure = computeSolutionMarketPressure();
      market_.bestWait = computeSolutionPrecedenceWait();
    }
  }

  constexpr size_t kMaxStoredIterationStatsUnbounded = 200000;
  auto appendIterationStat = [&](const IterationStats& stat) {
    if (numOfIterations_ <= 0 &&
        iterationStats.size() >= kMaxStoredIterationStatsUnbounded &&
        !iterationStats.empty()) {
      // Bound memory in unbounded-iteration mode while preserving
      // "latest-iteration" semantics used by ALNS updates.
      iterationStats.back() = stat;
      return;
    }
    iterationStats.push_back(stat);
  };

  bool checkpointHasFeasible = false;
  for (const auto& checkpoint : initialCheckpoints) {
    checkpointHasFeasible = checkpointHasFeasible || checkpoint.feasible;
    appendIterationStat(IterationStats(
        checkpoint.runtimeSec, checkpoint.label, instance_.getAgentNum(),
        instance_.getTasksNum(), checkpoint.soc, checkpoint.feasible,
        checkpoint.quality));
  }
  if (initialCheckpoints.empty() ||
      (!checkpointHasFeasible && feasibleSolutionUpdated)) {
    appendIterationStat(IterationStats(
        initialSolutionRuntime_, initialSolutionEffective_,
        instance_.getAgentNum(), instance_.getTasksNum(), solution_.sumOfCosts,
        feasibleSolutionUpdated, bestSolutionYet));
  }

  ConflictMap oldNeighborhood;

  // Needed to maintain a running mean and standard deviation which can then be used to standardize the sum of costs and conflicts in accepting criteria functions
  // Decouple moving-metrics stability from run-loop bounds:
  // when maxIterations is unbounded/unspecified (0), keep a reasonably sized
  // window so utility doesn't collapse to zero and freeze TA acceptance.
  constexpr int kDefaultMetricsWindowSize = 200;
  const int metricsWindowSize =
      (numOfIterations_ > 0) ? max(2, numOfIterations_)
                             : kDefaultMetricsWindowSize;
  const int initialConflictSignal =
      utilityUseConflictEventCount_
          ? currentValidationStats.totalConflictEvents()
                                    : (int)potentialNeighborhood.size();
  MovingMetrics metrics(metricsWindowSize, lnsConflictWeight_, lnsCostWeight_,
                        initialConflictSignal, solution_.sumOfCosts);
  solution_.utility =
      metrics.computeMovingMetrics(initialConflictSignal, solution_.sumOfCosts);

  constexpr double kMinTemperature = 1e-9;
  const double toleranceScale = tolerance_ / 100.0;
  temperature_ = std::abs(solution_.utility) * toleranceScale;
  if (!std::isfinite(temperature_) || temperature_ <= kMinTemperature) {
    // Moving utility can be ~0 at initialization when the rolling window is
    // prefilled with the same initial sample. Use a scale-aware fallback so
    // TA/SA are not effectively frozen from the first iteration.
    const double conflictScale =
        max(1.0, std::abs(static_cast<double>(initialConflictSignal)));
    const double costScale =
        max(1.0, std::abs(static_cast<double>(solution_.sumOfCosts)));
    const double blendedScale =
        lnsConflictWeight_ * conflictScale + lnsCostWeight_ * costScale;
    const double fallbackScale = max(1.0, blendedScale);
    temperature_ = max(kMinTemperature, fallbackScale * toleranceScale);
  }
  if (acceptanceCriteria == "SA") {
    temperature_ /= log(2);
  }
  initialTemperature_ = temperature_;
  maxTemperature_ = max(initialTemperature_, 1.0) * 1000.0;
  if (numOfIterations_ > 0) {
    greatDelugeDecay_ = initialTemperature_ / max(1, numOfIterations_);
  } else {
    greatDelugeDecay_ = initialTemperature_ / 1000.0;
  }
  if (!std::isfinite(greatDelugeDecay_) || greatDelugeDecay_ < 0.0) {
    greatDelugeDecay_ = 0.0;
  }

  previousSolution_ = solution_;

  const int64_t iterationLimit =
      (numOfIterations_ > 0)
          ? static_cast<int64_t>(numOfIterations_) * 2
          : std::numeric_limits<int64_t>::max();

  // LNS loop
  while (runtime < timeLimit_ &&
         static_cast<int64_t>(iterationStats.size()) < iterationLimit) {
    iterationRollbackHintAgents_.clear();
    market_.candidateUpdateConsumed = false;
    const int previousSocForIter = previousSolution_.sumOfCosts;
    const bool previousValidForIter = currentSolutionValid;
    const ValidationStats previousValidationStatsForIter =
        currentValidationStats;
    const double previousPressureForIter =
        market_.heuristics ? computeSolutionMarketPressure() : 0.0;
    const double previousWaitForIter =
        market_.heuristics ? computeSolutionPrecedenceWait() : 0.0;
    int alnsHeuristicForIter = -1;

    // These functions populate the LNS neighborhoods' removedTask parameter
    if (destroyHeuristic == "conflict") {
      conflictRemoval(&potentialNeighborhood);
    } else if (destroyHeuristic == "worst") {
      worstRemoval();
    } else if (destroyHeuristic == "random") {
      randomRemoval();
    } else if (destroyHeuristic == "shaw") {
      shawRemoval(neighborSize_ * 3);
    } else if (destroyHeuristic == "precedence_wait") {
      precedenceWaitRemoval(&potentialNeighborhood);
    } else if (destroyHeuristic == "low_slack") {
      lowSlackRemoval(&potentialNeighborhood);
    } else if (destroyHeuristic == "market_tatonnement") {
      if (!market_.heuristics) {
        PLOGE << "destroyHeuristic='market_tatonnement' requires market "
                 "heuristics to be enabled\n";
        return false;
      }
      marketTatonnementRemoval(&potentialNeighborhood);
    } else if (destroyHeuristic == "alns") {
      alnsRemoval(&potentialNeighborhood);
      alnsHeuristicForIter = adaptiveLNS_.recentDestroyHeuristic;
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.selections[alnsHeuristicForIter]++;
      }
    } else {
      PLOGE << "Unknown destroy heuristic: " << destroyHeuristic << "\n";
      return false;
    }

    oldNeighborhood = lnsNeighborhood_.removedTasks;
    IterationQuality quality = IterationQuality::none;

    PLOGD << "Printing neighborhood conflict tasks\n";
    PLOGD << "Size: " << lnsNeighborhood_.removedTasks.size() << "\n";
    for (const auto& [_, conflictTask] : lnsNeighborhood_.removedTasks) {
      PLOGD << "Conflicted Task : " << conflictTask.task << "\n";
    }

    if (!prepareNextIteration()) {
      const bool cascadeAbort = lastPrepareAbortedByCascade_;
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        if (cascadeAbort) {
          adaptiveLNS_.cascadeAborted[alnsHeuristicForIter]++;
        } else {
          adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
        }
      }
      if (!cascadeAbort) {
        iterationRollbackHintAgents_ =
            buildRollbackAgentHints(lastPrepareAffectedAgents_);
        restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
      }
      feasibleSolutionUpdated = false;
      quality = IterationQuality::couldNotFind;
      runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
      appendIterationStat(IterationStats(
          runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
          solution_.sumOfCosts, feasibleSolutionUpdated, quality));
      if (cascadeAbort) {
        PLOGD << "prepareNextIteration aborted due to cascade budget\n";
      }
      maybeUpdateMarketState(false);
      continue;
    }

    // This needs to happen after prepare iteration since we updated the conflictedTasks variable in the prepare next iteration function
    for (const auto& [_, conflictedTask] : lnsNeighborhood_.removedTasks) {
      if (conflictedTask.task >= 0 &&
          conflictedTask.task < (int)lnsNeighborhood_.committedTasks.size()) {
        lnsNeighborhood_.committedTasks[conflictedTask.task] = 0;
      }
    }
    lnsNeighborhood_.immutableRemovedTasks = lnsNeighborhood_.removedTasks;
    // Collect agents impacted by this neighborhood once and reuse across:
    // 1) terminal-path invalidation, 2) path-join scope, 3) terminal replanning.
    vector<int> agentsToCompute;
    vector<char> agentMarked(instance_.getAgentNum(), 0);
    auto markAgent = [&](int agent) {
      if (agent >= 0 && agent < instance_.getAgentNum() && !agentMarked[agent]) {
        agentMarked[agent] = 1;
        agentsToCompute.push_back(agent);
      }
    };
    vector<char> visitedTask(instance_.getTasksNum(), 0);
    stack<int> taskStack;
    for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
      taskStack.push(conflict.task);
      markAgent(conflict.agent);
    }
    const auto& ancestors = instance_.getAncestorsRef();
    const auto& successors = instance_.getSuccessorsRef();
    while (!taskStack.empty()) {
      const int task = taskStack.top();
      taskStack.pop();
      if (task < 0 || task >= instance_.getTasksNum() || visitedTask[task]) {
        continue;
      }
      visitedTask[task] = 1;
      const int curAgent =
          (task >= 0 && task < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[task]
              : UNASSIGNED;
      if (curAgent != UNASSIGNED) {
        markAgent(curAgent);
      }
      const int prevAgent =
          (task >= 0 && task < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[task]
              : UNASSIGNED;
      if (prevAgent != UNASSIGNED) {
        markAgent(prevAgent);
      }
      for (int parent : ancestors[task]) {
        if (parent >= 0 && parent < instance_.getTasksNum() &&
            !visitedTask[parent]) {
          taskStack.push(parent);
        }
      }
      for (int child : successors[task]) {
        if (child >= 0 && child < instance_.getTasksNum() &&
            !visitedTask[child]) {
          taskStack.push(child);
        }
      }
    }
    if (agentsToCompute.empty()) {
      agentsToCompute.resize(instance_.getAgentNum());
      std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
    }
    const vector<int> rollbackBaseAgents = agentsToCompute;
    // Phase-D optimization: join only agents that are provably dirty.
    // prepareNextIteration clears service paths for impacted agents, so an
    // empty joined path is a reliable dirty signal.
    vector<int> filteredAgentsToCompute;
    filteredAgentsToCompute.reserve(agentsToCompute.size());
    for (int agent : agentsToCompute) {
      if (agent < 0 || agent >= instance_.getAgentNum()) {
        continue;
      }
      const bool assignmentsChanged =
          (solution_.agents[agent].taskAssignments !=
           previousSolution_.agents[agent].taskAssignments);
      if (solution_.agents[agent].path.empty() || assignmentsChanged) {
        filteredAgentsToCompute.push_back(agent);
      }
    }
    if (!filteredAgentsToCompute.empty()) {
      agentsToCompute.swap(filteredAgentsToCompute);
    }

    // Repair: commit removed tasks back using regret.
    //
    // Default behavior recomputes regrets from scratch every commit.
    // Optional incremental mode recomputes regrets only for a dirty subset.
    lnsNeighborhood_.regretMaxHeap.clear();
    std::fill(regretBestOption_.begin(), regretBestOption_.end(),
              std::make_pair(UNASSIGNED, -1));
    std::fill(regretSecondBestOption_.begin(), regretSecondBestOption_.end(),
              std::make_pair(UNASSIGNED, -1));
    bool repairFailed = false;
    regretEvalStatsCurrent_.reset();
    {
      const int removedCount = (int)lnsNeighborhood_.removedTasks.size();
      regretEvalStatsCurrent_.neighborhoods++;
      regretEvalStatsTotal_.neighborhoods++;
      regretEvalStatsCurrent_.removedTasksSum += removedCount;
      regretEvalStatsTotal_.removedTasksSum += removedCount;
      regretEvalStatsCurrent_.removedTasksMax =
          max(regretEvalStatsCurrent_.removedTasksMax, (int64_t)removedCount);
      regretEvalStatsTotal_.removedTasksMax =
          max(regretEvalStatsTotal_.removedTasksMax, (int64_t)removedCount);
    }
    if (!incrementalRegret_) {
      // Compute regret for each of the tasks that are in the conflicting set
      // Pick the best one and repeat the whole process again
      while (!lnsNeighborhood_.removedTasks.empty()) {
        if (runtimeBudgetExhausted()) {
          repairFailed = true;
          break;
        }
        bool enoughSpace = computeRegret();
        if (!enoughSpace) {
          // We could not compute enough regrets for each task so we need to try
          // and reset the solution and try potentially with a different
          // neighborhood!
          repairFailed = true;
          break;
        }
        if (lnsNeighborhood_.regretMaxHeap.empty()) {
          PLOGE << "regretMaxHeap is empty after computeRegret\n";
          repairFailed = true;
          break;
        }
        assert(!lnsNeighborhood_.regretMaxHeap.empty());
        Regret bestRegret = lnsNeighborhood_.regretMaxHeap.top();
        // Use the best regret task and insert it in its correct location
        if (!commitBestRegretTask(bestRegret)) {
          PLOGE << "run: failed to commit best-regret task "
                << bestRegret.task << "\n";
          repairFailed = true;
          break;
        }
      }
    } else {
      // Initial regret computation for the neighborhood.
      incrementalRegretStatsCurrent_.reset();
      if (runtimeBudgetExhausted() ||
          !recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
        repairFailed = true;
      }

      int64_t stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
      int64_t stalePopsSinceRefresh = 0;
      int64_t commitsSinceRefresh = 0;

      while (!repairFailed && !lnsNeighborhood_.removedTasks.empty()) {
        if (runtimeBudgetExhausted()) {
          repairFailed = true;
          break;
        }
        // If we are spending too much effort discarding stale heap entries,
        // do a full refresh of all remaining regrets.
        //
        // This helps when dirty rules miss some dependencies (regret drift),
        // and also prevents the heap from filling up with stale entries.
        if (stalePopsSinceRefresh >= 100 &&
            stalePopsSinceRefresh > 2 * commitsSinceRefresh + 50) {
          lnsNeighborhood_.regretMaxHeap.clear();
          incrementalRegretStatsCurrent_.fullRefreshes++;
          incrementalRegretStatsTotal_.fullRefreshes++;
          if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
            repairFailed = true;
            break;
          }
          stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
          stalePopsSinceRefresh = 0;
          commitsSinceRefresh = 0;
        }

        const auto bestRegret = popNextValidRegret();
        const int64_t staleDelta =
            incrementalRegretStatsCurrent_.stalePops - stalePopsAtLastCheck;
        stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
        stalePopsSinceRefresh += staleDelta;

        if (!bestRegret.has_value()) {
          // Heap may have been exhausted by stale entries; rebuild for whatever
          // is left.
          incrementalRegretStatsCurrent_.heapRebuilds++;
          incrementalRegretStatsTotal_.heapRebuilds++;
          if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
            repairFailed = true;
          }
          continue;
        }

        const vector<int> endTimesBefore = computeCurrentTaskEndTimes();
        const vector<int> lastTaskBefore = computeCurrentLastTaskPerAgent();
        if (!commitBestRegretTask(*bestRegret)) {
          PLOGE << "run: failed to commit best-regret task "
                << bestRegret->task << "\n";
          repairFailed = true;
          break;
        }
        incrementalRegretStatsCurrent_.commits++;
        incrementalRegretStatsTotal_.commits++;
        commitsSinceRefresh++;
        const vector<int> endTimesAfter = computeCurrentTaskEndTimes();
        const vector<int> lastTaskAfter = computeCurrentLastTaskPerAgent();

        const vector<int> dirtyTasks = computeDirtyTasksAfterCommit(
            endTimesBefore, endTimesAfter, lastTaskBefore, lastTaskAfter);
        if (!recomputeRegretsForTasks(dirtyTasks)) {
          repairFailed = true;
        }
      }

      // Stats are collected and printed in a dedicated summary section.
    }

    // If we could not successfully compute the regrets and commit to all the tasks in the neighborhood then we need to reset this neighborhood!
    if (repairFailed || !lnsNeighborhood_.removedTasks.empty()) {
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
      }
      // Reject whatever we done till now
      iterationRollbackHintAgents_ =
          buildRollbackAgentHints(rollbackBaseAgents);
      restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
      feasibleSolutionUpdated = false;
      quality = IterationQuality::couldNotFind;
      runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
      appendIterationStat(IterationStats(
          runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
          solution_.sumOfCosts, feasibleSolutionUpdated, quality));
      // Skip everything after this statement
      PLOGD << "Could not find paths for the neighborhood! Attempting a new "
               "neighborhood computation\n";
      maybeUpdateMarketState(false);
      continue;
    }

    // Join only agents impacted by this neighborhood.
    if (!solution_.joinPaths(agentsToCompute)) {
      PLOGE << "run: failed to join agent paths for candidate solution\n";
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
      }
      iterationRollbackHintAgents_ =
          buildRollbackAgentHints(rollbackBaseAgents);
      restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
      feasibleSolutionUpdated = false;
      quality = IterationQuality::couldNotFind;
      runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
      appendIterationStat(IterationStats(
          runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
          solution_.sumOfCosts, feasibleSolutionUpdated, quality));
      maybeUpdateMarketState(false);
      continue;
    }
    if (goalOccupationMode_ == "reposition_true") {
      const vector<int> terminalReplanAgents =
          selectTerminalReplanAgents(agentsToCompute);
      if (!terminalReplanAgents.empty() &&
          !planTerminalReposition(terminalReplanAgents, false)) {
        PLOGE << "run: failed to replan terminal reposition paths for "
                 "candidate solution\n";
        if (alnsHeuristicForIter >= 0 &&
            alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
          adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
        }
        iterationRollbackHintAgents_ =
            buildRollbackAgentHints(rollbackBaseAgents);
        restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
        feasibleSolutionUpdated = false;
        quality = IterationQuality::couldNotFind;
        runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
        appendIterationStat(IterationStats(
            runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
            solution_.sumOfCosts, feasibleSolutionUpdated, quality));
        maybeUpdateMarketState(false);
        continue;
      }
    }

    // Compute the updated sum of costs
    long long recomputedSoc = 0;
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      recomputedSoc +=
          static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
    }
    if (recomputedSoc > std::numeric_limits<int>::max()) {
      PLOGW << "LNS::run: sum of costs overflowed int during recomputation;"
               " clamping to INT_MAX\n";
      solution_.sumOfCosts = std::numeric_limits<int>::max();
    } else if (recomputedSoc < std::numeric_limits<int>::min()) {
      PLOGW << "LNS::run: sum of costs underflowed int during recomputation;"
               " clamping to INT_MIN\n";
      solution_.sumOfCosts = std::numeric_limits<int>::min();
    } else {
      solution_.sumOfCosts = static_cast<int>(recomputedSoc);
    }

    PLOGD << "Old sum of costs = " << previousSolution_.sumOfCosts << "\n";
    PLOGD << "New sum of costs = " << solution_.sumOfCosts << "\n";

    const int previousConflictSignalForIter =
        utilityUseConflictEventCount_
            ? previousValidationStatsForIter.totalConflictEvents()
                                      : (int)potentialNeighborhood.size();
    PLOGD << "Conflict signal in old solution: "
          << previousConflictSignalForIter << "\n";

    // Extract the set of conflicting tasks
    potentialNeighborhood.clear();
    ValidationStats candidateValidationStats;
    useTerminalPathsInValidation_ = (goalOccupationMode_ == "reposition_true");
    const bool candidateValid =
        validateSolution(&potentialNeighborhood, &candidateValidationStats);
    useTerminalPathsInValidation_ = false;

    const int candidateConflictSignal =
        utilityUseConflictEventCount_
            ? candidateValidationStats.totalConflictEvents()
                                      : (int)potentialNeighborhood.size();
    PLOGD << "Conflict signal in new solution: " << candidateConflictSignal
          << "\n";

    // Accept the solution only if the new one has higher utility compared to the old solution where utility is a weighted combination of the number of conflicts and sum of costs.
    // Compute the utility of this solution
    solution_.utility =
        metrics.computeMovingMetrics(candidateConflictSignal, solution_.sumOfCosts);

    if (!candidateValid) {
      // Solution was not valid as we found some conflicts!
      feasibleSolutionUpdated = false;
      PLOGE << "The solution was not valid!\n";
    } else {
      if (extractFeasibleSolution()) {
        // This is the case when the feasible solution was updated!
        quality = IterationQuality::bestSolutionYet;
        feasibleSolutionUpdated = true;
      } else {
        feasibleSolutionUpdated = false;
      }
    }

    // Ensure that we are either accepting or rejecting a solution here!
    const int proposedSocForIter = solution_.sumOfCosts;
    double candidatePressure = 0.0;
    double candidateWait = 0.0;
    bool accepted = false;
    bool guardRejected = false;
    bool acceptedAsWorse = false;
    auto advanceTemperatureOnGuardReject = [&]() {
      if (acceptanceCriteria == "SA" || acceptanceCriteria == "TA") {
        temperature_ *= coolingCoefficient_;
      } else if (acceptanceCriteria == "OBA") {
        const double reheated = temperature_ * heatingCoefficient_;
        temperature_ = std::min(reheated, maxTemperature_);
      } else if (acceptanceCriteria == "GDA") {
        temperature_ = max(0.0, temperature_ - greatDelugeDecay_);
      }
    };
    iterationRollbackHintAgents_ = buildRollbackAgentHints(rollbackBaseAgents);
    if (!candidateValid && rejectInvalidCandidates_) {
      invalidCandidateRejections++;
      iterationRollbackHintAgents_ =
          buildRollbackAgentHints(rollbackBaseAgents);
      restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
      accepted = false;
      guardRejected = true;
      advanceTemperatureOnGuardReject();
      PLOGD << "Rejecting invalid candidate before acceptance criteria\n";
    } else {
      if (!candidateValid) {
        PLOGD << "Invalid candidate forwarded to acceptance criteria\n";
      }
      candidatePressure =
          market_.heuristics ? computeSolutionMarketPressure() : 0.0;
      candidateWait = market_.heuristics ? computeSolutionPrecedenceWait() : 0.0;
      if (market_.heuristics && !market_.updateOnAcceptedOnly &&
          market_.updateFromCandidate) {
        maybeUpdateMarketState(false, true);
      }
      if (market_.heuristics && market_.acceptanceGuards &&
          !passMarketAcceptanceGuards(previousPressureForIter, candidatePressure,
                                      previousWaitForIter, candidateWait,
                                      previousSolution_.utility <
                                          solution_.utility)) {
        marketGuardRejections++;
        iterationRollbackHintAgents_ =
            buildRollbackAgentHints(rollbackBaseAgents);
        restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
        accepted = false;
        guardRejected = true;
        advanceTemperatureOnGuardReject();
        PLOGD << "Rejecting this solution due to market acceptance guards\n";
      } else {
        if (acceptanceFeasibilityFirstPrecedenceDebt_ &&
            candidateValid && !previousValidForIter) {
          acceptanceDiagnostics_.feasibilityFirstDecisions++;
          acceptanceDiagnostics_.invalidToValidAccepted++;
          accepted = true;
          acceptedAsWorse = false;
          PLOGD << "Feasibility-first acceptance: valid candidate accepted "
                   "over invalid incumbent\n";
        } else if (acceptanceFeasibilityFirstPrecedenceDebt_ &&
                   !candidateValid && previousValidForIter) {
          acceptanceDiagnostics_.feasibilityFirstDecisions++;
          acceptanceDiagnostics_.validToInvalidCompared++;
          double previousSpatialNorm = 0.0;
          double previousDebtNorm = 0.0;
          double previousSocNorm = 0.0;
          double candidateSpatialNorm = 0.0;
          double candidateDebtNorm = 0.0;
          double candidateSocNorm = 0.0;

          const double previousInvalidScore = computeInvalidAcceptanceScore(
              previousValidationStatsForIter, previousSocForIter,
              candidateValidationStats, proposedSocForIter,
              &previousSpatialNorm, &previousDebtNorm, &previousSocNorm);
          const double candidateInvalidScore = computeInvalidAcceptanceScore(
              candidateValidationStats, proposedSocForIter,
              previousValidationStatsForIter, previousSocForIter,
              &candidateSpatialNorm, &candidateDebtNorm, &candidateSocNorm);
          const double scoreDelta =
              candidateInvalidScore - previousInvalidScore;
          const bool candidateWorseInvalid = scoreDelta > 0.0;
          double acceptanceTempBefore = temperature_;
          if (acceptanceUseDedicatedInvalidTemperature_) {
            ensureInvalidAcceptanceTemperatureInitialized(previousInvalidScore,
                                                         candidateInvalidScore);
            acceptanceTempBefore = invalidTemperature_;
            accepted = acceptScoreWithCurrentCriterion(
                candidateInvalidScore, previousInvalidScore, &invalidTemperature_,
                &invalidInitialTemperature_, &invalidMaxTemperature_,
                &invalidGreatDelugeDecay_);
          } else {
            accepted = acceptScoreWithCurrentCriterion(candidateInvalidScore,
                                                       previousInvalidScore);
          }
          const double acceptanceTempAfter =
              acceptanceUseDedicatedInvalidTemperature_ ? invalidTemperature_
                                                        : temperature_;
          acceptedAsWorse = candidateWorseInvalid;
          acceptanceDiagnostics_.invalidScoreComparisons++;
          acceptanceDiagnostics_.invalidScoreDeltaSum += scoreDelta;
          acceptanceDiagnostics_.invalidScoreAbsDeltaSum +=
              std::abs(scoreDelta);
          acceptanceDiagnostics_.invalidAcceptanceTempBeforeSum +=
              acceptanceTempBefore;
          acceptanceDiagnostics_.invalidAcceptanceTempAfterSum +=
              acceptanceTempAfter;
          if (candidateWorseInvalid) {
            acceptanceDiagnostics_.invalidScoreWorseComparisons++;
          }
          if (accepted) {
            acceptanceDiagnostics_.invalidScoreAccepted++;
            if (candidateWorseInvalid) {
              acceptanceDiagnostics_.invalidScoreWorseAccepted++;
            }
            acceptanceDiagnostics_.validToInvalidAccepted++;
            PLOGD << "Feasibility-first valid->invalid accepted by "
                     "SA/TA/OBA/GDA (previous="
                  << previousInvalidScore
                  << ", candidate=" << candidateInvalidScore
                  << ", temp_before=" << acceptanceTempBefore
                  << ", temp_after=" << acceptanceTempAfter
                  << ", dedicated_temp="
                  << (acceptanceUseDedicatedInvalidTemperature_ ? "true"
                                                                : "false")
                  << ")\n";
          } else {
            acceptanceDiagnostics_.invalidScoreRejected++;
            acceptanceDiagnostics_.validToInvalidRejected++;
            iterationRollbackHintAgents_ =
                buildRollbackAgentHints(rollbackBaseAgents);
            restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
            PLOGD << "Feasibility-first valid->invalid rejected by "
                     "SA/TA/OBA/GDA (previous="
                  << previousInvalidScore
                  << ", candidate=" << candidateInvalidScore
                  << ", temp_before=" << acceptanceTempBefore
                  << ", temp_after=" << acceptanceTempAfter
                  << ", dedicated_temp="
                  << (acceptanceUseDedicatedInvalidTemperature_ ? "true"
                                                                : "false")
                  << ")\n";
          }
        } else if (acceptanceFeasibilityFirstPrecedenceDebt_ &&
                   !candidateValid && !previousValidForIter) {
          acceptanceDiagnostics_.feasibilityFirstDecisions++;
          acceptanceDiagnostics_.invalidVsInvalidComparisons++;
          double previousSpatialNorm = 0.0;
          double previousDebtNorm = 0.0;
          double previousSocNorm = 0.0;
          double candidateSpatialNorm = 0.0;
          double candidateDebtNorm = 0.0;
          double candidateSocNorm = 0.0;

          const double previousInvalidScore = computeInvalidAcceptanceScore(
              previousValidationStatsForIter, previousSocForIter,
              candidateValidationStats, proposedSocForIter,
              &previousSpatialNorm, &previousDebtNorm, &previousSocNorm);
          const double candidateInvalidScore = computeInvalidAcceptanceScore(
              candidateValidationStats, proposedSocForIter,
              previousValidationStatsForIter, previousSocForIter,
              &candidateSpatialNorm, &candidateDebtNorm, &candidateSocNorm);

          acceptanceDiagnostics_.previousInvalidScoreSum += previousInvalidScore;
          acceptanceDiagnostics_.candidateInvalidScoreSum += candidateInvalidScore;
          acceptanceDiagnostics_.previousSpatialNormSum += previousSpatialNorm;
          acceptanceDiagnostics_.candidateSpatialNormSum += candidateSpatialNorm;
          acceptanceDiagnostics_.previousPrecedenceDebtNormSum +=
              previousDebtNorm;
          acceptanceDiagnostics_.candidatePrecedenceDebtNormSum +=
              candidateDebtNorm;
          acceptanceDiagnostics_.previousSocNormSum += previousSocNorm;
          acceptanceDiagnostics_.candidateSocNormSum += candidateSocNorm;

          const double scoreDelta =
              candidateInvalidScore - previousInvalidScore;
          const bool candidateWorseInvalid = scoreDelta > 0.0;
          double acceptanceTempBefore = temperature_;
          if (acceptanceUseDedicatedInvalidTemperature_) {
            ensureInvalidAcceptanceTemperatureInitialized(previousInvalidScore,
                                                         candidateInvalidScore);
            acceptanceTempBefore = invalidTemperature_;
            accepted = acceptScoreWithCurrentCriterion(
                candidateInvalidScore, previousInvalidScore, &invalidTemperature_,
                &invalidInitialTemperature_, &invalidMaxTemperature_,
                &invalidGreatDelugeDecay_);
          } else {
            accepted = acceptScoreWithCurrentCriterion(candidateInvalidScore,
                                                       previousInvalidScore);
          }
          const double acceptanceTempAfter =
              acceptanceUseDedicatedInvalidTemperature_ ? invalidTemperature_
                                                        : temperature_;
          acceptedAsWorse = candidateWorseInvalid;
          acceptanceDiagnostics_.invalidScoreComparisons++;
          acceptanceDiagnostics_.invalidScoreDeltaSum += scoreDelta;
          acceptanceDiagnostics_.invalidScoreAbsDeltaSum +=
              std::abs(scoreDelta);
          acceptanceDiagnostics_.invalidAcceptanceTempBeforeSum +=
              acceptanceTempBefore;
          acceptanceDiagnostics_.invalidAcceptanceTempAfterSum +=
              acceptanceTempAfter;
          if (candidateWorseInvalid) {
            acceptanceDiagnostics_.invalidScoreWorseComparisons++;
          }
          if (accepted) {
            acceptanceDiagnostics_.invalidScoreAccepted++;
            if (candidateWorseInvalid) {
              acceptanceDiagnostics_.invalidScoreWorseAccepted++;
            }
            acceptanceDiagnostics_.invalidVsInvalidAccepted++;
            PLOGD << "Feasibility-first invalid acceptance score: previous="
                  << previousInvalidScore
                  << ", candidate=" << candidateInvalidScore
                  << ", temp_before=" << acceptanceTempBefore
                  << ", temp_after=" << acceptanceTempAfter
                  << ", dedicated_temp="
                  << (acceptanceUseDedicatedInvalidTemperature_ ? "true"
                                                                : "false")
                  << "\n";
          } else {
            acceptanceDiagnostics_.invalidScoreRejected++;
            acceptanceDiagnostics_.invalidVsInvalidRejected++;
            iterationRollbackHintAgents_ =
                buildRollbackAgentHints(rollbackBaseAgents);
            restoreSolutionFromPrevious(&iterationRollbackHintAgents_);
            PLOGD << "Feasibility-first invalid rejection score: previous="
                  << previousInvalidScore
                  << ", candidate=" << candidateInvalidScore
                  << ", temp_before=" << acceptanceTempBefore
                  << ", temp_after=" << acceptanceTempAfter
                  << ", dedicated_temp="
                  << (acceptanceUseDedicatedInvalidTemperature_ ? "true"
                                                                : "false")
                  << "\n";
          }
        } else {
          if (acceptanceCriteria == "SA") {
            accepted = simulatedAnnealing();
          } else if (acceptanceCriteria == "TA") {
            accepted = thresholdAcceptance();
          } else if (acceptanceCriteria == "OBA") {
            accepted = oldBachelorsAcceptance();
          } else if (acceptanceCriteria == "GDA") {
            accepted = greatDelugeAlgorithm();
          } else {
            PLOGE << "Unknown acceptance criteria: " << acceptanceCriteria
                  << "\n";
            return false;
          }
          if (accepted) {
            acceptedAsWorse = previousSolution_.utility < solution_.utility;
          }
        }
      }
    }

    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      const double deltaSoc = (double)(previousSocForIter - proposedSocForIter);
      adaptiveLNS_.deltaSocAll[alnsHeuristicForIter] += deltaSoc;
      if (deltaSoc > 0.0) {
        adaptiveLNS_.proposedBetter[alnsHeuristicForIter]++;
      } else if (deltaSoc < 0.0) {
        adaptiveLNS_.proposedWorse[alnsHeuristicForIter]++;
      } else {
        adaptiveLNS_.proposedEqual[alnsHeuristicForIter]++;
      }
      if (candidateValid) {
        adaptiveLNS_.feasible[alnsHeuristicForIter]++;
      }
      if (!accepted) {
        adaptiveLNS_.rejected[alnsHeuristicForIter]++;
      } else {
        adaptiveLNS_.accepted[alnsHeuristicForIter]++;
        adaptiveLNS_.deltaSocAccepted[alnsHeuristicForIter] += deltaSoc;
        if (deltaSoc < 0.0) {
          adaptiveLNS_.acceptedWorse[alnsHeuristicForIter]++;
        }
        if (acceptedAsWorse) {
          adaptiveLNS_.downgradedAccepted[alnsHeuristicForIter]++;
        } else {
          adaptiveLNS_.improvedAccepted[alnsHeuristicForIter]++;
        }
        if (feasibleSolutionUpdated) {
          adaptiveLNS_.bestUpdates[alnsHeuristicForIter]++;
        }
      }
    }

    if (!accepted) {
      quality = IterationQuality::none;
      potentialNeighborhood = oldNeighborhood;
      if (guardRejected && market_.heuristics) {
        // Candidate utility was discarded; keep previous temperature behavior
        // untouched and preserve previous solution.
      }
    } else {
      if (acceptedAsWorse) {
        // We accepted a potentially worse solution to get out of local minima
        quality = IterationQuality::downgradedButAccepted;
      } else {
        // We accepted a strictly better solution!
        quality = IterationQuality::improvedSolution;
      }
      previousSolution_ = solution_;
      currentSolutionValid = candidateValid;
      currentValidationStats = candidateValidationStats;
      if (market_.heuristics) {
        market_.bestPressure = min(market_.bestPressure, candidatePressure);
        market_.bestWait = min(market_.bestWait, candidateWait);
      }
    }

    maybeUpdateMarketState(accepted);

    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    double costToLog = (feasibleSolutionUpdated) ? incumbentSolution_.sumOfCosts
                                                 : solution_.sumOfCosts;
    appendIterationStat(IterationStats(
        runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
        costToLog, feasibleSolutionUpdated, quality));
  }

  // printPaths();
  return !incumbentSolution_.agentPaths.empty();
}

bool LNS::prepareNextIteration() {
  PLOGI << "Preparing the solution object for the next iteration\n";
  lastPrepareAbortedByCascade_ = false;
  lastPrepareSeedTasks_ = 0;
  lastPrepareClosureTasks_ = 0;
  lastPrepareClosureAdded_ = 0;
  lastPrepareAffectedAgents_.clear();

  const int taskCount = instance_.getTasksNum();
  const vector<int> taskToPositionBefore =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);

  // Find the tasks that are following the earliest conflicting task as their paths need to be invalidated
  const auto& successors = instance_.getSuccessorsRef();

  // Include all successors of the original conflicted tasks in one multi-source
  // traversal to avoid re-traversing shared descendant subgraphs.
  const ConflictMap originalRemovedTasks = lnsNeighborhood_.removedTasks;
  ConflictMap closureRemovedTasks = originalRemovedTasks;
  vector<Conflicts> closureSeeds;
  closureSeeds.reserve(originalRemovedTasks.size());
  for (const auto& [_, conflict] : originalRemovedTasks) {
    closureSeeds.push_back(conflict);
  }
  vector<char> visitedSuccessor(taskCount, 0);
  stack<int> successorStack;
  for (const Conflicts& conflictTask : closureSeeds) {
    if (conflictTask.task >= 0 && conflictTask.task < taskCount) {
      successorStack.push(conflictTask.task);
    }
  }
  while (!successorStack.empty()) {
    const int successorTask = successorStack.top();
    successorStack.pop();
    if (successorTask < 0 || successorTask >= taskCount ||
        visitedSuccessor[successorTask]) {
      continue;
    }
    visitedSuccessor[successorTask] = 1;

    const int successorAgent =
        (successorTask >= 0 &&
         successorTask < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[successorTask]
            : UNASSIGNED;
    if (successorAgent != UNASSIGNED) {
      const int successorTaskPosition =
          (successorTask >= 0 &&
           successorTask < (int)taskToPositionBefore.size())
              ? taskToPositionBefore[successorTask]
              : UNASSIGNED;
      if (successorTaskPosition != UNASSIGNED) {
        closureRemovedTasks.emplace(
            successorTask,
            Conflicts(successorTask, successorAgent, successorTaskPosition));
      }
    }

    for (int nextTask : successors[successorTask]) {
      if (nextTask >= 0 && nextTask < taskCount &&
          !visitedSuccessor[nextTask]) {
        successorStack.push(nextTask);
      }
    }
  }
  const int closureSeedCount = (int)originalRemovedTasks.size();
  const int closureTaskCount = (int)closureRemovedTasks.size();
  const int closureAddedCount =
      max(0, closureTaskCount - closureSeedCount);
  lastPrepareSeedTasks_ = closureSeedCount;
  lastPrepareClosureTasks_ = closureTaskCount;
  lastPrepareClosureAdded_ = closureAddedCount;
  cascadeStats_.prepareCalls++;
  cascadeStats_.seedTasksSum += closureSeedCount;
  cascadeStats_.closureTasksSum += closureTaskCount;
  cascadeStats_.closureAddedSum += closureAddedCount;
  cascadeStats_.closureTasksMax =
      max(cascadeStats_.closureTasksMax, (int64_t)closureTaskCount);
  cascadeStats_.closureAddedMax =
      max(cascadeStats_.closureAddedMax, (int64_t)closureAddedCount);

  const int staticCascadeBudget = cascadeTaskBudget();
  int cascadeBudget = staticCascadeBudget;
  int adaptiveBudgetLower = 1;
  int adaptiveBudgetUpper = staticCascadeBudget;
  if (adaptiveCascadeBudget_) {
    const bool explicitHardCap = (maxCascadeTasks_ > 0);
    adaptiveBudgetUpper =
        explicitHardCap ? staticCascadeBudget
                        : max(staticCascadeBudget, instance_.getTasksNum());
    adaptiveBudgetLower = max(1, min(staticCascadeBudget, closureSeedCount + 1));
    if (adaptiveCascadeBudgetCurrent_ <= 0) {
      adaptiveCascadeBudgetCurrent_ = staticCascadeBudget;
    }
    adaptiveCascadeBudgetCurrent_ =
        min(adaptiveBudgetUpper, max(adaptiveBudgetLower,
                                     adaptiveCascadeBudgetCurrent_));
    cascadeBudget = adaptiveCascadeBudgetCurrent_;
  }

  cascadeStats_.budgetUsedSum += cascadeBudget;
  if (cascadeStats_.prepareCalls == 1) {
    cascadeStats_.budgetUsedMin = cascadeBudget;
    cascadeStats_.budgetUsedMax = cascadeBudget;
  } else {
    cascadeStats_.budgetUsedMin =
        min(cascadeStats_.budgetUsedMin, (int64_t)cascadeBudget);
    cascadeStats_.budgetUsedMax =
        max(cascadeStats_.budgetUsedMax, (int64_t)cascadeBudget);
  }
  adaptiveCascadeBudgetLastUsed_ = cascadeBudget;

  if (closureAddedCount > cascadeBudget) {
    if (adaptiveCascadeBudget_) {
      const int growthStep = max(1, adaptiveCascadeBudgetCurrent_ / 4);
      const int targetBudget =
          max(closureSeedCount + 1, adaptiveCascadeBudgetCurrent_ + growthStep);
      const int nextBudget =
          min(adaptiveBudgetUpper, max(adaptiveBudgetLower, targetBudget));
      if (nextBudget > adaptiveCascadeBudgetCurrent_) {
        cascadeStats_.adaptiveBudgetIncreases++;
      }
      adaptiveCascadeBudgetCurrent_ = nextBudget;
    }
    lastPrepareAbortedByCascade_ = true;
    cascadeStats_.budgetAborts++;
    PLOGW << "prepareNextIteration: cascade budget exceeded (seed="
          << closureSeedCount << ", closure_total=" << closureTaskCount
          << ", closure_added=" << closureAddedCount
          << ", budget=" << cascadeBudget << ", baseline="
          << staticCascadeBudget << ")\n";
    return false;
  }

  if (adaptiveCascadeBudget_) {
    int nextBudget = adaptiveCascadeBudgetCurrent_;
    const double closurePressure =
        (cascadeBudget > 0) ? ((double)closureAddedCount / (double)cascadeBudget)
                            : 1.0;
    if (closurePressure < 0.35) {
      nextBudget = max(adaptiveBudgetLower, adaptiveCascadeBudgetCurrent_ - 1);
    } else if (closurePressure > 0.85) {
      nextBudget = min(adaptiveBudgetUpper, adaptiveCascadeBudgetCurrent_ + 1);
    }

    // Avoid shrinking budget immediately after a productive iteration.
    if (!iterationStats.empty() &&
        (iterationStats.back().quality == IterationQuality::bestSolutionYet ||
         iterationStats.back().quality == IterationQuality::improvedSolution) &&
        nextBudget < adaptiveCascadeBudgetCurrent_) {
      nextBudget = adaptiveCascadeBudgetCurrent_;
    }

    if (nextBudget > adaptiveCascadeBudgetCurrent_) {
      cascadeStats_.adaptiveBudgetIncreases++;
    } else if (nextBudget < adaptiveCascadeBudgetCurrent_) {
      cascadeStats_.adaptiveBudgetDecreases++;
    }
    adaptiveCascadeBudgetCurrent_ = nextBudget;
  }

  lnsNeighborhood_.removedTasks = std::move(closureRemovedTasks);

  // If t_id is deleted then t_id + 1 task needs to be fixed
  set<int> tasksToFix, affectedAgents;
  for (const auto& [_, invalidTask] : lnsNeighborhood_.removedTasks) {

    int agent = invalidTask.agent, taskPosition = invalidTask.taskPosition;
    if (agent < 0 || agent >= instance_.getAgentNum() || taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size() ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size()) {
      PLOGE << "prepareNextIteration: invalid conflict entry (task="
            << invalidTask.task << ", agent=" << agent
            << ", position=" << taskPosition << ")\n";
      continue;
    }
    PLOGD << "Invalidating task: " << invalidTask.task << ", Agent: " << agent
          << " at position: " << taskPosition << " with path length = "
          << solution_.agents[agent].taskPaths[taskPosition].size() << "\n";

    // If the invalidated task was not the last local task of this agent then t_id + 1 exists
    // If the invalid task was not the last task
    if (invalidTask.task != solution_.getAgentGlobalTasks(agent).back()) {
      int nextTask = UNDEFINED, nextTaskPosition = taskPosition + 1;
      while (nextTask == UNDEFINED &&
             nextTaskPosition <
                 (int)solution_.getAgentGlobalTasks(agent).size()) {
        nextTask = solution_.getAgentGlobalTasks(agent)[nextTaskPosition];
        nextTaskPosition++;
      }
      PLOGD << "Found a potential next task!\n";
      // Next task can still be undefined in the case where that task was removed in the previous iteration of this loop
      if (lnsNeighborhood_.removedTasks.count(nextTask) == 0 &&
          nextTask != UNDEFINED) {
        tasksToFix.insert(nextTask);
        PLOGD << "Next task: " << nextTask << "\n";
      }
    }

    affectedAgents.insert(agent);

    // Marking past information about this conflicting task
    solution_.taskAgentMap[invalidTask.task] = UNASSIGNED;
    solution_.agents[agent].clearIntraAgentPrecedenceConstraint(
        invalidTask.task);
    // Needs to happen after clearing precedence constraints
    solution_.agents[agent].taskAssignments[taskPosition] = UNDEFINED;

    if (invalidTask.task >= 0 &&
        invalidTask.task < (int)lnsNeighborhood_.removedTasksPathSize.size()) {
      lnsNeighborhood_.removedTasksPathSize[invalidTask.task] =
          (int)solution_.agents[agent].taskPaths[taskPosition].size();
    }
    solution_.agents[agent].taskPaths[taskPosition] = AgentTaskPath();
  }
  lastPrepareAffectedAgents_.assign(affectedAgents.begin(),
                                    affectedAgents.end());

  // Marking past information about conflicting tasks
  for (int affAgent : affectedAgents) {

    // For an affected agent there can be multiple conflicting tasks so need to do it this way
    solution_.agents[affAgent].path = AgentTaskPath();
    solution_.agents[affAgent].terminalPath = AgentTaskPath();
    solution_.agents[affAgent].terminalPathActive = false;
    solution_.agents[affAgent].taskAssignments.erase(
        std::remove_if(solution_.agents[affAgent].taskAssignments.begin(),
                       solution_.agents[affAgent].taskAssignments.end(),
                       [](int task) { return task == UNDEFINED; }),
        solution_.agents[affAgent].taskAssignments.end());
    solution_.agents[affAgent].taskPaths.erase(
        std::remove_if(solution_.agents[affAgent].taskPaths.begin(),
                       solution_.agents[affAgent].taskPaths.end(),
                       [](const Path& p) { return p.empty(); }),
        solution_.agents[affAgent].taskPaths.end());

    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(affAgent));
    solution_.agents[affAgent].pathPlanner->setGoalLocations(taskLocations);
  }

  const vector<int> taskToPositionAfter =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);

  lnsNeighborhood_.patchedTasks = tasksToFix;

  // Find the paths for the tasks whose previous tasks were removed
  for (int task : instance_.getInputPlanningOrderRef()) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    if (tasksToFix.count(task) > 0) {

      PLOGD << "Going to find path for next task: " << task << "\n";

      int startTime = 0, agent = solution_.getAgentWithTask(task);
      if (agent == UNASSIGNED) {
        PLOGE << "prepareNextIteration: patched task " << task
              << " is not assigned to any agent\n";
        return false;
      }
      int taskPosition = (task >= 0 && task < (int)taskToPositionAfter.size())
                             ? taskToPositionAfter[task]
                             : UNASSIGNED;
      const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
      if (taskPosition < 0 || taskPosition >= (int)agentTasks.size() ||
          taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
        PLOGE << "prepareNextIteration: invalid task position " << taskPosition
              << " for task " << task << " (agent " << agent << ")\n";
        return false;
      }

      if (taskPosition != 0) {
        startTime = solution_.agents[agent].taskPaths[taskPosition - 1].endTime();
      }
      assert(taskPosition <= (int)agentTasks.size() - 1);

      ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
      if (!buildConstraintTable(constraintTable, task)) {
        PLOGE << "prepareNextIteration: failed to build constraint table for "
              << "task " << task << " (agent " << agent << ")\n";
        return false;
      }
      AgentTaskPath path = runLowLevelSearch(
          *solution_.agents[agent].pathPlanner, constraintTable, startTime,
          taskPosition, 0);
      // We must be able to find the path for the next task. If not then we cannot move forward!
      if (path.empty()) {
        PLOGE << "prepareNextIteration: path finding failed for patched task "
              << task << " (agent " << agent << ", position " << taskPosition
              << ")\n";
        return false;
      }
      assert(!path.empty());
      solution_.agents[agent].taskPaths[taskPosition] = path;

      // Once the path was found fix the begin times for subsequent tasks of the agent
      patchAgentTaskPaths(agent, taskPosition);
    }
  }
  return true;
}
