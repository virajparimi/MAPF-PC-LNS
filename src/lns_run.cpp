#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"
#include <cmath>
#include <limits>

void LNS::appendIterationStatBounded(const IterationStats& stat) {
  constexpr size_t kMaxStoredIterationStatsUnbounded = 200000;
  if (numOfIterations_ <= 0 &&
      iterationStats.size() >= kMaxStoredIterationStatsUnbounded &&
      !iterationStats.empty()) {
    // Bound memory in unbounded-iteration mode while preserving
    // "latest-iteration" semantics used by ALNS updates.
    iterationStats.back() = stat;
    return;
  }
  iterationStats.push_back(stat);
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

  bool checkpointHasFeasible = false;
  for (const auto& checkpoint : initialCheckpoints) {
    checkpointHasFeasible = checkpointHasFeasible || checkpoint.feasible;
    appendIterationStatBounded(IterationStats(
        checkpoint.runtimeSec, checkpoint.label, instance_.getAgentNum(),
        instance_.getTasksNum(), checkpoint.soc, checkpoint.feasible,
        checkpoint.quality));
  }
  if (initialCheckpoints.empty() ||
      (!checkpointHasFeasible && feasibleSolutionUpdated)) {
    appendIterationStatBounded(IterationStats(
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
    if (!runOneIteration(potentialNeighborhood, oldNeighborhood, metrics,
                         currentSolutionValid, currentValidationStats,
                         feasibleSolutionUpdated)) {
      return false;
    }
  }

  // printPaths();
  return !incumbentSolution_.agentPaths.empty();
}
