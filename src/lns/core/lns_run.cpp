#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_set>

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

uint64_t LNS::computeSolutionFingerprint(const Solution& solution) const {
  auto mix64 = [](uint64_t x) -> uint64_t {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  };
  uint64_t hashValue = 0x9e3779b97f4a7c15ULL;
  auto combine = [&](uint64_t value) {
    hashValue ^= mix64(value + 0x9e3779b97f4a7c15ULL + (hashValue << 6) +
                       (hashValue >> 2));
  };

  combine((uint64_t)instance_.getAgentNum());
  combine((uint64_t)instance_.getTasksNum());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution.agents[agent].taskAssignments;
    combine((uint64_t)agent);
    combine((uint64_t)assignments.size());
    for (int task : assignments) {
      combine((uint64_t)(uint32_t)(task + 1));
    }

    const auto& servicePath = solution.agents[agent].path;
    combine((uint64_t)servicePath.size());
    combine((uint64_t)(uint32_t)(servicePath.endTimeOrZero() + 1));
    if (!servicePath.empty()) {
      combine((uint64_t)(uint32_t)(servicePath.front().location + 1));
      combine((uint64_t)(uint32_t)(servicePath.back().location + 1));
    }
  }
  return hashValue;
}

uint64_t LNS::computeNeighborhoodFingerprint(
    const ConflictMap& removedTasks) const {
  auto mix64 = [](uint64_t x) -> uint64_t {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  };
  uint64_t hashValue = 0x517cc1b727220a95ULL;
  auto combine = [&](uint64_t value) {
    hashValue ^= mix64(value + 0x9e3779b97f4a7c15ULL + (hashValue << 6) +
                       (hashValue >> 2));
  };

  combine((uint64_t)removedTasks.size());
  for (const auto& [task, conflict] : removedTasks) {
    combine((uint64_t)(uint32_t)(task + 1));
    combine((uint64_t)(uint32_t)(conflict.agent + 1));
    combine((uint64_t)(uint32_t)(conflict.taskPosition + 1));
  }
  return hashValue;
}

string LNS::iterationQualityName(IterationQuality quality) const {
  switch (quality) {
    case IterationQuality::bestSolutionYet:
      return "bestSolutionYet";
    case IterationQuality::improvedSolution:
      return "improvedSolution";
    case IterationQuality::downgradedButAccepted:
      return "downgradedButAccepted";
    case IterationQuality::couldNotFind:
      return "couldNotFind";
    case IterationQuality::none:
    default:
      return "none";
  }
}

bool LNS::writeIterationDebugTsv(const string& outputPath) const {
  std::filesystem::path path(outputPath);
  if (path.empty()) {
    return true;
  }
  const auto parent = path.parent_path();
  std::error_code ec;
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      PLOGE << "writeIterationDebugTsv: failed to create directory '"
            << parent.string() << "': " << ec.message() << "\n";
      return false;
    }
  }

  std::ofstream out(outputPath);
  if (!out.is_open()) {
    PLOGE << "writeIterationDebugTsv: unable to open '" << outputPath
          << "' for writing\n";
    return false;
  }
  out << "iteration\truntime_s\tprevious_soc\tcandidate_soc\tincumbent_soc_before\t"
         "candidate_valid\taccepted\tguard_rejected\taccepted_as_worse_utility\t"
         "feasible_best_update\tquality\tearly_abort_reason\tprevious_conflict\t"
         "candidate_conflict\tnrr_soft_candidate\tnrr_soft_only_invalid\t"
         "nrr_soft_conflicts\tsoft_recovery_mode_before\tsoft_recovery_mode_after\t"
         "soft_recovery_decision_reason\tdestroy_heuristic_id\tdestroy_heuristic_name\t"
         "destroy_selected_in_soft_mode\tremoved_tasks\tremoved_task_ids_csv\tremoved_tasks_changed_agent\t"
         "removed_tasks_changed_order\tremoved_tasks_unchanged\t"
         "neighborhood_fingerprint_seen_before\tneighborhood_fingerprint_hex\t"
         "neighborhood_jaccard_prev\tneighborhood_repeat_streak\t"
         "fingerprint_seen_before\tfingerprint_hex\t"
         "time_destroy_prepare_s\ttime_repair_commit_s\t"
         "time_regret_candidate_eval_s\ttime_regret_commit_s\t"
         "time_regret_low_level_s\ttime_join_s\t"
         "time_terminal_replan_s\ttime_recompute_soc_s\ttime_validation_s\t"
         "time_acceptance_s\ttime_bookkeeping_s\n";

  std::ios::fmtflags oldFlags = out.flags();
  std::streamsize oldPrecision = out.precision();
  out << std::fixed << std::setprecision(6);
  for (const auto& row : iterationDebugRecords_) {
    out << row.iteration << '\t'
        << row.runtimeSec << '\t'
        << row.previousSoc << '\t'
        << row.candidateSoc << '\t'
        << row.incumbentSocBefore << '\t'
        << (row.candidateValid ? 1 : 0) << '\t'
        << (row.accepted ? 1 : 0) << '\t'
        << (row.guardRejected ? 1 : 0) << '\t'
        << (row.acceptedAsWorseUtility ? 1 : 0) << '\t'
        << (row.feasibleBestUpdate ? 1 : 0) << '\t'
        << row.quality << '\t'
        << row.earlyAbortReason << '\t'
        << row.previousConflictSignal << '\t'
        << row.candidateConflictSignal << '\t'
        << (row.nrrSoftCandidate ? 1 : 0) << '\t'
        << (row.nrrSoftOnlyInvalid ? 1 : 0) << '\t'
        << row.nrrSoftConflictCount << '\t'
        << (row.softRecoveryModeBefore ? 1 : 0) << '\t'
        << (row.softRecoveryModeAfter ? 1 : 0) << '\t'
        << row.softRecoveryDecisionReason << '\t'
        << row.destroyHeuristicId << '\t'
        << row.destroyHeuristicName << '\t'
        << (row.destroySelectedInSoftMode ? 1 : 0) << '\t'
        << row.removedTasks << '\t'
        << row.removedTaskIdsCsv << '\t'
        << row.removedTasksChangedAgent << '\t'
        << row.removedTasksChangedOrder << '\t'
        << row.removedTasksUnchanged << '\t'
        << (row.neighborhoodFingerprintSeenBefore ? 1 : 0) << '\t';
    out << std::hex << row.neighborhoodFingerprint << std::dec << '\t'
        << row.neighborhoodJaccardPrev << '\t'
        << row.neighborhoodRepeatStreak << '\t'
        << (row.fingerprintSeenBefore ? 1 : 0) << '\t';
    out << std::hex << row.fingerprint << std::dec << '\t'
        << row.timeDestroyAndPrepareSec << '\t'
        << row.timeRepairAndCommitSec << '\t'
        << row.timeRegretCandidateEvalSec << '\t'
        << row.timeRegretCommitSec << '\t'
        << row.timeRegretLowLevelSec << '\t'
        << row.timeJoinPathsSec << '\t'
        << row.timeTerminalReplanSec << '\t'
        << row.timeRecomputeSocSec << '\t'
        << row.timeValidationSec << '\t'
        << row.timeAcceptanceSec << '\t'
        << row.timeBookkeepingSec << '\n';
  }
  out.flags(oldFlags);
  out.precision(oldPrecision);
  out.close();
  return out.good();
}

bool LNS::run() {
  constexpr double kPortfolioMinArmTimeSec = 1.0;
  constexpr double kAdaptivePortfolioMinFraction = 0.05;
  constexpr double kAdaptivePortfolioMaxFraction = 0.35;

  invalidateCurrentTaskAssignmentIndexCache();
  invalidCandidateRejections = 0;
  marketGuardRejections = 0;
  improvementDiagnosticsStats_.reset();
  initialSolutionRuntime_ = 0.0;
  initialSolutionRuntimeReported_ = 0.0;
  initialSeedRuntimeFromLogSec_ = -1.0;
  initialAssignmentSnapshotAvailable_ = false;
  initialMetricsAvailable_ = false;
  initialObjectiveValue_ = std::numeric_limits<int>::max();
  initialSoc_ = std::numeric_limits<int>::max();
  initialMakespan_ = -1;
  initialPrecedenceWait_ = std::numeric_limits<double>::quiet_NaN();
  initialAssignmentsByAgent_.clear();
  initialTaskOwnerByTask_.clear();
  initialTaskPosByTask_.clear();
  lnsLoopRuntimeSec_ = 0.0;
  postRefineRuntimeSec_ = 0.0;
  postRefineAttempted_ = false;
  postRefineAccepted_ = false;
  cumulativeRegretCandidateEvalSec_ = 0.0;
  cumulativeRegretCommitSec_ = 0.0;
  cumulativeLowLevelSearchSec_ = 0.0;
  improvementDiagnosticsStats_.softModeSelectionsByDestroy.assign(
      adaptiveLNS_.numDestroyHeuristics, 0);
  improvementDiagnosticsStats_.softModeAcceptedByDestroy.assign(
      adaptiveLNS_.numDestroyHeuristics, 0);
  improvementDiagnosticsStats_.softModeBestUpdatesByDestroy.assign(
      adaptiveLNS_.numDestroyHeuristics, 0);
  acceptedSolutionFingerprints_.clear();
  seenNeighborhoodFingerprints_.clear();
  previousNeighborhoodTasksSorted_.clear();
  currentNeighborhoodRepeatStreak_ = 0;
  iterationDebugRecords_.clear();
  softRecoveryActive_ = false;
  softRecoveryCurrentConflicts_ = -1;
  lastNrrSoftCandidate_ = false;
  lastNrrSoftConflictCount_ = -1;
  lastNrrSoftOnlyInvalid_ = false;
  lastDestroySampledInSoftMode_ = false;
  persistentConflictPairs_.clear();
  persistentConflictAgents_.clear();
  lastValidationCollisionPairs_.clear();
  lastValidationConflictAgents_.clear();
  lastValidationConflictTasks_.clear();
  lastSoftFailureConflictAgents_.clear();
  lastSoftFailureConflictTasks_.clear();
  auto flushDebugTsv = [&]() {
    if (debugIterationTsvPath_.empty()) {
      return;
    }
    if (!writeIterationDebugTsv(debugIterationTsvPath_)) {
      PLOGE << "Failed to write iteration debug TSV to '"
            << debugIterationTsvPath_ << "'\n";
    }
  };

  auto runInitialSolutionStrategy =
      [&](const string& strategy,
          std::optional<double> armBudgetSec = std::nullopt) -> bool {
    if (strategy == "seeded_mapfpc_log") {
      if (initialSeedFromMapfpcLog_.empty()) {
        PLOGE << "seeded_mapfpc_log requested but no seed log path was provided\n";
        return false;
      }
      const bool loaded =
          buildSeededSolutionFromMAPFPCLog(initialSeedFromMapfpcLog_);
      if (loaded) {
        const std::filesystem::path logPath(initialSeedFromMapfpcLog_);
        const std::string logName = logPath.filename().string();
        initialSolutionEffective_ = logName.empty()
                                        ? "seeded_mapfpc_log"
                                        : ("seeded_mapfpc_log(" + logName + ")");
      }
      return loaded;
    }
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

  auto captureFeasibleSnapshotFromCurrent = [&]() -> FeasibleSolution {
    FeasibleSolution snapshot;
    snapshot.numOfCols = instance_.numOfCols;
    snapshot.sumOfCosts = solution_.sumOfCosts;
    snapshot.agentPaths.resize(instance_.getAgentNum());
    snapshot.agentTaskAssignments.resize(instance_.getAgentNum());
    snapshot.agentTaskPaths.resize(instance_.getAgentNum());
    snapshot.taskAgentMap = solution_.taskAgentMap;
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      snapshot.agentPaths[agent] = solution_.agents[agent].path;
      snapshot.agentTaskAssignments[agent] =
          solution_.agents[agent].taskAssignments;
      snapshot.agentTaskPaths[agent] = solution_.agents[agent].taskPaths;
    }
    return snapshot;
  };

  auto restoreCurrentFromFeasibleSnapshot =
      [&](const FeasibleSolution& snapshot) -> bool {
    if ((int)snapshot.agentTaskAssignments.size() != instance_.getAgentNum() ||
        (int)snapshot.agentTaskPaths.size() != instance_.getAgentNum() ||
        (int)snapshot.agentPaths.size() != instance_.getAgentNum()) {
      PLOGE << "portfolio restore: malformed snapshot size (agents="
            << instance_.getAgentNum()
            << ", assignments=" << snapshot.agentTaskAssignments.size()
            << ", taskPaths=" << snapshot.agentTaskPaths.size()
            << ", paths=" << snapshot.agentPaths.size() << ")\n";
      return false;
    }
    if ((int)snapshot.taskAgentMap.size() != instance_.getTasksNum()) {
      PLOGE << "portfolio restore: malformed taskAgentMap size (expected "
            << instance_.getTasksNum() << ", got "
            << snapshot.taskAgentMap.size() << ")\n";
      return false;
    }

    solution_.sumOfCosts = snapshot.sumOfCosts;
    solution_.taskAgentMap = snapshot.taskAgentMap;
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      auto& dstAgent = solution_.agents[agent];
      dstAgent.taskAssignments = snapshot.agentTaskAssignments[agent];
      dstAgent.taskPaths = snapshot.agentTaskPaths[agent];
      dstAgent.path = snapshot.agentPaths[agent];
      dstAgent.terminalPath = AgentTaskPath();
      dstAgent.terminalPathActive = false;
      dstAgent.intraPrecedenceConstraints.clear();
      dstAgent.intraPrecedenceDirty = true;
      if (dstAgent.pathPlanner == nullptr) {
        PLOGE << "portfolio restore: missing planner for agent " << agent
              << "\n";
        return false;
      }
      dstAgent.pathPlanner->setGoalLocations(
          instance_.getTaskLocations(dstAgent.taskAssignments));
    }
    return true;
  };

  struct InitialCheckpoint {
    double runtimeSec = 0.0;
    string label;
    int objective = 0;
    bool feasible = false;
    IterationQuality quality = IterationQuality::none;
  };
  vector<InitialCheckpoint> initialCheckpoints;

  string requestedInitialStrategy = initialSolutionStrategy;
  if (!initialSeedFromMapfpcLog_.empty()) {
    requestedInitialStrategy = "seeded_mapfpc_log";
  }
  initialSolutionRequested_ = requestedInitialStrategy;
  initialSolutionEffective_ = requestedInitialStrategy;
  initialSolutionFallbackUsed_ = false;
  initialSolutionFallbackReason_ = "none";
  bool terminalPreparedDuringPortfolio = false;

  bool success = false;
  if (requestedInitialStrategy == "portfolio") {
    // Adaptive portfolio arm selection by instance hardness.
    const int agentCount = std::max(1, instance_.getAgentNum());
    const int taskCount = std::max(1, instance_.getTasksNum());
    const int precedenceCount = std::max(
        1, static_cast<int>(instance_.getInputPrecedenceConstraintsRef().size()));

    const double agents = static_cast<double>(agentCount);
    const double tasks = static_cast<double>(taskCount);
    const double precedence = static_cast<double>(precedenceCount);
    const double normalizedAgents = agents / 30.0;
    const double normalizedTasks = tasks / 200.0;
    const double normalizedPrecedence = precedence / 120.0;
    const double difficultyScore = 0.45 * std::log1p(normalizedAgents) +
                                   0.35 * std::log1p(normalizedTasks) +
                                   0.20 * std::log1p(normalizedPrecedence);
    const double referenceDifficulty = std::log1p(1.0);  // (30, 200, 120)
    double difficultyScale =
        (referenceDifficulty > 0.0) ? (difficultyScore / referenceDifficulty)
                                    : 1.0;
    if (!std::isfinite(difficultyScale) || difficultyScale <= 0.0) {
      difficultyScale = 1.0;
    }

    auto classifyByPc = [&](int pc) -> string {
      if (pc >= 450) return "hard";
      if (pc >= 120 && pc <= 280) return "medium";
      if (pc >= 80 && pc <= 100) return "easy";
      return "unknown";
    };
    auto classifyByAgents = [&](int agentsCount) -> string {
      if (agentsCount >= 200) return "hard";
      if (agentsCount >= 50 && agentsCount <= 100) return "medium";
      if (agentsCount >= 10 && agentsCount <= 30) return "easy";
      return "unknown";
    };
    auto classifyByTasks = [&](int tasksCount) -> string {
      if (tasksCount >= 800) return "hard";
      if (tasksCount >= 200 && tasksCount < 800) return "medium";
      if (tasksCount <= 100) return "easy";
      return "unknown";
    };

    const string pcTier = classifyByPc(precedenceCount);
    const string agentsTier = classifyByAgents(agentCount);
    const string tasksTier = classifyByTasks(taskCount);

    // Match dataset tiering preference order: PC first, then agents, then tasks.
    string portfolioHardness = "easy";
    if (pcTier != "unknown") {
      portfolioHardness = pcTier;
    } else if (agentsTier != "unknown") {
      portfolioHardness = agentsTier;
    } else if (tasksTier != "unknown") {
      portfolioHardness = tasksTier;
    }

    vector<string> portfolioArms = {"prioritized", "sota_pbs", "sota_cbs"};
    if (portfolioHardness == "hard") {
      portfolioHardness = "hard";
      portfolioArms = {"sota_pbs"};
    } else if (portfolioHardness == "medium") {
      portfolioArms = {"prioritized", "sota_pbs"};
    }
    string portfolioArmsCsv;
    for (size_t armIdx = 0; armIdx < portfolioArms.size(); armIdx++) {
      if (armIdx > 0) {
        portfolioArmsCsv += ",";
      }
      portfolioArmsCsv += portfolioArms[armIdx];
    }
    PLOGI << "portfolio arm selection: hardness=" << portfolioHardness
          << ", difficulty_scale=" << difficultyScale
          << ", pc_tier=" << pcTier << ", agents_tier=" << agentsTier
          << ", tasks_tier=" << tasksTier
          << ", arms=" << portfolioArmsCsv
          << " (agents=" << agentCount << ", tasks=" << taskCount
          << ", precedence_edges=" << precedenceCount << ")\n";

    const bool singleArmPortfolio = (portfolioArms.size() == 1);
    double portfolioBudgetFraction = initialPortfolioTimeFraction_;
    if (adaptiveInitialPortfolioBudget_) {
      if (initialPortfolioTimeFraction_ <= 0.0) {
        portfolioBudgetFraction = 0.0;
      } else {
        if (std::isfinite(difficultyScale)) {
          portfolioBudgetFraction = std::clamp(
              initialPortfolioTimeFraction_ * difficultyScale,
              kAdaptivePortfolioMinFraction,
              kAdaptivePortfolioMaxFraction);
        } else {
          portfolioBudgetFraction = initialPortfolioTimeFraction_;
        }
      }
      PLOGI << "adaptiveInitialPortfolioBudget enabled: base_fraction="
            << initialPortfolioTimeFraction_ << ", effective_fraction="
            << portfolioBudgetFraction << " (agents=" << agentCount
            << ", tasks=" << taskCount << ", precedence_edges="
            << precedenceCount << ")\n";
    }
    const double remainingBudget = remainingRuntimeBudgetSec();
    double portfolioBudgetSec = 0.0;
    if (singleArmPortfolio) {
      // With a single portfolio arm there is no exploration/exploitation split.
      // Ignore the portfolio fraction and let initializer use full remaining
      // runtime budget.
      portfolioBudgetSec = remainingBudget;
    } else {
      portfolioBudgetSec = std::min(remainingBudget,
                                    max(0.0, timeLimit_ * portfolioBudgetFraction));
      if (portfolioBudgetSec <= 0.0 && remainingBudget > 0.0) {
        // Ensure at least one short arm when portfolio is explicitly requested.
        portfolioBudgetSec =
            std::min(remainingBudget, kPortfolioMinArmTimeSec);
      }
    }

    struct PortfolioCandidate {
      string arm;
      int objective;
      FeasibleSolution snapshot;

      PortfolioCandidate(const string& armName, int objectiveValue,
                         FeasibleSolution solutionSnapshot)
          : arm(armName),
            objective(objectiveValue),
            snapshot(std::move(solutionSnapshot)) {}
    };
    vector<PortfolioCandidate> feasiblePortfolioCandidates;

    bool haveBestPortfolioSolution = false;
    string bestPortfolioArm;
    Solution bestPortfolioSolution(instance_);
    int bestPortfolioObjective = std::numeric_limits<int>::max();

    for (int i = 0; i < (int)portfolioArms.size(); i++) {
      if (runtimeBudgetExhausted() ||
          (!singleArmPortfolio && portfolioBudgetSec <= 0.0)) {
        break;
      }
      double armBudgetSec = 0.0;
      if (singleArmPortfolio) {
        armBudgetSec = remainingRuntimeBudgetSec();
      } else {
        const int armsLeft = (int)portfolioArms.size() - i;
        const double fairShare = portfolioBudgetSec / max(1, armsLeft);
        armBudgetSec = std::max(kPortfolioMinArmTimeSec, fairShare);
        armBudgetSec = std::min(armBudgetSec, portfolioBudgetSec);
        armBudgetSec = std::min(armBudgetSec, remainingRuntimeBudgetSec());
      }
      if (armBudgetSec <= 0.0) {
        break;
      }

      const string& arm = portfolioArms[i];
      const double armStartSec = elapsedRuntimeSec();
      const bool armSuccess =
          runInitialSolutionStrategyWithBudget(arm, armBudgetSec);

      bool armFeasible = false;
      int armObjective = 0;
      IterationQuality armQuality = IterationQuality::none;
      if (armSuccess) {
        armObjective = currentObjectiveValue();
        ValidationStats armValidationStats;
        ConflictMap armPotentialNeighborhood;
        const bool previousTerminalValidationFlag =
            useTerminalPathsInValidation_;
        useTerminalPathsInValidation_ = false;
        armFeasible = validateSolution(&armPotentialNeighborhood,
                                       &armValidationStats);
        useTerminalPathsInValidation_ = previousTerminalValidationFlag;
        if (armFeasible) {
          feasiblePortfolioCandidates.emplace_back(
              arm, armObjective, captureFeasibleSnapshotFromCurrent());
          if (armObjective < bestPortfolioObjective) {
            bestPortfolioObjective = armObjective;
            armQuality = IterationQuality::bestSolutionYet;
          }
        }
      }
      const double armEndSec = elapsedRuntimeSec();
      const double consumedBudget = max(0.0, armEndSec - armStartSec);
      if (!singleArmPortfolio) {
        portfolioBudgetSec = max(0.0, portfolioBudgetSec - consumedBudget);
      }

      InitialCheckpoint checkpoint;
      checkpoint.runtimeSec = armEndSec;
      checkpoint.label = "InitPortfolio:" + arm;
      checkpoint.objective = armObjective;
      checkpoint.feasible = armFeasible;
      checkpoint.quality = armQuality;
      initialCheckpoints.push_back(std::move(checkpoint));
    }

    if (!feasiblePortfolioCandidates.empty()) {
      std::stable_sort(
          feasiblePortfolioCandidates.begin(), feasiblePortfolioCandidates.end(),
          [](const PortfolioCandidate& a, const PortfolioCandidate& b) {
            return a.objective < b.objective;
          });

      for (const auto& candidate : feasiblePortfolioCandidates) {
        if (!restoreCurrentFromFeasibleSnapshot(candidate.snapshot)) {
          continue;
        }
        bool candidateFeasible = false;
        if (isGoalOccupationRepositionTrue()) {
          vector<int> allAgents(instance_.getAgentNum());
          std::iota(allAgents.begin(), allAgents.end(), 0);
          if (!planTerminalReposition(allAgents, true)) {
            continue;
          }
          terminalPreparedDuringPortfolio = true;
        }

        ValidationStats validationStats;
        ConflictMap potentialNeighborhood;
        const bool previousTerminalValidationFlag =
            useTerminalPathsInValidation_;
        useTerminalPathsInValidation_ = isGoalOccupationRepositionTrue();
        candidateFeasible =
            validateSolution(&potentialNeighborhood, &validationStats);
        useTerminalPathsInValidation_ = previousTerminalValidationFlag;

        if (candidateFeasible) {
          bestPortfolioArm = candidate.arm;
          bestPortfolioSolution = solution_;
          haveBestPortfolioSolution = true;
          break;
        }
      }
    }

    if (haveBestPortfolioSolution) {
      solution_ = bestPortfolioSolution;
      invalidateCurrentTaskAssignmentIndexCache();
      initialSolutionEffective_ = "portfolio(" + bestPortfolioArm + ")";
      initialSolutionFallbackReason_ = "none";
      success = true;
    } else if (!feasiblePortfolioCandidates.empty()) {
      initialSolutionFallbackReason_ = "portfolio_no_terminal_feasible_arm";
      success = false;
    } else {
      initialSolutionFallbackReason_ = "portfolio_no_feasible_arm";
      success = false;
    }
  } else {
    success = runInitialSolutionStrategy(requestedInitialStrategy);
  }

  // If the requested initial solution strategy fails, terminate directly.
  if (!success) {
    if (initialSolutionFallbackReason_ == "none") {
      initialSolutionFallbackReason_ = "initializer_failed";
    }
    flushDebugTsv();
    return success;
  }

  initialSolutionRuntime_ = ((fsec)(Time::now() - plannerStartTime_)).count();
  initialSolutionRuntimeReported_ = initialSolutionRuntime_;
  if (requestedInitialStrategy == "seeded_mapfpc_log" &&
      initialSeedRuntimeFromLogSec_ >= 0.0 &&
      std::isfinite(initialSeedRuntimeFromLogSec_)) {
    initialSolutionRuntimeReported_ = initialSeedRuntimeFromLogSec_;
  }
  improvementDiagnosticsStats_.timeInitialSolutionSec = initialSolutionRuntime_;
  improvementDiagnosticsStats_.timeInitialSolutionReportedSec =
      initialSolutionRuntimeReported_;
  runtime = initialSolutionRuntime_;

  PLOGD << "Initial solution cost = " << solution_.sumOfCosts
        << ", Objective(" << optimizationObjective_
        << ") = " << currentObjectiveValue()
        << ", Runtime(measured) = " << initialSolutionRuntime_
        << ", Runtime(reported) = " << initialSolutionRuntimeReported_
        << "\n";

  if (isGoalOccupationRepositionTrue() &&
      !terminalPreparedDuringPortfolio) {
    vector<int> allAgents(instance_.getAgentNum());
    std::iota(allAgents.begin(), allAgents.end(), 0);
    if (!planTerminalReposition(allAgents, true)) {
      PLOGE << "run: true terminal reposition planning failed during "
               "initialization\n";
      flushDebugTsv();
      return false;
    }
  }

  ConflictMap potentialNeighborhood;  // Need for the conflict removal case
  ValidationStats currentValidationStats;
  vector<pair<int, int>> initialCollisionPairs;
  useTerminalPathsInValidation_ = isGoalOccupationRepositionTrue();
  bool currentSolutionValid =
      validateSolution(&potentialNeighborhood, &currentValidationStats,
                       &initialCollisionPairs);
  useTerminalPathsInValidation_ = false;
  std::sort(initialCollisionPairs.begin(), initialCollisionPairs.end());
  initialCollisionPairs.erase(
      std::unique(initialCollisionPairs.begin(), initialCollisionPairs.end()),
      initialCollisionPairs.end());
  lastValidationCollisionPairs_ = initialCollisionPairs;
  lastValidationConflictTasks_.clear();
  lastValidationConflictTasks_.reserve(potentialNeighborhood.size());
  std::unordered_set<int> initialConflictAgents;
  for (const auto& [task, conflict] : potentialNeighborhood) {
    lastValidationConflictTasks_.push_back(task);
    if (conflict.agent >= 0 && conflict.agent < instance_.getAgentNum()) {
      initialConflictAgents.insert(conflict.agent);
    } else if (task >= 0 && task < (int)solution_.taskAgentMap.size()) {
      const int owner = solution_.taskAgentMap[task];
      if (owner >= 0 && owner < instance_.getAgentNum()) {
        initialConflictAgents.insert(owner);
      }
    }
  }
  for (const auto& [a, b] : initialCollisionPairs) {
    if (a >= 0 && a < instance_.getAgentNum()) {
      initialConflictAgents.insert(a);
    }
    if (b >= 0 && b < instance_.getAgentNum()) {
      initialConflictAgents.insert(b);
    }
  }
  lastValidationConflictAgents_.assign(initialConflictAgents.begin(),
                                       initialConflictAgents.end());
  std::sort(lastValidationConflictAgents_.begin(),
            lastValidationConflictAgents_.end());
  if (softPersistentConflictGraph_) {
    persistentConflictPairs_ = initialCollisionPairs;
    persistentConflictAgents_ = lastValidationConflictAgents_;
  }

  bool feasibleSolutionUpdated = false;
  if (currentSolutionValid) {
    feasibleSolutionUpdated = true;
    extractFeasibleSolution();
  }

  // Snapshot the initial assignment state for end-of-run delta diagnostics.
  const int taskCount = instance_.getTasksNum();
  const int agentCount = instance_.getAgentNum();
  initialAssignmentsByAgent_.assign(agentCount, {});
  for (int agent = 0; agent < agentCount; agent++) {
    initialAssignmentsByAgent_[agent] = solution_.agents[agent].taskAssignments;
  }
  initialTaskOwnerByTask_.assign(taskCount, UNASSIGNED);
  initialTaskPosByTask_.assign(taskCount, UNASSIGNED);
  for (int agent = 0; agent < agentCount; agent++) {
    const auto& tasks = initialAssignmentsByAgent_[agent];
    for (int pos = 0; pos < (int)tasks.size(); pos++) {
      const int task = tasks[pos];
      if (task < 0 || task >= taskCount) {
        continue;
      }
      initialTaskOwnerByTask_[task] = agent;
      initialTaskPosByTask_[task] = pos;
    }
  }
  initialAssignmentSnapshotAvailable_ = true;
  if (currentSolutionValid) {
    long long initialMakespanLl = 0;
    for (const auto& agent : solution_.agents) {
      initialMakespanLl = std::max(
          initialMakespanLl, static_cast<long long>(agent.path.endTimeOrZero()));
    }
    initialMakespan_ =
        initialMakespanLl > std::numeric_limits<int>::max()
            ? std::numeric_limits<int>::max()
            : static_cast<int>(initialMakespanLl);
    initialSoc_ = solution_.sumOfCosts;
    initialObjectiveValue_ = currentObjectiveValue();
    initialPrecedenceWait_ = computeSolutionPrecedenceWait();
    initialMetricsAvailable_ = true;
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
        instance_.getTasksNum(), checkpoint.objective, checkpoint.feasible,
        checkpoint.quality));
  }
  if (initialCheckpoints.empty() ||
      (!checkpointHasFeasible && feasibleSolutionUpdated)) {
    long long initialMakespanLl = 0;
    for (const auto& agent : solution_.agents) {
      initialMakespanLl = std::max(
          initialMakespanLl, static_cast<long long>(agent.path.endTimeOrZero()));
    }
    const int initialMakespan =
        initialMakespanLl > std::numeric_limits<int>::max()
            ? std::numeric_limits<int>::max()
            : static_cast<int>(initialMakespanLl);
    appendIterationStatBounded(IterationStats(
        initialSolutionRuntime_, initialSolutionEffective_,
        instance_.getAgentNum(), instance_.getTasksNum(), currentObjectiveValue(),
        feasibleSolutionUpdated, IterationQuality::bestSolutionYet, 0, 0,
        feasibleSolutionUpdated ? initialMakespan : -1,
        feasibleSolutionUpdated ? computeSolutionPrecedenceWait()
                                : std::numeric_limits<double>::quiet_NaN()));
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
  const int initialConflictSignal = currentValidationStats.totalConflictEvents();
  const int initialObjective = currentObjectiveValue();
  MovingMetrics metrics(metricsWindowSize, lnsConflictWeight_, lnsCostWeight_,
                        initialConflictSignal, initialObjective);
  solution_.utility =
      metrics.computeMovingMetrics(initialConflictSignal, initialObjective);

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
        max(1.0, std::abs(static_cast<double>(initialObjective)));
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

  int64_t executedLnsIterations = 0;

  // LNS loop
  const Time::time_point lnsLoopStart = Time::now();
  while (runtime < timeLimit_ &&
         (numOfIterations_ <= 0 ||
          executedLnsIterations < static_cast<int64_t>(numOfIterations_))) {
    if (!runOneIteration(potentialNeighborhood, oldNeighborhood, metrics,
                         currentSolutionValid, currentValidationStats,
                         feasibleSolutionUpdated)) {
      flushDebugTsv();
      return false;
    }
    executedLnsIterations++;
  }
  lnsLoopRuntimeSec_ = ((fsec)(Time::now() - lnsLoopStart)).count();
  improvementDiagnosticsStats_.timeLnsLoopSec = lnsLoopRuntimeSec_;

  postRefineRuntimeSec_ = 0.0;
  if (postRefineWithMapfpc_) {
    postRefineAttempted_ = true;
    const Time::time_point postRefineStart = Time::now();
    const bool accepted = runPostMAPFPCRefinement();
    postRefineAccepted_ = accepted;
    postRefineRuntimeSec_ = ((fsec)(Time::now() - postRefineStart)).count();
    improvementDiagnosticsStats_.timePostRefineSec = postRefineRuntimeSec_;
    PLOGI << "post_refine_mapfpc: accepted="
          << (accepted ? "true" : "false") << "\n";
  }

  runtime = elapsedRuntimeSec();
  // printPaths();
  flushDebugTsv();
  return !incumbentSolution_.agentPaths.empty();
}
