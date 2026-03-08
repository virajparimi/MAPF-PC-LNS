#include "lns.hpp"
#include "lns_repair_engine.hpp"

RepairPlan RepairEngine::buildPlan(bool enableNrrRepair,
                                   bool nrrFallbackToStandard,
                                   bool incrementalRegret,
                                   int repairHeuristicMode) {
  RepairPlan plan;
  plan.tryNrr = enableNrrRepair;
  plan.allowFallbackToStandard = nrrFallbackToStandard;

  const auto mode =
      static_cast<LNS::RepairHeuristicMode>(repairHeuristicMode);
  switch (mode) {
    case LNS::RepairHeuristicMode::mapfpc_fixed:
      plan.strategy = RepairStrategy::mapfpc_fixed;
      break;
    case LNS::RepairHeuristicMode::mapfpc_neighborhood_fixed:
      plan.strategy = RepairStrategy::mapfpc_neighborhood_fixed;
      break;
    case LNS::RepairHeuristicMode::mapfpc_neighborhood_reassign_greedy:
      plan.strategy = RepairStrategy::mapfpc_neighborhood_reassign_greedy;
      break;
    default:
      plan.strategy = incrementalRegret ? RepairStrategy::incremental_regret
                                        : RepairStrategy::full_regret;
      break;
  }
  return plan;
}

bool RepairEngine::run(LNS& lns, bool& repairFailed, bool& nrrRepairSucceeded) {
  return lns.runRepairEngine(repairFailed, nrrRepairSucceeded);
}

bool LNS::runRepairEngine(bool& repairFailed, bool& nrrRepairSucceeded) {
  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };
  lnsNeighborhood_.regretMaxHeap.clear();
  std::fill(regretBestOption_.begin(), regretBestOption_.end(),
            std::make_pair(UNASSIGNED, -1));
  std::fill(regretSecondBestOption_.begin(),
            regretSecondBestOption_.end(),
            std::make_pair(UNASSIGNED, -1));
  for (auto& candidateAgents : regretCandidateAgents_) {
    candidateAgents.clear();
  }
  repairFailed = false;
  nrrRepairSucceeded = false;
  regretEvalStatsCurrent_.reset();
  lastNrrSoftCandidate_ = false;
  lastNrrSoftConflictCount_ = -1;
  lastNrrSoftOnlyInvalid_ = false;
  {
    const int removedCount = (int)lnsNeighborhood_.removedTasks.size();
    regretEvalStatsCurrent_.neighborhoods++;
    regretEvalStatsTotal_.neighborhoods++;
    regretEvalStatsCurrent_.removedTasksSum += removedCount;
    regretEvalStatsTotal_.removedTasksSum += removedCount;
    regretEvalStatsCurrent_.removedTasksMax =
        max(regretEvalStatsCurrent_.removedTasksMax,
            (int64_t)removedCount);
    regretEvalStatsTotal_.removedTasksMax =
        max(regretEvalStatsTotal_.removedTasksMax, (int64_t)removedCount);
  }

  const RepairPlan plan = RepairEngine::buildPlan(
      enableNrrRepair_, nrrFallbackToStandard_, incrementalRegret_,
      static_cast<int>(getRepairHeuristicMode()));

  if (plan.tryNrr && !lnsNeighborhood_.removedTasks.empty()) {
    if (runtimeBudgetExhausted()) {
      nrrRepairSucceeded = false;
      PLOGW << "nrr_repair: skipped due to exhausted runtime budget\n";
    } else {
      nrrRepairSucceeded = runNeighborhoodReoptimizationRepair();
    }
    if (nrrRepairSucceeded) {
      lnsNeighborhood_.removedTasks.clear();
      lnsNeighborhood_.regretMaxHeap.clear();
    } else {
      if (plan.allowFallbackToStandard) {
        nrrStats_.fallbackToStandard++;
        PLOGI << "nrr_repair: fallback to standard repair\n";
      } else {
        PLOGI << "nrr_repair: fallback disabled; terminating repair for this "
                 "iteration\n";
        repairFailed = true;
      }
    }
  }

  if (repairFailed || nrrRepairSucceeded) {
    return !repairFailed;
  }

  if (plan.strategy == RepairStrategy::mapfpc_fixed) {
    if (runtimeBudgetExhausted() || !runFixedAssignmentMapfpcRepair()) {
      repairFailed = true;
    } else {
      lnsNeighborhood_.removedTasks.clear();
      lnsNeighborhood_.regretMaxHeap.clear();
    }
    return !repairFailed;
  }
  if (plan.strategy == RepairStrategy::mapfpc_neighborhood_fixed) {
    if (runtimeBudgetExhausted() ||
        !runNeighborhoodFixedMapfpcRepair()) {
      repairFailed = true;
    } else {
      lnsNeighborhood_.removedTasks.clear();
      lnsNeighborhood_.regretMaxHeap.clear();
    }
    return !repairFailed;
  }
  if (plan.strategy == RepairStrategy::mapfpc_neighborhood_reassign_greedy) {
    if (runtimeBudgetExhausted() ||
        !runNeighborhoodReassignGreedyMapfpcRepair()) {
      repairFailed = true;
    } else {
      lnsNeighborhood_.removedTasks.clear();
      lnsNeighborhood_.regretMaxHeap.clear();
    }
    return !repairFailed;
  }

  if (plan.strategy == RepairStrategy::full_regret) {
    while (!lnsNeighborhood_.removedTasks.empty()) {
      if (runtimeBudgetExhausted()) {
        repairFailed = true;
        break;
      }
      const Time::time_point evalStart = Time::now();
      bool enoughSpace = computeRegret();
      cumulativeRegretCandidateEvalSec_ += elapsedSecSince(evalStart);
      if (!enoughSpace) {
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
      const Time::time_point commitStart = Time::now();
      if (!commitBestRegretTask(bestRegret)) {
        cumulativeRegretCommitSec_ += elapsedSecSince(commitStart);
        PLOGE << "run: failed to commit best-regret task " << bestRegret.task
              << "\n";
        repairFailed = true;
        break;
      }
      cumulativeRegretCommitSec_ += elapsedSecSince(commitStart);
    }
    return !repairFailed;
  }

  incrementalRegretStatsCurrent_.reset();
  const Time::time_point initialRecomputeStart = Time::now();
  if (runtimeBudgetExhausted() ||
      !recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
    cumulativeRegretCandidateEvalSec_ +=
        elapsedSecSince(initialRecomputeStart);
    repairFailed = true;
    return false;
  }
  cumulativeRegretCandidateEvalSec_ +=
      elapsedSecSince(initialRecomputeStart);

  int64_t stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
  int64_t stalePopsSinceRefresh = 0;
  int64_t commitsSinceRefresh = 0;
  int64_t consecutiveStaleGrowthCommits = 0;
  int refreshCooldownCommits = 0;
  bool endgameFullRefreshDone = false;
  enum class RefreshReason { high_stale, stale_growth, periodic };
  auto refreshRemainingRegrets = [&](RefreshReason reason) {
    lnsNeighborhood_.regretMaxHeap.clear();
    incrementalRegretStatsCurrent_.fullRefreshes++;
    incrementalRegretStatsTotal_.fullRefreshes++;
    if (reason == RefreshReason::high_stale) {
      incrementalRegretStatsCurrent_.refreshByHighStale++;
      incrementalRegretStatsTotal_.refreshByHighStale++;
    } else if (reason == RefreshReason::stale_growth) {
      incrementalRegretStatsCurrent_.refreshByStaleGrowth++;
      incrementalRegretStatsTotal_.refreshByStaleGrowth++;
    } else if (reason == RefreshReason::periodic) {
      incrementalRegretStatsCurrent_.refreshByPeriodic++;
      incrementalRegretStatsTotal_.refreshByPeriodic++;
    }
    const Time::time_point refreshRecomputeStart = Time::now();
    if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
      cumulativeRegretCandidateEvalSec_ +=
          elapsedSecSince(refreshRecomputeStart);
      repairFailed = true;
      return;
    }
    cumulativeRegretCandidateEvalSec_ +=
        elapsedSecSince(refreshRecomputeStart);
    stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
    stalePopsSinceRefresh = 0;
    commitsSinceRefresh = 0;
    consecutiveStaleGrowthCommits = 0;
    refreshCooldownCommits = 2;
  };

  while (!repairFailed && !lnsNeighborhood_.removedTasks.empty()) {
    if (runtimeBudgetExhausted()) {
      repairFailed = true;
      break;
    }

    if (refreshCooldownCommits > 0) {
      refreshCooldownCommits--;
    } else {
      const bool highStaleLoad =
          stalePopsSinceRefresh >= 120 &&
          (double)stalePopsSinceRefresh >
              (2.5 * (double)commitsSinceRefresh + 60.0);
      const bool staleGrowthStall =
          consecutiveStaleGrowthCommits >= 8 && stalePopsSinceRefresh >= 80 &&
          commitsSinceRefresh >= 10;
      const bool periodicSafety =
          stalePopsSinceRefresh >= 40 && commitsSinceRefresh >= 20;
      if (highStaleLoad) {
        refreshRemainingRegrets(RefreshReason::high_stale);
      } else if (staleGrowthStall) {
        refreshRemainingRegrets(RefreshReason::stale_growth);
      } else if (periodicSafety) {
        refreshRemainingRegrets(RefreshReason::periodic);
      }
      if (repairFailed) {
        break;
      }
    }

    const auto bestRegret = popNextValidRegret();
    const int64_t staleDelta =
        incrementalRegretStatsCurrent_.stalePops - stalePopsAtLastCheck;
    stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
    stalePopsSinceRefresh += staleDelta;
    if (staleDelta > 0) {
      consecutiveStaleGrowthCommits++;
    } else {
      consecutiveStaleGrowthCommits = 0;
    }

    if (!bestRegret.has_value()) {
      incrementalRegretStatsCurrent_.heapRebuilds++;
      incrementalRegretStatsTotal_.heapRebuilds++;
      const Time::time_point rebuildStart = Time::now();
      if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
        cumulativeRegretCandidateEvalSec_ += elapsedSecSince(rebuildStart);
        repairFailed = true;
      } else {
        cumulativeRegretCandidateEvalSec_ += elapsedSecSince(rebuildStart);
        stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
        stalePopsSinceRefresh = 0;
        commitsSinceRefresh = 0;
        consecutiveStaleGrowthCommits = 0;
        refreshCooldownCommits = 2;
      }
      continue;
    }

    const Time::time_point commitStart = Time::now();
    const vector<int> endTimesBefore = computeCurrentTaskEndTimes();
    const vector<uint64_t> agentSignaturesBefore =
        computeCurrentAgentScheduleSignatures();
    if (!commitBestRegretTask(*bestRegret)) {
      cumulativeRegretCommitSec_ += elapsedSecSince(commitStart);
      PLOGE << "run: failed to commit best-regret task " << bestRegret->task
            << "\n";
      repairFailed = true;
      break;
    }
    cumulativeRegretCommitSec_ += elapsedSecSince(commitStart);
    incrementalRegretStatsCurrent_.commits++;
    incrementalRegretStatsTotal_.commits++;
    commitsSinceRefresh++;
    const vector<int> endTimesAfter = computeCurrentTaskEndTimes();
    const vector<uint64_t> agentSignaturesAfter =
        computeCurrentAgentScheduleSignatures();

    const vector<int> dirtyTasks = computeDirtyTasksAfterCommit(
        endTimesBefore, endTimesAfter, agentSignaturesBefore,
        agentSignaturesAfter);
    vector<int> tasksToRecompute = dirtyTasks;
    const int remainingRemovedTasks =
        (int)lnsNeighborhood_.removedTasks.size();
    if (!endgameFullRefreshDone && remainingRemovedTasks <= 4 &&
        (stalePopsSinceRefresh >= 30 || commitsSinceRefresh >= 10)) {
      incrementalRegretStatsCurrent_.endgameFullRecomputes++;
      incrementalRegretStatsTotal_.endgameFullRecomputes++;
      tasksToRecompute = collectRemainingRemovedTasks();
      endgameFullRefreshDone = true;
    }
    const Time::time_point localRecomputeStart = Time::now();
    if (!recomputeRegretsForTasks(tasksToRecompute)) {
      cumulativeRegretCandidateEvalSec_ +=
          elapsedSecSince(localRecomputeStart);
      repairFailed = true;
    } else {
      cumulativeRegretCandidateEvalSec_ +=
          elapsedSecSince(localRecomputeStart);
    }
  }
  return !repairFailed;
}
