#include "lns.hpp"
#include "lns_repair_engine.hpp"

RepairPlan RepairEngine::buildPlan(bool enableNrrRepair,
                                   bool nrrFallbackToStandard,
                                   bool incrementalRegret,
                                   const std::string& repairHeuristic) {
  RepairPlan plan;
  plan.tryNrr = enableNrrRepair;
  plan.allowFallbackToStandard = nrrFallbackToStandard;

  if (repairHeuristic == "mapfpc_fixed") {
    plan.strategy = RepairStrategy::mapfpc_fixed;
  } else if (repairHeuristic == "mapfpc_neighborhood_fixed") {
    plan.strategy = RepairStrategy::mapfpc_neighborhood_fixed;
  } else if (repairHeuristic == "mapfpc_neighborhood_reassign_greedy") {
    plan.strategy = RepairStrategy::mapfpc_neighborhood_reassign_greedy;
  } else if (incrementalRegret) {
    plan.strategy = RepairStrategy::incremental_regret;
  } else {
    plan.strategy = RepairStrategy::full_regret;
  }
  return plan;
}

bool RepairEngine::run(LNS& lns, bool& repairFailed, bool& nrrRepairSucceeded) {
  lns.lnsNeighborhood_.regretMaxHeap.clear();
  std::fill(lns.regretBestOption_.begin(), lns.regretBestOption_.end(),
            std::make_pair(UNASSIGNED, -1));
  std::fill(lns.regretSecondBestOption_.begin(),
            lns.regretSecondBestOption_.end(),
            std::make_pair(UNASSIGNED, -1));
  for (auto& candidateAgents : lns.regretCandidateAgents_) {
    candidateAgents.clear();
  }
  repairFailed = false;
  nrrRepairSucceeded = false;
  lns.regretEvalStatsCurrent_.reset();
  {
    const int removedCount = (int)lns.lnsNeighborhood_.removedTasks.size();
    lns.regretEvalStatsCurrent_.neighborhoods++;
    lns.regretEvalStatsTotal_.neighborhoods++;
    lns.regretEvalStatsCurrent_.removedTasksSum += removedCount;
    lns.regretEvalStatsTotal_.removedTasksSum += removedCount;
    lns.regretEvalStatsCurrent_.removedTasksMax =
        max(lns.regretEvalStatsCurrent_.removedTasksMax,
            (int64_t)removedCount);
    lns.regretEvalStatsTotal_.removedTasksMax =
        max(lns.regretEvalStatsTotal_.removedTasksMax, (int64_t)removedCount);
  }

  const RepairPlan plan = RepairEngine::buildPlan(
      lns.enableNrrRepair_, lns.nrrFallbackToStandard_, lns.incrementalRegret_,
      lns.repairHeuristic);

  if (plan.tryNrr && !lns.lnsNeighborhood_.removedTasks.empty()) {
    if (lns.runtimeBudgetExhausted()) {
      nrrRepairSucceeded = false;
      PLOGW << "nrr_repair: skipped due to exhausted runtime budget\n";
    } else {
      nrrRepairSucceeded = lns.runNeighborhoodReoptimizationRepair();
    }
    if (nrrRepairSucceeded) {
      lns.lnsNeighborhood_.removedTasks.clear();
      lns.lnsNeighborhood_.regretMaxHeap.clear();
    } else {
      if (plan.allowFallbackToStandard) {
        lns.nrrStats_.fallbackToStandard++;
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
    if (lns.runtimeBudgetExhausted() || !lns.runFixedAssignmentMapfpcRepair()) {
      repairFailed = true;
    } else {
      lns.lnsNeighborhood_.removedTasks.clear();
      lns.lnsNeighborhood_.regretMaxHeap.clear();
    }
    return !repairFailed;
  }
  if (plan.strategy == RepairStrategy::mapfpc_neighborhood_fixed) {
    if (lns.runtimeBudgetExhausted() ||
        !lns.runNeighborhoodFixedMapfpcRepair()) {
      repairFailed = true;
    } else {
      lns.lnsNeighborhood_.removedTasks.clear();
      lns.lnsNeighborhood_.regretMaxHeap.clear();
    }
    return !repairFailed;
  }
  if (plan.strategy == RepairStrategy::mapfpc_neighborhood_reassign_greedy) {
    if (lns.runtimeBudgetExhausted() ||
        !lns.runNeighborhoodReassignGreedyMapfpcRepair()) {
      repairFailed = true;
    } else {
      lns.lnsNeighborhood_.removedTasks.clear();
      lns.lnsNeighborhood_.regretMaxHeap.clear();
    }
    return !repairFailed;
  }

  if (plan.strategy == RepairStrategy::full_regret) {
    while (!lns.lnsNeighborhood_.removedTasks.empty()) {
      if (lns.runtimeBudgetExhausted()) {
        repairFailed = true;
        break;
      }
      bool enoughSpace = lns.computeRegret();
      if (!enoughSpace) {
        repairFailed = true;
        break;
      }
      if (lns.lnsNeighborhood_.regretMaxHeap.empty()) {
        PLOGE << "regretMaxHeap is empty after computeRegret\n";
        repairFailed = true;
        break;
      }
      assert(!lns.lnsNeighborhood_.regretMaxHeap.empty());
      Regret bestRegret = lns.lnsNeighborhood_.regretMaxHeap.top();
      if (!lns.commitBestRegretTask(bestRegret)) {
        PLOGE << "run: failed to commit best-regret task " << bestRegret.task
              << "\n";
        repairFailed = true;
        break;
      }
    }
    return !repairFailed;
  }

  lns.incrementalRegretStatsCurrent_.reset();
  if (lns.runtimeBudgetExhausted() ||
      !lns.recomputeRegretsForTasks(lns.collectRemainingRemovedTasks())) {
    repairFailed = true;
    return false;
  }

  int64_t stalePopsAtLastCheck = lns.incrementalRegretStatsCurrent_.stalePops;
  int64_t stalePopsSinceRefresh = 0;
  int64_t commitsSinceRefresh = 0;
  int64_t consecutiveStaleGrowthCommits = 0;
  int refreshCooldownCommits = 0;
  bool endgameFullRefreshDone = false;
  enum class RefreshReason { high_stale, stale_growth, periodic };
  auto refreshRemainingRegrets = [&](RefreshReason reason) {
    lns.lnsNeighborhood_.regretMaxHeap.clear();
    lns.incrementalRegretStatsCurrent_.fullRefreshes++;
    lns.incrementalRegretStatsTotal_.fullRefreshes++;
    if (reason == RefreshReason::high_stale) {
      lns.incrementalRegretStatsCurrent_.refreshByHighStale++;
      lns.incrementalRegretStatsTotal_.refreshByHighStale++;
    } else if (reason == RefreshReason::stale_growth) {
      lns.incrementalRegretStatsCurrent_.refreshByStaleGrowth++;
      lns.incrementalRegretStatsTotal_.refreshByStaleGrowth++;
    } else if (reason == RefreshReason::periodic) {
      lns.incrementalRegretStatsCurrent_.refreshByPeriodic++;
      lns.incrementalRegretStatsTotal_.refreshByPeriodic++;
    }
    if (!lns.recomputeRegretsForTasks(lns.collectRemainingRemovedTasks())) {
      repairFailed = true;
      return;
    }
    stalePopsAtLastCheck = lns.incrementalRegretStatsCurrent_.stalePops;
    stalePopsSinceRefresh = 0;
    commitsSinceRefresh = 0;
    consecutiveStaleGrowthCommits = 0;
    refreshCooldownCommits = 2;
  };

  while (!repairFailed && !lns.lnsNeighborhood_.removedTasks.empty()) {
    if (lns.runtimeBudgetExhausted()) {
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

    const auto bestRegret = lns.popNextValidRegret();
    const int64_t staleDelta =
        lns.incrementalRegretStatsCurrent_.stalePops - stalePopsAtLastCheck;
    stalePopsAtLastCheck = lns.incrementalRegretStatsCurrent_.stalePops;
    stalePopsSinceRefresh += staleDelta;
    if (staleDelta > 0) {
      consecutiveStaleGrowthCommits++;
    } else {
      consecutiveStaleGrowthCommits = 0;
    }

    if (!bestRegret.has_value()) {
      lns.incrementalRegretStatsCurrent_.heapRebuilds++;
      lns.incrementalRegretStatsTotal_.heapRebuilds++;
      if (!lns.recomputeRegretsForTasks(lns.collectRemainingRemovedTasks())) {
        repairFailed = true;
      } else {
        stalePopsAtLastCheck = lns.incrementalRegretStatsCurrent_.stalePops;
        stalePopsSinceRefresh = 0;
        commitsSinceRefresh = 0;
        consecutiveStaleGrowthCommits = 0;
        refreshCooldownCommits = 2;
      }
      continue;
    }

    const vector<int> endTimesBefore = lns.computeCurrentTaskEndTimes();
    const vector<uint64_t> agentSignaturesBefore =
        lns.computeCurrentAgentScheduleSignatures();
    if (!lns.commitBestRegretTask(*bestRegret)) {
      PLOGE << "run: failed to commit best-regret task " << bestRegret->task
            << "\n";
      repairFailed = true;
      break;
    }
    lns.incrementalRegretStatsCurrent_.commits++;
    lns.incrementalRegretStatsTotal_.commits++;
    commitsSinceRefresh++;
    const vector<int> endTimesAfter = lns.computeCurrentTaskEndTimes();
    const vector<uint64_t> agentSignaturesAfter =
        lns.computeCurrentAgentScheduleSignatures();

    const vector<int> dirtyTasks = lns.computeDirtyTasksAfterCommit(
        endTimesBefore, endTimesAfter, agentSignaturesBefore,
        agentSignaturesAfter);
    vector<int> tasksToRecompute = dirtyTasks;
    const int remainingRemovedTasks =
        (int)lns.lnsNeighborhood_.removedTasks.size();
    if (!endgameFullRefreshDone && remainingRemovedTasks <= 4 &&
        (stalePopsSinceRefresh >= 30 || commitsSinceRefresh >= 10)) {
      lns.incrementalRegretStatsCurrent_.endgameFullRecomputes++;
      lns.incrementalRegretStatsTotal_.endgameFullRecomputes++;
      tasksToRecompute = lns.collectRemainingRemovedTasks();
      endgameFullRefreshDone = true;
    }
    if (!lns.recomputeRegretsForTasks(tasksToRecompute)) {
      repairFailed = true;
    }
  }
  return !repairFailed;
}
