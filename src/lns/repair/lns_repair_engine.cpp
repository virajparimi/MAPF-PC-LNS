#include "lns.hpp"
#include "lns_repair_engine.hpp"

RepairPlan RepairEngine::buildPlan(bool enableNrrRepair,
                                   bool nrrFallbackToStandard,
                                   int repairHeuristicMode) {
  RepairPlan plan;
  plan.tryNrr = enableNrrRepair;
  plan.allowFallbackToStandard = nrrFallbackToStandard;
  (void)repairHeuristicMode;
  plan.strategy = RepairStrategy::full_regret;
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
      enableNrrRepair_, nrrFallbackToStandard_,
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

  return !repairFailed;
}
