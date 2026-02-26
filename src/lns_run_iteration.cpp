#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"
#include <cmath>
#include <limits>

bool LNS::runOneIteration(ConflictMap& potentialNeighborhood,
                          ConflictMap& oldNeighborhood,
                          MovingMetrics& metrics,
                          bool& currentSolutionValid,
                          ValidationStats& currentValidationStats,
                          bool& feasibleSolutionUpdated) {
  market_.candidateUpdateConsumed = false;
  const int previousSocForIter = previousSolution_.sumOfCosts;
  const ValidationStats previousValidationStatsForIter =
      currentValidationStats;
  const double previousPressureForIter =
      market_.heuristics ? computeSolutionMarketPressure() : 0.0;
  const double previousWaitForIter =
      market_.heuristics ? computeSolutionPrecedenceWait() : 0.0;
  int alnsHeuristicForIter = -1;
  const int64_t iterationIndex =
      static_cast<int64_t>(iterationDebugRecords_.size());
  const int incumbentSocBeforeIter =
      incumbentSolution_.agentPaths.empty()
          ? std::numeric_limits<int>::max()
          : incumbentSolution_.sumOfCosts;
  IterationQuality quality = IterationQuality::none;
  const bool collectIterationDebug =
      debugImprovementDiagnostics_ || !debugIterationTsvPath_.empty();
  IterationDebugRecord debugRow;
  debugRow.iteration = iterationIndex;
  debugRow.previousSoc = previousSocForIter;
  debugRow.candidateSoc = previousSocForIter;
  debugRow.incumbentSocBefore = incumbentSocBeforeIter;
  improvementDiagnosticsStats_.iterationsStarted++;
  improvementDiagnosticsStats_.sumPreviousSoc +=
      static_cast<double>(previousSocForIter);
  if (incumbentSocBeforeIter != std::numeric_limits<int>::max()) {
    improvementDiagnosticsStats_.sumIncumbentSoc +=
        static_cast<double>(incumbentSocBeforeIter);
    improvementDiagnosticsStats_.incumbentSocSamples++;
  }
  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };
  double timeDestroyAndPrepareSec = 0.0;
  double timeRepairAndCommitSec = 0.0;
  double timeJoinPathsSec = 0.0;
  double timeTerminalReplanSec = 0.0;
  double timeRecomputeSocSec = 0.0;
  double timeValidationSec = 0.0;
  double timeAcceptanceSec = 0.0;
  double timeBookkeepingSec = 0.0;
  bool iterationTimingCommitted = false;
  bool iterationRowCommitted = false;
  auto commitIterationTiming = [&]() {
    if (iterationTimingCommitted) {
      return;
    }
    improvementDiagnosticsStats_.timeDestroyAndPrepareSec +=
        timeDestroyAndPrepareSec;
    improvementDiagnosticsStats_.timeRepairAndCommitSec +=
        timeRepairAndCommitSec;
    improvementDiagnosticsStats_.timeJoinPathsSec += timeJoinPathsSec;
    improvementDiagnosticsStats_.timeTerminalReplanSec +=
        timeTerminalReplanSec;
    improvementDiagnosticsStats_.timeRecomputeSocSec += timeRecomputeSocSec;
    improvementDiagnosticsStats_.timeValidationSec += timeValidationSec;
    improvementDiagnosticsStats_.timeAcceptanceSec += timeAcceptanceSec;
    improvementDiagnosticsStats_.timeBookkeepingSec += timeBookkeepingSec;
    iterationTimingCommitted = true;
    if (collectIterationDebug && !iterationRowCommitted) {
      debugRow.runtimeSec = runtime;
      debugRow.quality = iterationQualityName(quality);
      debugRow.timeDestroyAndPrepareSec = timeDestroyAndPrepareSec;
      debugRow.timeRepairAndCommitSec = timeRepairAndCommitSec;
      debugRow.timeJoinPathsSec = timeJoinPathsSec;
      debugRow.timeTerminalReplanSec = timeTerminalReplanSec;
      debugRow.timeRecomputeSocSec = timeRecomputeSocSec;
      debugRow.timeValidationSec = timeValidationSec;
      debugRow.timeAcceptanceSec = timeAcceptanceSec;
      debugRow.timeBookkeepingSec = timeBookkeepingSec;
      iterationDebugRecords_.push_back(debugRow);
      iterationRowCommitted = true;
    }
  };
  Time::time_point phaseStart = Time::now();

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
    debugRow.earlyAbortReason = "destroy_heuristic_error";
    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    commitIterationTiming();
    return false;
  }

  oldNeighborhood = lnsNeighborhood_.removedTasks;

  PLOGD << "Printing neighborhood conflict tasks\n";
  PLOGD << "Size: " << lnsNeighborhood_.removedTasks.size() << "\n";
  for (const auto& [_, conflictTask] : lnsNeighborhood_.removedTasks) {
    PLOGD << "Conflicted Task : " << conflictTask.task << "\n";
  }

  if (!prepareNextIteration()) {
    timeDestroyAndPrepareSec += elapsedSecSince(phaseStart);
    improvementDiagnosticsStats_.earlyAbortPrepare++;
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
      restoreSolutionFromPrevious();
    }
    feasibleSolutionUpdated = false;
    quality = IterationQuality::couldNotFind;
    const Time::time_point bookkeepingStart = Time::now();
    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    appendIterationStatBounded(IterationStats(
        runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
        solution_.sumOfCosts, feasibleSolutionUpdated, quality));
    timeBookkeepingSec += elapsedSecSince(bookkeepingStart);
    if (cascadeAbort) {
      PLOGD << "prepareNextIteration aborted due to cascade budget\n";
      debugRow.earlyAbortReason = "cascade_budget_abort";
    } else {
      debugRow.earlyAbortReason = "prepare_failed";
    }
    maybeUpdateMarketState(false);
    commitIterationTiming();
    return true;
  }
  timeDestroyAndPrepareSec += elapsedSecSince(phaseStart);
  phaseStart = Time::now();

  // This needs to happen after prepare iteration since we updated the conflictedTasks variable in the prepare next iteration function
  for (const auto& [_, conflictedTask] : lnsNeighborhood_.removedTasks) {
    if (conflictedTask.task >= 0 &&
        conflictedTask.task < (int)lnsNeighborhood_.committedTasks.size()) {
      lnsNeighborhood_.committedTasks[conflictedTask.task] = 0;
    }
  }
  lnsNeighborhood_.immutableRemovedTasks = lnsNeighborhood_.removedTasks;
  debugRow.removedTasks = (int)lnsNeighborhood_.immutableRemovedTasks.size();

  const uint64_t neighborhoodFingerprint =
      computeNeighborhoodFingerprint(lnsNeighborhood_.immutableRemovedTasks);
  debugRow.neighborhoodFingerprint = neighborhoodFingerprint;
  const auto [_, neighborhoodInserted] =
      seenNeighborhoodFingerprints_.insert(neighborhoodFingerprint);
  const bool neighborhoodSeenBefore = !neighborhoodInserted;
  debugRow.neighborhoodFingerprintSeenBefore = neighborhoodSeenBefore;
  if (neighborhoodSeenBefore) {
    improvementDiagnosticsStats_.neighborhoodFingerprintRepeat++;
    currentNeighborhoodRepeatStreak_++;
    debugRow.neighborhoodRepeatStreak = currentNeighborhoodRepeatStreak_;
    improvementDiagnosticsStats_.neighborhoodRepeatStreakMax =
        max(improvementDiagnosticsStats_.neighborhoodRepeatStreakMax,
            (int64_t)currentNeighborhoodRepeatStreak_);
  } else {
    improvementDiagnosticsStats_.neighborhoodFingerprintUnique++;
    currentNeighborhoodRepeatStreak_ = 0;
    debugRow.neighborhoodRepeatStreak = 0;
  }

  vector<int> currentNeighborhoodTasksSorted;
  currentNeighborhoodTasksSorted.reserve(
      lnsNeighborhood_.immutableRemovedTasks.size());
  for (const auto& [task, _] : lnsNeighborhood_.immutableRemovedTasks) {
    currentNeighborhoodTasksSorted.push_back(task);
  }
  if (!currentNeighborhoodTasksSorted.empty()) {
    string taskIdsCsv;
    taskIdsCsv.reserve(currentNeighborhoodTasksSorted.size() * 6);
    for (size_t idx = 0; idx < currentNeighborhoodTasksSorted.size(); idx++) {
      if (idx > 0) {
        taskIdsCsv.push_back(',');
      }
      taskIdsCsv += std::to_string(currentNeighborhoodTasksSorted[idx]);
    }
    debugRow.removedTaskIdsCsv = std::move(taskIdsCsv);
  } else {
    debugRow.removedTaskIdsCsv.clear();
  }

  if (!previousNeighborhoodTasksSorted_.empty()) {
    size_t i = 0;
    size_t j = 0;
    size_t intersection = 0;
    while (i < currentNeighborhoodTasksSorted.size() &&
           j < previousNeighborhoodTasksSorted_.size()) {
      if (currentNeighborhoodTasksSorted[i] ==
          previousNeighborhoodTasksSorted_[j]) {
        intersection++;
        i++;
        j++;
      } else if (currentNeighborhoodTasksSorted[i] <
                 previousNeighborhoodTasksSorted_[j]) {
        i++;
      } else {
        j++;
      }
    }
    const size_t unionCount =
        currentNeighborhoodTasksSorted.size() +
        previousNeighborhoodTasksSorted_.size() - intersection;
    const double jaccardPrev =
        unionCount > 0 ? (double)intersection / (double)unionCount : 1.0;
    debugRow.neighborhoodJaccardPrev = jaccardPrev;
    improvementDiagnosticsStats_.neighborhoodJaccardPrevSum += jaccardPrev;
    improvementDiagnosticsStats_.neighborhoodJaccardPrevSamples++;
    if (neighborhoodSeenBefore) {
      improvementDiagnosticsStats_.neighborhoodJaccardPrevRepeatSum +=
          jaccardPrev;
      improvementDiagnosticsStats_.neighborhoodJaccardPrevRepeatSamples++;
    }
  }
  previousNeighborhoodTasksSorted_.swap(currentNeighborhoodTasksSorted);

  improvementDiagnosticsStats_.neighborhoodsCount++;
  improvementDiagnosticsStats_.removedTasksTotal += debugRow.removedTasks;
  improvementDiagnosticsStats_.removedTasksMax =
      max(improvementDiagnosticsStats_.removedTasksMax,
          (int64_t)debugRow.removedTasks);
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
  for (auto& candidateAgents : regretCandidateAgents_) {
    candidateAgents.clear();
  }
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
      if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
        repairFailed = true;
        return;
      }
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
            consecutiveStaleGrowthCommits >= 8 &&
            stalePopsSinceRefresh >= 80 && commitsSinceRefresh >= 10;
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
        // Heap may have been exhausted by stale entries; rebuild for whatever
        // is left.
        incrementalRegretStatsCurrent_.heapRebuilds++;
        incrementalRegretStatsTotal_.heapRebuilds++;
        if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
          repairFailed = true;
        } else {
          stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
          stalePopsSinceRefresh = 0;
          commitsSinceRefresh = 0;
          consecutiveStaleGrowthCommits = 0;
          refreshCooldownCommits = 2;
        }
        continue;
      }

      const vector<int> endTimesBefore = computeCurrentTaskEndTimes();
      const vector<uint64_t> agentSignaturesBefore =
          computeCurrentAgentScheduleSignatures();
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
      const vector<uint64_t> agentSignaturesAfter =
          computeCurrentAgentScheduleSignatures();

      const vector<int> dirtyTasks = computeDirtyTasksAfterCommit(
          endTimesBefore, endTimesAfter, agentSignaturesBefore,
          agentSignaturesAfter);
      vector<int> tasksToRecompute = dirtyTasks;
      const int remainingRemovedTasks = (int)lnsNeighborhood_.removedTasks.size();
      if (!endgameFullRefreshDone && remainingRemovedTasks <= 4 &&
          (stalePopsSinceRefresh >= 30 || commitsSinceRefresh >= 10)) {
        incrementalRegretStatsCurrent_.endgameFullRecomputes++;
        incrementalRegretStatsTotal_.endgameFullRecomputes++;
        tasksToRecompute = collectRemainingRemovedTasks();
        endgameFullRefreshDone = true;
      }
      if (!recomputeRegretsForTasks(tasksToRecompute)) {
        repairFailed = true;
      }
    }

    // Stats are collected and printed in a dedicated summary section.
  }

  timeRepairAndCommitSec += elapsedSecSince(phaseStart);

  // If we could not successfully compute the regrets and commit to all the tasks in the neighborhood then we need to reset this neighborhood!
  if (repairFailed || !lnsNeighborhood_.removedTasks.empty()) {
    debugRow.earlyAbortReason = "repair_failed_or_incomplete";
    improvementDiagnosticsStats_.earlyAbortRepair++;
    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
    }
    // Reject whatever we done till now
    restoreSolutionFromPrevious();
    feasibleSolutionUpdated = false;
    quality = IterationQuality::couldNotFind;
    const Time::time_point bookkeepingStart = Time::now();
    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    appendIterationStatBounded(IterationStats(
        runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
        solution_.sumOfCosts, feasibleSolutionUpdated, quality));
    timeBookkeepingSec += elapsedSecSince(bookkeepingStart);
    // Skip everything after this statement
    PLOGD << "Could not find paths for the neighborhood! Attempting a new "
             "neighborhood computation\n";
    maybeUpdateMarketState(false);
    commitIterationTiming();
    return true;
  }

  // Refresh impacted agents after repair/commit. The pre-repair set can miss
  // agents that only appear due to reassignment decisions made during repair.
  {
    vector<char> joinMarked(instance_.getAgentNum(), 0);
    vector<int> refreshedAgents;
    refreshedAgents.reserve(instance_.getAgentNum());
    auto markForJoin = [&](int agent) {
      if (agent < 0 || agent >= instance_.getAgentNum() || joinMarked[agent]) {
        return;
      }
      joinMarked[agent] = 1;
      refreshedAgents.push_back(agent);
    };

    for (int agent : agentsToCompute) {
      markForJoin(agent);
    }

    const int taskCount = instance_.getTasksNum();
    for (int task = 0; task < taskCount; task++) {
      const int prevOwner =
          (task >= 0 && task < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[task]
              : UNASSIGNED;
      const int curOwner =
          (task >= 0 && task < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[task]
              : UNASSIGNED;
      if (prevOwner != curOwner) {
        markForJoin(prevOwner);
        markForJoin(curOwner);
      }
    }

    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      if (solution_.agents[agent].taskAssignments !=
              previousSolution_.agents[agent].taskAssignments ||
          solution_.agents[agent].path.empty()) {
        markForJoin(agent);
      }
    }

    if (refreshedAgents.empty()) {
      refreshedAgents.resize(instance_.getAgentNum());
      std::iota(refreshedAgents.begin(), refreshedAgents.end(), 0);
    }
    agentsToCompute.swap(refreshedAgents);
  }

  // Join only agents impacted by this neighborhood.
  const Time::time_point joinStart = Time::now();
  if (!solution_.joinPaths(agentsToCompute)) {
    timeJoinPathsSec += elapsedSecSince(joinStart);
    improvementDiagnosticsStats_.earlyAbortJoin++;
    PLOGE << "run: failed to join agent paths for candidate solution\n";
    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
    }
    restoreSolutionFromPrevious();
    debugRow.earlyAbortReason = "join_failed";
    feasibleSolutionUpdated = false;
    quality = IterationQuality::couldNotFind;
    const Time::time_point bookkeepingStart = Time::now();
    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    appendIterationStatBounded(IterationStats(
        runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
        solution_.sumOfCosts, feasibleSolutionUpdated, quality));
    timeBookkeepingSec += elapsedSecSince(bookkeepingStart);
    maybeUpdateMarketState(false);
    commitIterationTiming();
    return true;
  }
  timeJoinPathsSec += elapsedSecSince(joinStart);
  if (goalOccupationMode_ == "reposition_true") {
    const vector<int> terminalReplanAgents =
        selectTerminalReplanAgents(agentsToCompute);
    if (!terminalReplanAgents.empty()) {
      const Time::time_point terminalStart = Time::now();
      const bool terminalOk =
          planTerminalReposition(terminalReplanAgents, false);
      timeTerminalReplanSec += elapsedSecSince(terminalStart);
      if (!terminalOk) {
        improvementDiagnosticsStats_.earlyAbortTerminal++;
        debugRow.earlyAbortReason = "terminal_replan_failed";
        PLOGE << "run: failed to replan terminal reposition paths for "
                 "candidate solution\n";
        if (alnsHeuristicForIter >= 0 &&
            alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
          adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
        }
        restoreSolutionFromPrevious();
        feasibleSolutionUpdated = false;
        quality = IterationQuality::couldNotFind;
        const Time::time_point bookkeepingStart = Time::now();
        runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
        appendIterationStatBounded(IterationStats(
            runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
            solution_.sumOfCosts, feasibleSolutionUpdated, quality));
        timeBookkeepingSec += elapsedSecSince(bookkeepingStart);
        maybeUpdateMarketState(false);
        commitIterationTiming();
        return true;
      }
    }
  }

  // Compute the updated sum of costs
  const Time::time_point recomputeSocStart = Time::now();
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
  timeRecomputeSocSec += elapsedSecSince(recomputeSocStart);

  const vector<int> previousTaskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(
          previousSolution_, instance_.getTasksNum());
  const vector<int> candidateTaskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(
          solution_, instance_.getTasksNum());
  int removedTasksChangedAgent = 0;
  int removedTasksChangedOrder = 0;
  int removedTasksUnchanged = 0;
  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    const int task = conflict.task;
    if (task < 0 || task >= instance_.getTasksNum()) {
      continue;
    }
    const int prevAgent =
        (task < (int)previousSolution_.taskAgentMap.size())
            ? previousSolution_.taskAgentMap[task]
            : UNASSIGNED;
    const int candAgent =
        (task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (prevAgent != candAgent) {
      removedTasksChangedAgent++;
      continue;
    }
    const int prevPos =
        (task < (int)previousTaskPosByTask.size())
            ? previousTaskPosByTask[task]
            : UNASSIGNED;
    const int candPos =
        (task < (int)candidateTaskPosByTask.size())
            ? candidateTaskPosByTask[task]
            : UNASSIGNED;
    if (prevPos != candPos) {
      removedTasksChangedOrder++;
    } else {
      removedTasksUnchanged++;
    }
  }
  debugRow.removedTasksChangedAgent = removedTasksChangedAgent;
  debugRow.removedTasksChangedOrder = removedTasksChangedOrder;
  debugRow.removedTasksUnchanged = removedTasksUnchanged;
  const int changedRemovedTasks =
      removedTasksChangedAgent + removedTasksChangedOrder;
  improvementDiagnosticsStats_.removedTasksChangedAgentTotal +=
      removedTasksChangedAgent;
  improvementDiagnosticsStats_.removedTasksChangedOrderTotal +=
      removedTasksChangedOrder;
  improvementDiagnosticsStats_.removedTasksUnchangedTotal +=
      removedTasksUnchanged;
  improvementDiagnosticsStats_.changedTasksPerNeighborhoodMax =
      max(improvementDiagnosticsStats_.changedTasksPerNeighborhoodMax,
          (int64_t)changedRemovedTasks);
  if (changedRemovedTasks > 0) {
    improvementDiagnosticsStats_.neighborhoodsWithChanges++;
  } else {
    improvementDiagnosticsStats_.neighborhoodsWithoutChanges++;
  }

  const int previousConflictSignalForIter =
      previousValidationStatsForIter.totalConflictEvents();
  debugRow.previousConflictSignal = previousConflictSignalForIter;
  PLOGD << "Conflict signal in old solution: "
        << previousConflictSignalForIter << "\n";

  // Extract the set of conflicting tasks
  const Time::time_point validationStart = Time::now();
  potentialNeighborhood.clear();
  ValidationStats candidateValidationStats;
  useTerminalPathsInValidation_ = (goalOccupationMode_ == "reposition_true");
  const bool candidateValid =
      validateSolution(&potentialNeighborhood, &candidateValidationStats);
  useTerminalPathsInValidation_ = false;

  const int candidateConflictSignal =
      candidateValidationStats.totalConflictEvents();
  debugRow.candidateConflictSignal = candidateConflictSignal;
  PLOGD << "Conflict signal in new solution: " << candidateConflictSignal
        << "\n";

  // Accept the solution only if the new one has higher utility compared to the old solution where utility is a weighted combination of the number of conflicts and sum of costs.
  // Compute the utility of this solution
  solution_.utility =
      metrics.computeMovingMetrics(candidateConflictSignal, solution_.sumOfCosts);

  if (!candidateValid) {
    // Solution was not valid as we found some conflicts!
    feasibleSolutionUpdated = false;
    debugRow.feasibleBestUpdate = false;
    PLOGE << "The solution was not valid!\n";
  } else {
    if (extractFeasibleSolution()) {
      // This is the case when the feasible solution was updated!
      quality = IterationQuality::bestSolutionYet;
      feasibleSolutionUpdated = true;
      debugRow.feasibleBestUpdate = true;
    } else {
      feasibleSolutionUpdated = false;
      debugRow.feasibleBestUpdate = false;
    }
  }
  timeValidationSec += elapsedSecSince(validationStart);

  // Ensure that we are either accepting or rejecting a solution here!
  const int proposedSocForIter = solution_.sumOfCosts;
  debugRow.candidateSoc = proposedSocForIter;
  improvementDiagnosticsStats_.sumCandidateSoc +=
      static_cast<double>(proposedSocForIter);
  improvementDiagnosticsStats_.sumPreviousConflictSignal +=
      static_cast<double>(previousConflictSignalForIter);
  improvementDiagnosticsStats_.sumCandidateConflictSignal +=
      static_cast<double>(candidateConflictSignal);
  if (candidateConflictSignal < previousConflictSignalForIter) {
    improvementDiagnosticsStats_.conflictSignalBetter++;
  } else if (candidateConflictSignal > previousConflictSignalForIter) {
    improvementDiagnosticsStats_.conflictSignalWorse++;
  } else {
    improvementDiagnosticsStats_.conflictSignalEqual++;
  }
  if (candidateValid) {
    debugRow.candidateValid = true;
    improvementDiagnosticsStats_.candidateValid++;
    if (feasibleSolutionUpdated) {
      improvementDiagnosticsStats_.feasibleBestUpdates++;
    } else {
      improvementDiagnosticsStats_.feasibleNoBestUpdate++;
    }
  } else {
    debugRow.candidateValid = false;
    improvementDiagnosticsStats_.candidateInvalid++;
  }
  double candidatePressure = 0.0;
  double candidateWait = 0.0;
  bool accepted = false;
  bool guardRejected = false;
  bool invalidGuardRejected = false;
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
  const Time::time_point acceptanceStart = Time::now();
  if (!candidateValid && !acceptOnlyValidCandidates_) {
    PLOGD << "Invalid candidate forwarded to acceptance criteria\n";
  }
  if (acceptOnlyValidCandidates_ && !candidateValid) {
    restoreSolutionFromPrevious();
    accepted = false;
    invalidGuardRejected = true;
    advanceTemperatureOnGuardReject();
    PLOGD << "Rejecting this solution due to strict-valid acceptance guard\n";
  } else {
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
      restoreSolutionFromPrevious();
      accepted = false;
      guardRejected = true;
      advanceTemperatureOnGuardReject();
      PLOGD << "Rejecting this solution due to market acceptance guards\n";
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
        PLOGE << "Unknown acceptance criteria: " << acceptanceCriteria << "\n";
        debugRow.earlyAbortReason = "acceptance_criteria_error";
        runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
        timeAcceptanceSec += elapsedSecSince(acceptanceStart);
        commitIterationTiming();
        return false;
      }
      if (accepted) {
        acceptedAsWorse = previousSolution_.utility < solution_.utility;
      }
    }
  }
  timeAcceptanceSec += elapsedSecSince(acceptanceStart);

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

  if (accepted) {
    debugRow.accepted = true;
    debugRow.acceptedAsWorseUtility = acceptedAsWorse;
    improvementDiagnosticsStats_.accepted++;
    if (acceptedAsWorse) {
      improvementDiagnosticsStats_.acceptedAsWorseUtility++;
    } else {
      improvementDiagnosticsStats_.acceptedAsBetterOrEqualUtility++;
    }
    if (proposedSocForIter < previousSocForIter) {
      improvementDiagnosticsStats_.acceptedSocBetterVsPrevious++;
    } else if (proposedSocForIter > previousSocForIter) {
      improvementDiagnosticsStats_.acceptedSocWorseVsPrevious++;
    } else {
      improvementDiagnosticsStats_.acceptedSocEqualVsPrevious++;
    }
    if (incumbentSocBeforeIter != std::numeric_limits<int>::max()) {
      if (proposedSocForIter < incumbentSocBeforeIter) {
        improvementDiagnosticsStats_.acceptedSocBetterVsIncumbent++;
      } else if (proposedSocForIter > incumbentSocBeforeIter) {
        improvementDiagnosticsStats_.acceptedSocWorseVsIncumbent++;
      } else {
        improvementDiagnosticsStats_.acceptedSocEqualVsIncumbent++;
      }
    }
    if (candidateValid && !feasibleSolutionUpdated) {
      improvementDiagnosticsStats_.acceptedFeasibleNoBestUpdate++;
    }
    if (!candidateValid) {
      improvementDiagnosticsStats_.acceptedInvalid++;
    }
    if (candidateConflictSignal < previousConflictSignalForIter) {
      improvementDiagnosticsStats_.acceptedConflictSignalBetter++;
    } else if (candidateConflictSignal > previousConflictSignalForIter) {
      improvementDiagnosticsStats_.acceptedConflictSignalWorse++;
    } else {
      improvementDiagnosticsStats_.acceptedConflictSignalEqual++;
    }

    const uint64_t fingerprint = computeSolutionFingerprint(solution_);
    debugRow.fingerprint = fingerprint;
    const auto [_, inserted] =
        acceptedSolutionFingerprints_.insert(fingerprint);
    debugRow.fingerprintSeenBefore = !inserted;
    if (inserted) {
      improvementDiagnosticsStats_.acceptedFingerprintUnique++;
    } else {
      improvementDiagnosticsStats_.acceptedFingerprintRepeat++;
    }

    improvementDiagnosticsStats_.acceptedRemovedTasksTotal +=
        debugRow.removedTasks;
    if (debugRow.removedTasksChangedAgent >= 0) {
      improvementDiagnosticsStats_.acceptedRemovedTasksChangedAgentTotal +=
          debugRow.removedTasksChangedAgent;
      improvementDiagnosticsStats_.acceptedRemovedTasksChangedOrderTotal +=
          debugRow.removedTasksChangedOrder;
      improvementDiagnosticsStats_.acceptedRemovedTasksUnchangedTotal +=
          debugRow.removedTasksUnchanged;
      const int acceptedChangedRemovedTasks =
          debugRow.removedTasksChangedAgent +
          debugRow.removedTasksChangedOrder;
      improvementDiagnosticsStats_.acceptedChangedTasksPerNeighborhoodMax =
          max(improvementDiagnosticsStats_.acceptedChangedTasksPerNeighborhoodMax,
              (int64_t)acceptedChangedRemovedTasks);
    }
    improvementDiagnosticsStats_.acceptedRemovedTasksMax =
        max(improvementDiagnosticsStats_.acceptedRemovedTasksMax,
            (int64_t)debugRow.removedTasks);
  } else {
    debugRow.accepted = false;
    debugRow.acceptedAsWorseUtility = false;
    improvementDiagnosticsStats_.rejected++;
    if (invalidGuardRejected) {
      debugRow.earlyAbortReason = "invalid_candidate_guard_reject";
    }
    if (guardRejected) {
      debugRow.guardRejected = true;
      improvementDiagnosticsStats_.guardRejected++;
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

  if (debugRow.earlyAbortReason.empty()) {
    debugRow.earlyAbortReason = "none";
  }
  const Time::time_point bookkeepingStart = Time::now();
  runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
  double costToLog = (feasibleSolutionUpdated) ? incumbentSolution_.sumOfCosts
                                               : solution_.sumOfCosts;
  appendIterationStatBounded(IterationStats(
      runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
      costToLog, feasibleSolutionUpdated, quality));
  timeBookkeepingSec += elapsedSecSince(bookkeepingStart);
  commitIterationTiming();
  return true;
}
