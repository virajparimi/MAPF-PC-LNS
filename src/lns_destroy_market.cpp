#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"

void LNS::marketTatonnementRemoval(const ConflictMap* potentialNeighborhood) {
  PLOGD << "Using market tatonnement removal\n";

  if (!market_.heuristics) {
    PLOGW << "marketTatonnementRemoval requested while market heuristics are "
             "disabled; falling back to random removal\n";
    randomRemoval();
    return;
  }

  clearNeighborhood();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  vector<TaskScheduleMetrics> perTask;
  vector<double> blockedWaitSum;
  computeTaskScheduleMetrics(perTask, &blockedWaitSum);
  unordered_map<uint64_t, int> vertexDemand;
  unordered_map<uint64_t, int> edgeDemand;
  buildMarketDemandFromCurrentOccupancy(vertexDemand, edgeDemand);
  const auto& predecessors = instance_.getAncestorsRef();
  const auto& successors = instance_.getSuccessorsRef();
  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, UNASSIGNED);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    for (int pos = 0; pos < (int)assignments.size(); pos++) {
      const int task = assignments[pos];
      if (task >= 0 && task < taskCount) {
        taskToAgent[task] = agent;
        taskToPosition[task] = pos;
      }
    }
  }

  struct RankedTask {
    int task = UNASSIGNED;
    double burden = 0.0;
    double tieBreak = 0.0;
  };
  vector<RankedTask> rankedTasks;
  rankedTasks.reserve(taskCount);
  vector<double> rawRelief(taskCount, 0.0);
  vector<double> rawWait(taskCount, 0.0);
  vector<double> rawRoot(taskCount, 0.0);
  double maxRelief = 0.0, maxWait = 0.0, maxRoot = 0.0;
  const int currentIter = (int)iterationStats.size();
  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int pos = taskToPosition[task];
    if (agent == UNASSIGNED || pos == UNASSIGNED ||
        pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
    const double relief = taskPath.empty()
                              ? 0.0
                              : computeMarketMarginalReliefFromPath(
                                    taskPath, vertexDemand, edgeDemand, true);
    const double wait = perTask[task].valid ? (double)perTask[task].waitPrec : 0.0;
    const int blocker = perTask[task].blocker;
    const double root = (blocker >= 0 && blocker < taskCount)
                            ? blockedWaitSum[blocker]
                            : 0.0;
    rawRelief[task] = max(0.0, relief);
    rawWait[task] = max(0.0, wait);
    rawRoot[task] = max(0.0, root);
    maxRelief = max(maxRelief, rawRelief[task]);
    maxWait = max(maxWait, rawWait[task]);
    maxRoot = max(maxRoot, rawRoot[task]);
    rankedTasks.push_back(RankedTask{task, 0.0, 0.0});
  }

  if (rankedTasks.empty()) {
    randomRemoval();
    return;
  }

  std::uniform_real_distribution<double> tieBreakDist(0.0, 1.0);
  const double invRelief = (maxRelief > 0.0) ? (1.0 / maxRelief) : 0.0;
  const double invWait = (maxWait > 0.0) ? (1.0 / maxWait) : 0.0;
  const double invRoot = (maxRoot > 0.0) ? (1.0 / maxRoot) : 0.0;
  for (auto& entry : rankedTasks) {
    const int task = entry.task;
    const double normRelief = rawRelief[task] * invRelief;
    const double normWait = rawWait[task] * invWait;
    const double normRoot = rawRoot[task] * invRoot;
    double burden = market_.destroyWeightPrice * normRelief +
                    market_.destroyWeightWait * normWait +
                    market_.destroyWeightRoot * normRoot;
    if (task < (int)market_.taskCooldownUntilIter.size() &&
        market_.taskCooldownUntilIter[task] > currentIter) {
      burden *= 0.25;
    }
    entry.burden = burden;
    entry.tieBreak = tieBreakDist(rng_);
  }

  std::sort(rankedTasks.begin(), rankedTasks.end(),
            [](const RankedTask& lhs, const RankedTask& rhs) {
              if (lhs.burden != rhs.burden) {
                return lhs.burden > rhs.burden;
              }
              if (lhs.tieBreak != rhs.tieBreak) {
                return lhs.tieBreak > rhs.tieBreak;
              }
              return lhs.task < rhs.task;
            });

  vector<char> selected(taskCount, 0);
  auto addTask = [&](int task) -> bool {
    if (task < 0 || task >= taskCount || selected[task]) {
      return false;
    }
    const int agent = taskToAgent[task];
    const int pos = taskToPosition[task];
    if (agent == UNASSIGNED || pos == UNASSIGNED || pos < 0 ||
        pos >= (int)solution_.agents[agent].taskAssignments.size()) {
      return false;
    }
    selected[task] = 1;
    lnsNeighborhood_.removedTasks.emplace(task, Conflicts(task, agent, pos));
    return true;
  };

  if (potentialNeighborhood != nullptr && !potentialNeighborhood->empty() &&
      incumbentSolution_.agentPaths.empty()) {
    const int quota = max(1, cappedNeighborSize / 2);
    const ConflictMap conflictSeeds =
        extractNConflicts(quota, *potentialNeighborhood);
    for (const auto& [_, conflict] : conflictSeeds) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      addTask(conflict.task);
    }
  }

  const int randomSlots = min(
      cappedNeighborSize,
      max(0, (int)std::round((double)cappedNeighborSize * market_.randomDestroyQuota)));
  const int targetNonRandom = max(0, cappedNeighborSize - randomSlots);

  int seedPoolSize = (int)std::ceil((double)rankedTasks.size() * market_.seedTopFrac);
  seedPoolSize = max(1, min(seedPoolSize, (int)rankedTasks.size()));
  vector<int> seedPoolTasks;
  vector<double> seedPoolWeights;
  seedPoolTasks.reserve(seedPoolSize);
  seedPoolWeights.reserve(seedPoolSize);
  for (int i = 0; i < seedPoolSize; i++) {
    seedPoolTasks.push_back(rankedTasks[i].task);
    seedPoolWeights.push_back(max(1e-6, rankedTasks[i].burden + 1e-6));
  }

  auto expandAncestors = [&](int rootTask, int maxDepth, int* closureCount) {
    if (rootTask < 0 || rootTask >= taskCount || maxDepth <= 0) {
      return;
    }
    std::queue<pair<int, int>> q;
    q.push({rootTask, 0});
    while (!q.empty() && (int)lnsNeighborhood_.removedTasks.size() < targetNonRandom) {
      const auto [task, depth] = q.front();
      q.pop();
      if (depth >= maxDepth || task < 0 || task >= taskCount) {
        continue;
      }
      for (int pred : predecessors[task]) {
        if ((int)lnsNeighborhood_.removedTasks.size() >= targetNonRandom ||
            *closureCount >= market_.closureCap) {
          return;
        }
        if (addTask(pred)) {
          (*closureCount)++;
          q.push({pred, depth + 1});
        }
      }
    }
  };

  auto expandSuccessors = [&](int rootTask, int maxDepth, int* closureCount) {
    if (rootTask < 0 || rootTask >= taskCount || maxDepth <= 0) {
      return;
    }
    std::queue<pair<int, int>> q;
    q.push({rootTask, 0});
    while (!q.empty() && (int)lnsNeighborhood_.removedTasks.size() < targetNonRandom) {
      const auto [task, depth] = q.front();
      q.pop();
      if (depth >= maxDepth || task < 0 || task >= taskCount) {
        continue;
      }
      for (int succ : successors[task]) {
        if ((int)lnsNeighborhood_.removedTasks.size() >= targetNonRandom ||
            *closureCount >= market_.closureCap) {
          return;
        }
        if (addTask(succ)) {
          (*closureCount)++;
          q.push({succ, depth + 1});
        }
      }
    }
  };

  while ((int)lnsNeighborhood_.removedTasks.size() < targetNonRandom &&
         !seedPoolTasks.empty()) {
    std::discrete_distribution<int> seedDist(seedPoolWeights.begin(),
                                             seedPoolWeights.end());
    const int sampledIdx = seedDist(rng_);
    const int seedTask = seedPoolTasks[sampledIdx];
    const int blocker =
        (seedTask >= 0 && seedTask < taskCount) ? perTask[seedTask].blocker
                                                : UNASSIGNED;
    const bool addedSeed = addTask(seedTask);
    if (addedSeed) {
      int closureCount = 0;
      if (blocker != UNASSIGNED && addTask(blocker)) {
        closureCount++;
      }
      expandAncestors(blocker, market_.dUp, &closureCount);
      expandSuccessors(blocker, market_.dDown, &closureCount);
      expandSuccessors(seedTask, market_.dDown, &closureCount);
    }
    seedPoolTasks[sampledIdx] = seedPoolTasks.back();
    seedPoolTasks.pop_back();
    seedPoolWeights[sampledIdx] = seedPoolWeights.back();
    seedPoolWeights.pop_back();
  }

  std::uniform_int_distribution<int> randomTaskDist(0, taskCount - 1);
  int randomFillAttempts = 0;
  const int maxRandomFillAttempts = max(64, taskCount * 8);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
         randomFillAttempts < maxRandomFillAttempts) {
    addTask(randomTaskDist(rng_));
    randomFillAttempts++;
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    for (int task = 0;
         task < taskCount &&
         (int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize;
         task++) {
      addTask(task);
    }
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    PLOGW << "marketTatonnementRemoval: could only remove "
          << lnsNeighborhood_.removedTasks.size() << " out of requested "
          << cappedNeighborSize << " tasks\n";
  }

  if (market_.cooldownIters > 0) {
    for (int task = 0; task < taskCount; task++) {
      if (selected[task]) {
        market_.taskCooldownUntilIter[task] = currentIter + market_.cooldownIters;
      }
    }
  }
}

