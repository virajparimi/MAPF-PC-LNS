#include "lns.hpp"
#include "utils.hpp"

void LNS::clearNeighborhood() {
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.committedTasks.assign(instance_.getTasksNum(), -1);
  lnsNeighborhood_.removedTasksPathSize.assign(instance_.getTasksNum(), -1);
  lnsNeighborhood_.removedTasks.clear();
  lnsNeighborhood_.immutableRemovedTasks.clear();
}

void LNS::marketTatonnementRemoval(
    std::optional<ConflictMap> potentialNeighborhood) {
  PLOGD << "Using market tatonnement removal\n";

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

  vector<pair<int, double>> rankedTasks;
  rankedTasks.reserve(taskCount);
  const int currentIter = (int)iterationStats.size();
  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int pos = taskToPosition[task];
    if (agent == UNASSIGNED || pos == UNASSIGNED ||
        pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
    const double exposure =
        taskPath.empty() ? 0.0 : computeMarketExposureFromPath(taskPath, true);
    const double wait = perTask[task].valid ? (double)perTask[task].waitPrec : 0.0;
    const int blocker = perTask[task].blocker;
    const double root = (blocker >= 0 && blocker < taskCount)
                            ? blockedWaitSum[blocker]
                            : 0.0;
    double burden = market_.destroyWeightPrice * exposure +
                    market_.destroyWeightWait * wait +
                    market_.destroyWeightRoot * root;
    if (task < (int)market_.taskCooldownUntilIter.size() &&
        market_.taskCooldownUntilIter[task] > currentIter) {
      burden *= 0.25;
    }
    rankedTasks.emplace_back(task, burden);
  }

  if (rankedTasks.empty()) {
    randomRemoval();
    return;
  }

  std::sort(rankedTasks.begin(), rankedTasks.end(),
            [](const pair<int, double>& lhs, const pair<int, double>& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
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

  if (potentialNeighborhood.has_value() && !potentialNeighborhood->empty() &&
      incumbentSolution_.agentPaths.empty()) {
    const int quota = max(1, cappedNeighborSize / 2);
    const ConflictMap conflictSeeds =
        extractNConflicts(quota, potentialNeighborhood.value());
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
    seedPoolTasks.push_back(rankedTasks[i].first);
    seedPoolWeights.push_back(max(1e-6, rankedTasks[i].second + 1e-6));
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
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    addTask(randomTaskDist(rng_));
  }

  if (market_.cooldownIters > 0) {
    for (int task = 0; task < taskCount; task++) {
      if (selected[task]) {
        market_.taskCooldownUntilIter[task] = currentIter + market_.cooldownIters;
      }
    }
  }
}

void LNS::randomRemoval() {

  PLOGD << "Using random removal\n";

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  // Sample tasks uniformly without replacement (partial Fisher-Yates),
  // avoiding repeated draws/duplicate checks.
  const int taskCount = instance_.getTasksNum();
  const int numToRemove = (neighborSize_ < taskCount) ? neighborSize_ : taskCount;
  vector<int> taskIds(taskCount);
  std::iota(taskIds.begin(), taskIds.end(), 0);
  int removed = 0;
  for (int i = 0; i < taskCount && removed < numToRemove; i++) {
    std::uniform_int_distribution<int> distribution(i, taskCount - 1);
    const int j = distribution(rng_);
    std::swap(taskIds[i], taskIds[j]);
    const int randomTask = taskIds[i];
    const int randomTaskAgent =
        (randomTask >= 0 && randomTask < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[randomTask]
            : UNASSIGNED;
    if (randomTaskAgent == UNASSIGNED) {
      continue;
    }
    const int randomTaskPosition =
        solution_.getLocalTaskIndex(randomTaskAgent, randomTask);
    if (randomTaskPosition == UNASSIGNED) {
      continue;
    }
    lnsNeighborhood_.removedTasks.emplace(
        randomTask, Conflicts(randomTask, randomTaskAgent, randomTaskPosition));
    removed++;
  }
}

void LNS::conflictRemoval(std::optional<ConflictMap> potentialNeighborhood) {

  PLOGD << "Using conflict-based removal\n";

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  // Conflict removal operator should always be sent this argument!
  assert(potentialNeighborhood.has_value());

  // Extract N conflicts first. This can return N tasks where N <= neighborhood size
  lnsNeighborhood_.removedTasks =
      extractNConflicts(neighborSize_, potentialNeighborhood.value());

  const int taskCount = instance_.getTasksNum();
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    // In this case we have less conflicts than the neighborhood size of the LNS
    // so we need to augment this list with more tasks using random removal.
    if (taskCount <= 0) {
      return;
    }
    std::uniform_int_distribution<int> distribution(0, taskCount - 1);
    while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      const int randomTask = distribution(rng_);
      if (lnsNeighborhood_.removedTasks.count(randomTask) != 0) {
        continue;
      }
      const int randomTaskAgent =
          (randomTask >= 0 && randomTask < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[randomTask]
              : UNASSIGNED;
      if (randomTaskAgent == UNASSIGNED) {
        continue;
      }
      const int randomTaskPosition =
          solution_.getLocalTaskIndex(randomTaskAgent, randomTask);
      if (randomTaskPosition == UNASSIGNED) {
        continue;
      }
      lnsNeighborhood_.removedTasks.emplace(
          randomTask,
          Conflicts(randomTask, randomTaskAgent, randomTaskPosition));
    }
  }
  // The else case should not happen since the 'extractNConflict' will never return more than neighborhood size set
}

void LNS::worstRemoval() {

  PLOGD << "Using worst removal\n";

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  const int taskCount = instance_.getTasksNum();
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  // Maintain a priority queue of (key, value) where key is the path length of a task and the value is the task. We need to do a reverse way to avoid making our own comparator
  ppq worstTasksOrder;
  for (int task = 0; task < taskCount; task++) {
    const int taskAgent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (taskAgent == UNASSIGNED) {
      continue;
    }
    int taskPosition = solution_.getLocalTaskIndex(taskAgent, task);
    if (taskPosition == UNASSIGNED) {
      continue;
    }
    int taskPathSize =
        (int)solution_.agents[taskAgent].taskPaths[taskPosition].size();
    worstTasksOrder.emplace(taskPathSize, task);
  }
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
         !worstTasksOrder.empty()) {
    pair<int, int> worstTaskFromOrder = worstTasksOrder.top();
    int worstTask = worstTaskFromOrder.second;
    // No need to check whether this task was part of the removed tasks already or not since that cannot happen ever!
    const int worstTaskAgent =
        (worstTask >= 0 && worstTask < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[worstTask]
            : UNASSIGNED;
    if (worstTaskAgent == UNASSIGNED) {
      worstTasksOrder.pop();
      continue;
    }
    int worstTaskPosition =
        solution_.getLocalTaskIndex(worstTaskAgent, worstTask);
    if (worstTaskPosition == UNASSIGNED) {
      worstTasksOrder.pop();
      continue;
    }
    Conflicts conflict(worstTask, worstTaskAgent, worstTaskPosition);
    lnsNeighborhood_.removedTasks.emplace(conflict.task, conflict);
    worstTasksOrder.pop();
  }
}

void LNS::shawRemoval(int prioritySize) {
  /*
  Shaw removal works by using the relatedness parameter ->
  r(task_i, task_j) = w1 * distance(task_i_goal, task_j_goal) 
                    + w2 * (abs(task_i_start_time - task_j_start_time) + abs(task_i_end_time - task_j_end_time))

  -> w1 and w2 are parameters that can be tuned
  -> distance(task_i_goal, task_j_goal) is the manhattan distance between the tasks
  -> task_i_start_time is the begin time of the task
  -> task_i_end_time is the end time of the task

  Need to remove N - 1 tasks after selecting the first task randomly. N can be user input parameter
  */

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  // Randomly choose a task and remove it from the solution and add it to the neighborhood
  std::uniform_int_distribution<int> distribution(0,
                                                  instance_.getTasksNum() - 1);

  const int taskCount = instance_.getTasksNum();
  // Sample a random task and remove it!
  int randomTask = -1;
  int randomTaskAgent = UNASSIGNED;
  int randomTaskPosition = -1;
  for (int attempts = 0; attempts < taskCount; ++attempts) {
    const int candidate = distribution(rng_);
    const int candidateAgent =
        (candidate >= 0 && candidate < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[candidate]
            : UNASSIGNED;
    if (candidateAgent == UNASSIGNED) {
      continue;
    }
    randomTask = candidate;
    randomTaskAgent = candidateAgent;
    randomTaskPosition =
        solution_.getLocalTaskIndex(randomTaskAgent, randomTask);
    if (randomTaskPosition == UNASSIGNED) {
      randomTask = -1;
      continue;
    }
    break;
  }
  if (randomTask < 0) {
    PLOGE << "Shaw removal: no assigned task found for removal.\n";
    return;
  }
  Conflicts randomConflict(randomTask, randomTaskAgent, randomTaskPosition);
  lnsNeighborhood_.removedTasks.emplace(randomConflict.task, randomConflict);
  PLOGD << "Shaw Removal Step -> Random Task " << randomTask << " is removed!"
        << "\n";

  const int cappedPrioritySize = min(prioritySize, taskCount);
  const int cappedNeighborSize = min(neighborSize_, taskCount);

  // Get information about random task
  int randomTaskLocation = instance_.getTaskLocations(randomTask);
  int randomTaskST =
      solution_.agents[randomTaskAgent].taskPaths[randomTaskPosition].beginTime;
  int randomTaskET =
      solution_.agents[randomTaskAgent].taskPaths[randomTaskPosition].endTime();

  // Initialize a queue to hold the related tasks and rank by relatedness
  pqRelatedTasks relatedQ;  // TODO: can change to ascending or descending here
  set<RelatedTasks, RelatedTasks::RelatedTasksComparator> expandedTasks;
  vector<bool> alreadyExpanded(taskCount, false);

  // Adding the random task first
  RelatedTasks randomRelatedTask(randomTask, randomTaskAgent,
                                 randomTaskPosition, randomTaskST, randomTaskET,
                                 -1, -1);
  expandedTasks.insert(randomRelatedTask);
  alreadyExpanded[randomTask] = true;

  auto pushRelatedCandidate = [&](int relatedTask) {
    // Get information about related task
    const int relatedTaskAgent =
        (relatedTask >= 0 && relatedTask < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[relatedTask]
            : UNASSIGNED;
    if (relatedTaskAgent == UNASSIGNED) {
      return;
    }
    const int relatedTaskPosition =
        solution_.getLocalTaskIndex(relatedTaskAgent, relatedTask);
    if (relatedTaskPosition == UNASSIGNED) {
      return;
    }

    // Compute the manhattan distance
    const int relatedTaskLocation = instance_.getTaskLocations(relatedTask);
    const int relatedManhattanDistance =
        instance_.getManhattanDistance(randomTaskLocation, relatedTaskLocation);

    // Get the temporal values
    const int relatedTaskST = solution_.agents[relatedTaskAgent]
                                  .taskPaths[relatedTaskPosition]
                                  .beginTime;
    const int relatedTaskET = solution_.agents[relatedTaskAgent]
                                  .taskPaths[relatedTaskPosition]
                                  .endTime();

    // Compute the relatedness (smaller => more related).
    const int temporalDiff =
        abs(randomTaskST - relatedTaskST) + abs(randomTaskET - relatedTaskET);
    const int relatedness =
        (int)(shawDistanceWeight_ * relatedManhattanDistance +
              shawTemporalWeight_ * temporalDiff);

    // Store information
    RelatedTasks relatedToRandomTask(relatedTask, relatedTaskAgent,
                                     relatedTaskPosition, relatedTaskST,
                                     relatedTaskET, relatedManhattanDistance,
                                     relatedness);
    expandedTasks.insert(relatedToRandomTask);
    relatedQ.emplace(relatedness, relatedToRandomTask);
    alreadyExpanded[relatedTask] = true;
  };

  // Fill candidates uniformly at random.
  while ((int)expandedTasks.size() < cappedPrioritySize) {
    const int relatedTask = distribution(rng_);
    if (alreadyExpanded[relatedTask]) {
      continue;
    }
    pushRelatedCandidate(relatedTask);
  }

  // Now remove the most-related tasks.
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
         !relatedQ.empty()) {
    RelatedTasks relatedTask = relatedQ.top().second;
    PLOGD << "Shaw Removal Step -> Related Task " << relatedTask.task
          << " is removed!\n";
    relatedQ.pop();
    // Add the related task to the neighborhood
    Conflicts relatedConflict(relatedTask.task, relatedTask.agent,
                              relatedTask.taskPosition);
    lnsNeighborhood_.removedTasks.emplace(relatedConflict.task, relatedConflict);
  }

  // If the candidate queue was exhausted (e.g., very small instances), augment
  // with random tasks to reach the requested neighborhood size.
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    const int t = distribution(rng_);
    if (lnsNeighborhood_.removedTasks.count(t) != 0) {
      continue;
    }
    const int agent =
        (t >= 0 && t < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[t]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    const int pos = solution_.getLocalTaskIndex(agent, t);
    if (pos == UNASSIGNED) {
      continue;
    }
    lnsNeighborhood_.removedTasks.emplace(t, Conflicts(t, agent, pos));
  }
}

void LNS::precedenceWaitRemoval(
    std::optional<ConflictMap> potentialNeighborhood) {
  PLOGD << "Using precedence-wait removal\n";

  // Clear old information about the LNS neighborhood. This should be the
  // first thing that any removal operator must do!
  clearNeighborhood();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  const auto& predecessors = instance_.getAncestorsRef();
  const auto& successors = instance_.getSuccessorsRef();

  vector<int> criticalPred(taskCount, UNASSIGNED);
  vector<int> releaseTime(taskCount, 0);
  vector<int> precedenceWait(taskCount, 0);
  vector<char> taskFeasible(taskCount, 1);
  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, -1);

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& taskPaths = solution_.agents[agent].taskPaths;
    const int maxPos = min((int)assignments.size(), (int)taskPaths.size());
    for (int pos = 0; pos < maxPos; pos++) {
      const int task = assignments[pos];
      if (task >= 0 && task < taskCount) {
        taskToAgent[task] = agent;
        taskToPosition[task] = pos;
      }
    }
  }

  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0 ||
        taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      taskFeasible[task] = 0;
      continue;
    }

    int prevEnd = 0;
    int prevLocation = instance_.getStartLocationsRef()[agent];
    if (taskPos > 0) {
      const AgentTaskPath& previousTaskPath =
          solution_.agents[agent].taskPaths[taskPos - 1];
      if (!previousTaskPath.empty()) {
        prevEnd = previousTaskPath.endTime();
        prevLocation = previousTaskPath.back().location;
      } else {
        // Defensive fallback for malformed intermediate schedules.
        const int previousTask =
            solution_.agents[agent].taskAssignments[taskPos - 1];
        prevLocation = instance_.getTaskLocations(previousTask);
        prevEnd = 0;
      }
    }

    const int taskLocation = instance_.getTaskLocations(task);
    const int travelToTask =
        instance_.getManhattanDistance(prevLocation, taskLocation);
    const int earliestArrivalNoPrec = prevEnd + travelToTask;

    int maxPredEnd = std::numeric_limits<int>::min();
    int blocker = UNASSIGNED;
    for (int pred : predecessors[task]) {
      if (pred < 0 || pred >= taskCount) {
        continue;
      }
      const int predAgent = taskToAgent[pred];
      const int predPos = taskToPosition[pred];
      if (predAgent == UNASSIGNED) {
        continue;
      }
      if (predPos < 0 ||
          predPos >= (int)solution_.agents[predAgent].taskPaths.size()) {
        continue;
      }
      const AgentTaskPath& predPath = solution_.agents[predAgent].taskPaths[predPos];
      if (predPath.empty()) {
        continue;
      }
      const int predEnd = predPath.endTime();
      if (predEnd > maxPredEnd) {
        maxPredEnd = predEnd;
        blocker = pred;
      }
    }

    if (blocker != UNASSIGNED) {
      criticalPred[task] = blocker;
      releaseTime[task] = maxPredEnd;
      precedenceWait[task] = max(0, maxPredEnd - earliestArrivalNoPrec);
    }
  }

  vector<int> order(taskCount);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int lhs, int rhs) {
                     if (taskFeasible[lhs] != taskFeasible[rhs]) {
                       return taskFeasible[lhs] > taskFeasible[rhs];
                     }
                     if (precedenceWait[lhs] != precedenceWait[rhs]) {
                       return precedenceWait[lhs] > precedenceWait[rhs];
                     }
                     if (releaseTime[lhs] != releaseTime[rhs]) {
                       return releaseTime[lhs] > releaseTime[rhs];
                     }
                     return lhs < rhs;
                   });

  vector<char> selected(taskCount, 0);
  auto addTask = [&](int task) -> bool {
    if (task < 0 || task >= taskCount || selected[task]) {
      return false;
    }
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0) {
      return false;
    }
    selected[task] = 1;
    lnsNeighborhood_.removedTasks.emplace(task, Conflicts(task, agent, taskPos));
    return true;
  };

  // If the current solution is infeasible, keep a conflict-focused anchor so
  // this operator can still drive towards feasibility instead of only
  // precedence reshaping.
  if (potentialNeighborhood.has_value() && !potentialNeighborhood->empty()) {
    // Bootstrap phase: until we find the first feasible incumbent, prioritize
    // conflict-driven neighborhoods to quickly recover feasibility.
    const bool noFeasibleIncumbent = incumbentSolution_.agentPaths.empty();
    const int conflictQuota =
        noFeasibleIncumbent ? cappedNeighborSize : max(1, cappedNeighborSize / 2);
    const ConflictMap conflictSeeds =
        extractNConflicts(conflictQuota, potentialNeighborhood.value());
    for (const auto& [_, conflict] : conflictSeeds) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      addTask(conflict.task);
    }
  }

  auto addOneHopNeighborhood = [&](int task) {
    if (task < 0 || task >= taskCount) {
      return;
    }
    for (int pred : predecessors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(pred);
    }
    for (int succ : successors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(succ);
    }
  };

  // Pass 1: prioritize tasks with positive precedence wait. Include the critical
  // predecessor and one-hop precedence neighbors to address root causes.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    if (!taskFeasible[task] || precedenceWait[task] <= 0) {
      continue;
    }
    const bool insertedSeed = addTask(task);
    if (!insertedSeed) {
      continue;
    }
    const int blocker = criticalPred[task];
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        blocker != UNASSIGNED) {
      addTask(blocker);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      addOneHopNeighborhood(task);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        blocker != UNASSIGNED) {
      addOneHopNeighborhood(blocker);
    }
  }

  // Pass 2: if needed, fill from global precedence-wait order.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    addTask(task);
  }

  // Pass 3: fallback random fill (defensive; should almost never trigger).
  std::uniform_int_distribution<int> distribution(0, taskCount - 1);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    addTask(distribution(rng_));
  }
}

void LNS::lowSlackRemoval(std::optional<ConflictMap> potentialNeighborhood) {
  PLOGD << "Using low-slack removal\n";

  // Clear old information about the LNS neighborhood. This should be the
  // first thing that any removal operator must do!
  clearNeighborhood();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  const auto& predecessors = instance_.getAncestorsRef();
  const auto& successors = instance_.getSuccessorsRef();
  const int INF = std::numeric_limits<int>::max() / 4;

  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, -1);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& taskPaths = solution_.agents[agent].taskPaths;
    const int maxPos = min((int)assignments.size(), (int)taskPaths.size());
    for (int pos = 0; pos < maxPos; pos++) {
      const int task = assignments[pos];
      if (task >= 0 && task < taskCount) {
        taskToAgent[task] = agent;
        taskToPosition[task] = pos;
      }
    }
  }

  vector<char> taskFeasible(taskCount, 1);
  vector<char> hasSuccessor(taskCount, 0);
  vector<int> slack(taskCount, INF);
  vector<int> tightSuccessor(taskCount, UNASSIGNED);

  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0 ||
        taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      taskFeasible[task] = 0;
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[taskPos];
    if (taskPath.empty()) {
      taskFeasible[task] = 0;
      continue;
    }

    const int endTask = taskPath.endTime();
    int minSuccStart = INF;
    int minSucc = UNASSIGNED;
    for (int succ : successors[task]) {
      if (succ < 0 || succ >= taskCount) {
        continue;
      }
      const int succAgent = taskToAgent[succ];
      const int succPos = taskToPosition[succ];
      if (succAgent == UNASSIGNED || succPos < 0 ||
          succPos >= (int)solution_.agents[succAgent].taskPaths.size()) {
        continue;
      }
      const AgentTaskPath& succPath = solution_.agents[succAgent].taskPaths[succPos];
      if (succPath.empty()) {
        continue;
      }
      hasSuccessor[task] = 1;
      const int succStart = succPath.beginTime;
      if (succStart < minSuccStart) {
        minSuccStart = succStart;
        minSucc = succ;
      }
    }

    if (minSucc != UNASSIGNED) {
      tightSuccessor[task] = minSucc;
      slack[task] = minSuccStart - endTask;
    }
  }

  vector<int> order(taskCount);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    if (taskFeasible[lhs] != taskFeasible[rhs]) {
      return taskFeasible[lhs] > taskFeasible[rhs];
    }
    if (hasSuccessor[lhs] != hasSuccessor[rhs]) {
      return hasSuccessor[lhs] > hasSuccessor[rhs];
    }
    if (slack[lhs] != slack[rhs]) {
      return slack[lhs] < slack[rhs];  // lower slack = tighter
    }
    return lhs < rhs;
  });

  vector<char> selected(taskCount, 0);
  auto addTask = [&](int task) -> bool {
    if (task < 0 || task >= taskCount || selected[task]) {
      return false;
    }
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0) {
      return false;
    }
    selected[task] = 1;
    lnsNeighborhood_.removedTasks.emplace(task, Conflicts(task, agent, taskPos));
    return true;
  };

  // If the current solution is infeasible, keep a conflict-focused anchor so
  // this operator can recover feasibility.
  if (potentialNeighborhood.has_value() && !potentialNeighborhood->empty()) {
    const int conflictQuota = max(1, cappedNeighborSize / 2);
    const ConflictMap conflictSeeds =
        extractNConflicts(conflictQuota, potentialNeighborhood.value());
    for (const auto& [_, conflict] : conflictSeeds) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      addTask(conflict.task);
    }
  }

  auto addOneHopNeighborhood = [&](int task) {
    if (task < 0 || task >= taskCount) {
      return;
    }
    for (int pred : predecessors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(pred);
    }
    for (int succ : successors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(succ);
    }
  };

  // Pass 1: remove low-slack tasks and their tight successors / local DAG
  // neighbors to unlock precedence bottlenecks.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    if (!taskFeasible[task] || !hasSuccessor[task]) {
      continue;
    }
    const bool insertedSeed = addTask(task);
    if (!insertedSeed) {
      continue;
    }
    const int succ = tightSuccessor[task];
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        succ != UNASSIGNED) {
      addTask(succ);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      addOneHopNeighborhood(task);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        succ != UNASSIGNED) {
      addOneHopNeighborhood(succ);
    }
  }

  // Pass 2: if needed, fill from sorted slack order.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    addTask(task);
  }

  // Pass 3: random fallback to guarantee neighborhood size.
  std::uniform_int_distribution<int> distribution(0, taskCount - 1);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    addTask(distribution(rng_));
  }
}

void LNS::alnsRemoval(std::optional<ConflictMap> potentialNeighborhood) {

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  adaptiveLNS_.alnsCounter++;

  // Cannot update the successes in the first iteration!
  if (iterationStats.size() > 1) {
    // Incorporate the results of the heuristic performance in the last iteration
    switch (iterationStats.back().quality) {
      case bestSolutionYet:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta1;
        break;
      case improvedSolution:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta2;
        break;
      case downgradedButAccepted:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta3;
        break;
      default:
        break;
    }
  }

  if (adaptiveLNS_.alnsCounter >= adaptiveLNS_.alnsCounterThreshold) {
    // Need to update the weights here!
    for (int i = 0; i < adaptiveLNS_.numDestroyHeuristics; i++) {
      if (adaptiveLNS_.used[i] > 0) {
        adaptiveLNS_.weights[i] =
            (1.0 - adaptiveLNS_.reactionFactor) * adaptiveLNS_.weights[i] +
            adaptiveLNS_.reactionFactor *
                (adaptiveLNS_.success[i] / adaptiveLNS_.used[i]);
      } else {
        adaptiveLNS_.weights[i] =
            (1.0 - adaptiveLNS_.reactionFactor) * adaptiveLNS_.weights[i];
      }
      adaptiveLNS_.used[i] = 0;
      adaptiveLNS_.success[i] = 0;
    }
    adaptiveLNS_.alnsCounter = 0;
  }
  // Sample the destroy heuristic and extract the neighborhood
  std::discrete_distribution<> distribution(adaptiveLNS_.weights.begin(),
                                            adaptiveLNS_.weights.end());

  int sampledDestroyHeuristic = distribution(rng_);
  adaptiveLNS_.recentDestroyHeuristic = sampledDestroyHeuristic;
  switch (sampledDestroyHeuristic) {
    case DestroyHeuristic::randomRemoval:  // RANDOM
      randomRemoval();
      break;
    case DestroyHeuristic::worstRemoval:  // WORST
      worstRemoval();
      break;
    case DestroyHeuristic::conflictRemoval:  // CONFLICT
      conflictRemoval(std::move(potentialNeighborhood));
      break;
    case DestroyHeuristic::shawRemoval:  // SHAW
      shawRemoval(neighborSize_ * 3);
      break;
    case DestroyHeuristic::precedenceWaitRemoval:  // PRECEDENCE WAIT
      precedenceWaitRemoval(std::move(potentialNeighborhood));
      break;
    case DestroyHeuristic::lowSlackRemoval:  // LOW SLACK
      lowSlackRemoval(std::move(potentialNeighborhood));
      break;
    case DestroyHeuristic::marketTatonnementRemoval:  // MARKET TATONNEMENT
      marketTatonnementRemoval(std::move(potentialNeighborhood));
      break;
    default:
      PLOGE << "Sampled a non-existent destroy heuristic: "
            << sampledDestroyHeuristic << "\n";
      assert(false);
      return;
  }

  adaptiveLNS_.used[sampledDestroyHeuristic] += 1;
  adaptiveLNS_.destroyHeuristicHistory.push_back(sampledDestroyHeuristic);
}
