#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"

void LNS::randomRemoval() {

  PLOGD << "Using random removal\n";

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  // Sample tasks uniformly without replacement (partial Fisher-Yates),
  // avoiding repeated draws/duplicate checks.
  const int taskCount = instance_.getTasksNum();
  const vector<int> taskToPosition =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);
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
        (randomTask >= 0 && randomTask < (int)taskToPosition.size())
            ? taskToPosition[randomTask]
            : UNASSIGNED;
    if (randomTaskPosition == UNASSIGNED) {
      continue;
    }
    lnsNeighborhood_.removedTasks.emplace(
        randomTask, Conflicts(randomTask, randomTaskAgent, randomTaskPosition));
    removed++;
  }
}

void LNS::conflictRemoval(const ConflictMap* potentialNeighborhood) {

  PLOGD << "Using conflict-based removal\n";

  // Clear old information about the LNS neighborhood. This should be the first
  // thing that any removal operator must do!
  clearNeighborhood();

  if (potentialNeighborhood == nullptr) {
    PLOGE << "conflictRemoval requires a non-null conflict neighborhood\n";
    return;
  }

  // Extract N conflicts first. This can return N tasks where N <= neighborhood size
  lnsNeighborhood_.removedTasks =
      extractNConflicts(neighborSize_, *potentialNeighborhood);

  const int taskCount = instance_.getTasksNum();
  const vector<int> taskToPosition =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);
  const int cappedNeighborSize = min(neighborSize_, taskCount);

  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    // Augment conflicts without unbounded rejection-sampling loops.
    vector<int> candidates;
    candidates.reserve(taskCount);
    for (int task = 0; task < taskCount; task++) {
      if (lnsNeighborhood_.removedTasks.count(task) != 0) {
        continue;
      }
      const int taskAgent =
          (task >= 0 && task < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[task]
              : UNASSIGNED;
      if (taskAgent == UNASSIGNED) {
        continue;
      }
      const int taskPosition =
          (task >= 0 && task < (int)taskToPosition.size()) ? taskToPosition[task]
                                                           : UNASSIGNED;
      if (taskPosition == UNASSIGNED) {
        continue;
      }
      candidates.push_back(task);
    }
    std::shuffle(candidates.begin(), candidates.end(), rng_);
    for (int task : candidates) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      const int taskAgent = solution_.taskAgentMap[task];
      const int taskPosition =
          (task >= 0 && task < (int)taskToPosition.size()) ? taskToPosition[task]
                                                           : UNASSIGNED;
      if (taskPosition == UNASSIGNED) {
        continue;
      }
      lnsNeighborhood_.removedTasks.emplace(task,
                                            Conflicts(task, taskAgent, taskPosition));
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      PLOGW << "conflictRemoval: could only remove "
            << lnsNeighborhood_.removedTasks.size() << " out of requested "
            << cappedNeighborSize << " tasks\n";
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
  const vector<int> taskToPosition =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);
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
    int taskPosition = (task >= 0 && task < (int)taskToPosition.size())
                           ? taskToPosition[task]
                           : UNASSIGNED;
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
    int worstTaskPosition = (worstTask >= 0 &&
                             worstTask < (int)taskToPosition.size())
                                ? taskToPosition[worstTask]
                                : UNASSIGNED;
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

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedPrioritySize = min(prioritySize, taskCount);
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  std::uniform_int_distribution<int> distribution(0, taskCount - 1);
  const vector<int> taskToPosition =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_,
                                                                  taskCount);

  vector<int> taskAgent(taskCount, UNASSIGNED);
  vector<int> taskPosition(taskCount, UNASSIGNED);
  vector<int> taskStart(taskCount, UNASSIGNED);
  vector<int> taskEnd(taskCount, UNASSIGNED);
  vector<char> taskUsable(taskCount, 0);
  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    const int pos = (task >= 0 && task < (int)taskToPosition.size())
                        ? taskToPosition[task]
                        : UNASSIGNED;
    if (pos == UNASSIGNED) {
      continue;
    }
    taskAgent[task] = agent;
    taskPosition[task] = pos;
    taskStart[task] = solution_.agents[agent].taskPaths[pos].beginTime;
    taskEnd[task] = solution_.agents[agent].taskPaths[pos].endTime();
    taskUsable[task] = 1;
  }

  // Sample a random assigned task as the seed.
  int referenceTask = -1;
  for (int attempts = 0; attempts < taskCount; ++attempts) {
    const int candidate = distribution(rng_);
    if (!taskUsable[candidate]) {
      continue;
    }
    referenceTask = candidate;
    break;
  }
  if (referenceTask < 0) {
    PLOGE << "Shaw removal: no assigned task found for removal.\n";
    return;
  }

  lnsNeighborhood_.removedTasks.emplace(
      referenceTask,
      Conflicts(referenceTask, taskAgent[referenceTask],
                taskPosition[referenceTask]));
  PLOGD << "Shaw Removal Step -> Random Task " << referenceTask
        << " is removed!\n";

  // Candidate pool: randomized one-pass permutation; we rescore this pool
  // against the current reference task after each removal (iterative Shaw).
  vector<int> candidateOrder(taskCount);
  std::iota(candidateOrder.begin(), candidateOrder.end(), 0);
  std::shuffle(candidateOrder.begin(), candidateOrder.end(), rng_);
  size_t candidateCursor = 0;
  vector<int> candidatePool;
  candidatePool.reserve(max(1, cappedPrioritySize));
  vector<char> inPool(taskCount, 0);

  auto refillPool = [&]() {
    while ((int)candidatePool.size() < cappedPrioritySize &&
           candidateCursor < candidateOrder.size()) {
      const int task = candidateOrder[candidateCursor++];
      if (!taskUsable[task] || inPool[task] ||
          lnsNeighborhood_.removedTasks.count(task) != 0) {
        continue;
      }
      candidatePool.push_back(task);
      inPool[task] = 1;
    }
  };
  refillPool();

  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    if (candidatePool.empty()) {
      refillPool();
      if (candidatePool.empty()) {
        break;
      }
    }

    const int referenceLocation = instance_.getTaskLocations(referenceTask);
    const int referenceST = taskStart[referenceTask];
    const int referenceET = taskEnd[referenceTask];

    int bestPoolIdx = -1;
    double bestRelatedness = std::numeric_limits<double>::infinity();
    for (int idx = 0; idx < (int)candidatePool.size(); idx++) {
      const int task = candidatePool[idx];
      const int relatedLocation = instance_.getTaskLocations(task);
      const int relatedManhattanDistance =
          instance_.getManhattanDistance(referenceLocation, relatedLocation);
      const int temporalDiff =
          abs(referenceST - taskStart[task]) + abs(referenceET - taskEnd[task]);
      const double relatedness =
          shawDistanceWeight_ * relatedManhattanDistance +
          shawTemporalWeight_ * temporalDiff;
      if (relatedness < bestRelatedness) {
        bestRelatedness = relatedness;
        bestPoolIdx = idx;
      }
    }

    if (bestPoolIdx < 0) {
      break;
    }

    const int chosenTask = candidatePool[bestPoolIdx];
    PLOGD << "Shaw Removal Step -> Related Task " << chosenTask
          << " is removed!\n";
    lnsNeighborhood_.removedTasks.emplace(
        chosenTask, Conflicts(chosenTask, taskAgent[chosenTask],
                              taskPosition[chosenTask]));
    referenceTask = chosenTask;

    const int lastPoolTask = candidatePool.back();
    candidatePool[bestPoolIdx] = lastPoolTask;
    inPool[lastPoolTask] = 1;
    candidatePool.pop_back();
    inPool[chosenTask] = 0;
    refillPool();
  }

  // If iterative-Shaw candidate pool is exhausted (small instance / sparse
  // assignments), augment with random then linear fallback to reach requested
  // neighborhood size.
  int finalFillAttempts = 0;
  const int maxFinalFillAttempts = max(64, taskCount * 8);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
         finalFillAttempts < maxFinalFillAttempts) {
    finalFillAttempts++;
    const int task = distribution(rng_);
    if (!taskUsable[task] || lnsNeighborhood_.removedTasks.count(task) != 0) {
      continue;
    }
    lnsNeighborhood_.removedTasks.emplace(
        task, Conflicts(task, taskAgent[task], taskPosition[task]));
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    for (int task = 0;
         task < taskCount &&
         (int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize;
         task++) {
      if (!taskUsable[task] || lnsNeighborhood_.removedTasks.count(task) != 0) {
        continue;
      }
      lnsNeighborhood_.removedTasks.emplace(
          task, Conflicts(task, taskAgent[task], taskPosition[task]));
    }
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    PLOGW << "shawRemoval: could only remove "
          << lnsNeighborhood_.removedTasks.size() << " out of requested "
          << cappedNeighborSize << " tasks\n";
  }
}
