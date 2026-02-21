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

  // Randomly choose a task and remove it from the solution and add it to the neighborhood
  std::uniform_int_distribution<int> distribution(0,
                                                  instance_.getTasksNum() - 1);

  const int taskCount = instance_.getTasksNum();
  const vector<int> taskToPosition =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);
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
        (randomTask >= 0 && randomTask < (int)taskToPosition.size())
            ? taskToPosition[randomTask]
            : UNASSIGNED;
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
        (relatedTask >= 0 && relatedTask < (int)taskToPosition.size())
            ? taskToPosition[relatedTask]
            : UNASSIGNED;
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
    const double relatedness =
        shawDistanceWeight_ * relatedManhattanDistance +
        shawTemporalWeight_ * temporalDiff;

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
  int candidateFillAttempts = 0;
  const int maxCandidateFillAttempts = max(64, taskCount * 8);
  while ((int)expandedTasks.size() < cappedPrioritySize &&
         candidateFillAttempts < maxCandidateFillAttempts) {
    const int relatedTask = distribution(rng_);
    candidateFillAttempts++;
    if (alreadyExpanded[relatedTask]) {
      continue;
    }
    pushRelatedCandidate(relatedTask);
  }
  if ((int)expandedTasks.size() < cappedPrioritySize) {
    for (int task = 0;
         task < taskCount && (int)expandedTasks.size() < cappedPrioritySize;
         task++) {
      if (!alreadyExpanded[task]) {
        pushRelatedCandidate(task);
      }
    }
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
  int finalFillAttempts = 0;
  const int maxFinalFillAttempts = max(64, taskCount * 8);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
         finalFillAttempts < maxFinalFillAttempts) {
    finalFillAttempts++;
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
    const int pos = (t >= 0 && t < (int)taskToPosition.size())
                        ? taskToPosition[t]
                        : UNASSIGNED;
    if (pos == UNASSIGNED) {
      continue;
    }
    lnsNeighborhood_.removedTasks.emplace(t, Conflicts(t, agent, pos));
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    for (int task = 0;
         task < taskCount &&
         (int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize;
         task++) {
      if (lnsNeighborhood_.removedTasks.count(task) != 0) {
        continue;
      }
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
      lnsNeighborhood_.removedTasks.emplace(task, Conflicts(task, agent, pos));
    }
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    PLOGW << "shawRemoval: could only remove " << lnsNeighborhood_.removedTasks.size()
          << " out of requested " << cappedNeighborSize << " tasks\n";
  }
}

