#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"

void LNS::precedenceWaitRemoval(const ConflictMap* potentialNeighborhood) {
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
  const vector<int> taskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_,
                                                                  taskCount);

  vector<int> criticalPred(taskCount, UNASSIGNED);
  vector<int> releaseTime(taskCount, 0);
  vector<int> precedenceWait(taskCount, 0);
  vector<char> taskFeasible(taskCount, 1);
  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, -1);
  vector<TaskScheduleMetrics> scheduleMetrics;
  computeTaskScheduleMetricsFromIndex(taskPosByTask, scheduleMetrics, nullptr);

  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    const int pos = (task >= 0 && task < (int)taskPosByTask.size())
                        ? taskPosByTask[task]
                        : -1;
    if (agent < 0 || agent >= instance_.getAgentNum() || pos < 0 ||
        pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    taskToAgent[task] = agent;
    taskToPosition[task] = pos;
  }

  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0 ||
        taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      taskFeasible[task] = 0;
      continue;
    }

    if (task >= (int)scheduleMetrics.size()) {
      taskFeasible[task] = 0;
      continue;
    }
    const TaskScheduleMetrics& metric = scheduleMetrics[task];
    if (!metric.valid) {
      taskFeasible[task] = 0;
      continue;
    }
    if (metric.blocker != UNASSIGNED) {
      criticalPred[task] = metric.blocker;
      releaseTime[task] = metric.release;
      precedenceWait[task] = metric.waitPrec;
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
  if (potentialNeighborhood != nullptr && !potentialNeighborhood->empty()) {
    // Bootstrap phase: until we find the first feasible incumbent, prioritize
    // conflict-driven neighborhoods to quickly recover feasibility.
    const bool noFeasibleIncumbent = incumbentSolution_.agentPaths.empty();
    const int conflictQuota =
        noFeasibleIncumbent ? cappedNeighborSize : max(1, cappedNeighborSize / 2);
    const ConflictMap conflictSeeds =
        extractNConflicts(conflictQuota, *potentialNeighborhood);
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

  // Pass 3: deterministic finite fill.
  for (int task = 0;
       task < taskCount &&
       (int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize;
       task++) {
    addTask(task);
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    PLOGW << "precedenceWaitRemoval: could only remove "
          << lnsNeighborhood_.removedTasks.size() << " out of requested "
          << cappedNeighborSize << " tasks\n";
  }
}

void LNS::lowSlackRemoval(const ConflictMap* potentialNeighborhood) {
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
  const vector<int> taskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_,
                                                                  taskCount);

  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, -1);
  vector<TaskScheduleMetrics> scheduleMetrics;
  computeTaskScheduleMetricsFromIndex(taskPosByTask, scheduleMetrics, nullptr);
  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    const int pos = (task >= 0 && task < (int)taskPosByTask.size())
                        ? taskPosByTask[task]
                        : -1;
    if (agent < 0 || agent >= instance_.getAgentNum() || pos < 0 ||
        pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    taskToAgent[task] = agent;
    taskToPosition[task] = pos;
  }

  struct CriticalEdge {
    int from = UNASSIGNED;
    int to = UNASSIGNED;
    int slack = INF;
  };
  vector<CriticalEdge> edgeOrder;
  edgeOrder.reserve(taskCount);

  vector<char> taskFeasible(taskCount, 0);
  vector<char> hasCriticalEdge(taskCount, 0);
  vector<int> bestIncidentSlack(taskCount, INF);

  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0 ||
        taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    if (task >= (int)scheduleMetrics.size() || !scheduleMetrics[task].valid) {
      continue;
    }
    taskFeasible[task] = 1;

    const int endTask = scheduleMetrics[task].end;
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
      if (succ >= (int)scheduleMetrics.size() || !scheduleMetrics[succ].valid) {
        continue;
      }
      // Critical-edge slack uses realized successor arrival and predecessor end.
      const int edgeSlack = scheduleMetrics[succ].arrive - endTask;
      edgeOrder.push_back({task, succ, edgeSlack});
      hasCriticalEdge[task] = 1;
      hasCriticalEdge[succ] = 1;
      bestIncidentSlack[task] = min(bestIncidentSlack[task], edgeSlack);
      bestIncidentSlack[succ] = min(bestIncidentSlack[succ], edgeSlack);
    }
  }

  std::stable_sort(edgeOrder.begin(), edgeOrder.end(),
                   [](const CriticalEdge& lhs, const CriticalEdge& rhs) {
                     if (lhs.slack != rhs.slack) {
                       return lhs.slack < rhs.slack;  // tighter edge first
                     }
                     if (lhs.from != rhs.from) {
                       return lhs.from < rhs.from;
                     }
                     return lhs.to < rhs.to;
                   });

  vector<int> order(taskCount);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    if (taskFeasible[lhs] != taskFeasible[rhs]) {
      return taskFeasible[lhs] > taskFeasible[rhs];
    }
    if (hasCriticalEdge[lhs] != hasCriticalEdge[rhs]) {
      return hasCriticalEdge[lhs] > hasCriticalEdge[rhs];
    }
    if (bestIncidentSlack[lhs] != bestIncidentSlack[rhs]) {
      return bestIncidentSlack[lhs] < bestIncidentSlack[rhs];
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
  if (potentialNeighborhood != nullptr && !potentialNeighborhood->empty()) {
    const int conflictQuota = max(1, cappedNeighborSize / 2);
    const ConflictMap conflictSeeds =
        extractNConflicts(conflictQuota, *potentialNeighborhood);
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

  // Pass 1: remove endpoints of low-slack precedence edges and their local DAG
  // neighborhoods to unlock tight handoffs.
  for (const auto& edge : edgeOrder) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    const int task = edge.from;
    const int succ = edge.to;
    if (!taskFeasible[task] || !taskFeasible[succ]) {
      continue;
    }
    const bool insertedTask = addTask(task);
    const bool insertedSucc = addTask(succ);
    if (!insertedTask && !insertedSucc) {
      continue;
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      addOneHopNeighborhood(task);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
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

  // Pass 3: deterministic finite fill.
  for (int task = 0;
       task < taskCount &&
       (int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize;
       task++) {
    addTask(task);
  }
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    PLOGW << "lowSlackRemoval: could only remove "
          << lnsNeighborhood_.removedTasks.size() << " out of requested "
          << cappedNeighborSize << " tasks\n";
  }
}
