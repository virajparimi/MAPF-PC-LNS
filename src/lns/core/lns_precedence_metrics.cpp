#include "lns.hpp"

#include <algorithm>

void LNS::computeTaskScheduleMetricsFromIndex(
    const vector<int>& taskPosByTask, vector<TaskScheduleMetrics>& perTask,
    vector<double>* blockedWaitSum) const {
  const int taskCount = instance_.getTasksNum();
  perTask.assign(taskCount, TaskScheduleMetrics());
  if (blockedWaitSum != nullptr) {
    blockedWaitSum->assign(taskCount, 0.0);
  }
  const auto& predecessors = instance_.getAncestorsRef();

  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    const int pos = (task >= 0 && task < (int)taskPosByTask.size())
                        ? taskPosByTask[task]
                        : -1;
    if (pos < 0) {
      continue;
    }
    if (pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
    if (taskPath.empty()) {
      continue;
    }

    TaskScheduleMetrics metric;
    metric.valid = true;
    metric.end = taskPath.endTime();
    const int taskLocation = instance_.getTaskLocations(task);
    int arrive = taskPath.endTime();
    for (int i = 0; i < (int)taskPath.size(); i++) {
      if (taskPath[i].location == taskLocation) {
        arrive = taskPath.beginTime + i;
        break;
      }
    }
    metric.arrive = arrive;
    metric.release = 0;
    metric.blocker = UNASSIGNED;

    if (task >= 0 && task < (int)predecessors.size()) {
      for (int pred : predecessors[task]) {
        const int predAgent =
            (pred >= 0 && pred < (int)solution_.taskAgentMap.size())
                ? solution_.taskAgentMap[pred]
                : UNASSIGNED;
        if (predAgent == UNASSIGNED) {
          continue;
        }
        const int predPos = (pred >= 0 && pred < (int)taskPosByTask.size())
                                ? taskPosByTask[pred]
                                : -1;
        if (predPos < 0) {
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
        if (predEnd > metric.release) {
          metric.release = predEnd;
          metric.blocker = pred;
        }
      }
    }

    metric.start = max(metric.arrive, metric.release);
    metric.waitPrec = max(0, metric.start - metric.arrive);
    perTask[task] = metric;
    if (blockedWaitSum != nullptr && metric.blocker != UNASSIGNED) {
      (*blockedWaitSum)[metric.blocker] += metric.waitPrec;
    }
  }
}

void LNS::computeTaskScheduleMetrics(vector<TaskScheduleMetrics>& perTask,
                                     vector<double>* blockedWaitSum) const {
  const vector<int>& taskPosByTask = getCurrentTaskPositionIndexByTask();
  computeTaskScheduleMetricsFromIndex(taskPosByTask, perTask, blockedWaitSum);
}

double LNS::computeSolutionPrecedenceWaitFromIndex(
    const vector<int>& taskPosByTask) const {
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetricsFromIndex(taskPosByTask, perTask, nullptr);
  double totalWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
  }
  return totalWait;
}

double LNS::computeSolutionPrecedenceWait() const {
  const vector<int>& taskPosByTask = getCurrentTaskPositionIndexByTask();
  return computeSolutionPrecedenceWaitFromIndex(taskPosByTask);
}
