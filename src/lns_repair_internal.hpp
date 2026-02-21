#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

struct LNS::RegretWorkspace {
  explicit RegretWorkspace(const Solution& baseSolution)
      : base(baseSolution),
        assignmentsOwned(baseSolution.agents.size()),
        pathsOwned(baseSolution.agents.size()),
        owns(baseSolution.agents.size(), 0) {}

  int numAgents() const { return static_cast<int>(owns.size()); }

  bool ownsAgent(int agent) const {
    return agent >= 0 && agent < static_cast<int>(owns.size()) &&
           owns[agent] != 0;
  }

  void ensureOwned(int agent) {
    if (agent < 0 || agent >= static_cast<int>(owns.size()) || owns[agent]) {
      return;
    }
    assignmentsOwned[agent] = base.agents[agent].taskAssignments;
    pathsOwned[agent] = base.agents[agent].taskPaths;
    owns[agent] = 1;
    clonedAgents_++;
  }

  const vector<int>& assignments(int agent) const {
    return ownsAgent(agent) ? assignmentsOwned[agent]
                            : base.agents[agent].taskAssignments;
  }

  const vector<AgentTaskPath>& taskPaths(int agent) const {
    return ownsAgent(agent) ? pathsOwned[agent] : base.agents[agent].taskPaths;
  }

  vector<int>& mutableAssignments(int agent) {
    ensureOwned(agent);
    return assignmentsOwned[agent];
  }

  vector<AgentTaskPath>& mutableTaskPaths(int agent) {
    ensureOwned(agent);
    return pathsOwned[agent];
  }

  int clonedAgents() const { return clonedAgents_; }

 private:
  const Solution& base;
  vector<vector<int>> assignmentsOwned;
  vector<vector<AgentTaskPath>> pathsOwned;
  vector<uint8_t> owns;
  int clonedAgents_ = 0;
};

namespace {
struct AssignmentLookup {
  vector<int> owner;
  vector<int> pos;
};

struct InsertTaskRollbackOp {
  enum class Kind {
    insertAssignment,
    insertTaskPath,
    setTaskPath,
    setTaskPathBeginTime
  };

  Kind kind = Kind::setTaskPath;
  int agent = UNASSIGNED;
  int index = -1;
  int oldBeginTime = 0;
  AgentTaskPath oldPath;
};

struct InsertTaskRollbackLog {
  vector<InsertTaskRollbackOp> ops;

  void rollback(LNS::RegretWorkspace& workspace) {
    for (int i = (int)ops.size() - 1; i >= 0; i--) {
      InsertTaskRollbackOp& op = ops[i];
      if (op.agent < 0 || op.agent >= workspace.numAgents()) {
        continue;
      }
      vector<int>& assignments = workspace.mutableAssignments(op.agent);
      vector<AgentTaskPath>& taskPaths = workspace.mutableTaskPaths(op.agent);
      switch (op.kind) {
        case InsertTaskRollbackOp::Kind::insertAssignment:
          if (op.index >= 0 && op.index < (int)assignments.size()) {
            assignments.erase(assignments.begin() + op.index);
          }
          break;
        case InsertTaskRollbackOp::Kind::insertTaskPath:
          if (op.index >= 0 && op.index < (int)taskPaths.size()) {
            taskPaths.erase(taskPaths.begin() + op.index);
          }
          break;
        case InsertTaskRollbackOp::Kind::setTaskPath:
          if (op.index >= 0 && op.index < (int)taskPaths.size()) {
            taskPaths[op.index] = std::move(op.oldPath);
          }
          break;
        case InsertTaskRollbackOp::Kind::setTaskPathBeginTime:
          if (op.index >= 0 && op.index < (int)taskPaths.size()) {
            taskPaths[op.index].beginTime = op.oldBeginTime;
          }
          break;
      }
    }
  }
};

struct ScopedInsertTaskRollback {
  InsertTaskRollbackLog* log = nullptr;
  LNS::RegretWorkspace* workspace = nullptr;
  bool enabled = false;

  ~ScopedInsertTaskRollback() {
    if (enabled && log != nullptr && workspace != nullptr) {
      log->rollback(*workspace);
    }
  }
};

[[maybe_unused]] AssignmentLookup buildAssignmentLookup(
    const LNS::RegretWorkspace& workspace, int taskCount) {
  AssignmentLookup lookup;
  lookup.owner.assign(taskCount, UNASSIGNED);
  lookup.pos.assign(taskCount, -1);
  for (int agent = 0; agent < workspace.numAgents(); agent++) {
    const auto& assignments = workspace.assignments(agent);
    for (int i = 0; i < (int)assignments.size(); i++) {
      const int task = assignments[i];
      if (task >= 0 && task < taskCount) {
        lookup.owner[task] = agent;
        lookup.pos[task] = i;
      }
    }
  }
  return lookup;
}

[[maybe_unused]] bool isPendingCommitState(const Neighbor& neighborhood,
                                           int task) {
  return task >= 0 && task < (int)neighborhood.committedTasks.size() &&
         neighborhood.committedTasks[task] == 0;
}

[[maybe_unused]] int resolveTaskEndTimeFromMixedState(
    int task, const LNS::RegretWorkspace& workspace,
    const AssignmentLookup& workspaceIndex, const Solution& previousSolution,
    const Neighbor& neighborhood, bool* usedPreviousFallback) {
  if (usedPreviousFallback != nullptr) {
    *usedPreviousFallback = false;
  }
  if (task < 0 || task >= (int)workspaceIndex.owner.size()) {
    return -1;
  }

  const int workspaceOwner = workspaceIndex.owner[task];
  const int workspacePos = workspaceIndex.pos[task];
  if (workspaceOwner != UNASSIGNED && workspaceOwner >= 0 &&
      workspaceOwner < workspace.numAgents() && workspacePos >= 0 &&
      workspacePos < (int)workspace.taskPaths(workspaceOwner).size()) {
    const auto& workspacePath = workspace.taskPaths(workspaceOwner)[workspacePos];
    if (!workspacePath.empty()) {
      return workspacePath.endTime();
    }
  }

  if (!isPendingCommitState(neighborhood, task)) {
    return -1;
  }

  const int previousOwner =
      (task >= 0 && task < (int)previousSolution.taskAgentMap.size())
          ? previousSolution.taskAgentMap[task]
          : UNASSIGNED;
  if (previousOwner == UNASSIGNED || previousOwner < 0 ||
      previousOwner >= (int)previousSolution.agents.size()) {
    return -1;
  }

  const int previousPos = previousSolution.getLocalTaskIndex(previousOwner, task);
  if (previousPos == UNASSIGNED || previousPos < 0 ||
      previousPos >= (int)previousSolution.agents[previousOwner].taskPaths.size()) {
    return -1;
  }

  const auto& previousPath =
      previousSolution.agents[previousOwner].taskPaths[previousPos];
  if (previousPath.empty()) {
    return -1;
  }

  if (usedPreviousFallback != nullptr) {
    *usedPreviousFallback = true;
  }
  return previousPath.endTime();
}

[[maybe_unused]] double computeMedian(vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

[[maybe_unused]] void normalizeSeriesByRobustScale(
    const vector<double>& rawValues, const vector<int>& validIndices,
    vector<double>& normalizedValues) {
  normalizedValues.assign(rawValues.size(), 0.0);
  if (validIndices.empty()) {
    return;
  }

  vector<double> samples;
  samples.reserve(validIndices.size());
  for (int idx : validIndices) {
    samples.push_back(rawValues[idx]);
  }

  const double median = computeMedian(samples);
  vector<double> absDeviations;
  absDeviations.reserve(samples.size());
  for (double value : samples) {
    absDeviations.push_back(std::abs(value - median));
  }

  double center = median;
  constexpr double kMadToSigma = 1.482602218505602;
  double scale = kMadToSigma * computeMedian(absDeviations);

  if (!(scale > 1e-9)) {
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        (double)samples.size();
    double variance = 0.0;
    for (double value : samples) {
      const double delta = value - mean;
      variance += delta * delta;
    }
    variance /= (double)samples.size();
    center = mean;
    scale = std::sqrt(variance);
  }

  if (!(scale > 1e-9)) {
    return;
  }

  for (int idx : validIndices) {
    normalizedValues[idx] = (rawValues[idx] - center) / scale;
  }
}
}  // namespace
