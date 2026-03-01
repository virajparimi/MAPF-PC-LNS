#include "lns.hpp"
#include "lns_repair_internal.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace {
vector<int> orderTasksTopologically(const vector<int>& tasks,
                                    const vector<pair<int, int>>& edges) {
  vector<int> uniqueTasks;
  uniqueTasks.reserve(tasks.size());
  unordered_set<int> seen;
  seen.reserve(tasks.size() * 2 + 1);
  for (int task : tasks) {
    if (task < 0) {
      continue;
    }
    if (seen.insert(task).second) {
      uniqueTasks.push_back(task);
    }
  }
  if (uniqueTasks.size() <= 1) {
    return uniqueTasks;
  }

  unordered_map<int, int> indexOf;
  indexOf.reserve(uniqueTasks.size() * 2 + 1);
  for (int i = 0; i < (int)uniqueTasks.size(); i++) {
    indexOf[uniqueTasks[i]] = i;
  }

  vector<vector<int>> succ(uniqueTasks.size());
  vector<int> indegree(uniqueTasks.size(), 0);
  unordered_set<uint64_t> dedupEdges;
  dedupEdges.reserve(edges.size() * 2 + 1);
  for (const auto& edge : edges) {
    const auto predIt = indexOf.find(edge.first);
    const auto succIt = indexOf.find(edge.second);
    if (predIt == indexOf.end() || succIt == indexOf.end()) {
      continue;
    }
    const int predIdx = predIt->second;
    const int succIdx = succIt->second;
    if (predIdx == succIdx) {
      continue;
    }
    const uint64_t packed =
        (static_cast<uint64_t>(predIdx) << 32) |
        static_cast<uint32_t>(succIdx);
    if (!dedupEdges.insert(packed).second) {
      continue;
    }
    succ[predIdx].push_back(succIdx);
    indegree[succIdx]++;
  }

  using RankedNode = std::pair<int, int>;  // (input-rank, node-index)
  std::priority_queue<RankedNode, vector<RankedNode>,
                      std::greater<RankedNode>>
      frontier;
  for (int i = 0; i < (int)uniqueTasks.size(); i++) {
    if (indegree[i] == 0) {
      frontier.emplace(i, i);
    }
  }

  vector<int> ordered;
  ordered.reserve(uniqueTasks.size());
  vector<char> emitted(uniqueTasks.size(), 0);
  while (!frontier.empty()) {
    const auto [_, nodeIdx] = frontier.top();
    frontier.pop();
    if (emitted[nodeIdx]) {
      continue;
    }
    emitted[nodeIdx] = 1;
    ordered.push_back(uniqueTasks[nodeIdx]);
    for (int nextIdx : succ[nodeIdx]) {
      indegree[nextIdx]--;
      if (indegree[nextIdx] == 0) {
        frontier.emplace(nextIdx, nextIdx);
      }
    }
  }

  if ((int)ordered.size() < (int)uniqueTasks.size()) {
    // Fallback in case of cycles/dirty state: preserve deterministic coverage.
    for (int i = 0; i < (int)uniqueTasks.size(); i++) {
      if (!emitted[i]) {
        ordered.push_back(uniqueTasks[i]);
      }
    }
  }
  return ordered;
}
}  // namespace

bool LNS::computeRegret() {
  if (runtimeBudgetExhausted()) {
    return false;
  }
  regretEvalStatsCurrent_.recomputeCalls++;
  regretEvalStatsTotal_.recomputeCalls++;
  lnsNeighborhood_.regretMaxHeap.clear();
  buildFullPrecedenceConstraints(fullPrecedenceConstraintsScratch_);
  const auto& fullPrecedenceConstraints = fullPrecedenceConstraintsScratch_;
  const vector<int> orderedTasks =
      orderTasksTopologically(collectRemainingRemovedTasks(),
                              fullPrecedenceConstraints);
  for (int task : orderedTasks) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    bool enoughSpace = computeRegretForTask(task, fullPrecedenceConstraints);
    if (!enoughSpace) {
      return false;
    }
  }
  return true;
}

bool LNS::recomputeRegretsForTasks(const vector<int>& tasks) {
  if (runtimeBudgetExhausted()) {
    return false;
  }
  regretEvalStatsCurrent_.recomputeCalls++;
  regretEvalStatsTotal_.recomputeCalls++;
  incrementalRegretStatsCurrent_.recomputeCalls++;
  incrementalRegretStatsTotal_.recomputeCalls++;
  buildFullPrecedenceConstraints(fullPrecedenceConstraintsScratch_);
  const auto& fullPrecedenceConstraints = fullPrecedenceConstraintsScratch_;
  const vector<int> orderedTasks =
      orderTasksTopologically(tasks, fullPrecedenceConstraints);
  for (int task : orderedTasks) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    if (!isPendingCommitState(lnsNeighborhood_, task)) {
      continue;
    }
    regretStamp_[task]++;
    incrementalRegretStatsCurrent_.recomputedTasks++;
    incrementalRegretStatsTotal_.recomputedTasks++;
    const bool enoughSpace =
        computeRegretForTask(task, fullPrecedenceConstraints);
    if (!enoughSpace) {
      return false;
    }
  }
  return true;
}

bool LNS::computeRegretForTask(int task) {
  buildFullPrecedenceConstraints(fullPrecedenceConstraintsScratch_);
  const auto& fullPrecedenceConstraints = fullPrecedenceConstraintsScratch_;
  return computeRegretForTask(task, fullPrecedenceConstraints);
}

bool LNS::computeRegretForTask(
    int task, const vector<pair<int, int>>& fullPrecedenceConstraints) {
  if (runtimeBudgetExhausted()) {
    return false;
  }
  regretEvalStatsCurrent_.tasksEvaluated++;
  regretEvalStatsTotal_.tasksEvaluated++;
  pairing_heap<Utility, compare<Utility::CompareUtilities>> serviceTimes;

  // The task has to start after the earliest time step but needs to finish
  // before the latest time step. However we cannot give any guarantee on the
  // latest timestep so we only work with the earliest timestep.
  int earliestTimestep = 0;

  RegretWorkspace workspace(solution_);
  struct WorkspaceCloneStatsScope {
    RegretEvalStats& current;
    RegretEvalStats& total;
    const RegretWorkspace& workspace;
    ~WorkspaceCloneStatsScope() {
      const int64_t cloned = workspace.clonedAgents();
      current.workspaceAgentsCloned += cloned;
      total.workspaceAgentsCloned += cloned;
      current.workspaceMaxClonedPerTask =
          max(current.workspaceMaxClonedPerTask, cloned);
      total.workspaceMaxClonedPerTask =
          max(total.workspaceMaxClonedPerTask, cloned);
    }
  } workspaceCloneStats{regretEvalStatsCurrent_, regretEvalStatsTotal_, workspace};
  vector<char> workspaceTouchedAgents(instance_.getAgentNum(), 0);
  if (!injectPendingAncestors(workspace, task, fullPrecedenceConstraints,
                              workspaceTouchedAgents, nullptr)) {
    return false;
  }

  vector<pair<int, int>> precedenceConstraints;
  vector<int> assignmentOwnerLookup(instance_.getTasksNum(), UNASSIGNED);
  vector<int> assignmentPosLookup(instance_.getTasksNum(), UNASSIGNED);
  if (!prepareRegretWorkspaceForTask(task, workspace, workspaceTouchedAgents,
                                     precedenceConstraints,
                                     assignmentOwnerLookup,
                                     assignmentPosLookup, earliestTimestep)) {
    return false;
  }

  TaskBaselineMetrics baselineMetrics;
  if (market_.repairTieBreak || market_.repairBlend) {
    baselineMetrics.oldExposure = computeTaskMarketExposure(task, true);
    baselineMetrics.oldWait = computeTaskPrecedenceWaitInCurrentSolution(task);
    baselineMetrics.valid = true;
  }

  vector<int> candidateAgents;
  candidateAgents.reserve(instance_.getAgentNum());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    evaluateAgentPositionCandidate(task, agent, earliestTimestep, workspace,
                                   &precedenceConstraints, baselineMetrics,
                                   &serviceTimes, &candidateAgents,
                                   &assignmentOwnerLookup,
                                   &assignmentPosLookup);
  }
  if (task >= 0 && task < (int)regretCandidateAgents_.size()) {
    regretCandidateAgents_[task] = std::move(candidateAgents);
  }
  return buildRegretEntry(task, serviceTimes);
}
