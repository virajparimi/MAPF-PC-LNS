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

vector<int> LNS::collectRemainingRemovedTasks() const {
  vector<int> tasks;
  tasks.reserve(lnsNeighborhood_.removedTasks.size());
  for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
    tasks.push_back(conflict.task);
  }
  return tasks;
}

vector<int> LNS::computeCurrentTaskEndTimes() const {
  vector<int> endTimes(instance_.getTasksNum(), -1);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int pos = 0; pos < (int)solution_.agents[agent].taskAssignments.size();
         pos++) {
      const int task = solution_.agents[agent].taskAssignments[pos];
      if (pos >= (int)solution_.agents[agent].taskPaths.size()) {
        continue;
      }
      if (solution_.agents[agent].taskPaths[pos].empty()) {
        continue;
      }
      endTimes[task] = solution_.agents[agent].taskPaths[pos].endTime();
    }
  }
  return endTimes;
}

vector<int> LNS::computeCurrentLastTaskPerAgent() const {
  vector<int> lastTasks(instance_.getAgentNum(), UNASSIGNED);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      lastTasks[agent] = solution_.agents[agent].taskAssignments.back();
    }
  }
  return lastTasks;
}

vector<uint64_t> LNS::computeCurrentAgentScheduleSignatures() const {
  vector<uint64_t> signatures(instance_.getAgentNum(), 0);
  auto mixHash = [](uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
  };

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    uint64_t sig = 1469598103934665603ULL;
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& taskPaths = solution_.agents[agent].taskPaths;
    sig = mixHash(sig, (uint64_t)assignments.size());
    for (int task : assignments) {
      sig = mixHash(sig, (uint64_t)(task + 3));
    }
    sig = mixHash(sig, (uint64_t)taskPaths.size());
    for (const auto& segment : taskPaths) {
      sig = mixHash(sig, (uint64_t)(segment.beginTime + 17));
      sig = mixHash(sig, (uint64_t)segment.size());
      for (const auto& entry : segment.path) {
        const uint64_t packed =
            ((uint64_t)(entry.location + 2) << 1) | (entry.isGoal ? 1ULL : 0ULL);
        sig = mixHash(sig, packed);
      }
      sig = mixHash(sig, (uint64_t)segment.timeStamps.size());
      for (int ts : segment.timeStamps) {
        sig = mixHash(sig, (uint64_t)(ts + 11));
      }
    }
    signatures[agent] = sig;
  }
  return signatures;
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

std::optional<Regret> LNS::popNextValidRegret() {
  while (!lnsNeighborhood_.regretMaxHeap.empty()) {
    Regret r = lnsNeighborhood_.regretMaxHeap.top();
    lnsNeighborhood_.regretMaxHeap.pop();

    if (!isPendingCommitState(lnsNeighborhood_, r.task)) {
      incrementalRegretStatsCurrent_.stalePops++;
      incrementalRegretStatsTotal_.stalePops++;
      continue;
    }
    if (!incrementalRegret_ || r.stamp == regretStamp_[r.task]) {
      return r;
    }
    incrementalRegretStatsCurrent_.stalePops++;
    incrementalRegretStatsTotal_.stalePops++;
  }
  return std::nullopt;
}

vector<int> LNS::computeDirtyTasksAfterCommit(
    const vector<int>& endTimesBefore, const vector<int>& endTimesAfter,
    const vector<uint64_t>& agentSignaturesBefore,
    const vector<uint64_t>& agentSignaturesAfter) {
  vector<int> changedTasks;
  changedTasks.reserve(instance_.getTasksNum());
  for (int task = 0; task < instance_.getTasksNum(); task++) {
    if (endTimesAfter[task] < 0) {
      continue;
    }
    if (endTimesAfter[task] != endTimesBefore[task]) {
      changedTasks.push_back(task);
    }
  }
  incrementalRegretStatsCurrent_.changedSum += (int64_t)changedTasks.size();
  incrementalRegretStatsTotal_.changedSum += (int64_t)changedTasks.size();
  incrementalRegretStatsCurrent_.changedMax =
      max(incrementalRegretStatsCurrent_.changedMax,
          (int64_t)changedTasks.size());
  incrementalRegretStatsTotal_.changedMax =
      max(incrementalRegretStatsTotal_.changedMax, (int64_t)changedTasks.size());

  vector<char> changedAgents(instance_.getAgentNum(), 0);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (agent >= (int)agentSignaturesBefore.size() ||
        agent >= (int)agentSignaturesAfter.size()) {
      changedAgents[agent] = 1;
      continue;
    }
    if (agentSignaturesBefore[agent] != agentSignaturesAfter[agent]) {
      changedAgents[agent] = 1;
    }
  }
  int64_t changedAgentCount = 0;
  for (char v : changedAgents) {
    if (v) {
      changedAgentCount++;
    }
  }
  incrementalRegretStatsCurrent_.changedAgentsSum += changedAgentCount;
  incrementalRegretStatsTotal_.changedAgentsSum += changedAgentCount;
  incrementalRegretStatsCurrent_.changedAgentsMax =
      max(incrementalRegretStatsCurrent_.changedAgentsMax, changedAgentCount);
  incrementalRegretStatsTotal_.changedAgentsMax =
      max(incrementalRegretStatsTotal_.changedAgentsMax, changedAgentCount);

  vector<char> isDirty(instance_.getTasksNum(), 0);
  vector<char> dirtyFromDescendants(instance_.getTasksNum(), 0);
  vector<char> dirtyFromCandidateAgents(instance_.getTasksNum(), 0);
  vector<char> dirtyFromAncestors(instance_.getTasksNum(), 0);

  const auto& successors = instance_.getSuccessorsRef();
  std::vector<int> stack;
  stack.reserve(changedTasks.size());
  for (int t : changedTasks) {
    stack.push_back(t);
  }
  while (!stack.empty()) {
    const int current = stack.back();
    stack.pop_back();
    if (current < 0 || current >= instance_.getTasksNum()) {
      continue;
    }
    if (dirtyFromDescendants[current]) {
      continue;
    }
    dirtyFromDescendants[current] = 1;
    isDirty[current] = 1;
    for (int succ : successors[current]) {
      if (!dirtyFromDescendants[succ]) {
        stack.push_back(succ);
      }
    }
  }

  if (incrementalRegretMode_ == IncrementalRegretMode::descendants_and_agent) {
    for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
      const int task = conflict.task;
      if (task < 0 || task >= instance_.getTasksNum()) {
        continue;
      }
      bool touchesChangedAgent = false;
      if (task >= 0 && task < (int)regretCandidateAgents_.size()) {
        for (int candidateAgent : regretCandidateAgents_[task]) {
          if (candidateAgent >= 0 && candidateAgent < instance_.getAgentNum() &&
              changedAgents[candidateAgent]) {
            touchesChangedAgent = true;
            break;
          }
        }
      }
      if (!touchesChangedAgent) {
        const int bestAgent =
            (task >= 0 && task < (int)regretBestOption_.size())
                ? regretBestOption_[task].first
                : UNASSIGNED;
        const int secondAgent =
            (task >= 0 && task < (int)regretSecondBestOption_.size())
                ? regretSecondBestOption_[task].first
                : UNASSIGNED;
        if ((bestAgent != UNASSIGNED && bestAgent < instance_.getAgentNum() &&
             changedAgents[bestAgent]) ||
            (secondAgent != UNASSIGNED &&
             secondAgent < instance_.getAgentNum() &&
             changedAgents[secondAgent])) {
          touchesChangedAgent = true;
        }
      }
      if (touchesChangedAgent) {
        dirtyFromCandidateAgents[task] = 1;
        isDirty[task] = 1;
      }
    }
  }

  const bool precedencePressureActive = (repairHeuristic == "regret");
  if (precedencePressureActive) {
    const auto& ancestors = instance_.getAncestorsRef();
    stack.clear();
    for (int t : changedTasks) {
      stack.push_back(t);
    }
    while (!stack.empty()) {
      const int current = stack.back();
      stack.pop_back();
      if (current < 0 || current >= instance_.getTasksNum()) {
        continue;
      }
      if (dirtyFromAncestors[current]) {
        continue;
      }
      dirtyFromAncestors[current] = 1;
      isDirty[current] = 1;
      for (int pred : ancestors[current]) {
        if (!dirtyFromAncestors[pred]) {
          stack.push_back(pred);
        }
      }
    }
  }

  vector<int> dirtyTasks;
  dirtyTasks.reserve(lnsNeighborhood_.removedTasks.size());
  int64_t descendantHits = 0;
  int64_t candidateAgentHits = 0;
  int64_t ancestorHits = 0;
  for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
    const int task = conflict.task;
    if (task < 0 || task >= instance_.getTasksNum()) {
      continue;
    }
    if (dirtyFromDescendants[task]) {
      descendantHits++;
    }
    if (dirtyFromCandidateAgents[task]) {
      candidateAgentHits++;
    }
    if (dirtyFromAncestors[task]) {
      ancestorHits++;
    }
    if (isDirty[task]) {
      dirtyTasks.push_back(task);
    }
  }
  incrementalRegretStatsCurrent_.dirtySum += (int64_t)dirtyTasks.size();
  incrementalRegretStatsTotal_.dirtySum += (int64_t)dirtyTasks.size();
  incrementalRegretStatsCurrent_.dirtyMax =
      max(incrementalRegretStatsCurrent_.dirtyMax, (int64_t)dirtyTasks.size());
  incrementalRegretStatsTotal_.dirtyMax =
      max(incrementalRegretStatsTotal_.dirtyMax, (int64_t)dirtyTasks.size());
  incrementalRegretStatsCurrent_.dirtyByDescendants += descendantHits;
  incrementalRegretStatsTotal_.dirtyByDescendants += descendantHits;
  incrementalRegretStatsCurrent_.dirtyByCandidateAgent += candidateAgentHits;
  incrementalRegretStatsTotal_.dirtyByCandidateAgent += candidateAgentHits;
  incrementalRegretStatsCurrent_.dirtyByAncestors += ancestorHits;
  incrementalRegretStatsTotal_.dirtyByAncestors += ancestorHits;
  return dirtyTasks;
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

  // The task has to start after the earliest time step but needs to finish before the latest time step. However we cannot give any guarantee on the latest timestep so we only work with the earliest timestep
  int earliestTimestep = 0;

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (const auto& precConstraint : fullPrecedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }
  vector<char> ancestorsOfTask = reachableSet(task, ancestors);
  if (task >= 0 && task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[task] = 0;
  }

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

  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }
    if (isPendingCommitState(lnsNeighborhood_, ancestorTask)) {
      // This task's path will not exist currently!
      const int ancestorTaskAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (ancestorTaskAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in previous solution\n";
        return false;
      }
      assert(ancestorTaskAgent != UNASSIGNED);
      int ancestorTaskLocalIndex =
          previousSolution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
      if (ancestorTaskLocalIndex == UNASSIGNED ||
          ancestorTaskLocalIndex >=
              (int)previousSolution_.agents[ancestorTaskAgent]
                  .taskPaths.size()) {
        PLOGE << "computeRegretForTask: invalid local index for ancestor task "
              << ancestorTask << " on agent " << ancestorTaskAgent << "\n";
        return false;
      }
      int ancestorTaskLocalIndexRelativeToSolution = extractOldLocalTaskIndex(
          ancestorTask,
          previousSolution_.agents[ancestorTaskAgent].taskAssignments,
          workspace.assignments(ancestorTaskAgent));
      if (ancestorTaskLocalIndexRelativeToSolution == UNASSIGNED) {
        PLOGE << "computeRegretForTask: failed to map ancestor task "
              << ancestorTask << " into current assignment order\n";
        return false;
      }
      workspace.mutableAssignments(ancestorTaskAgent).insert(
          workspace.mutableAssignments(ancestorTaskAgent).begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          ancestorTask);
      workspace.mutableTaskPaths(ancestorTaskAgent).insert(
          workspace.mutableTaskPaths(ancestorTaskAgent).begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          previousSolution_.agents[ancestorTaskAgent]
              .taskPaths[ancestorTaskLocalIndex]);
      workspaceTouchedAgents[ancestorTaskAgent] = 1;
    }
  }

  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.assign(inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());
  size_t intraPrecedenceCount = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignmentsView = workspace.assignments(agent);
    if (assignmentsView.size() > 1) {
      intraPrecedenceCount += (assignmentsView.size() - 1);
    }
  }
  precedenceConstraints.reserve(precedenceConstraints.size() +
                                intraPrecedenceCount);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignmentsView = workspace.assignments(agent);
    for (int localTask = 1; localTask < (int)assignmentsView.size();
         localTask++) {
      precedenceConstraints.emplace_back(assignmentsView[localTask - 1],
                                         assignmentsView[localTask]);
    }
  }
  const AssignmentLookup assignmentLookup =
      buildAssignmentLookup(workspace, instance_.getTasksNum());

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    const auto& assignmentsView = workspace.assignments(agent);

    bool normalizeAgent = workspaceTouchedAgents[agent] != 0;
    if (!normalizeAgent) {
      const auto& pathsView = workspace.taskPaths(agent);
      if (pathsView.size() != assignmentsView.size()) {
        normalizeAgent = true;
      } else {
        for (int localTask = 0; localTask < (int)assignmentsView.size();
             localTask++) {
          const bool broken = pathsView[localTask].empty() ||
                              (localTask == 0 &&
                               pathsView[localTask].front().location !=
                                   instance_.getStartLocationsRef()[agent]) ||
                              (localTask > 0 &&
                               (pathsView[localTask - 1].empty() ||
                                pathsView[localTask - 1].path.back().location !=
                                    pathsView[localTask].path.front().location));
          if (broken) {
            normalizeAgent = true;
            break;
          }
          const int expectedBegin =
              (localTask == 0) ? 0 : pathsView[localTask - 1].endTime();
          if (pathsView[localTask].beginTime != expectedBegin) {
            normalizeAgent = true;
            break;
          }
        }
      }
    }

    if (!normalizeAgent) {
      continue;
    }

    auto& assignments = workspace.mutableAssignments(agent);
    auto& paths = workspace.mutableTaskPaths(agent);
    vector<int> goalLocations = instance_.getTaskLocations(assignments);
    auto localPlanner = createLocalPlanner(agent);
    localPlanner->setGoalLocations(goalLocations);

    for (int localTask = 0; localTask < (int)assignments.size(); localTask++) {
      if (localTask > 0) {
        paths[localTask].beginTime = paths[localTask - 1].endTime();
      } else {
        paths[localTask].beginTime = 0;
      }

      const bool broken = paths[localTask].empty() ||
                          (localTask == 0 &&
                           paths[localTask].front().location !=
                               instance_.getStartLocationsRef()[agent]) ||
                          (localTask > 0 &&
                           (paths[localTask - 1].empty() ||
                            paths[localTask - 1].path.back().location !=
                                paths[localTask].path.front().location));
      if (!broken) {
        continue;
      }

      int startTime = 0;
      if (localTask > 0) {
        startTime = paths[localTask - 1].endTime();
      }
      ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
      TaskRegretPacket taskPacket = {assignments[localTask], agent, localTask,
                                     -1};
      if (!buildConstraintTable(constraintTable, taskPacket,
                                goalLocations[localTask], workspace,
                                &precedenceConstraints, false,
                                &assignmentLookup.owner, &assignmentLookup.pos)) {
        PLOGE << "computeRegretForTask: failed to build constraint table for "
              << "agent " << agent << ", task " << assignments[localTask]
              << " at position " << localTask << "\n";
        return false;
      }
      AgentTaskPath path = runLowLevelSearch(*localPlanner, constraintTable,
                                             startTime, localTask, 0);
      // We must be able to find the path for the next task. If not then we
      // cannot move forward!
      if (path.empty()) {
        PLOGE << "computeRegretForTask: empty path for agent " << agent
              << ", task " << assignments[localTask] << " at position "
              << localTask << "\n";
        return false;
      }
      paths[localTask] = std::move(path);
    }

  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = workspace.assignments(agent);
    const auto& paths = workspace.taskPaths(agent);
    for (int localTask = 0; localTask < (int)assignments.size();
         localTask++) {
      if (localTask > 0) {
        assert(paths[localTask - 1].path.back().location ==
               paths[localTask].path.front().location);
      } else {
        assert(paths[localTask].front().location ==
               instance_.getStartLocationsRef()[agent]);
      }
    }
  }

  ancestors.clear();
  ancestors.resize(instance_.getTasksNum());
  for (const auto& precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }
  ancestorsOfTask = reachableSet(task, ancestors);
  if (task >= 0 && task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[task] = 0;
  }

  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }
    int ancestorTaskAgent = UNDEFINED;
    if (isPendingCommitState(lnsNeighborhood_, ancestorTask)) {
      const int prevAssignedAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (prevAssignedAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in previous solution\n";
        return false;
      }
      ancestorTaskAgent = prevAssignedAgent;
    } else {
      const int curAssignedAgent =
          (ancestorTask >= 0 && ancestorTask < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (curAssignedAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in current solution\n";
        return false;
      }
      ancestorTaskAgent = curAssignedAgent;
    }
    if (ancestorTaskAgent < 0 || ancestorTaskAgent >= instance_.getAgentNum()) {
      PLOGE << "Invalid ancestor agent " << ancestorTaskAgent
            << " for task " << ancestorTask << "\n";
      return false;
    }
    assert(ancestorTaskAgent != UNASSIGNED);
    if (ancestorTask < 0 || ancestorTask >= instance_.getTasksNum()) {
      continue;
    }
    const int lookupAgent = assignmentLookup.owner[ancestorTask];
    const int ancestorTaskPosition = assignmentLookup.pos[ancestorTask];
    if (lookupAgent != ancestorTaskAgent || ancestorTaskPosition == UNASSIGNED) {
      PLOGE << "Ancestor task " << ancestorTask
            << " missing from temporary assignment for agent "
            << ancestorTaskAgent << "\n";
      return false;
    }
    if (ancestorTaskPosition >=
            (int)workspace.taskPaths(ancestorTaskAgent).size() ||
        workspace.taskPaths(ancestorTaskAgent)[ancestorTaskPosition].empty()) {
      PLOGE << "Ancestor path missing for task " << ancestorTask
            << " at position " << ancestorTaskPosition << "\n";
      return false;
    }
    earliestTimestep =
        max(earliestTimestep,
            workspace.taskPaths(ancestorTaskAgent)[ancestorTaskPosition]
                    .endTimeChecked() +
                1);
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

    TaskRegretPacket regretPacket = {task, agent, -1, earliestTimestep};
    const auto beforeOptions = serviceTimes.size();
    computeRegretForTaskWithAgent(regretPacket, workspace,
                                  &precedenceConstraints,
                                  baselineMetrics,
                                  &serviceTimes, &assignmentLookup.owner,
                                  &assignmentLookup.pos);
    if (serviceTimes.size() > beforeOptions) {
      candidateAgents.push_back(agent);
    }
  }
  if (task >= 0 && task < (int)regretCandidateAgents_.size()) {
    regretCandidateAgents_[task] = std::move(candidateAgents);
  }

  if (serviceTimes.empty()) {
    // No feasible insertion exists for this task under the current temporary
    // neighborhood state.
    PLOGD << "No service time options for task " << task
          << " inside computeRegretForTask\n";
    return false;
  }
  Utility bestUtility = serviceTimes.top();
  serviceTimes.pop();
  if (serviceTimes.empty()) {
    // Exactly one feasible insertion exists; treat it as forced and prioritize
    // it ahead of ambiguous tasks.
    regretBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
    regretSecondBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
    Regret forced(task, bestUtility.agent, bestUtility.taskPosition,
                  bestUtility.pathLength, bestUtility.agentTasksLen, 0,
                  std::numeric_limits<double>::infinity(), regretStamp_[task]);
    lnsNeighborhood_.regretMaxHeap.push(forced);
    return true;
  }
  Utility secondBestUtility = serviceTimes.top();

  regretBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
  regretSecondBestOption_[task] = {secondBestUtility.agent,
                                   secondBestUtility.taskPosition};

  double value = 0;
  if (regretType == "absolute") {
    value = secondBestUtility.value - bestUtility.value;
  } else {
    value = (secondBestUtility.value + 1) / (bestUtility.value + 1);
  }
  Regret regret(task, bestUtility.agent, bestUtility.taskPosition,
                bestUtility.pathLength, bestUtility.agentTasksLen,
                (int)serviceTimes.size(), value, regretStamp_[task]);
  lnsNeighborhood_.regretMaxHeap.push(regret);
  return true;
}

int LNS::computeTaskPrecedenceWaitFromWorkspace(
    int task, int taskLocation, const RegretWorkspace& workspace,
    const vector<int>* assignmentOwnerLookup,
    const vector<int>* assignmentPosLookup) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }

  const int taskCount = instance_.getTasksNum();
  AssignmentLookup builtLookup;
  const vector<int>* ownerLookup = assignmentOwnerLookup;
  const vector<int>* posLookup = assignmentPosLookup;
  if (ownerLookup == nullptr || posLookup == nullptr ||
      (int)ownerLookup->size() != taskCount || (int)posLookup->size() != taskCount) {
    builtLookup = buildAssignmentLookup(workspace, taskCount);
    ownerLookup = &builtLookup.owner;
    posLookup = &builtLookup.pos;
  }
  const int taskAgent =
      (task >= 0 && task < (int)ownerLookup->size()) ? (*ownerLookup)[task]
                                                      : UNASSIGNED;
  const int taskPos =
      (task >= 0 && task < (int)posLookup->size()) ? (*posLookup)[task] : -1;
  if (taskAgent == UNASSIGNED || taskPos < 0 ||
      taskPos >= (int)workspace.taskPaths(taskAgent).size()) {
    return 0;
  }

  const AgentTaskPath& taskPath = workspace.taskPaths(taskAgent)[taskPos];
  if (taskPath.empty()) {
    return 0;
  }

  int arrive = taskPath.endTime();
  for (int i = 0; i < (int)taskPath.size(); i++) {
    if (taskPath[i].location == taskLocation) {
      arrive = taskPath.beginTime + i;
      break;
    }
  }

  int release = 0;
  vector<char> seenPredecessor(taskCount, 0);
  auto consumePredecessor = [&](int pred) {
    if (pred < 0 || pred >= taskCount || seenPredecessor[pred]) {
      return;
    }
    seenPredecessor[pred] = 1;
    const int predAgent = (pred >= 0 && pred < (int)ownerLookup->size())
                              ? (*ownerLookup)[pred]
                              : UNASSIGNED;
    const int predPos = (pred >= 0 && pred < (int)posLookup->size())
                            ? (*posLookup)[pred]
                            : -1;
    if (predAgent != UNASSIGNED && predPos >= 0 &&
        predPos < (int)workspace.taskPaths(predAgent).size() &&
        !workspace.taskPaths(predAgent)[predPos].empty()) {
      release = max(release, workspace.taskPaths(predAgent)[predPos].endTime());
    }
  };

  const auto& baseAncestors = instance_.getAncestorsRef();
  if (task >= 0 && task < (int)baseAncestors.size()) {
    for (int pred : baseAncestors[task]) {
      consumePredecessor(pred);
    }
  }

  if (taskPos > 0 && taskAgent >= 0 && taskAgent < workspace.numAgents() &&
      taskPos - 1 < (int)workspace.assignments(taskAgent).size()) {
    consumePredecessor(workspace.assignments(taskAgent)[taskPos - 1]);
  }

  return max(0, release - arrive);
}
