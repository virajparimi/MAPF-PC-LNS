#include "lns.hpp"
#include "lns_repair_internal.hpp"

#include <algorithm>
#include <cstdint>

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
