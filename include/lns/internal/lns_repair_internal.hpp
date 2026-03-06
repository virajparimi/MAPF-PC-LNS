#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <sstream>

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

struct ConstraintTableDigest {
  uint64_t hash = 1469598103934665603ULL;
  int vertexBuckets = 0;
  int edgeBuckets = 0;
  int intervalCount = 0;
};

[[maybe_unused]] inline void mixConstraintDigest(uint64_t& h, uint64_t value) {
  // FNV-1a style mixing for compact debug signatures.
  h ^= value;
  h *= 1099511628211ULL;
}

[[maybe_unused]] ConstraintTableDigest computeConstraintTableDigest(
    const ConstraintTable& constraintTable, const Instance& instance) {
  ConstraintTableDigest digest;
  mixConstraintDigest(digest.hash,
                      (uint64_t)(constraintTable.goalLocation + 3));
  mixConstraintDigest(digest.hash, (uint64_t)(constraintTable.lengthMin + 5));
  mixConstraintDigest(digest.hash, (uint64_t)(constraintTable.lengthMax + 7));
  mixConstraintDigest(digest.hash,
                      (uint64_t)(constraintTable.latestTimestep + 11));
  mixConstraintDigest(digest.hash,
                      (uint64_t)(constraintTable.temporalExtent + 13));

  for (int loc = 0; loc < instance.mapSize; loc++) {
    const auto* vertexIntervals = constraintTable.getConstraintIntervals(loc);
    if (vertexIntervals != nullptr && !vertexIntervals->empty()) {
      digest.vertexBuckets++;
      digest.intervalCount += (int)vertexIntervals->size();
      mixConstraintDigest(digest.hash, (uint64_t)(loc + 17));
      mixConstraintDigest(digest.hash, (uint64_t)(vertexIntervals->size() + 19));
      for (const auto& [tMin, tMax] : *vertexIntervals) {
        mixConstraintDigest(digest.hash, (uint64_t)(tMin + 23));
        mixConstraintDigest(digest.hash, (uint64_t)(tMax + 29));
      }
    }

    const auto& neighbors = instance.getNeighbors(loc);
    for (int nxt : neighbors) {
      const auto* edgeIntervals =
          constraintTable.getEdgeConstraintIntervals(loc, nxt);
      if (edgeIntervals != nullptr && !edgeIntervals->empty()) {
        digest.edgeBuckets++;
        digest.intervalCount += (int)edgeIntervals->size();
        mixConstraintDigest(digest.hash, (uint64_t)(loc + 31));
        mixConstraintDigest(digest.hash, (uint64_t)(nxt + 37));
        mixConstraintDigest(digest.hash, (uint64_t)(edgeIntervals->size() + 41));
        for (const auto& [tMin, tMax] : *edgeIntervals) {
          mixConstraintDigest(digest.hash, (uint64_t)(tMin + 43));
          mixConstraintDigest(digest.hash, (uint64_t)(tMax + 47));
        }
      }
    }
  }
  return digest;
}

[[maybe_unused]] bool shouldTraceConstraintDebugTriple(int task, int agent,
                                                       int nextTask) {
  // Enable with:
  //   LNS_DEBUG_TRIPLE="<task>,<agent>,<nextTask>"
  // Example:
  //   LNS_DEBUG_TRIPLE="25,20,124"
  const char* spec = std::getenv("LNS_DEBUG_TRIPLE");
  if (spec == nullptr || *spec == '\0') {
    return false;
  }
  int tracedTask = std::numeric_limits<int>::min();
  int tracedAgent = std::numeric_limits<int>::min();
  int tracedNext = std::numeric_limits<int>::min();
  if (std::sscanf(spec, "%d,%d,%d", &tracedTask, &tracedAgent, &tracedNext) !=
      3) {
    return false;
  }
  return tracedTask == task && tracedAgent == agent && tracedNext == nextTask;
}

[[maybe_unused]] bool shouldTraceConstraintDebugTask(int task) {
  // Enable with:
  //   LNS_DEBUG_CT_TASK="<task>"
  // Example:
  //   LNS_DEBUG_CT_TASK="124"
  static const int tracedTask = []() {
    // Keep legacy alias support for older debug scripts.
    const char* spec = std::getenv("LNS_DEBUG_CT_TASK");
    if (spec == nullptr || *spec == '\0') {
      spec = std::getenv("MAPF_PC_DEBUG_TASK");
    }
    if (spec == nullptr || *spec == '\0') {
      return std::numeric_limits<int>::min();
    }

    char* end = nullptr;
    const long parsed = std::strtol(spec, &end, 10);
    if (end == spec || *end != '\0') {
      return std::numeric_limits<int>::min();
    }
    if (parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      return std::numeric_limits<int>::min();
    }
    return static_cast<int>(parsed);
  }();
  return tracedTask == task;
}

[[maybe_unused]] string summarizeTaskQueue(const vector<int>& assignments,
                                           int maxItems = 20) {
  std::ostringstream oss;
  const int count = (int)assignments.size();
  oss << "[";
  for (int i = 0; i < count && i < maxItems; i++) {
    if (i > 0) {
      oss << ",";
    }
    oss << assignments[i];
  }
  if (count > maxItems) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

[[maybe_unused]] string summarizeIntList(const vector<int>& values,
                                         int maxItems = 40) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < (int)values.size() && i < maxItems; i++) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  if ((int)values.size() > maxItems) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

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

[[maybe_unused]] AssignmentLookup buildAssignmentLookup(
    const Solution& solution, int taskCount) {
  AssignmentLookup lookup;
  lookup.owner.assign(taskCount, UNASSIGNED);
  lookup.pos.assign(taskCount, -1);
  for (int agent = 0; agent < (int)solution.agents.size(); agent++) {
    const auto& assignments = solution.agents[agent].taskAssignments;
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
    const vector<int>& workspaceOwnerLookup,
    const vector<int>& workspacePosLookup, const Solution& previousSolution,
    const Neighbor& neighborhood, bool* usedPreviousFallback,
    const vector<int>* previousOwnerLookup = nullptr,
    const vector<int>* previousPosLookup = nullptr) {
  if (usedPreviousFallback != nullptr) {
    *usedPreviousFallback = false;
  }
  if (task < 0 || task >= (int)workspaceOwnerLookup.size() ||
      task >= (int)workspacePosLookup.size()) {
    return -1;
  }

  const int workspaceOwner = workspaceOwnerLookup[task];
  const int workspacePos = workspacePosLookup[task];
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

  int previousOwner = UNASSIGNED;
  if (previousOwnerLookup != nullptr &&
      task < (int)previousOwnerLookup->size()) {
    previousOwner = (*previousOwnerLookup)[task];
  }
  if (previousOwner == UNASSIGNED) {
    previousOwner = (task >= 0 && task < (int)previousSolution.taskAgentMap.size())
                        ? previousSolution.taskAgentMap[task]
                        : UNASSIGNED;
  }
  if (previousOwner == UNASSIGNED || previousOwner < 0 ||
      previousOwner >= (int)previousSolution.agents.size()) {
    return -1;
  }

  int previousPos = UNASSIGNED;
  if (previousPosLookup != nullptr &&
      task < (int)previousPosLookup->size()) {
    const int candidatePos = (*previousPosLookup)[task];
    const bool ownerMatches =
        (previousOwnerLookup == nullptr || task >= (int)previousOwnerLookup->size() ||
         (*previousOwnerLookup)[task] == previousOwner);
    if (ownerMatches) {
      previousPos = candidatePos;
    }
  }
  if (previousPos == UNASSIGNED) {
    previousPos = previousSolution.getLocalTaskIndex(previousOwner, task);
  }
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
