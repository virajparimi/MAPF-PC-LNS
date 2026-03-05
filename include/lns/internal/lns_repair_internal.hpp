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

ConstraintTableDigest computeConstraintTableDigest(
    const ConstraintTable& constraintTable, const Instance& instance);

bool shouldTraceConstraintDebugTriple(int task, int agent, int nextTask);

bool shouldTraceConstraintDebugTask(int task);

string summarizeTaskQueue(const vector<int>& assignments, int maxItems = 20);

string summarizeIntList(const vector<int>& values, int maxItems = 40);

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

  void rollback(LNS::RegretWorkspace& workspace);
};

struct ScopedInsertTaskRollback {
  InsertTaskRollbackLog* log = nullptr;
  LNS::RegretWorkspace* workspace = nullptr;
  bool enabled = false;

  ~ScopedInsertTaskRollback();
};

AssignmentLookup buildAssignmentLookup(const LNS::RegretWorkspace& workspace,
                                       int taskCount);

AssignmentLookup buildAssignmentLookup(const Solution& solution,
                                       int taskCount);

bool isPendingCommitState(const Neighbor& neighborhood, int task);

int resolveTaskEndTimeFromMixedState(
    int task, const LNS::RegretWorkspace& workspace,
    const vector<int>& workspaceOwnerLookup,
    const vector<int>& workspacePosLookup, const Solution& previousSolution,
    const Neighbor& neighborhood, bool* usedPreviousFallback,
    const vector<int>* previousOwnerLookup = nullptr,
    const vector<int>* previousPosLookup = nullptr);

void normalizeSeriesByRobustScale(const vector<double>& rawValues,
                                  const vector<int>& validIndices,
                                  vector<double>& normalizedValues);
