#pragma once

#include <queue>
#include <sstream>

#include <utility>
#include "common.hpp"
#include "utils.hpp"

namespace {
struct TaskAssignmentIndex {
  vector<int> owner;
  vector<int> pos;
};

[[maybe_unused]] int clampSocToInt(long long soc, const char* context) {
  if (soc > std::numeric_limits<int>::max()) {
    PLOGW << context << ": sum of costs overflowed int; clamping to INT_MAX\n";
    return std::numeric_limits<int>::max();
  }
  if (soc < std::numeric_limits<int>::min()) {
    PLOGW << context << ": sum of costs underflowed int; clamping to INT_MIN\n";
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(soc);
}

[[maybe_unused]] TaskAssignmentIndex buildTaskAssignmentIndex(
    const vector<vector<int>>& assignments, int numTasks) {
  TaskAssignmentIndex index;
  index.owner.assign(numTasks, UNASSIGNED);
  index.pos.assign(numTasks, -1);
  int duplicateTaskOwners = 0;
  for (int agent = 0; agent < (int)assignments.size(); agent++) {
    for (int p = 0; p < (int)assignments[agent].size(); p++) {
      const int task = assignments[agent][p];
      if (task < 0 || task >= numTasks) {
        continue;
      }
      if (index.owner[task] != UNASSIGNED) {
        duplicateTaskOwners++;
        if (duplicateTaskOwners <= 5) {
          PLOGE << "Duplicate task ownership detected while building assignment index"
                << " for task " << task << " (existing agent="
                << index.owner[task] << ", new agent=" << agent << ")\n";
        }
      }
      index.owner[task] = agent;
      index.pos[task] = p;
    }
  }
  if (duplicateTaskOwners > 5) {
    PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
          << " tasks while building assignment index.\n";
  }
  return index;
}

[[maybe_unused]] TaskAssignmentIndex buildCurrentTaskAssignmentIndex(
    const Solution& solution, int numTasks) {
  TaskAssignmentIndex index;
  index.owner.assign(numTasks, UNASSIGNED);
  index.pos.assign(numTasks, -1);
  int duplicateTaskOwners = 0;
  for (int agent = 0; agent < solution.numOfAgents; agent++) {
    const auto& assignments = solution.agents[agent].taskAssignments;
    for (int p = 0; p < (int)assignments.size(); p++) {
      const int task = assignments[p];
      if (task < 0 || task >= numTasks) {
        continue;
      }
      if (index.owner[task] != UNASSIGNED) {
        duplicateTaskOwners++;
        if (duplicateTaskOwners <= 5) {
          PLOGE << "Duplicate task ownership detected in current solution for task "
                << task << " (existing agent=" << index.owner[task]
                << ", new agent=" << agent << ")\n";
        }
      }
      index.owner[task] = agent;
      index.pos[task] = p;
    }
  }
  if (duplicateTaskOwners > 5) {
    PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
          << " tasks in current solution assignment index.\n";
  }
  return index;
}

[[maybe_unused]] string summarizeIntervals(
    const vector<pair<int, int>>* intervals, int maxIntervals = 6) {
  if (intervals == nullptr || intervals->empty()) {
    return "(none)";
  }
  std::ostringstream oss;
  const int limit = min((int)intervals->size(), maxIntervals);
  for (int i = 0; i < limit; i++) {
    if (i > 0) {
      oss << ", ";
    }
    oss << "[" << (*intervals)[i].first << ", ";
    if ((*intervals)[i].second == MAX_TIMESTEP) {
      oss << "INF";
    } else {
      oss << (*intervals)[i].second;
    }
    oss << ")";
  }
  if ((int)intervals->size() > limit) {
    oss << ", ... (" << intervals->size() << " total)";
  }
  return oss.str();
}

[[maybe_unused]] int permanentOccupancyStart(
    const vector<pair<int, int>>* intervals) {
  if (intervals == nullptr) {
    return MAX_TIMESTEP;
  }
  int start = MAX_TIMESTEP;
  for (const auto& interval : *intervals) {
    if (interval.second == MAX_TIMESTEP) {
      start = min(start, interval.first);
    }
  }
  return start;
}

[[maybe_unused]] int firstFreeTimeAtOrAfter(
    const vector<pair<int, int>>* intervals, int timestep) {
  if (intervals == nullptr) {
    return timestep;
  }
  int t = timestep;
  for (const auto& interval : *intervals) {
    if (t < interval.first) {
      break;
    }
    if (interval.first <= t && t < interval.second) {
      if (interval.second == MAX_TIMESTEP) {
        return MAX_TIMESTEP;
      }
      t = interval.second;
    }
  }
  return t;
}

[[maybe_unused]] bool reachableWithPermanentBlocksByTime(
    const Instance& instance, const ConstraintTable& constraintTable, int start,
    int goal, int byTime) {
  if (start < 0 || goal < 0 || start >= instance.mapSize ||
      goal >= instance.mapSize) {
    return false;
  }

  const auto isBlocked = [&](int loc) {
    if (instance.isObstacle(loc)) {
      return true;
    }
    const int blockedFrom =
        permanentOccupancyStart(constraintTable.getConstraintIntervals(loc));
    return blockedFrom != MAX_TIMESTEP && blockedFrom <= byTime;
  };

  if (isBlocked(start) || isBlocked(goal)) {
    return false;
  }

  vector<char> seen(instance.mapSize, 0);
  std::queue<int> q;
  q.push(start);
  seen[start] = 1;
  while (!q.empty()) {
    const int curr = q.front();
    q.pop();
    if (curr == goal) {
      return true;
    }
    for (int nxt : instance.getNeighbors(curr)) {
      if (nxt < 0 || nxt >= instance.mapSize || seen[nxt] || isBlocked(nxt)) {
        continue;
      }
      seen[nxt] = 1;
      q.push(nxt);
    }
  }
  return false;
}
[[maybe_unused]] void logInitialSegmentFailureDiagnostics(
    const Instance& instance, const Solution& solution,
    const ConstraintTable& constraintTable, int agent, int task, int taskPosition,
    int startTime, SingleAgentSolver& solver, double remainingBudgetSec,
    double effectiveTimeoutSec) {
  const int goalLocation = instance.getTaskLocations(task);
  int startLocation = instance.getStartLocationsRef()[agent];
  if (taskPosition > 0) {
    const int prevTask = solution.getAgentGlobalTasks(agent, taskPosition - 1);
    startLocation = instance.getTaskLocations(prevTask);
  }

  const auto* goalIntervals = constraintTable.getConstraintIntervals(goalLocation);
  const auto* startIntervals = constraintTable.getConstraintIntervals(startLocation);
  const int hDist = solver.getStageGoalDistance(taskPosition, startLocation);
  const int earliestArrivalLb =
      (hDist >= MAX_TIMESTEP / 2) ? MAX_TIMESTEP : startTime + hDist;
  const int goalFreeAt = firstFreeTimeAtOrAfter(goalIntervals, earliestArrivalLb);
  const int goalPermanentFrom = permanentOccupancyStart(goalIntervals);
  const int stableTime = max(startTime, constraintTable.temporalExtent + 1);
  int constrainedVerticesAtStable = 0;
  int permanentVerticesByStable = 0;
  for (int loc = 0; loc < instance.mapSize; loc++) {
    if (constraintTable.constrained(loc, stableTime)) {
      constrainedVerticesAtStable++;
    }
    const int blockedFrom =
        permanentOccupancyStart(constraintTable.getConstraintIntervals(loc));
    if (blockedFrom != MAX_TIMESTEP && blockedFrom <= stableTime) {
      permanentVerticesByStable++;
    }
  }
  const bool permanentStaticReachable = reachableWithPermanentBlocksByTime(
      instance, constraintTable, startLocation, goalLocation, stableTime);

  int feasibleImmediateMoves = 0;
  for (int nxt : instance.getNeighbors(startLocation)) {
    const bool blockedVertex = constraintTable.constrained(nxt, startTime + 1);
    const bool blockedEdge =
        constraintTable.constrained(startLocation, nxt, startTime + 1);
    if (!blockedVertex && !blockedEdge) {
      feasibleImmediateMoves++;
    }
  }
  const bool canWaitAtStart =
      !constraintTable.constrained(startLocation, startTime + 1);

  PLOGE << "Init LL-failure diagnostics: agent=" << agent << ", task=" << task
        << ", stage=" << taskPosition << ", start_loc=" << startLocation
        << ", goal_loc=" << goalLocation << ", start_time=" << startTime
        << ", hdist_to_goal=" << hDist
        << ", earliest_arrival_lb=" << earliestArrivalLb
        << ", goal_first_free_at_or_after_lb="
        << (goalFreeAt == MAX_TIMESTEP ? -1 : goalFreeAt)
        << ", goal_permanent_block_from="
        << (goalPermanentFrom == MAX_TIMESTEP ? -1 : goalPermanentFrom)
        << ", stable_time=" << stableTime
        << ", constrained_vertices_at_stable=" << constrainedVerticesAtStable
        << ", permanent_vertices_by_stable=" << permanentVerticesByStable
        << ", permanent_static_reachable="
        << (permanentStaticReachable ? "true" : "false")
        << ", feasible_immediate_moves_from_start=" << feasibleImmediateMoves
        << ", can_wait_at_start=" << (canWaitAtStart ? "true" : "false")
        << ", remaining_budget_sec=" << remainingBudgetSec
        << ", effective_timeout_sec=" << effectiveTimeoutSec << "\n";
  PLOGE << "Init LL-failure diagnostics: goal_intervals="
        << summarizeIntervals(goalIntervals)
        << ", start_intervals=" << summarizeIntervals(startIntervals) << "\n";

  if (goalPermanentFrom != MAX_TIMESTEP &&
      goalPermanentFrom <= earliestArrivalLb) {
    PLOGE << "Init LL-failure diagnostics: certificate=goal_permanently_occupied"
          << "_before_earliest_arrival_lb\n";
  }
  if (!canWaitAtStart && feasibleImmediateMoves == 0) {
    PLOGE << "Init LL-failure diagnostics: certificate=start_trapped_at_t+1\n";
  }
  if (!permanentStaticReachable) {
    PLOGE << "Init LL-failure diagnostics: certificate=static_disconnected_under_"
             "permanent_blocks\n";
  }

  MultiLabelSIPPS sippsCrossCheck(instance, agent, false);
  sippsCrossCheck.setGoalLocations(instance.getTaskLocations(
      solution.getAgentGlobalTasks(agent)));
  sippsCrossCheck.setSegmentTimeout(max(1e-6, min(30.0, remainingBudgetSec)));
  ConstraintTable sippsCt(constraintTable);
  const AgentTaskPath sippsPath = sippsCrossCheck.findPathSegment(
      sippsCt, startTime, taskPosition, 0);
  PLOGE << "Init LL-failure diagnostics: sipps_crosscheck_outcome="
        << sippsCrossCheck.getLastSearchOutcomeName()
        << ", sipps_path_size=" << sippsPath.size() << "\n";
}
}  // namespace
