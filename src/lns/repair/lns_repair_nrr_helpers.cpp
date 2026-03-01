#include "lns_repair_nrr_helpers.hpp"

#include <algorithm>
#include <limits>
#include <set>

#include "lns_internal_helpers.hpp"

namespace lns_nrr_helpers {

long long distOrInf(const Instance& instance, int fromLocation, int toLocation) {
  const int d = instance.getDistanceToGoal(toLocation, fromLocation);
  if (d >= MAX_TIMESTEP / 2) {
    return kNrrInfCost;
  }
  return static_cast<long long>(d);
}

bool topologicalSortSubsetWithPriorityStrict(
    const Instance& instance, const vector<int>& tasks,
    const std::unordered_map<int, long long>& tiePriority,
    vector<int>& outOrdered) {
  std::unordered_map<int, int> indegree;
  std::unordered_map<int, vector<int>> children;
  indegree.reserve(tasks.size());
  children.reserve(tasks.size());
  std::unordered_set<int> taskSet;
  taskSet.reserve(tasks.size());
  for (int task : tasks) {
    taskSet.insert(task);
    indegree[task] = 0;
  }
  for (int task : tasks) {
    for (int succ : instance.getSuccessorsRef(task)) {
      if (taskSet.find(succ) == taskSet.end()) {
        continue;
      }
      indegree[succ]++;
      children[task].push_back(succ);
    }
  }

  auto priorityFor = [&](int task) -> long long {
    const auto it = tiePriority.find(task);
    if (it == tiePriority.end()) {
      return std::numeric_limits<long long>::max() / 4;
    }
    return it->second;
  };

  std::set<pair<long long, int>> ready;
  for (int task : tasks) {
    if (indegree[task] == 0) {
      ready.emplace(priorityFor(task), task);
    }
  }

  outOrdered.clear();
  outOrdered.reserve(tasks.size());
  while (!ready.empty()) {
    const int task = ready.begin()->second;
    ready.erase(ready.begin());
    outOrdered.push_back(task);
    const auto childIt = children.find(task);
    if (childIt == children.end()) {
      continue;
    }
    for (int succ : childIt->second) {
      indegree[succ]--;
      if (indegree[succ] == 0) {
        ready.emplace(priorityFor(succ), succ);
      }
    }
  }
  return (int)outOrdered.size() == (int)tasks.size();
}

bool differenceConstraintsFeasible(int numVars,
                                   const vector<DifferenceEdge>& edges) {
  if (numVars <= 0) {
    return true;
  }
  const long long kBound = (1LL << 60);
  auto clampAdd = [&](long long a, long long b) -> long long {
    if (b > 0 && a > kBound - b) {
      return kBound;
    }
    if (b < 0 && a < -kBound - b) {
      return -kBound;
    }
    const long long s = a + b;
    return std::max(-kBound, std::min(kBound, s));
  };

  vector<long long> dist(numVars, 0);
  for (int iter = 0; iter < numVars; iter++) {
    bool relaxed = false;
    for (const auto& e : edges) {
      if (e.from < 0 || e.from >= numVars || e.to < 0 || e.to >= numVars) {
        continue;
      }
      const long long cand = clampAdd(dist[e.from], e.weight);
      if (cand < dist[e.to]) {
        dist[e.to] = cand;
        relaxed = true;
        if (iter == numVars - 1) {
          return false;  // negative cycle -> infeasible
        }
      }
    }
    if (!relaxed) {
      break;
    }
  }
  return true;
}

std::string normalizeNrrFailureReason(const std::string& reason) {
  if (reason.empty()) {
    return "unknown";
  }
  const size_t pos = reason.find(':');
  const std::string key = reason.substr(0, pos);
  if (key.empty()) {
    return "unknown";
  }
  return key;
}

bool buildIterativeProposal(
    const Instance& instance, int numAgents, int numTasks,
    const vector<int>& destroyedTasks, const vector<char>& destroyedMask,
    const vector<int>& neighborhoodAgents,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    const ConstraintTable& frozenCt, vector<vector<int>>& proposedAssignments,
    std::string& failureReason) {
  vector<char> insertedDestroyed(numTasks, 0);
  vector<int> destroyedOrder;
  {
    std::unordered_map<int, long long> tie;
    tie.reserve(destroyedTasks.size());
    for (int task : destroyedTasks) {
      long long key = 0;
      if (task >= 0 && task < (int)boundaryWindows.releaseByTask.size()) {
        key = boundaryWindows.releaseByTask[task];
      }
      tie.emplace(task, key);
    }
    if (!topologicalSortSubsetWithPriorityStrict(instance, destroyedTasks, tie,
                                                 destroyedOrder)) {
      failureReason = "destroyed_subset_cycle";
      return false;
    }
  }

  vector<int> approxCompletion(numTasks, -1);
  auto recomputeApproxCompletion = [&]() -> bool {
    std::fill(approxCompletion.begin(), approxCompletion.end(), -1);
    for (int agent = 0; agent < numAgents; agent++) {
      const auto& queue = proposedAssignments[agent];
      int completion = 0;
      int prevLocation = instance.getStartLocationsRef()[agent];
      for (int task : queue) {
        if (task < 0 || task >= numTasks) {
          return false;
        }
        const long long travel =
            distOrInf(instance, prevLocation, instance.getTaskLocations(task));
        if (travel >= kNrrInfCost / 4) {
          return false;
        }
        long long nextCompletion64 =
            static_cast<long long>(completion) + std::max(0LL, travel);
        if (nextCompletion64 > std::numeric_limits<int>::max() / 2) {
          completion = std::numeric_limits<int>::max() / 2;
        } else {
          completion = static_cast<int>(nextCompletion64);
        }
        if (destroyedMask[task] && insertedDestroyed[task] &&
            task < (int)boundaryWindows.releaseByTask.size()) {
          completion = std::max(completion, boundaryWindows.releaseByTask[task]);
        }
        approxCompletion[task] = completion;
        prevLocation = instance.getTaskLocations(task);
      }
    }
    return true;
  };

  if (!recomputeApproxCompletion()) {
    failureReason = "approx_completion_initial_failed";
    return false;
  }

  for (int task : destroyedOrder) {
    if (!recomputeApproxCompletion()) {
      failureReason = "approx_completion_recompute_failed";
      return false;
    }
    const auto currentIndex =
        buildTaskAssignmentIndex(proposedAssignments, numTasks);

    int releaseBound = 0;
    if (task >= 0 && task < (int)boundaryWindows.releaseByTask.size()) {
      releaseBound = boundaryWindows.releaseByTask[task];
    }
    int deadlineBound = MAX_TIMESTEP;
    if (task >= 0 && task < (int)boundaryWindows.deadlineByTask.size()) {
      deadlineBound = boundaryWindows.deadlineByTask[task];
    }

    for (int pred : instance.getAncestorsRef(task)) {
      if (pred < 0 || pred >= numTasks) {
        continue;
      }
      if (destroyedMask[pred] && !insertedDestroyed[pred]) {
        failureReason = "destroyed_order_violation";
        return false;
      }
      if (approxCompletion[pred] >= 0 &&
          approxCompletion[pred] < std::numeric_limits<int>::max() - 1) {
        releaseBound = std::max(releaseBound, approxCompletion[pred] + 1);
      }
    }
    for (int succ : instance.getSuccessorsRef(task)) {
      if (succ < 0 || succ >= numTasks || destroyedMask[succ]) {
        continue;
      }
      if (approxCompletion[succ] >= 0 &&
          approxCompletion[succ] > std::numeric_limits<int>::min() + 1) {
        deadlineBound = std::min(deadlineBound, approxCompletion[succ] - 1);
      }
    }
    if (releaseBound > deadlineBound) {
      failureReason = "task_window_infeasible_after_prefix";
      return false;
    }

    bool foundCandidate = false;
    NrrSlot bestSlot;
    long long bestScore = kNrrInfCost;
    const int taskLocation = instance.getTaskLocations(task);
    const int gapCap =
        std::max(1, std::min((int)numTasks + 1, 2 * (int)destroyedTasks.size()));

    for (int agent : neighborhoodAgents) {
      const auto& queue = proposedAssignments[agent];
      const int gapCount = (int)queue.size() + 1;
      vector<pair<long long, int>> rankedGaps;
      rankedGaps.reserve(gapCount);
      for (int gap = 0; gap < gapCount; gap++) {
        const int prevLoc =
            (gap == 0) ? instance.getStartLocationsRef()[agent]
                       : instance.getTaskLocations(queue[gap - 1]);
        long long gapScore = 0;
        if (gap < (int)queue.size()) {
          const int nextLoc = instance.getTaskLocations(queue[gap]);
          gapScore = distOrInf(instance, prevLoc, nextLoc);
        }
        rankedGaps.emplace_back(gapScore, gap);
      }
      std::sort(rankedGaps.begin(), rankedGaps.end(),
                [](const pair<long long, int>& lhs,
                   const pair<long long, int>& rhs) {
                  if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                  }
                  return lhs.second < rhs.second;
                });
      const int agentGapCap = std::max(1, std::min(gapCount, gapCap));
      vector<int> selectedGaps;
      selectedGaps.reserve(agentGapCap);
      for (int i = 0; i < agentGapCap; i++) {
        selectedGaps.push_back(rankedGaps[i].second);
      }
      std::sort(selectedGaps.begin(), selectedGaps.end());

      int minGap = 0;
      int maxGap = (int)queue.size();
      for (int pred : instance.getAncestorsRef(task)) {
        if (pred < 0 || pred >= numTasks) {
          continue;
        }
        if (currentIndex.owner[pred] == agent) {
          minGap = std::max(minGap, currentIndex.pos[pred] + 1);
        }
      }
      for (int succ : instance.getSuccessorsRef(task)) {
        if (succ < 0 || succ >= numTasks) {
          continue;
        }
        if (currentIndex.owner[succ] == agent) {
          maxGap = std::min(maxGap, currentIndex.pos[succ]);
        }
      }

      for (int gap : selectedGaps) {
        if (gap < minGap || gap > maxGap) {
          continue;
        }

        const int prevLoc =
            (gap == 0) ? instance.getStartLocationsRef()[agent]
                       : instance.getTaskLocations(queue[gap - 1]);
        const int nextLoc =
            (gap < (int)queue.size()) ? instance.getTaskLocations(queue[gap]) : -1;
        const long long prevToTask = distOrInf(instance, prevLoc, taskLocation);
        if (prevToTask >= kNrrInfCost / 4) {
          continue;
        }

        int prevCompletion = 0;
        if (gap > 0) {
          const int prevTask = queue[gap - 1];
          if (prevTask < 0 || prevTask >= numTasks ||
              approxCompletion[prevTask] < 0) {
            continue;
          }
          prevCompletion = approxCompletion[prevTask];
        }
        int approxArrival = prevCompletion;
        if (prevCompletion < std::numeric_limits<int>::max() / 2) {
          const long long arrival64 =
              static_cast<long long>(prevCompletion) + std::max(0LL, prevToTask);
          if (arrival64 > std::numeric_limits<int>::max() / 2) {
            approxArrival = std::numeric_limits<int>::max() / 2;
          } else {
            approxArrival = static_cast<int>(arrival64);
          }
        }

        if (approxArrival < releaseBound || approxArrival > deadlineBound) {
          continue;
        }

        long long insertionCost = prevToTask;
        if (nextLoc >= 0) {
          const long long taskToNext = distOrInf(instance, taskLocation, nextLoc);
          const long long prevToNext = distOrInf(instance, prevLoc, nextLoc);
          if (taskToNext >= kNrrInfCost / 4 || prevToNext >= kNrrInfCost / 4) {
            continue;
          }
          insertionCost += (taskToNext - prevToNext);
        }

        if (frozenCt.constrained(taskLocation, approxArrival)) {
          continue;
        }
        if (frozenCt.constrained(prevLoc, taskLocation, approxArrival)) {
          continue;
        }
        if (nextLoc >= 0) {
          const long long taskToNext = distOrInf(instance, taskLocation, nextLoc);
          if (taskToNext >= kNrrInfCost / 4) {
            continue;
          }
          int approxNextArrival = std::numeric_limits<int>::max() / 2;
          if (approxArrival < std::numeric_limits<int>::max() / 2) {
            const long long nextArrival64 =
                static_cast<long long>(approxArrival) + std::max(0LL, taskToNext);
            if (nextArrival64 <= std::numeric_limits<int>::max() / 2) {
              approxNextArrival = static_cast<int>(nextArrival64);
            }
          }
          if (frozenCt.constrained(taskLocation, nextLoc, approxNextArrival) ||
              frozenCt.constrained(nextLoc, approxNextArrival)) {
            continue;
          }
        }

        const long long total = insertionCost;
        if (!foundCandidate || total < bestScore) {
          foundCandidate = true;
          bestScore = total;
          bestSlot = NrrSlot{agent, gap, 0, prevLoc, nextLoc, prevCompletion,
                             approxArrival};
        }
      }
    }

    if (!foundCandidate) {
      failureReason = "no_feasible_slot_for_destroyed_task_iterative";
      return false;
    }

    auto& seq = proposedAssignments[bestSlot.agent];
    const int insertPos = std::max(0, std::min(bestSlot.gap, (int)seq.size()));
    seq.insert(seq.begin() + insertPos, task);
    insertedDestroyed[task] = 1;
  }

  for (int task : destroyedTasks) {
    if (!insertedDestroyed[task]) {
      failureReason = "destroyed_task_not_inserted";
      return false;
    }
  }

  vector<char> finalSeen(numTasks, 0);
  int finalCount = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    for (int task : proposedAssignments[agent]) {
      if (task < 0 || task >= numTasks) {
        failureReason = "proposed_assignment_out_of_range";
        return false;
      }
      if (finalSeen[task]) {
        failureReason = "proposed_assignment_duplicate";
        return false;
      }
      finalSeen[task] = 1;
      finalCount++;
    }
  }
  if (finalCount != numTasks) {
    failureReason = "proposed_assignment_incomplete";
    return false;
  }

  return true;
}

bool runTemporalPrecheck(
    const Instance& instance, int numTasks,
    const vector<int>& neighborhoodAgents, const vector<int>& destroyedTasks,
    const vector<vector<int>>& proposedAssignments,
    const vector<int>& incumbentCompletion,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    std::string& failureReason) {
  vector<char> mutableTaskMask(numTasks, 0);
  for (int agent : neighborhoodAgents) {
    for (int task : proposedAssignments[agent]) {
      if (task >= 0 && task < numTasks) {
        mutableTaskMask[task] = 1;
      }
    }
  }

  vector<int> mutableTasks;
  mutableTasks.reserve(numTasks);
  for (int task = 0; task < numTasks; task++) {
    if (mutableTaskMask[task]) {
      mutableTasks.push_back(task);
    }
  }
  std::unordered_map<int, int> taskToVar;
  taskToVar.reserve(mutableTasks.size() * 2 + 1);
  for (int i = 0; i < (int)mutableTasks.size(); i++) {
    taskToVar[mutableTasks[i]] = i + 1;
  }

  vector<DifferenceEdge> precheckEdges;
  precheckEdges.reserve(8 * std::max(1, (int)mutableTasks.size()));
  auto addUpper = [&](int leftTask, int rightTask, long long c) {
    const auto itL = taskToVar.find(leftTask);
    const auto itR = taskToVar.find(rightTask);
    if (itL == taskToVar.end() || itR == taskToVar.end()) {
      return;
    }
    precheckEdges.push_back({itR->second, itL->second, c});
  };
  auto addLowerFromSource = [&](int task, long long lb) {
    const auto it = taskToVar.find(task);
    if (it == taskToVar.end()) {
      return;
    }
    precheckEdges.push_back({it->second, 0, -lb});
  };
  auto addUpperToSource = [&](int task, long long ub) {
    const auto it = taskToVar.find(task);
    if (it == taskToVar.end()) {
      return;
    }
    precheckEdges.push_back({0, it->second, ub});
  };

  for (int agent : neighborhoodAgents) {
    const auto& queue = proposedAssignments[agent];
    if (queue.empty()) {
      continue;
    }
    const int startLoc = instance.getStartLocationsRef()[agent];
    const int firstTask = queue.front();
    const long long startToFirst =
        distOrInf(instance, startLoc, instance.getTaskLocations(firstTask));
    if (startToFirst >= kNrrInfCost / 4) {
      failureReason = "precheck_temporal_infeasible";
      return false;
    }
    addLowerFromSource(firstTask, startToFirst);

    for (int i = 0; i + 1 < (int)queue.size(); i++) {
      const int u = queue[i];
      const int v = queue[i + 1];
      const long long travel =
          distOrInf(instance, instance.getTaskLocations(u),
                    instance.getTaskLocations(v));
      if (travel >= kNrrInfCost / 4) {
        failureReason = "precheck_temporal_infeasible";
        return false;
      }
      addUpper(u, v, -travel);
    }
  }

  for (int task : destroyedTasks) {
    if (!mutableTaskMask[task]) {
      continue;
    }
    if (task < 0 || task >= numTasks) {
      continue;
    }
    const int lb = boundaryWindows.releaseByTask[task];
    const int ub = boundaryWindows.deadlineByTask[task];
    if (lb > ub) {
      failureReason = "precheck_temporal_infeasible";
      return false;
    }
    addLowerFromSource(task, lb);
    addUpperToSource(task, ub);
  }

  for (const auto& edge : instance.getInputPrecedenceConstraintsRef()) {
    const int u = edge.first;
    const int v = edge.second;
    if (u < 0 || u >= numTasks || v < 0 || v >= numTasks) {
      failureReason = "precheck_temporal_infeasible";
      return false;
    }
    const bool uMutable = (mutableTaskMask[u] != 0);
    const bool vMutable = (mutableTaskMask[v] != 0);
    if (!uMutable && !vMutable) {
      continue;
    }
    if (uMutable && vMutable) {
      addUpper(u, v, -1);
      continue;
    }
    if (!uMutable && vMutable) {
      const int tauU = incumbentCompletion[u];
      if (tauU < 0) {
        failureReason = "precheck_temporal_infeasible";
        return false;
      }
      addLowerFromSource(v, static_cast<long long>(tauU) + 1);
      continue;
    }
    if (uMutable && !vMutable) {
      const int tauV = incumbentCompletion[v];
      if (tauV < 0) {
        failureReason = "precheck_temporal_infeasible";
        return false;
      }
      addUpperToSource(u, static_cast<long long>(tauV) - 1);
      continue;
    }
  }

  const int stnVars = (int)mutableTasks.size() + 1;
  if (!differenceConstraintsFeasible(stnVars, precheckEdges)) {
    failureReason = "precheck_temporal_infeasible";
    return false;
  }

  return true;
}

}  // namespace lns_nrr_helpers
