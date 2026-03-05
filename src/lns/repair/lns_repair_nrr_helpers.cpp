#include "lns_repair_nrr_helpers.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <unordered_set>

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

void NrrFrozenOccupancyIndex::clear() {
  perAgentContrib_.clear();
  vertexCounts_.clear();
  edgeCounts_.clear();
  tailStartsByLocation_.clear();
}

void NrrFrozenOccupancyIndex::decrementVertexKey(const VertexTimeKey& key) {
  const auto it = vertexCounts_.find(key);
  if (it == vertexCounts_.end()) {
    return;
  }
  it->second--;
  if (it->second <= 0) {
    vertexCounts_.erase(it);
  }
}

void NrrFrozenOccupancyIndex::decrementEdgeKey(const EdgeTimeKey& key) {
  const auto it = edgeCounts_.find(key);
  if (it == edgeCounts_.end()) {
    return;
  }
  it->second--;
  if (it->second <= 0) {
    edgeCounts_.erase(it);
  }
}

void NrrFrozenOccupancyIndex::decrementTailKey(int location, int startTimestep) {
  const auto itLoc = tailStartsByLocation_.find(location);
  if (itLoc == tailStartsByLocation_.end()) {
    return;
  }
  auto& starts = itLoc->second;
  const auto itStart = starts.find(startTimestep);
  if (itStart == starts.end()) {
    return;
  }
  itStart->second--;
  if (itStart->second <= 0) {
    starts.erase(itStart);
  }
  if (starts.empty()) {
    tailStartsByLocation_.erase(itLoc);
  }
}

void NrrFrozenOccupancyIndex::addPathContribution(int agent,
                                                  const AgentTaskPath& path,
                                                  bool reserveTail) {
  if (agent < 0 || agent >= (int)perAgentContrib_.size() || path.empty()) {
    return;
  }
  auto& contrib = perAgentContrib_[agent];
  contrib.active = true;

  int prevLoc = path.front().location;
  for (int i = 0; i < (int)path.size(); i++) {
    const int t = path.beginTime + i;
    const int loc = path.at(i).location;
    if (loc >= 0 && t >= 0) {
      const VertexTimeKey vKey{loc, t};
      vertexCounts_[vKey]++;
      contrib.vertexKeys.push_back(vKey);
    }
    if (i > 0 && prevLoc >= 0 && loc >= 0 && t >= 0) {
      // Reverse edge occupancy for swap checks:
      // frozen move prev->loc at t blocks candidate move loc->prev at t.
      const EdgeTimeKey eKey{loc, prevLoc, t};
      edgeCounts_[eKey]++;
      contrib.edgeKeys.push_back(eKey);
    }
    prevLoc = loc;
  }

  if (reserveTail) {
    const int goal = path.back().location;
    const int start = path.endTime();
    if (goal >= 0 && start >= 0) {
      tailStartsByLocation_[goal][start]++;
      contrib.tailStartsByLoc.emplace_back(goal, start);
    }
  }
}

void NrrFrozenOccupancyIndex::buildFromPreviousSolution(
    const Instance& instance, const Solution& previousSolution,
    bool stayGoalOccupationMode) {
  clear();
  const int numAgents = instance.getAgentNum();
  perAgentContrib_.assign(numAgents, AgentContribution());
  for (int agent = 0; agent < numAgents; agent++) {
    if (agent >= (int)previousSolution.agents.size()) {
      continue;
    }
    const auto& servicePath = previousSolution.agents[agent].path;
    addPathContribution(agent, servicePath, stayGoalOccupationMode);

    if (!stayGoalOccupationMode &&
        previousSolution.agents[agent].terminalPathActive) {
      addPathContribution(agent, previousSolution.agents[agent].terminalPath,
                          true);
    }
  }
}

bool NrrFrozenOccupancyIndex::isVertexConstrained(int location,
                                                  int timestep) const {
  if (location < 0 || timestep < 0) {
    return false;
  }
  const VertexTimeKey exactKey{location, timestep};
  const auto it = vertexCounts_.find(exactKey);
  if (it != vertexCounts_.end() && it->second > 0) {
    return true;
  }

  const auto itLoc = tailStartsByLocation_.find(location);
  if (itLoc == tailStartsByLocation_.end() || itLoc->second.empty()) {
    return false;
  }
  const auto& starts = itLoc->second;
  auto itStart = starts.upper_bound(timestep);
  if (itStart == starts.begin()) {
    return false;
  }
  --itStart;
  return itStart->second > 0;
}

bool NrrFrozenOccupancyIndex::isEdgeConstrained(int fromLocation, int toLocation,
                                                int nextTimestep) const {
  if (fromLocation < 0 || toLocation < 0 || nextTimestep <= 0 ||
      fromLocation == toLocation) {
    return false;
  }
  const EdgeTimeKey key{fromLocation, toLocation, nextTimestep};
  const auto it = edgeCounts_.find(key);
  return it != edgeCounts_.end() && it->second > 0;
}

bool NrrFrozenOccupancyIndex::demoteAgent(int agent) {
  if (agent < 0 || agent >= (int)perAgentContrib_.size()) {
    return false;
  }
  auto& contrib = perAgentContrib_[agent];
  if (!contrib.active) {
    return false;
  }
  for (const auto& key : contrib.vertexKeys) {
    decrementVertexKey(key);
  }
  for (const auto& key : contrib.edgeKeys) {
    decrementEdgeKey(key);
  }
  for (const auto& [loc, start] : contrib.tailStartsByLoc) {
    decrementTailKey(loc, start);
  }
  contrib.vertexKeys.clear();
  contrib.edgeKeys.clear();
  contrib.tailStartsByLoc.clear();
  contrib.active = false;
  return true;
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

bool buildIterativeProposalGlobal(
    const Instance& instance, int numAgents, int numTasks,
    const vector<int>& destroyedTasks, const vector<char>& destroyedMask,
    const vector<int>& candidateAgents, const vector<int>& initialTouchedAgents,
    const vector<int>& incumbentTaskOwnerByTask,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    NrrFrozenOccupancyIndex& frozenOccupancy,
    vector<vector<int>>& proposedAssignments, ProposalMutationTrace& trace,
    std::string& failureReason) {
  trace = ProposalMutationTrace();
  vector<char> touchedMask(numAgents, 0);
  for (int agent : initialTouchedAgents) {
    if (agent < 0 || agent >= numAgents || touchedMask[agent]) {
      continue;
    }
    touchedMask[agent] = 1;
    trace.touchedAgentsFinal.push_back(agent);
    if (frozenOccupancy.demoteAgent(agent)) {
      trace.globalDemotions++;
    }
  }
  if (trace.touchedAgentsFinal.empty()) {
    failureReason = "empty_initial_touched_agents_global";
    return false;
  }

  if (candidateAgents.empty()) {
    failureReason = "empty_candidate_agents_global";
    return false;
  }

  vector<char> candidateMask(numAgents, 0);
  for (int agent : candidateAgents) {
    if (agent >= 0 && agent < numAgents) {
      candidateMask[agent] = 1;
    }
  }

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

  const int gapCap =
      std::max(1, std::min((int)numTasks + 1, 2 * (int)destroyedTasks.size()));
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
    for (int agent = 0; agent < numAgents; agent++) {
      if (!candidateMask[agent]) {
        continue;
      }
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
        trace.globalSlotEvaluations++;

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

        if (frozenOccupancy.isVertexConstrained(taskLocation, approxArrival)) {
          continue;
        }
        if (frozenOccupancy.isEdgeConstrained(prevLoc, taskLocation,
                                              approxArrival)) {
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
          if (frozenOccupancy.isEdgeConstrained(taskLocation, nextLoc,
                                                approxNextArrival) ||
              frozenOccupancy.isVertexConstrained(nextLoc, approxNextArrival)) {
            continue;
          }
        }

        if (!foundCandidate || insertionCost < bestScore) {
          foundCandidate = true;
          bestScore = insertionCost;
          bestSlot = NrrSlot{agent, gap, 0, prevLoc, nextLoc, prevCompletion,
                             approxArrival};
        }
      }
    }

    if (!foundCandidate) {
      trace.globalNoFeasibleSlotCount++;
      failureReason = "no_feasible_slot_for_destroyed_task_iterative_global";
      return false;
    }

    auto& seq = proposedAssignments[bestSlot.agent];
    const int oldSize = (int)seq.size();
    const int insertPos = std::max(0, std::min(bestSlot.gap, oldSize));
    if (insertPos < oldSize) {
      const int successorTask = seq[insertPos];
      if (successorTask >= 0 && successorTask < numTasks &&
          !destroyedMask[successorTask]) {
        trace.patchedImmediateSuccessorTasks.push_back(successorTask);
      }
    }
    seq.insert(seq.begin() + insertPos, task);
    insertedDestroyed[task] = 1;
    if (!touchedMask[bestSlot.agent]) {
      touchedMask[bestSlot.agent] = 1;
      trace.touchedAgentsFinal.push_back(bestSlot.agent);
      if (frozenOccupancy.demoteAgent(bestSlot.agent)) {
        trace.globalDemotions++;
      }
    }
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

  std::sort(trace.touchedAgentsFinal.begin(), trace.touchedAgentsFinal.end());
  trace.touchedAgentsFinal.erase(
      std::unique(trace.touchedAgentsFinal.begin(), trace.touchedAgentsFinal.end()),
      trace.touchedAgentsFinal.end());

  std::sort(trace.patchedImmediateSuccessorTasks.begin(),
            trace.patchedImmediateSuccessorTasks.end());
  trace.patchedImmediateSuccessorTasks.erase(
      std::unique(trace.patchedImmediateSuccessorTasks.begin(),
                  trace.patchedImmediateSuccessorTasks.end()),
      trace.patchedImmediateSuccessorTasks.end());

  const auto finalIndex = buildTaskAssignmentIndex(proposedAssignments, numTasks);
  if ((int)incumbentTaskOwnerByTask.size() == numTasks) {
    for (int task : destroyedTasks) {
      if (task < 0 || task >= numTasks) {
        continue;
      }
      const int oldOwner = incumbentTaskOwnerByTask[task];
      const int newOwner = finalIndex.owner[task];
      if (oldOwner != newOwner) {
        trace.destroyedTaskOwnerChanges.push_back(
            DestroyedTaskOwnerChange{task, oldOwner, newOwner});
      }
    }
  }
  return true;
}

bool runTemporalPrecheck(
    const Instance& instance, int numTasks,
    const vector<int>& mutableAgents, const vector<int>& destroyedTasks,
    const vector<vector<int>>& proposedAssignments,
    const vector<int>& incumbentCompletion,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    std::string& failureReason) {
  vector<char> mutableTaskMask(numTasks, 0);
  for (int agent : mutableAgents) {
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

  for (int agent : mutableAgents) {
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
