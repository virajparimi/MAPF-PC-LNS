#include "lns.hpp"
#include "internal/task_position_index.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace {

int pickRandomIndex(std::mt19937& rng, int upperExclusive) {
  std::uniform_int_distribution<int> dist(0, upperExclusive - 1);
  return dist(rng);
}

void addUndirectedEdge(std::vector<std::unordered_set<int>>& adjacency, int a,
                       int b) {
  if (a < 0 || b < 0 || a >= (int)adjacency.size() || b >= (int)adjacency.size() ||
      a == b) {
    return;
  }
  adjacency[a].insert(b);
  adjacency[b].insert(a);
}

std::vector<int> collectAgentsFromPotential(const ConflictMap* potentialNeighborhood,
                                            int numAgents) {
  std::unordered_set<int> agentSet;
  if (potentialNeighborhood != nullptr) {
    for (const auto& [_, conflict] : *potentialNeighborhood) {
      if (conflict.agent >= 0 && conflict.agent < numAgents) {
        agentSet.insert(conflict.agent);
      }
    }
  }
  std::vector<int> agents(agentSet.begin(), agentSet.end());
  std::sort(agents.begin(), agents.end());
  return agents;
}

std::vector<int> bfsComponent(const std::vector<std::unordered_set<int>>& adjacency,
                              int seed) {
  std::vector<int> component;
  if (seed < 0 || seed >= (int)adjacency.size()) {
    return component;
  }
  std::vector<char> visited(adjacency.size(), 0);
  std::queue<int> q;
  q.push(seed);
  visited[seed] = 1;
  while (!q.empty()) {
    const int current = q.front();
    q.pop();
    component.push_back(current);
    for (int next : adjacency[current]) {
      if (next < 0 || next >= (int)adjacency.size() || visited[next]) {
        continue;
      }
      visited[next] = 1;
      q.push(next);
    }
  }
  return component;
}

void fillRemovedTasksFromAgents(Solution& solution, Neighbor& neighborhood,
                                const Instance& instance,
                                const std::vector<int>& selectedAgents,
                                const ConflictMap* potentialNeighborhood,
                                int targetTasks) {
  const int taskCount = instance.getTasksNum();
  if (targetTasks <= 0 || taskCount <= 0) {
    return;
  }
  const int cappedTarget = std::min(targetTasks, taskCount);
  const int numAgents = instance.getAgentNum();
  std::vector<char> selectedAgentMask(numAgents, 0);
  for (int agent : selectedAgents) {
    if (agent >= 0 && agent < numAgents) {
      selectedAgentMask[agent] = 1;
    }
  }

  const std::vector<int> taskPosByTask =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution,
                                                                  taskCount);
  auto addTask = [&](int task) -> bool {
    if ((int)neighborhood.removedTasks.size() >= cappedTarget) {
      return false;
    }
    if (task < 0 || task >= taskCount ||
        neighborhood.removedTasks.count(task) > 0) {
      return false;
    }
    const int owner = (task < (int)solution.taskAgentMap.size())
                          ? solution.taskAgentMap[task]
                          : UNASSIGNED;
    if (owner == UNASSIGNED || owner < 0 || owner >= numAgents) {
      return false;
    }
    const int pos = (task < (int)taskPosByTask.size()) ? taskPosByTask[task]
                                                       : UNASSIGNED;
    if (pos == UNASSIGNED) {
      return false;
    }
    neighborhood.removedTasks.emplace(task, Conflicts(task, owner, pos));
    return true;
  };

  std::vector<std::vector<int>> conflictedPositionsByAgent(numAgents);
  if (potentialNeighborhood != nullptr) {
    for (const auto& [task, conflict] : *potentialNeighborhood) {
      if (conflict.agent < 0 || conflict.agent >= numAgents ||
          !selectedAgentMask[conflict.agent]) {
        continue;
      }
      addTask(task);
      const int pos = (task >= 0 && task < (int)taskPosByTask.size())
                          ? taskPosByTask[task]
                          : UNASSIGNED;
      if (pos != UNASSIGNED) {
        conflictedPositionsByAgent[conflict.agent].push_back(pos);
      }
    }
  }

  for (int agent : selectedAgents) {
    if (agent < 0 || agent >= numAgents) {
      continue;
    }
    auto& positions = conflictedPositionsByAgent[agent];
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()),
                    positions.end());
    const auto& assignments = solution.agents[agent].taskAssignments;
    for (int pos : positions) {
      if ((int)neighborhood.removedTasks.size() >= cappedTarget) {
        break;
      }
      if (pos - 1 >= 0 && pos - 1 < (int)assignments.size()) {
        addTask(assignments[pos - 1]);
      }
      if (pos + 1 >= 0 && pos + 1 < (int)assignments.size()) {
        addTask(assignments[pos + 1]);
      }
    }
  }

  for (int agent : selectedAgents) {
    if ((int)neighborhood.removedTasks.size() >= cappedTarget) {
      break;
    }
    if (agent < 0 || agent >= numAgents) {
      continue;
    }
    const auto& assignments = solution.agents[agent].taskAssignments;
    for (int task : assignments) {
      if ((int)neighborhood.removedTasks.size() >= cappedTarget) {
        break;
      }
      addTask(task);
    }
  }

  for (int task = 0;
       task < taskCount &&
       (int)neighborhood.removedTasks.size() < cappedTarget;
       task++) {
    addTask(task);
  }
}

std::vector<int> buildConflictAdjacencyAndSampleAgents(
    const std::vector<int>& conflictAgents,
    const std::vector<std::pair<int, int>>& collisionPairs, std::mt19937& rng,
    int numAgents,
    int targetAgentCount) {
  std::vector<std::unordered_set<int>> adjacency(numAgents);
  std::vector<char> inConflict(numAgents, 0);
  for (int a : conflictAgents) {
    if (a >= 0 && a < numAgents) {
      inConflict[a] = 1;
    }
  }

  for (const auto& [a, b] : collisionPairs) {
    if (a < 0 || b < 0 || a >= numAgents || b >= numAgents) {
      continue;
    }
    if (!inConflict[a] || !inConflict[b]) {
      continue;
    }
    addUndirectedEdge(adjacency, a, b);
  }

  bool hasAnyEdge = false;
  for (int a : conflictAgents) {
    if (a >= 0 && a < numAgents && !adjacency[a].empty()) {
      hasAnyEdge = true;
      break;
    }
  }
  if (!hasAnyEdge && conflictAgents.size() > 1) {
    for (size_t i = 0; i + 1 < conflictAgents.size(); i++) {
      const int a = conflictAgents[i];
      const int b = conflictAgents[i + 1];
      addUndirectedEdge(adjacency, a, b);
    }
  }

  std::vector<int> nonIsolated;
  for (int a : conflictAgents) {
    if (a >= 0 && a < numAgents && !adjacency[a].empty()) {
      nonIsolated.push_back(a);
    }
  }
  const std::vector<int>& seeds = nonIsolated.empty() ? conflictAgents : nonIsolated;
  if (seeds.empty()) {
    return {};
  }

  const int seed = seeds[pickRandomIndex(rng, (int)seeds.size())];
  std::vector<int> component = bfsComponent(adjacency, seed);
  if (component.empty()) {
    component.push_back(seed);
  }

  std::unordered_set<int> selectedSet;
  auto tryInsert = [&](int agent) {
    if (agent >= 0 && agent < numAgents && inConflict[agent]) {
      selectedSet.insert(agent);
    }
  };

  if ((int)component.size() <= targetAgentCount) {
    for (int a : component) {
      tryInsert(a);
    }
    int attempts = 0;
    const int maxAttempts = std::max(50, targetAgentCount * 20);
    while ((int)selectedSet.size() < targetAgentCount && attempts < maxAttempts &&
           !selectedSet.empty()) {
      attempts++;
      const int pivotIndex = pickRandomIndex(rng, (int)selectedSet.size());
      auto it = selectedSet.begin();
      std::advance(it, pivotIndex);
      const int pivot = *it;
      if (!adjacency[pivot].empty()) {
        const int nextIndex = pickRandomIndex(rng, (int)adjacency[pivot].size());
        auto jt = adjacency[pivot].begin();
        std::advance(jt, nextIndex);
        tryInsert(*jt);
      }
    }
  } else {
    tryInsert(component[pickRandomIndex(rng, (int)component.size())]);
    int attempts = 0;
    const int maxAttempts = std::max(100, targetAgentCount * 30);
    while ((int)selectedSet.size() < targetAgentCount && attempts < maxAttempts) {
      attempts++;
      const int pivotIndex = pickRandomIndex(rng, (int)selectedSet.size());
      auto it = selectedSet.begin();
      std::advance(it, pivotIndex);
      const int pivot = *it;
      if (adjacency[pivot].empty()) {
        continue;
      }
      const int nextIndex = pickRandomIndex(rng, (int)adjacency[pivot].size());
      auto jt = adjacency[pivot].begin();
      std::advance(jt, nextIndex);
      tryInsert(*jt);
    }
  }

  if ((int)selectedSet.size() < targetAgentCount) {
    std::vector<int> shuffled = conflictAgents;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    for (int a : shuffled) {
      if ((int)selectedSet.size() >= targetAgentCount) {
        break;
      }
      tryInsert(a);
    }
  }

  std::vector<int> selected(selectedSet.begin(), selectedSet.end());
  std::sort(selected.begin(), selected.end());
  return selected;
}

std::vector<int> collectTargetRelatedAgents(
    const Solution& solution, int numAgents, int agent,
    const std::unordered_map<int, std::vector<int>>& goalLocationToAgents,
    const std::vector<char>& allowedAgentMask) {
  std::unordered_set<int> related;
  if (agent < 0 || agent >= numAgents) {
    return {};
  }
  const auto& path = solution.agents[agent].path;
  for (const auto& state : path.path) {
    const auto it = goalLocationToAgents.find(state.location);
    if (it == goalLocationToAgents.end()) {
      continue;
    }
    for (int other : it->second) {
      if (other >= 0 && other < (int)allowedAgentMask.size() &&
          allowedAgentMask[other]) {
        related.insert(other);
      }
    }
  }
  related.erase(agent);
  std::vector<int> result(related.begin(), related.end());
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace

void LNS::collisionSoftRemoval(const ConflictMap* potentialNeighborhood) {
  PLOGD << "Using collision-soft removal\n";
  clearNeighborhood();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = std::min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  const std::vector<int> conflictAgents =
      collectAgentsFromPotential(potentialNeighborhood, instance_.getAgentNum());
  if (conflictAgents.empty()) {
    if (potentialNeighborhood != nullptr) {
      conflictRemoval(potentialNeighborhood);
    } else {
      randomRemoval();
    }
    return;
  }

  const int targetAgentCount =
      std::max(1, std::min((int)conflictAgents.size(), cappedNeighborSize));
  const std::vector<int> selectedAgents =
      buildConflictAdjacencyAndSampleAgents(
          conflictAgents, lastValidationCollisionPairs_, rng_,
          instance_.getAgentNum(),
          targetAgentCount);
  if (selectedAgents.empty()) {
    if (potentialNeighborhood != nullptr) {
      conflictRemoval(potentialNeighborhood);
    } else {
      randomRemoval();
    }
    return;
  }

  fillRemovedTasksFromAgents(solution_, lnsNeighborhood_, instance_,
                             selectedAgents, potentialNeighborhood,
                             cappedNeighborSize);
  if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    PLOGW << "collisionSoftRemoval: could only remove "
          << lnsNeighborhood_.removedTasks.size() << " out of requested "
          << cappedNeighborSize << " tasks\n";
  }
}

void LNS::failureSoftRemoval(const ConflictMap* potentialNeighborhood) {
  PLOGD << "Using failure-soft removal\n";
  clearNeighborhood();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = std::min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  std::vector<int> sourceAgents = lastSoftFailureConflictAgents_;
  if (sourceAgents.empty()) {
    sourceAgents =
        collectAgentsFromPotential(potentialNeighborhood, instance_.getAgentNum());
  }
  std::sort(sourceAgents.begin(), sourceAgents.end());
  sourceAgents.erase(std::unique(sourceAgents.begin(), sourceAgents.end()),
                     sourceAgents.end());

  if (sourceAgents.empty()) {
    collisionSoftRemoval(potentialNeighborhood);
    if (lnsNeighborhood_.removedTasks.empty()) {
      if (potentialNeighborhood != nullptr) {
        conflictRemoval(potentialNeighborhood);
      } else {
        randomRemoval();
      }
    }
    return;
  }

  std::unordered_map<int, int> degree;
  for (int a : sourceAgents) {
    degree[a] = 0;
  }
  for (const auto& [a, b] : lastValidationCollisionPairs_) {
    if (degree.count(a) == 0 || degree.count(b) == 0) {
      continue;
    }
    degree[a]++;
    degree[b]++;
  }

  int anchor = sourceAgents[pickRandomIndex(rng_, (int)sourceAgents.size())];
  int bestDegree = -1;
  std::vector<int> bestAnchors;
  for (int a : sourceAgents) {
    const int d = degree[a];
    if (d > bestDegree) {
      bestDegree = d;
      bestAnchors.clear();
      bestAnchors.push_back(a);
    } else if (d == bestDegree) {
      bestAnchors.push_back(a);
    }
  }
  if (!bestAnchors.empty()) {
    anchor = bestAnchors[pickRandomIndex(rng_, (int)bestAnchors.size())];
  }

  const int numAgents = instance_.getAgentNum();
  std::vector<char> sourceMask(numAgents, 0);
  for (int a : sourceAgents) {
    if (a >= 0 && a < numAgents) {
      sourceMask[a] = 1;
    }
  }

  std::set<std::pair<int, int>> A_start;
  const int anchorStartLocation = instance_.getStartLocationsRef()[anchor];
  const bool includeTerminal = (goalOccupationMode_ == "reposition_true");
  for (int other = 0; other < numAgents; other++) {
    if (other == anchor || !sourceMask[other]) {
      continue;
    }
    const int horizon = getAgentOccupancyHorizon(other, includeTerminal);
    for (int t = 0; t < horizon; t++) {
      const int loc = getAgentLocationAt(other, t, includeTerminal);
      if (loc == anchorStartLocation) {
        A_start.emplace(t, other);
        break;
      }
    }
  }

  std::unordered_map<int, std::vector<int>> goalLocationToAgents;
  goalLocationToAgents.reserve(taskCount);
  for (int agent = 0; agent < numAgents; agent++) {
    for (int task : solution_.agents[agent].taskAssignments) {
      if (task < 0 || task >= taskCount) {
        continue;
      }
      const int loc = instance_.getTaskLocations(task);
      goalLocationToAgents[loc].push_back(agent);
    }
  }
  for (auto& [_, owners] : goalLocationToAgents) {
    std::sort(owners.begin(), owners.end());
    owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  }

  std::unordered_set<int> A_target_set;
  const auto& anchorPath = solution_.agents[anchor].path;
  for (const auto& state : anchorPath.path) {
    const auto it = goalLocationToAgents.find(state.location);
    if (it == goalLocationToAgents.end()) {
      continue;
    }
    for (int other : it->second) {
      if (other != anchor && other >= 0 && other < numAgents &&
          sourceMask[other]) {
        A_target_set.insert(other);
      }
    }
  }

  std::set<int> neighborsSet;
  neighborsSet.insert(anchor);
  const int targetAgentCount =
      std::max(1, std::min((int)sourceAgents.size(), cappedNeighborSize));

  if ((int)(A_start.size() + A_target_set.size()) >= targetAgentCount - 1) {
    if (A_start.empty()) {
      std::vector<int> shuffled(A_target_set.begin(), A_target_set.end());
      std::shuffle(shuffled.begin(), shuffled.end(), rng_);
      for (int a : shuffled) {
        if ((int)neighborsSet.size() >= targetAgentCount) {
          break;
        }
        neighborsSet.insert(a);
      }
    } else if ((int)A_target_set.size() >= targetAgentCount) {
      std::vector<int> shuffled(A_target_set.begin(), A_target_set.end());
      std::shuffle(shuffled.begin(), shuffled.end(), rng_);
      for (int a : shuffled) {
        if ((int)neighborsSet.size() >= std::max(1, targetAgentCount - 1)) {
          break;
        }
        neighborsSet.insert(a);
      }
      neighborsSet.insert(A_start.begin()->second);
    } else {
      neighborsSet.insert(A_target_set.begin(), A_target_set.end());
      for (const auto& [_, a] : A_start) {
        if ((int)neighborsSet.size() >= targetAgentCount) {
          break;
        }
        neighborsSet.insert(a);
      }
    }
  } else if (!A_start.empty() || !A_target_set.empty()) {
    neighborsSet.insert(A_target_set.begin(), A_target_set.end());
    for (const auto& [_, a] : A_start) {
      neighborsSet.insert(a);
    }

    std::set<int> tabuSet;
    while ((int)neighborsSet.size() < targetAgentCount) {
      if (neighborsSet.empty() || tabuSet.size() == neighborsSet.size()) {
        break;
      }
      const int randIdx = pickRandomIndex(rng_, (int)neighborsSet.size());
      auto it = neighborsSet.begin();
      std::advance(it, randIdx);
      const int pivot = *it;
      tabuSet.insert(pivot);

      std::vector<int> targets =
          collectTargetRelatedAgents(solution_, numAgents, pivot,
                                     goalLocationToAgents,
                                     sourceMask);
      if (targets.empty()) {
        continue;
      }
      const int pick = targets[pickRandomIndex(rng_, (int)targets.size())];
      neighborsSet.insert(pick);
    }
  }

  if ((int)neighborsSet.size() < targetAgentCount) {
    std::vector<int> shuffled = sourceAgents;
    std::shuffle(shuffled.begin(), shuffled.end(), rng_);
    for (int a : shuffled) {
      if ((int)neighborsSet.size() >= targetAgentCount) {
        break;
      }
      neighborsSet.insert(a);
    }
  }

  std::vector<int> selectedAgents(neighborsSet.begin(), neighborsSet.end());
  fillRemovedTasksFromAgents(solution_, lnsNeighborhood_, instance_,
                             selectedAgents, potentialNeighborhood,
                             cappedNeighborSize);

  if (lnsNeighborhood_.removedTasks.empty()) {
    collisionSoftRemoval(potentialNeighborhood);
  }
  if (lnsNeighborhood_.removedTasks.empty()) {
    if (potentialNeighborhood != nullptr) {
      conflictRemoval(potentialNeighborhood);
    } else {
      randomRemoval();
    }
  }
}
