#include "lns.hpp"
#include "utils.hpp"

void LNS::addConflictingTask(int agent, int timestep, ConflictMap* out) const {
  if (out == nullptr || agent < 0 || agent >= instance_.getAgentNum()) {
    return;
  }
  const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
  if (agentTasks.empty()) {
    return;
  }
  const auto& timeStamps = solution_.agents[agent].path.timeStamps;
  int usable = (int)agentTasks.size();
  if ((int)timeStamps.size() < usable) {
    usable = (int)timeStamps.size();
  }
  if (usable <= 0) {
    return;
  }
  int taskIdx = -1;
  for (int i = 0; i < usable; i++) {
    if (timeStamps[i] > timestep) {
      taskIdx = i;
      break;
    }
  }
  if (taskIdx < 0) {
    taskIdx = usable - 1;
  }
  const int task = solution_.getAgentGlobalTasks(agent, taskIdx);
  out->emplace(task, Conflicts(task, agent, taskIdx));
}

bool LNS::validateSolution(ConflictMap* conflictedTasks) {

  bool result = true;

  vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints();

  for (int task = 0; task < instance_.getTasksNum(); task++) {
    const int taskAgent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (taskAgent == UNASSIGNED) {
      PLOGE << "validateSolution: missing agent assignment for task "
            << task << "\n";
      return false;
    }
    int taskPosition = solution_.getLocalTaskIndex(taskAgent, task);
    if (taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[taskAgent].taskPaths.size()) {
      PLOGE << "validateSolution: invalid local index " << taskPosition
            << " for task " << task << " on agent " << taskAgent << "\n";
      return false;
    }
    if (solution_.agents[taskAgent].taskPaths[taskPosition].empty()) {
      PLOGE << "validateSolution: empty path for task " << task
            << " on agent " << taskAgent << " at local index "
            << taskPosition << "\n";
      result = false;
      return result;
    }
  }

  // Check that the precedence constraints are not violated
  for (const auto& precedenceConstraint : precedenceConstraints) {

    int agentA = solution_.getAgentWithTask(precedenceConstraint.first),
        agentB = solution_.getAgentWithTask(precedenceConstraint.second);
    if (agentA == UNASSIGNED || agentB == UNASSIGNED) {
      PLOGE << "validateSolution: missing agent for precedence pair ("
            << precedenceConstraint.first << ", "
            << precedenceConstraint.second << ")\n";
      return false;
    }
    int taskPositionA =
            solution_.getLocalTaskIndex(agentA, precedenceConstraint.first),
        taskPositionB =
            solution_.getLocalTaskIndex(agentB, precedenceConstraint.second);
    if (taskPositionA < 0 || taskPositionB < 0 ||
        taskPositionA >= (int)solution_.agents[agentA].path.timeStamps.size() ||
        taskPositionB >= (int)solution_.agents[agentB].path.timeStamps.size()) {
      PLOGE << "validateSolution: invalid local task index for precedence pair ("
            << precedenceConstraint.first << ", "
            << precedenceConstraint.second << ")\n";
      return false;
    }

    if (solution_.agents[agentA].path.timeStamps[taskPositionA] >=
        solution_.agents[agentB].path.timeStamps[taskPositionB]) {
      PLOGE << "Temporal conflict between agent " << agentA << " doing task "
            << precedenceConstraint.first << " and agent " << agentB
            << " doing task " << precedenceConstraint.second << "\n";
      result = false;
      if (conflictedTasks == nullptr) {
        return false;
      }
      Conflicts conflictA(precedenceConstraint.first, agentA, taskPositionA);
      Conflicts conflictB(precedenceConstraint.second, agentB, taskPositionB);
      conflictedTasks->emplace(conflictA.task, conflictA);
      conflictedTasks->emplace(conflictB.task, conflictB);
    }
  }

  vector<int> activeAgents;
  activeAgents.reserve(instance_.getAgentNum());
  int maxPathLength = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (solution_.agents[agent].taskAssignments.empty() ||
        solution_.agents[agent].path.empty()) {
      continue;
    }
    activeAgents.push_back(agent);
    maxPathLength = max(maxPathLength, (int)solution_.agents[agent].path.size());
  }

  auto getLocationAt = [this](int agent, int timestep) {
    const auto& path = solution_.agents[agent].path;
    if (path.empty()) {
      return UNDEFINED;
    }
    if (timestep < (int)path.size()) {
      return path.at(timestep).location;
    }
    // After path end, agent waits at its final location.
    return path.back().location;
  };

  auto edgeKey = [](int from, int to) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(from)) << 32) |
           static_cast<uint32_t>(to);
  };

  for (int timestep = 0; timestep < maxPathLength; timestep++) {
    // Vertex collisions at timestep.
    unordered_map<int, vector<int>> occupancy;
    occupancy.reserve(activeAgents.size() * 2);
    for (int agent : activeAgents) {
      const int location = getLocationAt(agent, timestep);
      if (location == UNDEFINED) {
        continue;
      }
      occupancy[location].push_back(agent);
    }
    for (const auto& it : occupancy) {
      const int location = it.first;
      const auto& agentsAtLocation = it.second;
      if (agentsAtLocation.size() < 2) {
        continue;
      }
      for (int i = 0; i < (int)agentsAtLocation.size(); i++) {
        for (int j = i + 1; j < (int)agentsAtLocation.size(); j++) {
          const int agentI = agentsAtLocation[i];
          const int agentJ = agentsAtLocation[j];
          pair<int, int> coord = instance_.getCoordinate(location);
          PLOGE << "Agents " << agentI << " and " << agentJ
                << " collide with each other at (" << coord.first << ", "
                << coord.second << ") at timestep " << timestep << "\n";
          result = false;
          if (conflictedTasks == nullptr) {
            return false;
          }
          addConflictingTask(agentI, timestep, conflictedTasks);
          addConflictingTask(agentJ, timestep, conflictedTasks);
        }
      }
    }

    // Edge swap collisions between timestep and timestep + 1.
    if (timestep + 1 >= maxPathLength) {
      continue;
    }
    unordered_map<uint64_t, int> directedEdgeOwner;
    directedEdgeOwner.reserve(activeAgents.size() * 2);
    for (int agent : activeAgents) {
      const int from = getLocationAt(agent, timestep);
      const int to = getLocationAt(agent, timestep + 1);
      if (from == UNDEFINED || to == UNDEFINED || from == to) {
        // Waiting cannot create a swap conflict.
        continue;
      }
      const uint64_t reverse = edgeKey(to, from);
      const auto reverseIt = directedEdgeOwner.find(reverse);
      if (reverseIt != directedEdgeOwner.end()) {
        const int otherAgent = reverseIt->second;
        pair<int, int> coordI = instance_.getCoordinate(from);
        pair<int, int> coordJ = instance_.getCoordinate(to);
        PLOGE << "Agents " << agent << " and " << otherAgent
              << " collide with each other at (" << coordI.first << ", "
              << coordI.second << ") --> (" << coordJ.first << ", "
              << coordJ.second << ") at timestep " << timestep << "\n";
        result = false;
        if (conflictedTasks == nullptr) {
          return false;
        }
        addConflictingTask(agent, timestep, conflictedTasks);
        addConflictingTask(otherAgent, timestep, conflictedTasks);
      }
      directedEdgeOwner[edgeKey(from, to)] = agent;
    }
  }
  return result;
}

void LNS::printPaths() const {
  for (int i = 0; i < instance_.getAgentNum(); i++) {
    const int agentPathSize = solution_.agents[i].path.endTimeOrZero();
    PLOGI << "Agent " << i << " (cost = " << agentPathSize << "):";
    std::string pathsLine;
    for (int t = 0; t < (int)solution_.agents[i].path.size(); t++) {
      pair<int, int> coord =
          instance_.getCoordinate(solution_.agents[i].path.at(t).location);
      pathsLine += "(" + std::to_string(coord.first) + ", " +
                   std::to_string(coord.second) + ")@" + std::to_string(t);
      if (solution_.agents[i].path.at(t).isGoal) {
        pathsLine += "*";
      }
      if (t != (int)solution_.agents[i].path.size() - 1) {
        pathsLine += " -> ";
      }
    }
    PLOGI << "\tPaths:\n\t" << pathsLine;
    std::string timestampsLine;
    for (int j = 0; j < (int)solution_.getAgentGlobalTasks(i).size(); j++) {
      pair<int, int> goalCoord = instance_.getCoordinate(
          solution_.agents[i].pathPlanner->goalLocations[j]);
      timestampsLine +=
          "(" + std::to_string(goalCoord.first) + ", " +
          std::to_string(goalCoord.second) + ")@" +
          std::to_string(solution_.agents[i].path.timeStamps[j]);
      if (j != (int)solution_.getAgentGlobalTasks(i).size() - 1) {
        timestampsLine += " -> ";
      }
    }
    PLOGI << "\tTimestamps:\n\t" << timestampsLine;
    std::string tasksLine;
    for (int j = 0; j < (int)solution_.getAgentGlobalTasks(i).size(); j++) {
      tasksLine += std::to_string(solution_.getAgentGlobalTasks(i)[j]);
      if (j != (int)solution_.getAgentGlobalTasks(i).size() - 1) {
        tasksLine += " -> ";
      }
    }
    PLOGI << "\tTasks:\n\t" << tasksLine;
  }
}

Solution& Solution::operator=(const Solution& other) {
  if (this == &other) {
    return *this;
  }
  this->numOfTasks = other.numOfTasks;
  this->numOfAgents = other.numOfAgents;
  this->sumOfCosts = other.sumOfCosts;
  this->utility = other.utility;
  this->taskAgentMap = other.taskAgentMap;
  if (this->agents.size() != other.agents.size()) {
    // Fallback for unexpected shape mismatch; preserves old behavior.
    this->agents = other.agents;
    return *this;
  }
  // Hot-path snapshot copies happen every LNS iteration. Use Agent::operator=
  // so planner state (goalLocations/heuristics/timeout) stays in sync with
  // taskAssignments and taskPaths.
  for (int i = 0; i < (int)this->agents.size(); i++) {
    this->agents[i] = other.agents[i];
  }

  return *this;
}
