#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"

namespace {
const char* occupancySourceName(LNS::OccupancySource source) {
  switch (source) {
    case LNS::OccupancySource::service:
      return "service";
    case LNS::OccupancySource::terminal:
      return "terminal";
    case LNS::OccupancySource::undefined:
    default:
      return "undefined";
  }
}

const char* occupancyPairLabel(LNS::OccupancySource lhs,
                               LNS::OccupancySource rhs) {
  if (lhs == LNS::OccupancySource::terminal &&
      rhs == LNS::OccupancySource::terminal) {
    return "terminal-vs-terminal";
  }
  if ((lhs == LNS::OccupancySource::service &&
       rhs == LNS::OccupancySource::terminal) ||
      (lhs == LNS::OccupancySource::terminal &&
       rhs == LNS::OccupancySource::service)) {
    return "service-vs-terminal";
  }
  if (lhs == LNS::OccupancySource::service &&
      rhs == LNS::OccupancySource::service) {
    return "service-vs-service";
  }
  return "unknown";
}
}  // namespace

int LNS::getServiceOccupancyEndExclusive(int agent) const {
  if (agent < 0 || agent >= instance_.getAgentNum()) {
    return 0;
  }
  const auto& servicePath = solution_.agents[agent].path;
  if (servicePath.empty()) {
    return 0;
  }
  if (goalOccupationMode_ == "stay") {
    return MAX_TIMESTEP;
  }
  int endExclusive = (int)servicePath.size();
  if (!solution_.agents[agent].taskAssignments.empty() && goalTailSteps_ > 0) {
    if (endExclusive >= MAX_TIMESTEP - goalTailSteps_) {
      return MAX_TIMESTEP;
    }
    endExclusive += goalTailSteps_;
  }
  return endExclusive;
}

LNS::OccupancySource LNS::getAgentOccupancySourceAt(
    int agent, int timestep, bool includeTerminal) const {
  if (agent < 0 || agent >= instance_.getAgentNum() || timestep < 0) {
    return OccupancySource::undefined;
  }
  const auto& servicePath = solution_.agents[agent].path;
  if (servicePath.empty()) {
    return OccupancySource::undefined;
  }
  if (timestep < (int)servicePath.size()) {
    return OccupancySource::service;
  }
  if (timestep < getServiceOccupancyEndExclusive(agent)) {
    return OccupancySource::service;
  }

  if (includeTerminal && solution_.agents[agent].terminalPathActive) {
    const auto& terminalPath = solution_.agents[agent].terminalPath;
    if (!terminalPath.empty()) {
      if (timestep < terminalPath.beginTime) {
        return OccupancySource::undefined;
      }
      return OccupancySource::terminal;
    }
  }
  return OccupancySource::undefined;
}

int LNS::getAgentLocationAt(int agent, int timestep, bool includeTerminal) const {
  if (agent < 0 || agent >= instance_.getAgentNum() || timestep < 0) {
    return UNDEFINED;
  }
  const auto& servicePath = solution_.agents[agent].path;
  if (servicePath.empty()) {
    return UNDEFINED;
  }
  if (timestep < (int)servicePath.size()) {
    return servicePath.at(timestep).location;
  }
  if (timestep < getServiceOccupancyEndExclusive(agent)) {
    return servicePath.back().location;
  }

  if (includeTerminal && solution_.agents[agent].terminalPathActive) {
    const auto& terminalPath = solution_.agents[agent].terminalPath;
    if (!terminalPath.empty()) {
      if (timestep < terminalPath.beginTime) {
        return UNDEFINED;
      }
      const int terminalOffset = timestep - terminalPath.beginTime;
      if (terminalOffset >= 0 && terminalOffset < (int)terminalPath.size()) {
        return terminalPath.at(terminalOffset).location;
      }
      return terminalPath.back().location;
    }
  }
  return UNDEFINED;
}

int LNS::getAgentOccupancyHorizon(int agent, bool includeTerminal) const {
  if (agent < 0 || agent >= instance_.getAgentNum()) {
    return 0;
  }
  const auto& servicePath = solution_.agents[agent].path;
  if (servicePath.empty()) {
    return 0;
  }
  int horizon = (goalOccupationMode_ == "stay")
                    ? (int)servicePath.size()
                    : getServiceOccupancyEndExclusive(agent);
  if (includeTerminal && solution_.agents[agent].terminalPathActive) {
    const auto& terminalPath = solution_.agents[agent].terminalPath;
    if (!terminalPath.empty()) {
      horizon = max(horizon, terminalPath.endTimeOrZero() + 1);
    }
  }
  return horizon;
}

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
  const int taskCount = instance_.getTasksNum();
  const vector<int> taskToPosition =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);

  vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints();

  for (int task = 0; task < taskCount; task++) {
    const int taskAgent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (taskAgent == UNASSIGNED) {
      PLOGE << "validateSolution: missing agent assignment for task "
            << task << "\n";
      return false;
    }
    int taskPosition = (task >= 0 && task < (int)taskToPosition.size())
                           ? taskToPosition[task]
                           : UNASSIGNED;
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
    int taskPositionA = (precedenceConstraint.first >= 0 &&
                         precedenceConstraint.first < (int)taskToPosition.size())
                            ? taskToPosition[precedenceConstraint.first]
                            : UNASSIGNED,
        taskPositionB = (precedenceConstraint.second >= 0 &&
                         precedenceConstraint.second < (int)taskToPosition.size())
                            ? taskToPosition[precedenceConstraint.second]
                            : UNASSIGNED;
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
    maxPathLength = max(maxPathLength, getAgentOccupancyHorizon(
                                           agent, useTerminalPathsInValidation_));
  }

  auto edgeKey = [](int from, int to) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(from)) << 32) |
           static_cast<uint32_t>(to);
  };

  unordered_map<int, vector<int>> occupancy;
  occupancy.reserve(activeAgents.size() * 2);
  unordered_map<uint64_t, int> directedEdgeOwner;
  directedEdgeOwner.reserve(activeAgents.size() * 2);

  for (int timestep = 0; timestep < maxPathLength; timestep++) {
    // Vertex collisions at timestep.
    occupancy.clear();
    for (int agent : activeAgents) {
      const int location = getAgentLocationAt(
          agent, timestep, useTerminalPathsInValidation_);
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
          const OccupancySource sourceI = getAgentOccupancySourceAt(
              agentI, timestep, useTerminalPathsInValidation_);
          const OccupancySource sourceJ = getAgentOccupancySourceAt(
              agentJ, timestep, useTerminalPathsInValidation_);
          pair<int, int> coord = instance_.getCoordinate(location);
          PLOGE << "Agents " << agentI << " and " << agentJ
                << " collide with each other at (" << coord.first << ", "
                << coord.second << ") at timestep " << timestep
                << " [type=" << occupancyPairLabel(sourceI, sourceJ)
                << ", a" << agentI << "=" << occupancySourceName(sourceI)
                << ", a" << agentJ << "=" << occupancySourceName(sourceJ)
                << "]\n";
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
    directedEdgeOwner.clear();
    for (int agent : activeAgents) {
      const int from = getAgentLocationAt(
          agent, timestep, useTerminalPathsInValidation_);
      const int to = getAgentLocationAt(
          agent, timestep + 1, useTerminalPathsInValidation_);
      if (from == UNDEFINED || to == UNDEFINED || from == to) {
        // Waiting cannot create a swap conflict.
        continue;
      }
      const uint64_t reverse = edgeKey(to, from);
      const auto reverseIt = directedEdgeOwner.find(reverse);
      if (reverseIt != directedEdgeOwner.end()) {
        const int otherAgent = reverseIt->second;
        const OccupancySource sourceAFrom = getAgentOccupancySourceAt(
            agent, timestep, useTerminalPathsInValidation_);
        const OccupancySource sourceATo = getAgentOccupancySourceAt(
            agent, timestep + 1, useTerminalPathsInValidation_);
        const OccupancySource sourceBFrom = getAgentOccupancySourceAt(
            otherAgent, timestep, useTerminalPathsInValidation_);
        const OccupancySource sourceBTo = getAgentOccupancySourceAt(
            otherAgent, timestep + 1, useTerminalPathsInValidation_);
        const OccupancySource sourceA =
            (sourceAFrom == OccupancySource::terminal ||
             sourceATo == OccupancySource::terminal)
                ? OccupancySource::terminal
                : sourceAFrom;
        const OccupancySource sourceB =
            (sourceBFrom == OccupancySource::terminal ||
             sourceBTo == OccupancySource::terminal)
                ? OccupancySource::terminal
                : sourceBFrom;
        pair<int, int> coordI = instance_.getCoordinate(from);
        pair<int, int> coordJ = instance_.getCoordinate(to);
        PLOGE << "Agents " << agent << " and " << otherAgent
              << " collide with each other at (" << coordI.first << ", "
              << coordI.second << ") --> (" << coordJ.first << ", "
              << coordJ.second << ") at timestep " << timestep
              << " [type=" << occupancyPairLabel(sourceA, sourceB)
              << ", a" << agent << "=" << occupancySourceName(sourceA)
              << ", a" << otherAgent << "=" << occupancySourceName(sourceB)
              << "]\n";
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
