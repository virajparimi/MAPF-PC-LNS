#include "lns.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

#include "lns_internal_helpers.hpp"

bool LNS::parseMAPFPCStreamIntoSolution(std::istream& inputStream,
                                        const string& sourceLabel) {

  // Reset solution state in case this is called more than once.
  Solution freshSolution(instance_);
  solution_ = freshSolution;
  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());

  bool readingTaskAssignments = false, readingTaskPaths = false;
  int agent = -1;
  string line;

  auto parseInt = [](const std::string& s) -> std::optional<int> {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      return std::nullopt;
    }
    const size_t end = s.find_last_not_of(" \t\r\n");
    try {
      return std::stoi(s.substr(begin, end - begin + 1));
    } catch (...) {
      return std::nullopt;
    }
  };

  auto parseAgentHeader = [](const std::string& s) -> bool {
    // Expected format: "Agent <id>".
    std::istringstream iss(s);
    std::string tag;
    int id = -1;
    if (!(iss >> tag >> id)) {
      return false;
    }
    return tag == "Agent";
  };

  while (std::getline(inputStream, line)) {
    if (line.empty()) {
      continue;
    }

    if (agent >= instance_.getAgentNum()) {
      if (readingTaskAssignments) {
        readingTaskAssignments = false;
      } else if (readingTaskPaths) {
        readingTaskPaths = false;
      }
    }

    if (line == "TASK ASSIGNMENTS") {
      readingTaskAssignments = true;
      readingTaskPaths = false;
      agent = -1;
      continue;
    }
    if (line == "TASK PATHS") {
      readingTaskAssignments = false;
      readingTaskPaths = true;
      agent = -1;
      continue;
    }

    if (parseAgentHeader(line)) {
      agent++;
      continue;
    }

    if (readingTaskAssignments && agent > -1) {
      string token;
      size_t pos = 0;
      while ((pos = line.find(',')) != string::npos) {
        token = line.substr(0, pos);
        const auto task = parseInt(token);
        if (!task.has_value()) {
          PLOGE << "Failed to parse task assignment token from '" << sourceLabel
                << "': '" << token << "'\n";
          return false;
        }
        solution_.assignTaskToAgent(agent, *task);
        line.erase(0, pos + 1);
      }
      if (const auto task = parseInt(line); task.has_value()) {
        solution_.assignTaskToAgent(agent, *task);
      }
      solution_.agents[agent].taskPaths.resize(
          solution_.agents[agent].taskAssignments.size(), AgentTaskPath());
      continue;
    }

    if (readingTaskPaths && agent > -1) {
      const int expectedTaskCount =
          (agent >= 0 && agent < instance_.getAgentNum())
              ? (int)solution_.agents[agent].taskAssignments.size()
              : 0;
      // Some MAPF-PC runs emit an idle "Agent X" path line even when that
      // agent has zero assigned tasks in TASK ASSIGNMENTS. Ignore such lines:
      // there are no task segments to import for this agent.
      if (expectedTaskCount == 0) {
        continue;
      }

      bool ok = true;
      auto isWhitespaceOnly = [](const std::string& s) -> bool {
        return s.find_first_not_of(" \t\r\n") == std::string::npos;
      };
      auto hasDigit = [](const std::string& s) -> bool {
        return s.find_first_of("0123456789") != std::string::npos;
      };
      auto consumeToken = [&](const std::string& rawToken, AgentTaskPath& taskPath,
                              int& taskIndex) {
        if (rawToken.empty() || isWhitespaceOnly(rawToken)) {
          return;
        }
        string token = rawToken;
        if (isWhitespaceOnly(token)) {
          return;
        }
        if (token.find('@') != string::npos) {
          if (!taskPath.empty()) {
            taskPath.path.back().isGoal = true;
            if (taskIndex < 0 || taskIndex >= expectedTaskCount) {
              PLOGE << "parseMAPFPCStreamIntoSolution: task segment index out "
                       "of range while finalizing previous segment for agent "
                    << agent << " (taskIndex=" << taskIndex
                    << ", expectedTaskCount=" << expectedTaskCount << ") from '"
                    << sourceLabel << "'\n";
              ok = false;
              return;
            }
            solution_.agents[agent].taskPaths[taskIndex] = taskPath;
            initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
                taskPath;
            taskPath = AgentTaskPath();
          }
          taskIndex++;
          if (taskIndex < 0 || taskIndex >= expectedTaskCount) {
            PLOGE << "parseMAPFPCStreamIntoSolution: task segment index out "
                     "of range while starting new segment for agent "
                  << agent << " (taskIndex=" << taskIndex
                  << ", expectedTaskCount=" << expectedTaskCount << ") from '"
                  << sourceLabel << "'\n";
            ok = false;
            return;
          }

          const size_t atPos = token.find('@');
          const auto loc = parseInt(token.substr(0, atPos));
          if (!loc.has_value()) {
            if (!hasDigit(token.substr(0, atPos))) {
              return;
            }
            PLOGE << "Failed to parse path location token from '" << sourceLabel
                  << "': '" << token << "'\n";
            ok = false;
            return;
          }
          PathEntry pEntry = {false, *loc};

          if (taskIndex > 0) {
            if (taskIndex - 1 >=
                    (int)solution_.agents[agent].taskPaths.size() ||
                solution_.agents[agent].taskPaths[taskIndex - 1].empty()) {
              PLOGE << "parseMAPFPCStreamIntoSolution: previous task path "
                       "missing/empty for agent "
                    << agent << ", taskIndex " << taskIndex << "\n";
              ok = false;
              return;
            }
            const int previousLocation =
                solution_.agents[agent].taskPaths[taskIndex - 1]
                    .back()
                    .location;
            taskPath.path.push_back(PathEntry{false, previousLocation});
          }
          taskPath.path.push_back(pEntry);

          token.erase(0, atPos + 1);
          if (taskIndex > 0) {
            const auto beginTime = parseInt(token);
            if (!beginTime.has_value()) {
              PLOGE << "Failed to parse task begin time token from '"
                    << sourceLabel << "': '" << token << "'\n";
              ok = false;
              return;
            }
            taskPath.beginTime = *beginTime - 1;
          } else {
            taskPath.beginTime = 0;
          }
        } else {
          const auto loc = parseInt(token);
          if (!loc.has_value()) {
            if (!hasDigit(token)) {
              return;
            }
            PLOGE << "Failed to parse path location token from '" << sourceLabel
                  << "': '" << token << "'\n";
            ok = false;
            return;
          }
          taskPath.path.push_back(PathEntry{false, *loc});
        }
      };

      AgentTaskPath taskPath;
      size_t pos = 0;
      int taskIndex = -1;
      while ((pos = line.find("->")) != string::npos) {
        consumeToken(line.substr(0, pos), taskPath, taskIndex);
        line.erase(0, pos + 2);
      }
      consumeToken(line, taskPath, taskIndex);

      if (!taskPath.empty() && taskIndex >= 0) {
        if (taskIndex >= expectedTaskCount) {
          PLOGE << "parseMAPFPCStreamIntoSolution: task segment index out of "
                   "range at line end for agent "
                << agent << " (taskIndex=" << taskIndex
                << ", expectedTaskCount=" << expectedTaskCount << ") from '"
                << sourceLabel << "'\n";
          return false;
        }
        taskPath.path.back().isGoal = true;
        solution_.agents[agent].taskPaths[taskIndex] = taskPath;
        initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
            taskPath;
      }
      if (!ok) {
        return false;
      }
    }
  }

  for (int currentAgent = 0; currentAgent < instance_.getAgentNum();
       currentAgent++) {
    for (int taskIdx = 0;
         taskIdx < (int)solution_.agents[currentAgent].taskAssignments.size();
         taskIdx++) {
      if (solution_.agents[currentAgent].taskPaths[taskIdx].empty()) {
        PLOGE << "parseMAPFPCStreamIntoSolution: missing task segment for agent "
              << currentAgent << ", local task " << taskIdx << " from '"
              << sourceLabel << "'\n";
        return false;
      }
    }
    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(currentAgent));
    solution_.agents[currentAgent].pathPlanner->setGoalLocations(taskLocations);
    for (int task = 1;
         task < (int)solution_.getAgentGlobalTasks(currentAgent).size();
         task++) {
      solution_.agents[currentAgent].insertPrecedenceConstraint(
          solution_.agents[currentAgent].taskAssignments[task - 1],
          solution_.agents[currentAgent].taskAssignments[task]);
    }
  }

  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "parseMAPFPCStreamIntoSolution: failed to join agent paths from '"
          << sourceLabel << "'\n";
    return false;
  }

  long long initialSumOfCosts = 0;
  for (int currentAgent = 0; currentAgent < instance_.getAgentNum();
       currentAgent++) {
    initialSumOfCosts += static_cast<long long>(
        solution_.agents[currentAgent].path.endTimeOrZero());
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "parseMAPFPCStreamIntoSolution");
  return true;
}

bool LNS::parseMAPFPCAssignmentsFromStream(
    std::istream& inputStream, const string& sourceLabel,
    vector<vector<int>>& outAssignments) const {
  outAssignments.assign(instance_.getAgentNum(), {});

  bool readingTaskAssignments = false;
  int agent = -1;
  string line;

  auto parseInt = [](const std::string& s) -> std::optional<int> {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      return std::nullopt;
    }
    const size_t end = s.find_last_not_of(" \t\r\n");
    try {
      return std::stoi(s.substr(begin, end - begin + 1));
    } catch (...) {
      return std::nullopt;
    }
  };

  auto parseAgentHeader = [](const std::string& s) -> bool {
    std::istringstream iss(s);
    std::string tag;
    int id = -1;
    if (!(iss >> tag >> id)) {
      return false;
    }
    return tag == "Agent";
  };

  while (std::getline(inputStream, line)) {
    if (line.empty()) {
      continue;
    }
    if (line == "TASK ASSIGNMENTS") {
      readingTaskAssignments = true;
      agent = -1;
      continue;
    }
    if (line == "TASK PATHS") {
      if (readingTaskAssignments) {
        break;
      }
      continue;
    }
    if (line == "Feasible Solution") {
      continue;
    }
    if (!readingTaskAssignments) {
      continue;
    }
    if (parseAgentHeader(line)) {
      agent++;
      continue;
    }
    if (agent < 0 || agent >= instance_.getAgentNum()) {
      continue;
    }
    string token;
    size_t pos = 0;
    while ((pos = line.find(',')) != string::npos) {
      token = line.substr(0, pos);
      if (const auto task = parseInt(token); task.has_value()) {
        outAssignments[agent].push_back(*task);
      } else {
        PLOGE << "Failed to parse assignment token from '" << sourceLabel
              << "': '" << token << "'\n";
        return false;
      }
      line.erase(0, pos + 1);
    }
    if (const auto task = parseInt(line); task.has_value()) {
      outAssignments[agent].push_back(*task);
    }
  }

  vector<int> taskOwner(instance_.getTasksNum(), UNASSIGNED);
  int assignedCount = 0;
  for (int a = 0; a < instance_.getAgentNum(); a++) {
    for (int task : outAssignments[a]) {
      if (task < 0 || task >= instance_.getTasksNum()) {
        PLOGE << "Assignment task id out of range in '" << sourceLabel
              << "': task=" << task << "\n";
        return false;
      }
      if (taskOwner[task] != UNASSIGNED) {
        PLOGE << "Duplicate assignment for task " << task << " in '"
              << sourceLabel << "' (agents " << taskOwner[task] << " and " << a
              << ")\n";
        return false;
      }
      taskOwner[task] = a;
      assignedCount++;
    }
  }
  if (assignedCount != instance_.getTasksNum()) {
    PLOGE << "Incomplete assignments in '" << sourceLabel << "': assigned "
          << assignedCount << "/" << instance_.getTasksNum() << " tasks\n";
    return false;
  }
  return true;
}

bool LNS::writeMAPFPCAssignmentFile(const vector<vector<int>>& assignments,
                                    const string& outputPath,
                                    string& errorMessage) const {
  errorMessage.clear();
  if ((int)assignments.size() != instance_.getAgentNum()) {
    errorMessage = "assignment vector size does not match agent count";
    return false;
  }

  vector<int> taskAgent(instance_.getTasksNum(), UNASSIGNED);
  vector<int> taskLocalIndex(instance_.getTasksNum(), UNASSIGNED);
  int assignedCount = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localIdx = 0; localIdx < (int)assignments[agent].size();
         localIdx++) {
      const int task = assignments[agent][localIdx];
      if (task < 0 || task >= instance_.getTasksNum()) {
        errorMessage = "task id out of range in assignments";
        return false;
      }
      if (taskAgent[task] != UNASSIGNED) {
        errorMessage = "duplicate task assignment detected";
        return false;
      }
      taskAgent[task] = agent;
      taskLocalIndex[task] = localIdx;
      assignedCount++;
    }
  }
  if (assignedCount != instance_.getTasksNum()) {
    errorMessage = "not all tasks are assigned";
    return false;
  }

  std::ofstream out(outputPath);
  if (!out.is_open()) {
    errorMessage = "failed to open output file";
    return false;
  }

  out << instance_.getAgentNum() << " # number of agents\n";
  out << "# Format:  num_of_goals sx sy g1x g1y g2x g2y ...\n";
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto startCoord =
        instance_.getCoordinate(instance_.getStartLocationsRef()[agent]);
    out << assignments[agent].size() << "\t" << startCoord.second << "\t"
        << startCoord.first << "\t";
    for (int task : assignments[agent]) {
      const int taskLocation = instance_.getTaskLocations(task);
      const auto taskCoord = instance_.getCoordinate(taskLocation);
      out << taskCoord.second << "\t" << taskCoord.first << "\t";
    }
    out << "\n";
  }

  out << "temporal cons:\n";
  const auto& precedence = instance_.getInputPrecedenceConstraintsRef();
  for (const auto& edge : precedence) {
    const int predecessor = edge.first;
    const int successor = edge.second;
    if (predecessor < 0 || predecessor >= instance_.getTasksNum() ||
        successor < 0 || successor >= instance_.getTasksNum()) {
      errorMessage = "precedence task id out of range";
      return false;
    }
    const int predAgent = taskAgent[predecessor];
    const int succAgent = taskAgent[successor];
    const int predLocal = taskLocalIndex[predecessor];
    const int succLocal = taskLocalIndex[successor];
    if (predAgent == UNASSIGNED || succAgent == UNASSIGNED ||
        predLocal == UNASSIGNED || succLocal == UNASSIGNED) {
      errorMessage = "precedence edge references unassigned task";
      return false;
    }
    out << predAgent << "\t" << predLocal << "\t" << succAgent << "\t"
        << succLocal << "\n";
  }

  out.flush();
  if (!out.good()) {
    errorMessage = "write failure while saving assignment file";
    return false;
  }
  return true;
}
