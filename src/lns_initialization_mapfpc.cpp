#include "lns.hpp"
#include <boost/process.hpp>
#if defined(__has_include)
#if __has_include(<boost/process/null.hpp>)
#include <boost/process/null.hpp>
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 1
#else
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 0
#endif
#else
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 0
#endif
#include <cmath>
#include <deque>
#include <limits>
#include <filesystem>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <utility>

#include "common.hpp"
#include "lns_internal_helpers.hpp"
#include "utils.hpp"

bool LNS::buildGreedySolutionWithMAPFPC(const string& variant,
                                        int solverTimeoutSec) {

  // Reset solution state in case this is called more than once.
  Solution freshSolution(instance_);
  solution_ = freshSolution;

  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  bool readingTaskAssignments = false, readingTaskPaths = false;

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

  string solver;
  if (variant == "sota_cbs") {
    solver = "CBS";
  } else if (variant == "sota_pbs") {
    solver = "PBS";
  } else {
    PLOGE << "Initial solution solver using MAPF-PC variant not supported\n";
    return false;
  }

  // Run a child process to spawn the MAPC-PC codebase with the current map and agent informations
  namespace bp = boost::process;
  bp::ipstream inputStream;
  std::string taskAssignmentExe;
  if (std::filesystem::exists("./MAPF-PC/build_local/bin/task_assignment")) {
    taskAssignmentExe = "./MAPF-PC/build_local/bin/task_assignment";
  } else {
    taskAssignmentExe = "./MAPF-PC/build/bin/task_assignment";
  }
  const std::vector<std::string> args = {
      "-m", instance_.getMapName(),
      "-a", instance_.getAgentTaskFName(),
      "-k", std::to_string(instance_.getAgentNum()),
      "-t", std::to_string(max(1, solverTimeoutSec)),
      "-d", std::to_string(seed_),
      "--solver", solver,
  };
#if MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL
  bp::child child(taskAssignmentExe, bp::args(args), bp::std_out > inputStream,
                  bp::std_err > bp::null);
#else
  // Some Boost.Process installations don't ship <boost/process/null.hpp>.
  // In that case, don't suppress stderr.
  bp::child child(taskAssignmentExe, bp::args(args), bp::std_out > inputStream);
#endif

  // The output sequence of the MAPF-PC codebase is as follows:
  // 1. Output TASK ASSIGNMENTS
  // Output Agent # and then followed by the task sequences in order
  // 2. Output some other internal stuff
  // 3. Output TASK PATHS
  // Output Agent # and then followed by the task locations (non-linearized) with @ after the first location with begin time right after and -> between each location

  int agent = -1;
  string line;
  auto parseAgentHeader = [](const std::string& s) -> bool {
    // Expected format: "Agent <id>"
    // Avoid matching unrelated stderr/log lines that merely contain "Agent".
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

    // If the agent variable exceeds the total number of agents we are working with then we have read all the task assignments or all the task paths
    if (agent >= instance_.getAgentNum()) {
      if (readingTaskAssignments) {
        readingTaskAssignments = false;
      } else if (readingTaskPaths) {
        readingTaskPaths = false;
      }
    }

    // Check if the following set of lines will be for task assignments or task paths
    if (line == "TASK ASSIGNMENTS") {
      readingTaskAssignments = true;
      readingTaskPaths = false;
      agent = -1;
      continue;
    } else if (line == "TASK PATHS") {
      readingTaskAssignments = false;
      readingTaskPaths = true;
      agent = -1;
      continue;
    }

    // Use the Agent # to increment the agent variable
    if (parseAgentHeader(line)) {
      agent++;
    }
    // Otherwise if we are supposed to read the task assignments then we split that line using ',' as the delimiter and extract the tokens one by one
    // Eg: Agent 1
    //     1, 2, 3, 4,
    else if (readingTaskAssignments && agent > -1) {
      string token;
      size_t pos = 0;
      while ((pos = line.find(',')) != string::npos) {
        token = line.substr(0, pos);
        const auto task = parseInt(token);
        if (!task.has_value()) {
          PLOGE << "Failed to parse task assignment token: '" << token << "'\n";
          child.terminate();
          child.wait();
          return false;
        }
        solution_.assignTaskToAgent(agent, *task);
        line.erase(0, pos + 1);
      }
      // Handle the last token if the line doesn't end with a comma.
      if (const auto task = parseInt(line); task.has_value()) {
        solution_.assignTaskToAgent(agent, *task);
      }
      solution_.agents[agent].taskPaths.resize(
          solution_.agents[agent].taskAssignments.size(), AgentTaskPath());
    }
    // If we are not reading the task assignments then we must be reading the task paths.
    // Eg: Agent 1
    //     6 @ 0 -> 22 -> 23 @ 2 -> 24 @ 3 -> 25 -> 26 ->
    else if (readingTaskPaths && agent > -1) {
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
            // Mark the last location of the previous task as goal.
            taskPath.path.back().isGoal = true;
            solution_.agents[agent].taskPaths[taskIndex] = taskPath;
            initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
                taskPath;
            taskPath = AgentTaskPath();
          }
          taskIndex++;

          const size_t atPos = token.find('@');
          const auto loc = parseInt(token.substr(0, atPos));
          if (!loc.has_value()) {
            if (!hasDigit(token.substr(0, atPos))) {
              return;
            }
            PLOGE << "Failed to parse path location token: '" << token << "'\n";
            ok = false;
            return;
          }
          PathEntry pEntry = {false, *loc};

          if (taskIndex > 0) {
            if (taskIndex - 1 >=
                    (int)solution_.agents[agent].taskPaths.size() ||
                solution_.agents[agent].taskPaths[taskIndex - 1].empty()) {
              PLOGE << "buildGreedySolutionWithMAPFPC: previous task path "
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
              PLOGE << "Failed to parse task begin time token: '" << token
                    << "'\n";
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
            PLOGE << "Failed to parse path location token: '" << token << "'\n";
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

      // Process any trailing token after the last "->".
      consumeToken(line, taskPath, taskIndex);

      // Add the last task path.
      if (!taskPath.empty() && taskIndex >= 0) {
        taskPath.path.back().isGoal = true;
        solution_.agents[agent].taskPaths[taskIndex] = taskPath;
        initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
            taskPath;
      }
      if (!ok) {
        child.terminate();
        child.wait();
        return false;
      }
    }
  }

  child.wait();
  if (child.exit_code() != 0) {
    PLOGE << "MAPF-PC task_assignment exited with code " << child.exit_code()
          << "\n";
    return false;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(agent));
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolutionWithMAPFPC: failed to join agent paths\n";
    return false;
  }

  // Gather the information
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildGreedySolutionWithMAPFPC");
  return true;
}

