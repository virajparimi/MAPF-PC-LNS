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
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <filesystem>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <utility>

#include "common.hpp"
#include "lns_internal_helpers.hpp"
#include "utils.hpp"

namespace {
std::optional<std::filesystem::path> resolveTaskAssignmentExecutable() {
  namespace fs = std::filesystem;
  std::vector<fs::path> candidates;

  if (const char* envExe = std::getenv("MAPF_PC_TASK_ASSIGNMENT_EXE");
      envExe != nullptr && envExe[0] != '\0') {
    candidates.emplace_back(envExe);
  }

  // Relative to current working directory.
  candidates.emplace_back("./MAPF-PC/build_local/bin/task_assignment");
  candidates.emplace_back("./MAPF-PC/build/bin/task_assignment");

  // Relative to this binary location.
  std::error_code ec;
  const fs::path selfExe = fs::read_symlink("/proc/self/exe", ec);
  if (!ec && !selfExe.empty()) {
    const fs::path projectRoot = selfExe.parent_path().parent_path();
    if (!projectRoot.empty()) {
      candidates.emplace_back(projectRoot / "MAPF-PC/build_local/bin/task_assignment");
      candidates.emplace_back(projectRoot / "MAPF-PC/build/bin/task_assignment");
    }
  }

  for (const auto& candidate : candidates) {
    std::error_code candidateEc;
    if (candidate.empty() || !fs::exists(candidate, candidateEc) ||
        candidateEc || fs::is_directory(candidate, candidateEc)) {
      continue;
    }
    return fs::absolute(candidate, candidateEc);
  }
  return std::nullopt;
}

std::optional<int> parsePositiveEnvInt(const char* varName) {
  if (const char* raw = std::getenv(varName); raw != nullptr && raw[0] != '\0') {
    try {
      const int value = std::stoi(raw);
      if (value > 0) {
        return value;
      }
    } catch (...) {
    }
  }
  return std::nullopt;
}

std::optional<std::string> normalizeMapfpcSolverVariant(
    const std::string& solverVariant) {
  if (solverVariant == "sota_cbs" || solverVariant == "cbs" ||
      solverVariant == "CBS") {
    return std::string("CBS");
  }
  if (solverVariant == "sota_pbs" || solverVariant == "pbs" ||
      solverVariant == "PBS") {
    return std::string("PBS");
  }
  return std::nullopt;
}
}  // namespace

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

bool LNS::buildGreedySolutionWithMAPFPC(const string& variant,
                                        int solverTimeoutSec) {
  return runMAPFPCForAgentFile(instance_.getAgentTaskFName(), variant,
                               solverTimeoutSec,
                               "initialization:" + variant);
}

bool LNS::runMAPFPCForAgentFile(const string& agentFilePath,
                                const string& solverVariant,
                                int solverTimeoutSec,
                                const string& sourceLabel,
                                const string& fixedAssignmentFilePath) {
  const auto solver = normalizeMapfpcSolverVariant(solverVariant);
  if (!solver.has_value()) {
    PLOGE << "MAPF-PC solver variant not supported: '" << solverVariant
          << "'\n";
    return false;
  }

  namespace bp = boost::process;
  namespace fs = std::filesystem;
  const auto taskAssignmentExe = resolveTaskAssignmentExecutable();
  if (!taskAssignmentExe.has_value()) {
    PLOGE << "MAPF-PC task_assignment executable not found. "
          << "Expected under ./MAPF-PC/build_local/bin or ./MAPF-PC/build/bin, "
          << "or set MAPF_PC_TASK_ASSIGNMENT_EXE.\n";
    return false;
  }
  const int effectiveSolverTimeoutSec = max(1, solverTimeoutSec);
  int wallClockTimeoutSec = max(10, effectiveSolverTimeoutSec + 30);
  if (const auto timeoutFromEnv =
          parsePositiveEnvInt("MAPF_PC_TASK_ASSIGNMENT_WALL_TIMEOUT_SEC");
      timeoutFromEnv.has_value()) {
    wallClockTimeoutSec = *timeoutFromEnv;
  }

  std::error_code tempDirEc;
  fs::path tempDir = fs::temp_directory_path(tempDirEc);
  if (tempDirEc || tempDir.empty()) {
    tempDir = fs::current_path(tempDirEc);
  }
  if (tempDirEc || tempDir.empty()) {
    PLOGE << "Failed to resolve temporary directory for MAPF-PC stdout capture\n";
    return false;
  }
  const fs::path stdoutCapturePath =
      tempDir /
      ("mapf_pc_task_assignment_" + std::to_string(seed_) + "_" +
       std::to_string((long long)Time::now().time_since_epoch().count()) +
       ".log");
  auto cleanupStdoutCapture = [&stdoutCapturePath]() {
    std::error_code removeEc;
    fs::remove(stdoutCapturePath, removeEc);
  };

  std::vector<std::string> args = {
      "-m", instance_.getMapName(),
      "-a", agentFilePath,
      "-k", std::to_string(instance_.getAgentNum()),
      "-t", std::to_string(effectiveSolverTimeoutSec),
      "-d", std::to_string(seed_),
      "--solver", *solver,
  };
  if (!fixedAssignmentFilePath.empty()) {
    args.push_back("--fixedAssignmentFile");
    args.push_back(fixedAssignmentFilePath);
  }
  std::optional<bp::child> childProcess;
  try {
#if MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL
    childProcess.emplace(taskAssignmentExe->string(), bp::args(args),
                         bp::std_out > stdoutCapturePath.string(),
                         bp::std_err > bp::null);
#else
    // Some Boost.Process installations don't ship <boost/process/null.hpp>.
    // In that case, don't suppress stderr.
    childProcess.emplace(taskAssignmentExe->string(), bp::args(args),
                         bp::std_out > stdoutCapturePath.string());
#endif
  } catch (const bp::process_error& e) {
    PLOGE << "Failed to launch MAPF-PC task_assignment at '"
          << taskAssignmentExe->string() << "': " << e.what() << "\n";
    cleanupStdoutCapture();
    return false;
  }

  if (!sourceLabel.empty()) {
    PLOGI << "MAPF-PC task_assignment start: solver=" << *solver
          << ", timeout_sec=" << effectiveSolverTimeoutSec
          << ", source=" << sourceLabel << "\n";
  }
  bp::child& child = *childProcess;
  const auto wallClockDeadline =
      std::chrono::steady_clock::now() +
      std::chrono::seconds(wallClockTimeoutSec);
  while (child.running() && std::chrono::steady_clock::now() < wallClockDeadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (child.running()) {
    PLOGE << "MAPF-PC task_assignment exceeded wall-clock timeout ("
          << wallClockTimeoutSec << "s); terminating process\n";
    child.terminate();
    child.wait();
    cleanupStdoutCapture();
    return false;
  }
  child.wait();
  if (child.exit_code() != 0) {
    PLOGE << "MAPF-PC task_assignment exited with code " << child.exit_code()
          << "\n";
    cleanupStdoutCapture();
    return false;
  }
  std::ifstream inputStream(stdoutCapturePath);
  if (!inputStream.is_open()) {
    PLOGE << "Failed to open captured MAPF-PC stdout at '"
          << stdoutCapturePath.string() << "'\n";
    cleanupStdoutCapture();
    return false;
  }

  const string parseSourceLabel =
      sourceLabel.empty() ? stdoutCapturePath.string() : sourceLabel;
  const bool parsed =
      parseMAPFPCStreamIntoSolution(inputStream, parseSourceLabel);
  inputStream.close();
  cleanupStdoutCapture();
  return parsed;
}

bool LNS::runMAPFPCOnAssignments(const vector<vector<int>>& assignments,
                                 const string& solverVariant,
                                 int solverTimeoutSec,
                                 const string& sourceLabel) {
  namespace fs = std::filesystem;
  std::error_code tempDirEc;
  fs::path tempDir = fs::temp_directory_path(tempDirEc);
  if (tempDirEc || tempDir.empty()) {
    tempDir = fs::current_path(tempDirEc);
  }
  if (tempDirEc || tempDir.empty()) {
    PLOGE << "Failed to resolve temporary directory for MAPF-PC assignment "
             "capture\n";
    return false;
  }

  const fs::path assignmentPath =
      tempDir /
      ("mapf_pc_assignment_" + std::to_string(seed_) + "_" +
       std::to_string((long long)Time::now().time_since_epoch().count()) +
       ".txt");

  string assignmentError;
  if (!writeMAPFPCAssignmentFile(assignments, assignmentPath.string(),
                                 assignmentError)) {
    PLOGE << "Failed to write MAPF-PC assignment file: " << assignmentError
          << "\n";
    std::error_code removeEc;
    fs::remove(assignmentPath, removeEc);
    return false;
  }

  const bool success = runMAPFPCForAgentFile(
      instance_.getAgentTaskFName(), solverVariant, solverTimeoutSec,
      sourceLabel.empty() ? assignmentPath.string() : sourceLabel,
      assignmentPath.string());
  std::error_code removeEc;
  fs::remove(assignmentPath, removeEc);
  return success;
}

bool LNS::buildSeededSolutionFromMAPFPCLog(const string& logFilePath) {
  if (logFilePath.empty()) {
    PLOGE << "buildSeededSolutionFromMAPFPCLog: empty log path\n";
    return false;
  }
  std::ifstream inputStream(logFilePath);
  if (!inputStream.is_open()) {
    PLOGE << "buildSeededSolutionFromMAPFPCLog: failed to open '" << logFilePath
          << "'\n";
    return false;
  }
  return parseMAPFPCStreamIntoSolution(inputStream, logFilePath);
}

bool LNS::runPostMAPFPCRefinement() {
  vector<vector<int>> assignments(instance_.getAgentNum());
  string assignmentSource = postRefineAssignmentSource_;

  if (assignmentSource == "log") {
    if (postRefineAssignmentLog_.empty()) {
      PLOGE << "post_refine_mapfpc: assignment source is 'log' but no log path "
               "was provided\n";
      return false;
    }
    std::ifstream inputStream(postRefineAssignmentLog_);
    if (!inputStream.is_open()) {
      PLOGE << "post_refine_mapfpc: failed to open assignment log '"
            << postRefineAssignmentLog_ << "'\n";
      return false;
    }
    if (!parseMAPFPCAssignmentsFromStream(inputStream, postRefineAssignmentLog_,
                                          assignments)) {
      PLOGE << "post_refine_mapfpc: failed to parse assignments from '"
            << postRefineAssignmentLog_ << "'\n";
      return false;
    }
  } else {
    bool usedIncumbent = false;
    if ((int)incumbentSolution_.agentTaskAssignments.size() ==
        instance_.getAgentNum()) {
      assignments = incumbentSolution_.agentTaskAssignments;
      usedIncumbent = true;
    } else {
      for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
        assignments[agent] = solution_.agents[agent].taskAssignments;
      }
    }
    assignmentSource = usedIncumbent ? "incumbent_solution" : "current_solution";
  }

  const Solution previousSolution = solution_;
  const FeasibleSolution previousIncumbent = incumbentSolution_;
  const int baselineSoc =
      previousIncumbent.agentPaths.empty() ? previousSolution.sumOfCosts
                                           : previousIncumbent.sumOfCosts;

  const string solverVariant = postRefineSolver_;
  const string sourceLabel =
      "post_refine_mapfpc(" + solverVariant + "," + assignmentSource + ")";
  if (!runMAPFPCOnAssignments(assignments, solverVariant, postRefineTimeoutSec_,
                              sourceLabel)) {
    solution_ = previousSolution;
    incumbentSolution_ = previousIncumbent;
    return false;
  }

  if (goalOccupationMode_ == "reposition_true") {
    vector<int> allAgents(instance_.getAgentNum());
    std::iota(allAgents.begin(), allAgents.end(), 0);
    if (!planTerminalReposition(allAgents, true)) {
      PLOGE << "post_refine_mapfpc: terminal reposition planning failed\n";
      solution_ = previousSolution;
      incumbentSolution_ = previousIncumbent;
      return false;
    }
  }

  ConflictMap potentialNeighborhood;
  ValidationStats validationStats;
  const bool previousTerminalValidationFlag = useTerminalPathsInValidation_;
  useTerminalPathsInValidation_ = (goalOccupationMode_ == "reposition_true");
  const bool valid = validateSolution(&potentialNeighborhood, &validationStats);
  useTerminalPathsInValidation_ = previousTerminalValidationFlag;
  if (!valid) {
    PLOGE << "post_refine_mapfpc: refined solution failed validation\n";
    solution_ = previousSolution;
    incumbentSolution_ = previousIncumbent;
    return false;
  }

  const int refinedSoc = solution_.sumOfCosts;
  const bool strictlyBetter =
      previousIncumbent.agentPaths.empty() ||
      refinedSoc < previousIncumbent.sumOfCosts;
  if (postRefineAcceptOnlyIfBetter_ && !strictlyBetter) {
    PLOGI << "post_refine_mapfpc: rejected non-improving refinement (baseline="
          << baselineSoc << ", refined=" << refinedSoc << ")\n";
    solution_ = previousSolution;
    incumbentSolution_ = previousIncumbent;
    return false;
  }

  if (strictlyBetter) {
    extractFeasibleSolution();
  } else {
    overwriteIncumbentFromCurrentSolution();
  }
  runtime = elapsedRuntimeSec();
  appendIterationStatBounded(IterationStats(
      runtime, sourceLabel, instance_.getAgentNum(), instance_.getTasksNum(),
      refinedSoc, true,
      strictlyBetter ? IterationQuality::bestSolutionYet
                     : IterationQuality::improvedSolution));
  PLOGI << "post_refine_mapfpc: accepted refinement (baseline=" << baselineSoc
        << ", refined=" << refinedSoc
        << ", accept_only_if_better="
        << (postRefineAcceptOnlyIfBetter_ ? "true" : "false") << ")\n";
  return true;
}
