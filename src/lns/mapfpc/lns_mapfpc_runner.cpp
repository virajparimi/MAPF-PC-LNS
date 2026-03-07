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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <vector>

#include "lns_mapfpc_internal.hpp"
#include "lns_temp_file_guard.hpp"

using lns_mapfpc_internal::normalizeMapfpcSolverVariant;
using lns_mapfpc_internal::parsePositiveEnvInt;
using lns_mapfpc_internal::resolveTaskAssignmentExecutable;

bool LNS::buildGreedySolutionWithMAPFPC(const string& variant,
                                        int solverTimeoutSec) {
  return runMAPFPCForAgentFile(instance_.getAgentTaskFName(), variant,
                               solverTimeoutSec, "initialization:" + variant,
                               "", "", "", "", "mlastar");
}

bool LNS::runMAPFPCForAgentFile(const string& agentFilePath,
                                const string& solverVariant,
                                int solverTimeoutSec,
                                const string& sourceLabel,
                                const string& fixedAssignmentFilePath,
                                const string& mutableAgentsFilePath,
                                const string& initialPathsFilePath,
                                const string& mutableTasksFilePath,
                                const string& lowLevelPlannerOverride,
                                const string& catBackendOverride,
                                int catBackendSmallMapsOverride) {
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
  string lowLevelPlanner =
      (lowLevelPlannerType_ == LowLevelPlannerType::sipps) ? "sipps"
                                                            : "mlastar";
  if (!lowLevelPlannerOverride.empty()) {
    lowLevelPlanner = lowLevelPlannerOverride;
  }
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
  const fs::path stderrCapturePath =
      tempDir /
      ("mapf_pc_task_assignment_err_" + std::to_string(seed_) + "_" +
       std::to_string((long long)Time::now().time_since_epoch().count()) +
       ".log");
  lns_temp_file::ScopedPathCleanup captureCleanup;
  captureCleanup.add(stdoutCapturePath);
  captureCleanup.add(stderrCapturePath);
  auto logCapturedStderr = [&stderrCapturePath]() {
    std::ifstream errStream(stderrCapturePath);
    if (!errStream.is_open()) {
      return;
    }
    std::string line;
    int emitted = 0;
    while (std::getline(errStream, line) && emitted < 8) {
      if (line.empty()) {
        continue;
      }
      PLOGE << "MAPF-PC stderr: " << line << "\n";
      emitted++;
    }
  };
  const bool keepCaptureOnFailure =
      std::getenv("MAPF_PC_KEEP_CAPTURE_ON_FAIL") != nullptr;

  std::vector<std::string> args = {
      "-m", instance_.getMapName(),
      "-a", agentFilePath,
      "-k", std::to_string(instance_.getAgentNum()),
      "-t", std::to_string(effectiveSolverTimeoutSec),
      "-d", std::to_string(seed_),
      "--solver", *solver,
      "--lowLevelPlanner", lowLevelPlanner,
  };
  if (optimizationObjective_ == "soc" || optimizationObjective_ == "makespan") {
    args.push_back("--optimizationObjective");
    args.push_back(optimizationObjective_);
  }
  if (lowLevelPlanner == "sipps") {
    args.push_back("--sippsSuboptimality");
    args.push_back(std::to_string(lowLevelSippsSuboptimality_));
  }
  if (!fixedAssignmentFilePath.empty()) {
    args.push_back("--fixedAssignmentFile");
    args.push_back(fixedAssignmentFilePath);
  }
  if (!mutableAgentsFilePath.empty()) {
    args.push_back("--mutableAgentsFile");
    args.push_back(mutableAgentsFilePath);
  }
  if (!initialPathsFilePath.empty()) {
    args.push_back("--initialPathsFile");
    args.push_back(initialPathsFilePath);
  }
  if (!mutableTasksFilePath.empty()) {
    args.push_back("--mutableTasksFile");
    args.push_back(mutableTasksFilePath);
  }
  if (!catBackendOverride.empty()) {
    args.push_back("--catBackend");
    args.push_back(catBackendOverride);
  }
  if (catBackendSmallMapsOverride == 0 || catBackendSmallMapsOverride == 1) {
    args.push_back("--catBackendSmallMaps");
    args.push_back(std::to_string(catBackendSmallMapsOverride));
  }
  const char* cbsVanillaReasoningEnv =
      std::getenv("MAPFPC_NRR_CBS_VANILLA_REASONING");
  const bool useVanillaCbsReasoning =
      cbsVanillaReasoningEnv != nullptr &&
      std::string(cbsVanillaReasoningEnv) != "0" &&
      std::string(cbsVanillaReasoningEnv) != "false" &&
      std::string(cbsVanillaReasoningEnv) != "False" &&
      std::string(cbsVanillaReasoningEnv) != "FALSE";
  const char* cbsPcEnv = std::getenv("MAPFPC_NRR_CBS_PC");
  const bool useCbsPc = cbsPcEnv != nullptr && std::string(cbsPcEnv) != "0" &&
                        std::string(cbsPcEnv) != "false" &&
                        std::string(cbsPcEnv) != "False" &&
                        std::string(cbsPcEnv) != "FALSE";
  // By default keep LNS-spawned CBS configuration aligned with MAPF-LNS2
  // defaults (rectangle/corridor/target/bypass on, mutex off, no disjoint
  // splitting). For A/B testing we can opt into vanilla task_assignment CBS
  // defaults via MAPFPC_NRR_CBS_VANILLA_REASONING=1.
  if (*solver == "CBS") {
    if (useCbsPc) {
      args.push_back("--pc");
      args.push_back("1");
    }
    if (!useVanillaCbsReasoning) {
      args.push_back("--rectangle");
      args.push_back("1");
      args.push_back("--corridor");
      args.push_back("1");
      args.push_back("--bypass");
      args.push_back("1");
      args.push_back("--mutex");
      args.push_back("0");
      args.push_back("--disjoint");
      args.push_back("0");
      args.push_back("--target");
      args.push_back("1");
      args.push_back("--stp");
      args.push_back("1");
    }
  }
  std::optional<bp::child> childProcess;
  try {
#if MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL
    childProcess.emplace(taskAssignmentExe->string(), bp::args(args),
                         bp::std_out > stdoutCapturePath.string(),
                         bp::std_err > stderrCapturePath.string());
#else
    // Some Boost.Process installations don't ship <boost/process/null.hpp>.
    // Capture stderr to a file for diagnostics.
    childProcess.emplace(taskAssignmentExe->string(), bp::args(args),
                         bp::std_out > stdoutCapturePath.string(),
                         bp::std_err > stderrCapturePath.string());
#endif
  } catch (const bp::process_error& e) {
    PLOGE << "Failed to launch MAPF-PC task_assignment at '"
          << taskAssignmentExe->string() << "': " << e.what() << "\n";
    if (keepCaptureOnFailure) {
      captureCleanup.release();
      PLOGE << "Keeping MAPF-PC captures after launch failure: stdout='"
            << stdoutCapturePath.string() << "', stderr='"
            << stderrCapturePath.string() << "'\n";
    }
    return false;
  }

  if (!sourceLabel.empty()) {
    PLOGI << "MAPF-PC task_assignment start: solver=" << *solver
          << ", ll=" << lowLevelPlanner
          << ", objective=" << optimizationObjective_
          << (lowLevelPlanner == "sipps"
                  ? (", ll_w=" + std::to_string(lowLevelSippsSuboptimality_))
                  : "")
          << (catBackendOverride.empty() ? ""
                                         : (", catBackend=" + catBackendOverride))
          << ((catBackendSmallMapsOverride == 0 ||
               catBackendSmallMapsOverride == 1)
                  ? (", catBackendSmallMaps=" +
                     std::to_string(catBackendSmallMapsOverride))
                  : "")
          << ((*solver == "CBS")
                  ? (std::string(", cbs_reasoning_profile=") +
                     (useVanillaCbsReasoning ? "vanilla" : "lns_forced"))
                  : "")
          << ((*solver == "CBS")
                  ? (std::string(", cbs_pc=") + (useCbsPc ? "1" : "0"))
                  : "")
          << ", timeout_sec=" << effectiveSolverTimeoutSec
          << ", source=" << sourceLabel << "\n";
  }
  bp::child& child = *childProcess;
  auto waitTask = std::async(std::launch::async, [&child]() {
    std::error_code ec;
    child.wait(ec);
    return ec;
  });
  const bool completedWithinTimeout =
      (waitTask.wait_for(std::chrono::seconds(wallClockTimeoutSec)) ==
       std::future_status::ready);
  if (!completedWithinTimeout) {
    PLOGE << "MAPF-PC task_assignment exceeded wall-clock timeout ("
          << wallClockTimeoutSec << "s); terminating process\n";
    child.terminate();
    waitTask.wait();
    (void)waitTask.get();
    logCapturedStderr();
    if (keepCaptureOnFailure) {
      captureCleanup.release();
      PLOGE << "Keeping MAPF-PC captures after timeout: stdout='"
            << stdoutCapturePath.string() << "', stderr='"
            << stderrCapturePath.string() << "'\n";
    }
    return false;
  }
  const std::error_code waitEc = waitTask.get();
  if (waitEc) {
    PLOGE << "MAPF-PC task_assignment wait failed: " << waitEc.message() << "\n";
    logCapturedStderr();
    if (keepCaptureOnFailure) {
      captureCleanup.release();
      PLOGE << "Keeping MAPF-PC captures after wait error: stdout='"
            << stdoutCapturePath.string() << "', stderr='"
            << stderrCapturePath.string() << "'\n";
    }
    return false;
  }
  if (child.exit_code() != 0) {
    PLOGE << "MAPF-PC task_assignment exited with code " << child.exit_code()
          << "\n";
    logCapturedStderr();
    if (keepCaptureOnFailure) {
      captureCleanup.release();
      PLOGE << "Keeping MAPF-PC captures after nonzero exit: stdout='"
            << stdoutCapturePath.string() << "', stderr='"
            << stderrCapturePath.string() << "'\n";
    }
    return false;
  }
  std::ifstream inputStream(stdoutCapturePath);
  if (!inputStream.is_open()) {
    PLOGE << "Failed to open captured MAPF-PC stdout at '"
          << stdoutCapturePath.string() << "'\n";
    return false;
  }
  {
    // MAPF-PC may return exit code 0 even when it prints an explicit
    // no-solution marker. Detect that before attempting to parse paths.
    std::string line;
    bool reportedNoSolution = false;
    while (std::getline(inputStream, line)) {
      if (line.find("No solutions") != std::string::npos ||
          line.find("No solution") != std::string::npos) {
        reportedNoSolution = true;
        break;
      }
    }
    inputStream.clear();
    inputStream.seekg(0, std::ios::beg);
    if (reportedNoSolution) {
      PLOGE << "MAPF-PC task_assignment reported no solution in stdout capture\n";
      if (keepCaptureOnFailure) {
        captureCleanup.release();
        PLOGE << "Keeping MAPF-PC captures after reported no-solution: stdout='"
              << stdoutCapturePath.string() << "', stderr='"
              << stderrCapturePath.string() << "'\n";
      }
      return false;
    }
  }

  const string parseSourceLabel =
      sourceLabel.empty() ? stdoutCapturePath.string() : sourceLabel;
  const bool parsed =
      parseMAPFPCStreamIntoSolution(inputStream, parseSourceLabel);
  inputStream.close();
  if (!parsed && keepCaptureOnFailure) {
    captureCleanup.release();
    PLOGE << "Keeping MAPF-PC captures after parse failure: stdout='"
          << stdoutCapturePath.string() << "', stderr='"
          << stderrCapturePath.string() << "'\n";
    return false;
  }
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
  lns_temp_file::ScopedPathCleanup assignmentCleanup;
  assignmentCleanup.add(assignmentPath);

  string assignmentError;
  if (!writeMAPFPCAssignmentFile(assignments, assignmentPath.string(),
                                 assignmentError)) {
    PLOGE << "Failed to write MAPF-PC assignment file: " << assignmentError
          << "\n";
    return false;
  }

  return runMAPFPCForAgentFile(
      instance_.getAgentTaskFName(), solverVariant, solverTimeoutSec,
      sourceLabel.empty() ? assignmentPath.string() : sourceLabel,
      assignmentPath.string());
}
