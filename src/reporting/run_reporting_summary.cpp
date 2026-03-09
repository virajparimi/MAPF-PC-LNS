#include "run_reporting.hpp"
#include "run_reporting_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

using run_reporting_internal::FinalSolutionScheduleMetrics;
using run_reporting_internal::computeFinalSolutionScheduleMetrics;
using run_reporting_internal::computeIterationDurations;
using run_reporting_internal::computeTimeToBestFeasible;
using run_reporting_internal::countLnsIterations;
using run_reporting_internal::percentileValue;
using run_reporting_internal::printMetric;

namespace {

std::tm localtimeSafe(std::time_t timeValue) {
  std::tm tmValue{};
#if defined(_WIN32)
  localtime_s(&tmValue, &timeValue);
#else
  localtime_r(&timeValue, &tmValue);
#endif
  return tmValue;
}

std::string currentTimestampCompact() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
  const std::tm tmValue = localtimeSafe(nowTime);
  std::ostringstream oss;
  oss << std::put_time(&tmValue, "%Y%m%d_%H%M%S");
  return oss.str();
}

std::string formatDoubleValue(double value, int precision = 4) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

std::string sanitizeTsvCell(const std::string& value) {
  std::string out = value;
  for (char& ch : out) {
    if (ch == '\t' || ch == '\n' || ch == '\r') {
      ch = ' ';
    }
  }
  return out;
}

std::string escapeJson(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    switch (c) {
      case '\"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::filesystem::path resolveMetricsOutputDir() {
  if (const char* envDir = std::getenv("MAPF_PC_REPORT_DIR")) {
    if (envDir[0] != '\0') {
      return std::filesystem::path(envDir);
    }
  }
  return std::filesystem::path("build/test_artifacts/run_metrics");
}

struct AssignmentDeltaStats {
  int tasksMovedBetweenAgents = 0;
  int tasksReorderedWithinAgent = 0;
  int agentsTouched = 0;
  int finalAssignmentDiffCount = 0;
  double finalAssignmentDiffFraction = 0.0;
};

AssignmentDeltaStats computeAssignmentDeltaStats(const LNS& lns,
                                                 const FeasibleSolution& final) {
  AssignmentDeltaStats stats;
  if (!lns.hasInitialAssignmentSnapshot()) {
    return stats;
  }
  const int taskCount = lns.getInstance().getTasksNum();
  const int agentCount = lns.getInstance().getAgentNum();
  if (taskCount <= 0 || agentCount <= 0) {
    return stats;
  }
  const auto& initialOwner = lns.getInitialTaskOwnerByTask();
  const auto& initialPos = lns.getInitialTaskPosByTask();
  std::vector<int> finalOwner(taskCount, UNASSIGNED);
  std::vector<int> finalPos(taskCount, UNASSIGNED);
  std::vector<char> touched(agentCount, 0);
  for (int agent = 0; agent < agentCount; agent++) {
    if (agent >= (int)final.agentTaskAssignments.size()) {
      continue;
    }
    const auto& tasks = final.agentTaskAssignments[agent];
    for (int pos = 0; pos < (int)tasks.size(); pos++) {
      const int task = tasks[pos];
      if (task < 0 || task >= taskCount) {
        continue;
      }
      finalOwner[task] = agent;
      finalPos[task] = pos;
    }
  }
  for (int task = 0; task < taskCount; task++) {
    const int initOwner = task < (int)initialOwner.size() ? initialOwner[task]
                                                          : UNASSIGNED;
    const int initPos =
        task < (int)initialPos.size() ? initialPos[task] : UNASSIGNED;
    const int curOwner = finalOwner[task];
    const int curPos = finalPos[task];
    if (initOwner != curOwner) {
      stats.tasksMovedBetweenAgents++;
      if (initOwner >= 0 && initOwner < agentCount) {
        touched[initOwner] = 1;
      }
      if (curOwner >= 0 && curOwner < agentCount) {
        touched[curOwner] = 1;
      }
      continue;
    }
    if (initOwner >= 0 && initOwner < agentCount && initPos != curPos) {
      stats.tasksReorderedWithinAgent++;
      touched[initOwner] = 1;
    }
  }
  for (char flag : touched) {
    if (flag) {
      stats.agentsTouched++;
    }
  }
  stats.finalAssignmentDiffCount =
      stats.tasksMovedBetweenAgents + stats.tasksReorderedWithinAgent;
  stats.finalAssignmentDiffFraction =
      taskCount > 0 ? (double)stats.finalAssignmentDiffCount / (double)taskCount
                    : 0.0;
  return stats;
}

double computeStddev(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  double sq = 0.0;
  for (double v : values) {
    const double d = v - mean;
    sq += d * d;
  }
  return std::sqrt(sq / static_cast<double>(values.size()));
}

double computeTasksPerAgentStddev(const FeasibleSolution& solution) {
  std::vector<double> counts;
  counts.reserve(solution.agentTaskAssignments.size());
  for (const auto& tasks : solution.agentTaskAssignments) {
    counts.push_back(static_cast<double>(tasks.size()));
  }
  return computeStddev(counts);
}

int computeMaxTasksPerAgent(const FeasibleSolution& solution) {
  int maxTasks = 0;
  for (const auto& tasks : solution.agentTaskAssignments) {
    maxTasks = std::max(maxTasks, static_cast<int>(tasks.size()));
  }
  return maxTasks;
}

double computeAgentCostStddev(const FeasibleSolution& solution) {
  std::vector<double> costs;
  costs.reserve(solution.agentPaths.size());
  for (const auto& path : solution.agentPaths) {
    const double cost = path.empty() ? 0.0 : static_cast<double>(path.size() - 1);
    costs.push_back(cost);
  }
  return computeStddev(costs);
}

void writeTrajectoryTsv(const std::filesystem::path& outPath,
                        const std::vector<IterationStats>& stats) {
  std::ofstream out(outPath);
  if (!out.is_open()) {
    return;
  }
  out << "iter\ttime_s\tbest_soc\tcurrent_soc\tbest_makespan\tcurrent_makespan\t"
         "best_precedence_wait\tcurrent_precedence_wait\t"
         "accepted_best_soc\taccepted_current_soc\t"
         "accepted_best_makespan\taccepted_current_makespan\t"
         "accepted_best_precedence_wait\taccepted_current_precedence_wait\n";
  int bestSoc = std::numeric_limits<int>::max();
  int bestMakespan = -1;
  double bestPrecWait = std::numeric_limits<double>::quiet_NaN();
  bool hasCurrent = false;
  int currentSoc = 0;
  int currentMakespan = -1;
  double currentPrecWait = std::numeric_limits<double>::quiet_NaN();
  int acceptedBestSoc = std::numeric_limits<int>::max();
  int acceptedBestMakespan = -1;
  double acceptedBestPrecWait = std::numeric_limits<double>::quiet_NaN();
  bool hasAcceptedCurrent = false;
  int acceptedCurrentSoc = 0;
  int acceptedCurrentMakespan = -1;
  double acceptedCurrentPrecWait = std::numeric_limits<double>::quiet_NaN();
  for (size_t iter = 0; iter < stats.size(); iter++) {
    const IterationStats& row = stats[iter];
    const bool acceptedIteration =
        row.quality == IterationQuality::bestSolutionYet ||
        row.quality == IterationQuality::improvedSolution ||
        row.quality == IterationQuality::downgradedButAccepted;
    if (row.feasibleSolutionFound) {
      hasCurrent = true;
      currentSoc = row.sumOfCosts;
      currentMakespan = row.makespan;
      currentPrecWait = row.precedenceWait;
      if (row.sumOfCosts < bestSoc) {
        bestSoc = row.sumOfCosts;
        bestMakespan = row.makespan;
        bestPrecWait = row.precedenceWait;
      }
    }
    if (acceptedIteration) {
      hasAcceptedCurrent = true;
      acceptedCurrentSoc = row.sumOfCosts;
      acceptedCurrentMakespan = row.makespan;
      acceptedCurrentPrecWait = row.precedenceWait;
      if (row.sumOfCosts < acceptedBestSoc) {
        acceptedBestSoc = row.sumOfCosts;
        acceptedBestMakespan = row.makespan;
        acceptedBestPrecWait = row.precedenceWait;
      }
    }
    out << iter << '\t' << formatDoubleValue(row.runtime, 6) << '\t';
    if (bestSoc == std::numeric_limits<int>::max()) {
      out << '\t';
    } else {
      out << bestSoc << '\t';
    }
    if (hasCurrent) {
      out << currentSoc << '\t';
    } else {
      out << '\t';
    }
    if (bestMakespan >= 0) {
      out << bestMakespan << '\t';
    } else {
      out << '\t';
    }
    if (hasCurrent && currentMakespan >= 0) {
      out << currentMakespan << '\t';
    } else {
      out << '\t';
    }
    if (std::isfinite(bestPrecWait)) {
      out << formatDoubleValue(bestPrecWait, 4) << '\t';
    } else {
      out << '\t';
    }
    if (hasCurrent && std::isfinite(currentPrecWait)) {
      out << formatDoubleValue(currentPrecWait, 4);
    }
    out << '\t';
    if (acceptedBestSoc == std::numeric_limits<int>::max()) {
      out << '\t';
    } else {
      out << acceptedBestSoc << '\t';
    }
    if (hasAcceptedCurrent) {
      out << acceptedCurrentSoc << '\t';
    } else {
      out << '\t';
    }
    if (acceptedBestMakespan >= 0) {
      out << acceptedBestMakespan << '\t';
    } else {
      out << '\t';
    }
    if (hasAcceptedCurrent && acceptedCurrentMakespan >= 0) {
      out << acceptedCurrentMakespan << '\t';
    } else {
      out << '\t';
    }
    if (std::isfinite(acceptedBestPrecWait)) {
      out << formatDoubleValue(acceptedBestPrecWait, 4) << '\t';
    } else {
      out << '\t';
    }
    if (hasAcceptedCurrent && std::isfinite(acceptedCurrentPrecWait)) {
      out << formatDoubleValue(acceptedCurrentPrecWait, 4);
    }
    out << '\n';
  }
}

void writeRunSummaryWideTsv(
    const std::filesystem::path& outPath,
    const std::vector<std::pair<std::string, std::string>>& fields) {
  std::ofstream out(outPath);
  if (!out.is_open()) {
    return;
  }
  for (size_t i = 0; i < fields.size(); i++) {
    if (i > 0) {
      out << '\t';
    }
    out << sanitizeTsvCell(fields[i].first);
  }
  out << '\n';
  for (size_t i = 0; i < fields.size(); i++) {
    if (i > 0) {
      out << '\t';
    }
    out << sanitizeTsvCell(fields[i].second);
  }
  out << '\n';
}

void writeSolutionJson(const std::filesystem::path& outPath,
                       const LNS& lns,
                       const FeasibleSolution& solution,
                       const FinalSolutionScheduleMetrics& metrics,
                       const std::string& objective,
                       int objectiveValue) {
  std::ofstream out(outPath);
  if (!out.is_open()) {
    return;
  }

  const int agentCount = lns.getInstance().getAgentNum();
  out << "{\n";
  out << "  \"objective\": \"" << escapeJson(objective) << "\",\n";
  out << "  \"objective_value\": " << objectiveValue << ",\n";
  out << "  \"sum_of_costs\": " << solution.sumOfCosts << ",\n";
  out << "  \"makespan\": " << metrics.makespan << ",\n";
  out << "  \"max_individual_cost\": " << metrics.maxIndividualCost << ",\n";
  out << "  \"agents\": [\n";

  for (int agent = 0; agent < agentCount; agent++) {
    const AgentTaskPath emptyPath;
    const std::vector<int> emptyTasks;
    const std::vector<AgentTaskPath> emptyTaskPaths;
    const AgentTaskPath& joinedPath =
        agent < (int)solution.agentPaths.size() ? solution.agentPaths[agent]
                                                : emptyPath;
    const std::vector<int>& tasks =
        agent < (int)solution.agentTaskAssignments.size()
            ? solution.agentTaskAssignments[agent]
            : emptyTasks;
    const std::vector<AgentTaskPath>& taskPaths =
        agent < (int)solution.agentTaskPaths.size()
            ? solution.agentTaskPaths[agent]
            : emptyTaskPaths;
    const int cost = joinedPath.empty() ? 0 : static_cast<int>(joinedPath.size()) - 1;

    out << "    {\n";
    out << "      \"agent\": " << agent << ",\n";
    out << "      \"cost\": " << cost << ",\n";
    out << "      \"task_assignments\": [";
    for (size_t i = 0; i < tasks.size(); i++) {
      if (i > 0) {
        out << ", ";
      }
      out << tasks[i];
    }
    out << "],\n";
    out << "      \"path\": [";
    for (size_t i = 0; i < joinedPath.size(); i++) {
      if (i > 0) {
        out << ", ";
      }
      out << "{\"location\": " << joinedPath.path[i].location
          << ", \"time\": " << (joinedPath.beginTime + static_cast<int>(i))
          << ", \"is_goal\": "
          << (joinedPath.path[i].isGoal ? "true" : "false") << "}";
    }
    out << "],\n";
    out << "      \"task_paths\": [";
    for (size_t taskIdx = 0; taskIdx < taskPaths.size(); taskIdx++) {
      if (taskIdx > 0) {
        out << ", ";
      }
      const int taskId =
          taskIdx < tasks.size() ? tasks[taskIdx] : static_cast<int>(taskIdx);
      out << "{\"task\": " << taskId << ", \"path\": [";
      const AgentTaskPath& taskPath = taskPaths[taskIdx];
      for (size_t i = 0; i < taskPath.size(); i++) {
        if (i > 0) {
          out << ", ";
        }
        out << "{\"location\": " << taskPath.path[i].location
            << ", \"time\": " << (taskPath.beginTime + static_cast<int>(i))
            << ", \"is_goal\": "
            << (taskPath.path[i].isGoal ? "true" : "false") << "}";
      }
      out << "]}";
    }
    out << "]\n";
    out << "    }";
    if (agent + 1 < agentCount) {
      out << ",";
    }
    out << "\n";
  }

  out << "  ]\n";
  out << "}\n";
}

}  // namespace

void printRunSummaryReport(const LNS& lns, const FeasibleSolution& solution,
                           bool success,
                           const FeasibleTrajectoryStats& stats) {
  const int lnsIterations = countLnsIterations(lns.iterationStats);
  const vector<double> iterDurations = computeIterationDurations(lns.iterationStats);
  const double avgIterSec = iterDurations.empty()
                                ? 0.0
                                : std::accumulate(iterDurations.begin(),
                                                  iterDurations.end(),
                                                  0.0) /
                                      static_cast<double>(iterDurations.size());
  const double p95IterSec = percentileValue(iterDurations, 0.95);
  const double iterPerSec =
      lns.runtime > 0.0 ? static_cast<double>(lnsIterations) / lns.runtime
                        : 0.0;
  const double initialRuntimeMeasuredSec = lns.getInitialSolutionRuntimeSec();
  const double initialRuntimeReportedSec =
      lns.getInitialSolutionRuntimeReportedSec();
  const double initialSeedRuntimeFromLogSec =
      lns.getInitialSeedRuntimeFromLogSec();
  const double lnsLoopRuntimeSec = lns.getLnsLoopRuntimeSec();
  const double postRefineRuntimeSec = lns.getPostRefineRuntimeSec();
  const double coveredRuntimeSec = initialRuntimeMeasuredSec + lnsLoopRuntimeSec +
                                   postRefineRuntimeSec;
  const double uncoveredRuntimeSec =
      std::max(0.0, lns.runtime - coveredRuntimeSec);
  const auto& timingStats = lns.getImprovementDiagnosticsStats();
  const bool fullTimingDetail = []() {
    const char* timingLevel = std::getenv("MAPF_PC_TIMING_LEVEL");
    return timingLevel != nullptr && std::string(timingLevel) == "full";
  }();
  const double timingMeasuredLnsPhasesSec =
      timingStats.timeDestroyAndPrepareSec +
      timingStats.timeRepairAndCommitSec + timingStats.timeJoinPathsSec +
      timingStats.timeRecomputeSocSec +
      timingStats.timeValidationSec + timingStats.timeAcceptanceSec +
      timingStats.timeBookkeepingSec;
  const double timeToBestFeasible = computeTimeToBestFeasible(lns.iterationStats);
  const FinalSolutionScheduleMetrics finalSolutionMetrics =
      computeFinalSolutionScheduleMetrics(solution, lns.getInstance());
  const std::string optimizationObjective = lns.getOptimizationObjective();
  const int objectiveValue =
      optimizationObjective == "makespan"
          ? finalSolutionMetrics.makespan
          : solution.sumOfCosts;
  const LNS::LowLevelSearchStats lowLevelStats = lns.getLowLevelSearchStats();
  const double lowLevelCallsPerSec =
      lns.runtime > 0.0 ? static_cast<double>(lowLevelStats.calls) / lns.runtime
                        : 0.0;
  const double lowLevelExpandedPerSec =
      lns.runtime > 0.0
          ? static_cast<double>(lowLevelStats.expanded) / lns.runtime
          : 0.0;
  const double lowLevelGeneratedPerSec =
      lns.runtime > 0.0
          ? static_cast<double>(lowLevelStats.generated) / lns.runtime
          : 0.0;
  const double expandedPerCall =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.expanded) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double generatedPerCall =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.generated) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llFoundRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.found) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llTimeoutRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.timeout) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llExhaustedRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.searchExhausted) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const double llBudgetExhaustedRate =
      lowLevelStats.calls > 0
          ? static_cast<double>(lowLevelStats.budgetExhausted) /
                static_cast<double>(lowLevelStats.calls)
          : 0.0;
  const std::string initialSeedLogPath = lns.getInitialSeedFromMapfpcLog();
  const bool hasInitialSeedRuntimeFromLog =
      std::isfinite(initialSeedRuntimeFromLogSec) &&
      initialSeedRuntimeFromLogSec >= 0.0;
  const std::string initialSeedSolver = [&]() -> std::string {
    if (initialSeedLogPath.empty()) {
      return "n/a";
    }
    if (initialSeedLogPath.find("cbs_pc__") != std::string::npos) {
      return "cbs";
    }
    if (initialSeedLogPath.find("pbs_pc__") != std::string::npos) {
      return "pbs";
    }
    return "unknown";
  }();
  const AssignmentDeltaStats assignmentDelta =
      computeAssignmentDeltaStats(lns, solution);
  const int maxTasksPerAgent = computeMaxTasksPerAgent(solution);
  const double tasksPerAgentStddev = computeTasksPerAgentStddev(solution);
  const double agentCostStddev = computeAgentCostStddev(solution);
  const bool initialMetricsAvailable = lns.hasInitialMetrics();
  const int initialSoc =
      initialMetricsAvailable ? lns.getInitialSoc() : std::numeric_limits<int>::max();
  const int initialMakespan =
      initialMetricsAvailable ? lns.getInitialMakespan() : -1;
  const double initialPrecWait =
      initialMetricsAvailable ? lns.getInitialPrecedenceWait()
                              : std::numeric_limits<double>::quiet_NaN();
  const int finalSoc = solution.sumOfCosts;
  const int finalMakespan = finalSolutionMetrics.makespan;
  const double finalPrecWait =
      finalSolutionMetrics.hasTaskScheduleData
          ? static_cast<double>(finalSolutionMetrics.totalPrecedenceWait)
          : std::numeric_limits<double>::quiet_NaN();
  const int deltaSoc =
      initialMetricsAvailable ? (finalSoc - initialSoc) : std::numeric_limits<int>::max();
  const double relativeImprovementSoc =
      (initialMetricsAvailable && initialSoc > 0)
          ? (100.0 * (static_cast<double>(initialSoc - finalSoc) /
                      static_cast<double>(initialSoc)))
          : std::numeric_limits<double>::quiet_NaN();
  const int deltaMakespan =
      initialMetricsAvailable ? (finalMakespan - initialMakespan)
                              : std::numeric_limits<int>::max();
  const double deltaPrecWait =
      (initialMetricsAvailable && std::isfinite(finalPrecWait))
          ? (finalPrecWait - initialPrecWait)
          : std::numeric_limits<double>::quiet_NaN();
  const int64_t acceptedBetter = timingStats.acceptedSocBetterVsPrevious;
  const int64_t acceptedEqual = timingStats.acceptedSocEqualVsPrevious;
  const int64_t acceptedWorse = timingStats.acceptedSocWorseVsPrevious;
  const int64_t rejectedInvalid = lns.invalidCandidateRejections;
  const int64_t rejectedFailFind = stats.couldNotFindIterations;
  const int64_t rejectedGuard = timingStats.guardRejected;
  const int64_t totalAccepts = timingStats.accepted;
  const auto& regretStats = lns.getRegretEvalStatsRef();
  const double candidateTriedPerRemovedTask =
      regretStats.removedTasksSum > 0
          ? static_cast<double>(regretStats.candidateInsertionsTried) /
                static_cast<double>(regretStats.removedTasksSum)
          : 0.0;
  const double lowLevelCallsPerRemovedTask =
      regretStats.removedTasksSum > 0
          ? static_cast<double>(lowLevelStats.calls) /
                static_cast<double>(regretStats.removedTasksSum)
          : 0.0;
  const double lowLevelExpandedPerRemovedTask =
      regretStats.removedTasksSum > 0
          ? static_cast<double>(lowLevelStats.expanded) /
                static_cast<double>(regretStats.removedTasksSum)
          : 0.0;

  std::cout << "\n=== Run Summary ===\n";
  printMetric("Requested initial solution",
              lns.getInitialSolutionRequested());
  printMetric("Effective initial solution", lns.getInitialSolutionEffective());
  printMetric("Initial fallback used",
              lns.wasInitialSolutionFallbackUsed() ? "true" : "false");
  printMetric("Initial fallback reason",
              lns.getInitialSolutionFallbackReason());
  printMetric("Runtime (s)", lns.runtime);
  printMetric("Initial runtime measured (s)", initialRuntimeMeasuredSec);
  printMetric("Initial runtime reported (s)", initialRuntimeReportedSec);
  printMetric("LNS loop runtime (s)", lnsLoopRuntimeSec);
  printMetric("Post-refine runtime (s)", postRefineRuntimeSec);
  printMetric("Uncovered runtime (s)", uncoveredRuntimeSec);
  printMetric("LNS measured phase time (s)", timingMeasuredLnsPhasesSec);
  printMetric("Destroy+prepare time (s)", timingStats.timeDestroyAndPrepareSec);
  printMetric("Repair+commit time (s)", timingStats.timeRepairAndCommitSec);
  printMetric("Regret candidate-eval time (s)",
              timingStats.timeRegretCandidateEvalSec);
  printMetric("Regret commit time (s)", timingStats.timeRegretCommitSec);
  printMetric("Regret low-level time (s)", timingStats.timeRegretLowLevelSec);
  printMetric("Join paths time (s)", timingStats.timeJoinPathsSec);
  printMetric("Recompute SoC time (s)", timingStats.timeRecomputeSocSec);
  printMetric("Validation time (s)", timingStats.timeValidationSec);
  printMetric("Acceptance time (s)", timingStats.timeAcceptanceSec);
  printMetric("Bookkeeping time (s)", timingStats.timeBookkeepingSec);
  printMetric("Iterations", lnsIterations);
  printMetric("Iterations/sec", iterPerSec);
  printMetric("Avg iteration runtime (ms)", avgIterSec * 1000.0);
  printMetric("P95 iteration runtime (ms)", p95IterSec * 1000.0);
  printMetric("First feasible runtime (s)", stats.firstFeasibleRuntime);
  printMetric("Time-to-best feasible (s)", timeToBestFeasible);
  printMetric("Improving feasible updates", stats.numImprovingFeasibleUpdates);
  printMetric("Total feasible iterations", stats.numFeasibleIterations);
  printMetric("Optimization objective", optimizationObjective);
  printMetric("Objective value", objectiveValue);
  printMetric("Solution cost", solution.sumOfCosts);
  printMetric("Makespan", finalSolutionMetrics.makespan);
  printMetric("Max individual cost", finalSolutionMetrics.maxIndividualCost);
  if (finalSolutionMetrics.hasTaskScheduleData) {
    printMetric("Total precedence wait",
                finalSolutionMetrics.totalPrecedenceWait);
    printMetric("Max precedence wait", finalSolutionMetrics.maxPrecedenceWait);
    printMetric("Precedence slack edges",
                static_cast<int64_t>(finalSolutionMetrics.precedenceSlacks.size()));
    if (!finalSolutionMetrics.precedenceSlacks.empty()) {
      const double minSlack =
          *std::min_element(finalSolutionMetrics.precedenceSlacks.begin(),
                            finalSolutionMetrics.precedenceSlacks.end());
      printMetric("Min precedence slack", minSlack);
      printMetric("Precedence slack p25",
                  percentileValue(finalSolutionMetrics.precedenceSlacks, 0.25));
      printMetric("Precedence slack p50",
                  percentileValue(finalSolutionMetrics.precedenceSlacks, 0.50));
      printMetric("Precedence slack p75",
                  percentileValue(finalSolutionMetrics.precedenceSlacks, 0.75));
    } else {
      printMetric("Min precedence slack", "n/a");
      printMetric("Precedence slack p25", "n/a");
      printMetric("Precedence slack p50", "n/a");
      printMetric("Precedence slack p75", "n/a");
    }
  } else {
    printMetric("Total precedence wait", "n/a");
    printMetric("Max precedence wait", "n/a");
    printMetric("Precedence slack edges", "n/a");
    printMetric("Min precedence slack", "n/a");
    printMetric("Precedence slack p25", "n/a");
    printMetric("Precedence slack p50", "n/a");
    printMetric("Precedence slack p75", "n/a");
  }
  printMetric("Number of failures", lns.numOfFailures);
  printMetric("Invalid candidate rejections",
              lns.invalidCandidateRejections);
  printMetric("Guard rejections", timingStats.guardRejected);
  printMetric("Success", success ? "true" : "false");

  std::cout << "\n=== Initial Solution ===\n";
  printMetric("Requested strategy", lns.getInitialSolutionRequested());
  printMetric("Effective strategy", lns.getInitialSolutionEffective());
  printMetric("Fallback used",
              lns.wasInitialSolutionFallbackUsed() ? "true" : "false");
  printMetric("Fallback reason", lns.getInitialSolutionFallbackReason());
  printMetric("Seed log path",
              initialSeedLogPath.empty() ? "n/a" : initialSeedLogPath);
  printMetric("Seed solver", initialSeedSolver);
  if (hasInitialSeedRuntimeFromLog) {
    printMetric("Seed runtime from log (s)", initialSeedRuntimeFromLogSec);
  } else {
    printMetric("Seed runtime from log (s)", "n/a");
  }
  printMetric("Runtime measured (s)", initialRuntimeMeasuredSec);
  printMetric("Runtime reported (s)", initialRuntimeReportedSec);

  std::cout << "\n=== Post-Refine ===\n";
  printMetric("Enabled", lns.isPostRefineEnabled() ? "true" : "false");
  printMetric("Attempted", lns.wasPostRefineAttempted() ? "true" : "false");
  printMetric("Accepted", lns.wasPostRefineAccepted() ? "true" : "false");
  printMetric("Assignment source", lns.getPostRefineAssignmentSource());
  if (lns.getPostRefineAssignmentSource() == "log") {
    printMetric("Assignment log", lns.getPostRefineAssignmentLog());
  }
  printMetric("Solver", lns.getPostRefineSolver());
  printMetric("Timeout (s)", lns.getPostRefineTimeoutSec());
  printMetric("Accept-only-if-better",
              lns.isPostRefineAcceptOnlyIfBetter() ? "true" : "false");
  printMetric("Runtime (s)", postRefineRuntimeSec);

  std::cout << "\n=== Seed-to-Final Delta ===\n";
  if (initialMetricsAvailable) {
    printMetric("Initial SoC", initialSoc);
    printMetric("Final SoC", finalSoc);
    printMetric("Delta SoC (final-initial)", deltaSoc);
    printMetric("Relative improvement SoC (%)", relativeImprovementSoc);
    printMetric("Initial makespan", initialMakespan);
    printMetric("Final makespan", finalMakespan);
    printMetric("Delta makespan (final-initial)", deltaMakespan);
    printMetric("Initial total precedence wait", initialPrecWait);
    if (std::isfinite(finalPrecWait)) {
      printMetric("Final total precedence wait", finalPrecWait);
      printMetric("Delta precedence wait (final-initial)", deltaPrecWait);
    } else {
      printMetric("Final total precedence wait", "n/a");
      printMetric("Delta precedence wait (final-initial)", "n/a");
    }
  } else {
    printMetric("Initial SoC", "n/a");
    printMetric("Final SoC", finalSoc);
    printMetric("Delta SoC (final-initial)", "n/a");
    printMetric("Relative improvement SoC (%)", "n/a");
    printMetric("Initial makespan", "n/a");
    printMetric("Final makespan", finalMakespan);
    printMetric("Delta makespan (final-initial)", "n/a");
    printMetric("Initial total precedence wait", "n/a");
    printMetric("Final total precedence wait",
                std::isfinite(finalPrecWait) ? formatDoubleValue(finalPrecWait)
                                             : "n/a");
    printMetric("Delta precedence wait (final-initial)", "n/a");
  }

  std::cout << "\n=== Assignment Change ===\n";
  printMetric("Tasks moved between agents", assignmentDelta.tasksMovedBetweenAgents);
  printMetric("Tasks reordered within agent",
              assignmentDelta.tasksReorderedWithinAgent);
  printMetric("Agents touched", assignmentDelta.agentsTouched);
  printMetric("Final assignment diff count",
              assignmentDelta.finalAssignmentDiffCount);
  printMetric("Final assignment diff fraction",
              assignmentDelta.finalAssignmentDiffFraction);

  std::cout << "\n=== Acceptance Totals ===\n";
  printMetric("Accepted better", acceptedBetter);
  printMetric("Accepted equal", acceptedEqual);
  printMetric("Accepted worse", acceptedWorse);
  printMetric("Rejected invalid", rejectedInvalid);
  printMetric("Rejected fail-find", rejectedFailFind);
  printMetric("Rejected guard", rejectedGuard);
  printMetric("Total accepts", totalAccepts);

  std::cout << "\n=== Balance Metrics ===\n";
  printMetric("Max tasks per agent", maxTasksPerAgent);
  printMetric("Stddev tasks per agent", tasksPerAgentStddev);
  printMetric("Stddev agent costs", agentCostStddev);

  std::cout << "\n=== Low-Level Planner Throughput ===\n";
  printMetric("Low-level calls",
              static_cast<int64_t>(lowLevelStats.calls));
  printMetric("Low-level calls/sec", lowLevelCallsPerSec);
  printMetric("Low-level nodes expanded",
              static_cast<int64_t>(lowLevelStats.expanded));
  printMetric("Low-level nodes generated",
              static_cast<int64_t>(lowLevelStats.generated));
  printMetric("Expanded/sec", lowLevelExpandedPerSec);
  printMetric("Generated/sec", lowLevelGeneratedPerSec);
  printMetric("Expanded/call", expandedPerCall);
  printMetric("Generated/call", generatedPerCall);
  printMetric("LL outcome found", static_cast<int64_t>(lowLevelStats.found));
  printMetric("LL outcome timeout", static_cast<int64_t>(lowLevelStats.timeout));
  printMetric("LL outcome exhausted",
              static_cast<int64_t>(lowLevelStats.searchExhausted));
  printMetric("LL outcome invalid-input",
              static_cast<int64_t>(lowLevelStats.invalidInput));
  printMetric("LL outcome budget-exhausted",
              static_cast<int64_t>(lowLevelStats.budgetExhausted));
  printMetric("LL outcome unknown", static_cast<int64_t>(lowLevelStats.unknown));
  printMetric("LL timeout reason goal-permanent-before-lb",
              static_cast<int64_t>(
                  lowLevelStats.timeoutGoalPermanentBeforeArrivalLb));
  printMetric("LL timeout reason start-trapped@t+1",
              static_cast<int64_t>(lowLevelStats.timeoutStartTrappedAtTPlus1));
  printMetric("LL timeout reason static-disconnected",
              static_cast<int64_t>(
                  lowLevelStats.timeoutStaticDisconnectedPermanent));
  printMetric("LL timeout reason other",
              static_cast<int64_t>(lowLevelStats.timeoutOther));
  printMetric("LL timeout reduced-by-global-budget",
              static_cast<int64_t>(lowLevelStats.timeoutReducedByGlobalBudget));
  printMetric("LL timeout multi-certificate",
              static_cast<int64_t>(lowLevelStats.timeoutMultiCertificate));
  printMetric("LL structural pre-pruned",
              static_cast<int64_t>(lowLevelStats.structuralPrePruned));
  printMetric("LL structural pre-pruned goal-permanent-before-lb",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedGoalPermanentBeforeArrivalLb));
  printMetric("LL structural pre-pruned start-trapped@t+1",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedStartTrappedAtTPlus1));
  printMetric("LL structural pre-pruned static-disconnected",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedStaticDisconnectedPermanent));
  printMetric("LL structural pre-pruned multi-certificate",
              static_cast<int64_t>(
                  lowLevelStats.structuralPrePrunedMultiCertificate));
  printMetric("LL found rate", llFoundRate);
  printMetric("LL timeout rate", llTimeoutRate);
  printMetric("LL exhausted rate", llExhaustedRate);
  printMetric("LL budget-exhausted rate", llBudgetExhaustedRate);
  printMetric("Candidate insertions tried / removed task",
              candidateTriedPerRemovedTask);
  printMetric("Low-level calls / removed task", lowLevelCallsPerRemovedTask);
  printMetric("Low-level expanded / removed task",
              lowLevelExpandedPerRemovedTask);

  const auto& cascadeStats = lns.getCascadeStatsRef();
  const int totalTasks = lns.getInstance().getTasksNum();
  const double cascadeAbortRate =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.budgetAborts /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgSeedTasks =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.seedTasksSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgClosureTasks =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.closureTasksSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgClosureAdded =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.closureAddedSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgCascadeBudgetUsed =
      cascadeStats.prepareCalls > 0
          ? (double)cascadeStats.budgetUsedSum /
                (double)cascadeStats.prepareCalls
          : 0.0;
  const double avgClosureFracOfTasks =
      (cascadeStats.prepareCalls > 0 && totalTasks > 0)
          ? (double)cascadeStats.closureAddedSum /
                ((double)cascadeStats.prepareCalls * (double)totalTasks)
          : 0.0;

  std::cout << "\n=== Cascade Stats ===\n";
  printMetric("Cascade budget avg used", avgCascadeBudgetUsed);
  printMetric("Cascade budget min used",
              cascadeStats.prepareCalls > 0 ? cascadeStats.budgetUsedMin : 0);
  printMetric("Cascade budget max used",
              cascadeStats.prepareCalls > 0 ? cascadeStats.budgetUsedMax : 0);
  printMetric("prepareNextIteration calls", cascadeStats.prepareCalls);
  printMetric("Cascade budget aborts", cascadeStats.budgetAborts);
  printMetric("Cascade abort rate", cascadeAbortRate);
  printMetric("Avg seed removed tasks", avgSeedTasks);
  printMetric("Avg closure removed tasks", avgClosureTasks);
  printMetric("Avg closure added tasks", avgClosureAdded);
  printMetric("Max closure removed tasks", cascadeStats.closureTasksMax);
  printMetric("Max closure added tasks", cascadeStats.closureAddedMax);
  printMetric("Avg closure_added/total_tasks", avgClosureFracOfTasks);

  const auto& restoreStats = lns.getSolutionRestoreStats();
  std::cout << "\n=== Solution Restore Stats ===\n";
  printMetric("Restore calls", restoreStats.restoreCalls);
  printMetric("Full restores", restoreStats.fullRestores);

  const double feasibleRate =
      regretStats.candidateInsertionsTried > 0
          ? (double)regretStats.candidateInsertionsFeasible /
                (double)regretStats.candidateInsertionsTried
          : 0.0;
  const double avgRemovedTasks =
      regretStats.neighborhoods > 0
          ? (double)regretStats.removedTasksSum /
                (double)regretStats.neighborhoods
          : 0.0;
  std::cout << "\n=== Regret Evaluation Stats ===\n";
  printMetric("Recompute calls", regretStats.recomputeCalls);
  printMetric("Tasks evaluated", regretStats.tasksEvaluated);
  printMetric("Task-agent evaluations", regretStats.agentEvaluations);
  printMetric("Candidate insertions tried",
              regretStats.candidateInsertionsTried);
  printMetric("Candidate insertions feasible",
              regretStats.candidateInsertionsFeasible);
  printMetric("Workspace agents cloned", regretStats.workspaceAgentsCloned);
  printMetric("Max cloned agents/task",
              regretStats.workspaceMaxClonedPerTask);
  printMetric("Feasible rate", feasibleRate);
  printMetric("Repair neighborhoods", regretStats.neighborhoods);
  printMetric("Avg removed tasks/neighborhood", avgRemovedTasks);
  printMetric("Max removed tasks/neighborhood", regretStats.removedTasksMax);
  const auto& nrrStats = lns.getNrrStats();
  if (nrrStats.calls > 0 || nrrStats.attempts > 0) {
    const int64_t nrrCalls = std::max<int64_t>(nrrStats.calls, nrrStats.attempts);
    const auto formatHistogram = [](const std::map<int, int64_t>& hist) {
      if (hist.empty()) {
        return string("(none)");
      }
      std::ostringstream oss;
      bool first = true;
      for (const auto& [key, count] : hist) {
        if (!first) {
          oss << ", ";
        }
        first = false;
        oss << key << "->" << count;
      }
      return oss.str();
    };
    const auto formatReasonHistogram =
        [](const std::map<std::string, int64_t>& hist) {
          if (hist.empty()) {
            return string("(none)");
          }
          std::ostringstream oss;
          bool first = true;
          for (const auto& [key, count] : hist) {
            if (!first) {
              oss << ", ";
            }
            first = false;
            oss << key << "->" << count;
          }
          return oss.str();
        };
    const double successRate =
        nrrStats.attempts > 0
            ? (double)nrrStats.success / (double)nrrStats.attempts
            : 0.0;
    const double miniSolverCallRate =
        nrrCalls > 0
            ? (double)nrrStats.miniSolverCalls / (double)nrrCalls
            : 0.0;
    const double avgAttemptTimeSec =
        nrrCalls > 0 ? nrrStats.runtimeSecSum / (double)nrrCalls : 0.0;
    const auto avgPerCall = [&](double totalSec) {
      return nrrCalls > 0 ? totalSec / (double)nrrCalls : 0.0;
    };
    const double avgDestroyed =
        nrrStats.attempts > 0
            ? (double)nrrStats.removedTasksSum / (double)nrrStats.attempts
            : 0.0;
    const double avgNeighborhoodAgents =
        nrrStats.attempts > 0
            ? (double)nrrStats.neighborhoodAgentsSum / (double)nrrStats.attempts
            : 0.0;
    const double avgGlobalTouchedAgents =
        nrrStats.attempts > 0
            ? (double)nrrStats.globalTouchedAgentsSum / (double)nrrStats.attempts
            : 0.0;
    const double avgAcceptedDeltaSoc =
        nrrStats.acceptedDeltaSocCount > 0
            ? nrrStats.acceptedDeltaSocSum /
                  (double)nrrStats.acceptedDeltaSocCount
            : 0.0;
    const double avgSoftCandidateConflicts =
        nrrStats.softCandidateConflictSamples > 0
            ? nrrStats.softCandidateConflictSum /
                  (double)nrrStats.softCandidateConflictSamples
            : 0.0;
    const double avgSoftConflictDescentPerResolvedEntry =
        nrrStats.softModeResolvedEntries > 0
            ? (double)(nrrStats.softModeEntryConflictSum -
                       nrrStats.softModeExitConflictSum) /
                  (double)nrrStats.softModeResolvedEntries
            : 0.0;
    auto formatSoftHeuristicHistogram = [](const std::vector<int64_t>& values) {
      if (values.empty()) {
        return string("(none)");
      }
      auto nameForId = [](int id) -> const char* {
        switch (id) {
          case DestroyHeuristic::randomRemoval:
            return "random";
          case DestroyHeuristic::worstRemoval:
            return "worst";
          case DestroyHeuristic::conflictRemoval:
            return "conflict";
          case DestroyHeuristic::shawRemoval:
            return "shaw";
          case DestroyHeuristic::precedenceWaitRemoval:
            return "precedence_wait";
          case DestroyHeuristic::lowSlackRemoval:
            return "low_slack";
          case DestroyHeuristic::collisionSoftRemoval:
            return "collision_soft";
          case DestroyHeuristic::failureSoftRemoval:
            return "failure_soft";
          default:
            return "unknown";
        }
      };
      std::ostringstream oss;
      bool first = true;
      for (int i = 0; i < (int)values.size(); i++) {
        if (values[i] <= 0) {
          continue;
        }
        if (!first) {
          oss << ", ";
        }
        first = false;
        oss << nameForId(i) << "->" << values[i];
      }
      if (first) {
        return string("(all_zero)");
      }
      return oss.str();
    };
    std::cout << "\n=== NRR Stats ===\n";
    printMetric("Global reassignment enabled",
                lns.isNrrGlobalReassignEnabled() ? "true" : "false");
    printMetric("NRR calls", nrrCalls);
    printMetric("Attempts", nrrStats.attempts);
    printMetric("Mini-solver calls", nrrStats.miniSolverCalls);
    printMetric("Mini-solver call rate", miniSolverCallRate);
    printMetric("Success", nrrStats.success);
    printMetric("Fallback to standard", nrrStats.fallbackToStandard);
    printMetric("Success rate", successRate);
    printMetric("NRR total wall time (s)", nrrStats.runtimeSecSum);
    printMetric("Avg attempt time (s)", avgAttemptTimeSec);
    printMetric("Time proposal build (s)", nrrStats.timeProposalBuildSecSum);
    printMetric("Time mini-solver wall (s)", nrrStats.timeMiniSolverSecSum);
    printMetric("Time validation (s)", nrrStats.timeValidationSecSum);
    printMetric("Avg proposal build/call (ms)",
                avgPerCall(nrrStats.timeProposalBuildSecSum) * 1000.0);
    printMetric("Avg mini-solver wall/call (ms)",
                avgPerCall(nrrStats.timeMiniSolverSecSum) * 1000.0);
    printMetric("Avg validation/call (ms)",
                avgPerCall(nrrStats.timeValidationSecSum) * 1000.0);
    printMetric("Solver PBS count", nrrStats.solverPbs);
    printMetric("Solver CBS count", nrrStats.solverCbs);
    printMetric("Avg |D|", avgDestroyed);
    printMetric("Max |D|", nrrStats.removedTasksMax);
    printMetric("Avg |A'|", avgNeighborhoodAgents);
    printMetric("Max |A'|", nrrStats.neighborhoodAgentsMax);
    printMetric("Failure reasons",
                formatReasonHistogram(nrrStats.failureReasonHistogram));

    if (fullTimingDetail) {
      printMetric("Time setup (s)", nrrStats.timeSetupSecSum);
      printMetric("Time boundary windows (s)",
                  nrrStats.timeBoundaryWindowsSecSum);
      printMetric("Time temporal precheck (s)",
                  nrrStats.timeTemporalPrecheckSecSum);
      printMetric("Time input build (s)", nrrStats.timeInputBuildSecSum);
      printMetric("Time stitch (s)", nrrStats.timeStitchSecSum);
      printMetric("Time task-map rebuild (s)",
                  nrrStats.timeTaskMapRebuildSecSum);
      printMetric("Time terminal replan (s)",
                  nrrStats.timeTerminalReplanSecSum);
      printMetric("Time recompute objective (s)",
                  nrrStats.timeRecomputeObjectiveSecSum);
      printMetric("Avg setup/call (ms)",
                  avgPerCall(nrrStats.timeSetupSecSum) * 1000.0);
      printMetric("Avg boundary windows/call (ms)",
                  avgPerCall(nrrStats.timeBoundaryWindowsSecSum) * 1000.0);
      printMetric("Avg temporal precheck/call (ms)",
                  avgPerCall(nrrStats.timeTemporalPrecheckSecSum) * 1000.0);
      printMetric("Avg input build/call (ms)",
                  avgPerCall(nrrStats.timeInputBuildSecSum) * 1000.0);
      printMetric("Avg stitch/call (ms)",
                  avgPerCall(nrrStats.timeStitchSecSum) * 1000.0);
      printMetric("Avg task-map rebuild/call (ms)",
                  avgPerCall(nrrStats.timeTaskMapRebuildSecSum) * 1000.0);
      printMetric("Avg terminal replan/call (ms)",
                  avgPerCall(nrrStats.timeTerminalReplanSecSum) * 1000.0);
      printMetric("Avg recompute objective/call (ms)",
                  avgPerCall(nrrStats.timeRecomputeObjectiveSecSum) * 1000.0);
      printMetric("|D| distribution",
                  formatHistogram(nrrStats.removedTasksHistogram));
      printMetric("|A'| distribution",
                  formatHistogram(nrrStats.neighborhoodAgentsHistogram));
      printMetric("Soft candidates produced", nrrStats.softCandidatesProduced);
      printMetric("Soft candidates accepted", nrrStats.softCandidatesAccepted);
      printMetric("Soft candidates rejected", nrrStats.softCandidatesRejected);
      printMetric("Soft-recovery entries", nrrStats.softRecoveryEntries);
      printMetric("Soft-recovery exits", nrrStats.softRecoveryExits);
      printMetric("Soft-recovery accepted entry",
                  nrrStats.softRecoveryAcceptedEntry);
      printMetric("Soft-recovery accepted descent",
                  nrrStats.softRecoveryAcceptedDescent);
      printMetric("Soft-recovery rejected", nrrStats.softRecoveryRejected);
      printMetric("Soft-mode entry conflict sum",
                  nrrStats.softModeEntryConflictSum);
      printMetric("Soft-mode exit conflict sum",
                  nrrStats.softModeExitConflictSum);
      printMetric("Soft-mode resolved entries",
                  nrrStats.softModeResolvedEntries);
      printMetric("Soft-mode avg conflict descent/resolved-entry",
                  avgSoftConflictDescentPerResolvedEntry);
      printMetric("Soft-mode heuristic selections",
                  formatSoftHeuristicHistogram(
                      nrrStats.softModeSelectionsByDestroy));
      printMetric("Soft-mode heuristic accepted",
                  formatSoftHeuristicHistogram(
                      nrrStats.softModeAcceptedByDestroy));
      printMetric("Soft-mode heuristic best updates",
                  formatSoftHeuristicHistogram(
                      nrrStats.softModeBestUpdatesByDestroy));
      printMetric("Soft-candidate avg conflicts", avgSoftCandidateConflicts);
      printMetric("Stitched-invalid attempts", nrrStats.stitchedInvalidAttempts);
      printMetric("Stitched-invalid precedence violations",
                  nrrStats.stitchedInvalidPrecedenceViolations);
      printMetric("Stitched-invalid vertex collisions",
                  nrrStats.stitchedInvalidVertexCollisions);
      printMetric("Stitched-invalid edge-swap collisions",
                  nrrStats.stitchedInvalidEdgeSwapCollisions);
      printMetric("Stitched-invalid structural violations",
                  nrrStats.stitchedInvalidStructuralViolations);
      printMetric("Accepted avg DeltaSoC", avgAcceptedDeltaSoc);
    }
    if (lns.isNrrGlobalReassignEnabled() && fullTimingDetail) {
      printMetric("Global avg touched agents", avgGlobalTouchedAgents);
      printMetric("Global max touched agents",
                  nrrStats.globalTouchedAgentsMax);
      printMetric("Global touched-agent distribution",
                  formatHistogram(nrrStats.globalTouchedAgentsHistogram));
      printMetric("Global demotions", nrrStats.globalDemotionsCount);
      printMetric("Global patched tasks sum", nrrStats.globalPatchedTasksSum);
      printMetric("Global slot evaluations", nrrStats.globalSlotEvalCount);
      printMetric("Global no-feasible-slot count",
                  nrrStats.globalNoFeasibleSlotCount);
    }
  }

  if (lns.isDebugImprovementDiagnosticsEnabled()) {
    const auto& debugStats = lns.getImprovementDiagnosticsStats();
    const int64_t started = debugStats.iterationsStarted;
    const int64_t candidateEvaluated =
        debugStats.candidateValid + debugStats.candidateInvalid;
    const int64_t accepted = debugStats.accepted;
    const int64_t rejected = debugStats.rejected;
    const int64_t decisions = accepted + rejected;
    const double acceptanceRate =
        decisions > 0 ? (double)accepted / (double)decisions : 0.0;
    const double candidateValidRate =
        candidateEvaluated > 0
            ? (double)debugStats.candidateValid / (double)candidateEvaluated
            : 0.0;
    const double acceptedFeasibleNoBestRate =
        accepted > 0
            ? (double)debugStats.acceptedFeasibleNoBestUpdate /
                  (double)accepted
            : 0.0;
    const double acceptedFingerprintRepeatRate =
        accepted > 0
            ? (double)debugStats.acceptedFingerprintRepeat / (double)accepted
            : 0.0;
    const int64_t neighborhoodFingerprintSamples =
        debugStats.neighborhoodFingerprintUnique +
        debugStats.neighborhoodFingerprintRepeat;
    const double neighborhoodFingerprintRepeatRate =
        neighborhoodFingerprintSamples > 0
            ? (double)debugStats.neighborhoodFingerprintRepeat /
                  (double)neighborhoodFingerprintSamples
            : 0.0;
    const double neighborhoodMeanJaccardPrev =
        debugStats.neighborhoodJaccardPrevSamples > 0
            ? debugStats.neighborhoodJaccardPrevSum /
                  (double)debugStats.neighborhoodJaccardPrevSamples
            : 0.0;
    const double neighborhoodMeanJaccardPrevWhenRepeat =
        debugStats.neighborhoodJaccardPrevRepeatSamples > 0
            ? debugStats.neighborhoodJaccardPrevRepeatSum /
                  (double)debugStats.neighborhoodJaccardPrevRepeatSamples
            : 0.0;
    const double avgRemovedTasksPerNeighborhood =
        debugStats.neighborhoodsCount > 0
            ? (double)debugStats.removedTasksTotal /
                  (double)debugStats.neighborhoodsCount
            : 0.0;
    const double avgChangedRemovedTasksPerNeighborhood =
        debugStats.neighborhoodsCount > 0
            ? (double)(debugStats.removedTasksChangedAgentTotal +
                       debugStats.removedTasksChangedOrderTotal) /
                  (double)debugStats.neighborhoodsCount
            : 0.0;
    const double changedNeighborhoodRate =
        debugStats.neighborhoodsCount > 0
            ? (double)debugStats.neighborhoodsWithChanges /
                  (double)debugStats.neighborhoodsCount
            : 0.0;
    const double avgAcceptedRemovedTasks =
        debugStats.accepted > 0
            ? (double)debugStats.acceptedRemovedTasksTotal /
                  (double)debugStats.accepted
            : 0.0;
    const double avgAcceptedChangedRemovedTasks =
        debugStats.accepted > 0
            ? (double)(debugStats.acceptedRemovedTasksChangedAgentTotal +
                       debugStats.acceptedRemovedTasksChangedOrderTotal) /
                  (double)debugStats.accepted
            : 0.0;
    const double meanPreviousSoc =
        started > 0 ? debugStats.sumPreviousSoc / (double)started : 0.0;
    const double meanCandidateSoc =
        candidateEvaluated > 0
            ? debugStats.sumCandidateSoc / (double)candidateEvaluated
            : 0.0;
    const double meanIncumbentSoc =
        debugStats.incumbentSocSamples > 0
            ? debugStats.sumIncumbentSoc /
                  (double)debugStats.incumbentSocSamples
            : 0.0;
    const double meanPreviousConflictSignal =
        candidateEvaluated > 0
            ? debugStats.sumPreviousConflictSignal /
                  (double)candidateEvaluated
            : 0.0;
    const double meanCandidateConflictSignal =
        candidateEvaluated > 0
            ? debugStats.sumCandidateConflictSignal /
                  (double)candidateEvaluated
            : 0.0;
    const double avgSoftCandidateConflict =
        debugStats.softCandidateConflictSamples > 0
            ? debugStats.softCandidateConflictSum /
                  (double)debugStats.softCandidateConflictSamples
            : 0.0;
    const double avgSoftConflictDescentPerResolvedEntry =
        debugStats.softModeResolvedEntries > 0
            ? (double)(debugStats.softModeEntryConflictSum -
                       debugStats.softModeExitConflictSum) /
                  (double)debugStats.softModeResolvedEntries
            : 0.0;

    const double measuredDebugSec =
        debugStats.timeDestroyAndPrepareSec +
        debugStats.timeRepairAndCommitSec + debugStats.timeJoinPathsSec +
        debugStats.timeRecomputeSocSec +
        debugStats.timeValidationSec + debugStats.timeAcceptanceSec +
        debugStats.timeBookkeepingSec;
    const double measuredShareOfRuntime =
        lns.runtime > 0.0 ? measuredDebugSec / lns.runtime : 0.0;
    const auto phaseShare = [&](double phaseSec) -> double {
      return measuredDebugSec > 0.0 ? phaseSec / measuredDebugSec : 0.0;
    };
    const double regretEvalShareOfRepair =
        debugStats.timeRepairAndCommitSec > 0.0
            ? debugStats.timeRegretCandidateEvalSec /
                  debugStats.timeRepairAndCommitSec
            : 0.0;
    const double regretCommitShareOfRepair =
        debugStats.timeRepairAndCommitSec > 0.0
            ? debugStats.timeRegretCommitSec / debugStats.timeRepairAndCommitSec
            : 0.0;
    const double regretEvalAndCommitShareOfRepair =
        debugStats.timeRepairAndCommitSec > 0.0
            ? (debugStats.timeRegretCandidateEvalSec +
               debugStats.timeRegretCommitSec) /
                  debugStats.timeRepairAndCommitSec
            : 0.0;
    const double regretLowLevelShareOfRepair =
        debugStats.timeRepairAndCommitSec > 0.0
            ? debugStats.timeRegretLowLevelSec /
                  debugStats.timeRepairAndCommitSec
            : 0.0;
    const double regretLowLevelShareOfRegretEvalAndCommit =
        (debugStats.timeRegretCandidateEvalSec +
         debugStats.timeRegretCommitSec) > 0.0
            ? debugStats.timeRegretLowLevelSec /
                  (debugStats.timeRegretCandidateEvalSec +
                   debugStats.timeRegretCommitSec)
            : 0.0;

    std::cout << "\n=== Improvement Diagnostics ===\n";
    printMetric("Iterations started", started);
    printMetric("Iterations with candidate eval", candidateEvaluated);
    printMetric("Candidate valid", debugStats.candidateValid);
    printMetric("Candidate invalid", debugStats.candidateInvalid);
    printMetric("Candidate valid rate", candidateValidRate);
    printMetric("Accepted", accepted);
    printMetric("Rejected", rejected);
    printMetric("Guard rejected", debugStats.guardRejected);
    printMetric("Acceptance rate", acceptanceRate);
    printMetric("Accepted utility better/equal",
                debugStats.acceptedAsBetterOrEqualUtility);
    printMetric("Accepted utility worse", debugStats.acceptedAsWorseUtility);
    printMetric("Accepted SoC better vs previous",
                debugStats.acceptedSocBetterVsPrevious);
    printMetric("Accepted SoC equal vs previous",
                debugStats.acceptedSocEqualVsPrevious);
    printMetric("Accepted SoC worse vs previous",
                debugStats.acceptedSocWorseVsPrevious);
    printMetric("Accepted SoC better vs incumbent",
                debugStats.acceptedSocBetterVsIncumbent);
    printMetric("Accepted SoC equal vs incumbent",
                debugStats.acceptedSocEqualVsIncumbent);
    printMetric("Accepted SoC worse vs incumbent",
                debugStats.acceptedSocWorseVsIncumbent);
    printMetric("Feasible best updates", debugStats.feasibleBestUpdates);
    printMetric("Feasible no-best updates", debugStats.feasibleNoBestUpdate);
    printMetric("Accepted feasible no-best updates",
                debugStats.acceptedFeasibleNoBestUpdate);
    printMetric("Accepted feasible-no-best rate",
                acceptedFeasibleNoBestRate);
    printMetric("Accepted invalid", debugStats.acceptedInvalid);
    printMetric("Soft candidates produced",
                debugStats.softCandidateProduced);
    printMetric("Soft candidates accepted",
                debugStats.softCandidateAccepted);
    printMetric("Soft candidates rejected",
                debugStats.softCandidateRejected);
    printMetric("Soft-recovery entries",
                debugStats.softRecoveryEntries);
    printMetric("Soft-recovery exits",
                debugStats.softRecoveryExits);
    printMetric("Soft-recovery accepted entry",
                debugStats.softRecoveryAcceptedEntry);
    printMetric("Soft-recovery accepted descent",
                debugStats.softRecoveryAcceptedDescent);
    printMetric("Soft-recovery rejected",
                debugStats.softRecoveryRejected);
    printMetric("Soft-mode entry conflict sum",
                debugStats.softModeEntryConflictSum);
    printMetric("Soft-mode exit conflict sum",
                debugStats.softModeExitConflictSum);
    printMetric("Soft-mode resolved entries",
                debugStats.softModeResolvedEntries);
    printMetric("Soft-mode avg conflict descent/resolved-entry",
                avgSoftConflictDescentPerResolvedEntry);
    printMetric("Avg soft-candidate conflicts",
                avgSoftCandidateConflict);
    printMetric("Conflict signal better", debugStats.conflictSignalBetter);
    printMetric("Conflict signal equal", debugStats.conflictSignalEqual);
    printMetric("Conflict signal worse", debugStats.conflictSignalWorse);
    printMetric("Accepted conflict better",
                debugStats.acceptedConflictSignalBetter);
    printMetric("Accepted conflict equal",
                debugStats.acceptedConflictSignalEqual);
    printMetric("Accepted conflict worse",
                debugStats.acceptedConflictSignalWorse);
    printMetric("Accepted fingerprint unique",
                debugStats.acceptedFingerprintUnique);
    printMetric("Accepted fingerprint repeat",
                debugStats.acceptedFingerprintRepeat);
    printMetric("Accepted fingerprint repeat rate",
                acceptedFingerprintRepeatRate);
    printMetric("Neighborhoods analyzed", debugStats.neighborhoodsCount);
    printMetric("Neighborhoods with changes",
                debugStats.neighborhoodsWithChanges);
    printMetric("Neighborhoods without changes",
                debugStats.neighborhoodsWithoutChanges);
    printMetric("Neighborhood fingerprint unique",
                debugStats.neighborhoodFingerprintUnique);
    printMetric("Neighborhood fingerprint repeat",
                debugStats.neighborhoodFingerprintRepeat);
    printMetric("Neighborhood fingerprint repeat rate",
                neighborhoodFingerprintRepeatRate);
    printMetric("Neighborhood repeat streak max",
                debugStats.neighborhoodRepeatStreakMax);
    printMetric("Neighborhood mean Jaccard vs previous",
                neighborhoodMeanJaccardPrev);
    printMetric("Neighborhood mean Jaccard vs previous (repeat only)",
                neighborhoodMeanJaccardPrevWhenRepeat);
    printMetric("Neighborhoods with changes rate",
                changedNeighborhoodRate);
    printMetric("Avg removed tasks/neighborhood",
                avgRemovedTasksPerNeighborhood);
    printMetric("Avg changed removed tasks/neighborhood",
                avgChangedRemovedTasksPerNeighborhood);
    printMetric("Max removed tasks/neighborhood",
                debugStats.removedTasksMax);
    printMetric("Max changed removed tasks/neighborhood",
                debugStats.changedTasksPerNeighborhoodMax);
    printMetric("Avg accepted removed tasks",
                avgAcceptedRemovedTasks);
    printMetric("Avg accepted changed removed tasks",
                avgAcceptedChangedRemovedTasks);
    printMetric("Max accepted removed tasks",
                debugStats.acceptedRemovedTasksMax);
    printMetric("Max accepted changed removed tasks",
                debugStats.acceptedChangedTasksPerNeighborhoodMax);
    printMetric("Early aborts (prepare)", debugStats.earlyAbortPrepare);
    printMetric("Early aborts (repair)", debugStats.earlyAbortRepair);
    printMetric("Early aborts (join)", debugStats.earlyAbortJoin);
    printMetric("Mean previous SoC", meanPreviousSoc);
    printMetric("Mean candidate SoC", meanCandidateSoc);
    printMetric("Mean incumbent SoC", meanIncumbentSoc);
    printMetric("Mean previous conflict signal",
                meanPreviousConflictSignal);
    printMetric("Mean candidate conflict signal",
                meanCandidateConflictSignal);

    std::cout << "\n=== Iteration Time Split (Debug) ===\n";
    printMetric("Measured debug time (s)", measuredDebugSec);
    printMetric("Measured time / runtime", measuredShareOfRuntime);
    printMetric("Destroy+prepare time (s)",
                debugStats.timeDestroyAndPrepareSec);
    printMetric("Repair+commit time (s)",
                debugStats.timeRepairAndCommitSec);
    printMetric("Regret candidate-eval time (s)",
                debugStats.timeRegretCandidateEvalSec);
    printMetric("Regret commit time (s)",
                debugStats.timeRegretCommitSec);
    printMetric("Regret low-level time (s)",
                debugStats.timeRegretLowLevelSec);
    printMetric("Join paths time (s)", debugStats.timeJoinPathsSec);
    printMetric("Recompute SoC time (s)", debugStats.timeRecomputeSocSec);
    printMetric("Validation time (s)", debugStats.timeValidationSec);
    printMetric("Acceptance time (s)", debugStats.timeAcceptanceSec);
    printMetric("Bookkeeping time (s)", debugStats.timeBookkeepingSec);
    printMetric("Destroy+prepare share", phaseShare(debugStats.timeDestroyAndPrepareSec));
    printMetric("Repair+commit share", phaseShare(debugStats.timeRepairAndCommitSec));
    printMetric("Regret candidate-eval / repair",
                regretEvalShareOfRepair);
    printMetric("Regret commit / repair",
                regretCommitShareOfRepair);
    printMetric("Regret (eval+commit) / repair",
                regretEvalAndCommitShareOfRepair);
    printMetric("Regret low-level / repair",
                regretLowLevelShareOfRepair);
    printMetric("Regret low-level / (regret eval+commit)",
                regretLowLevelShareOfRegretEvalAndCommit);
    printMetric("Join paths share", phaseShare(debugStats.timeJoinPathsSec));
    printMetric("Recompute SoC share", phaseShare(debugStats.timeRecomputeSocSec));
    printMetric("Validation share", phaseShare(debugStats.timeValidationSec));
    printMetric("Acceptance share", phaseShare(debugStats.timeAcceptanceSec));
    printMetric("Bookkeeping share", phaseShare(debugStats.timeBookkeepingSec));
  }

  const auto nrrStatsForSummary = lns.getNrrStats();
  std::error_code metricsEc;
  const std::filesystem::path metricsDir = resolveMetricsOutputDir();
  std::filesystem::create_directories(metricsDir, metricsEc);
  if (metricsEc) {
    std::cout << "\n=== Artifacts ===\n";
    printMetric("Run summary TSV", "write_failed");
    printMetric("Trajectory TSV", "write_failed");
    printMetric("Artifacts error", metricsEc.message());
    return;
  }

  const std::string baseTag =
      currentTimestampCompact() + "_seed" + std::to_string(lns.getSeed());
  auto makeUniqueArtifactPath = [&](const std::string& stem,
                                    const std::string& extension) {
    std::filesystem::path outPath =
        metricsDir / (stem + "_" + baseTag + extension);
    int suffix = 1;
    std::error_code existsEc;
    while (std::filesystem::exists(outPath, existsEc) && !existsEc) {
      outPath = metricsDir /
                (stem + "_" + baseTag + "_" + std::to_string(suffix) +
                 extension);
      suffix++;
    }
    return outPath;
  };

  const std::filesystem::path trajectoryPath =
      makeUniqueArtifactPath("trajectory", ".tsv");
  const std::filesystem::path runSummaryPath =
      makeUniqueArtifactPath("run_summary", ".tsv");
  const std::filesystem::path solutionJsonPath =
      makeUniqueArtifactPath("solution", ".json");

  writeTrajectoryTsv(trajectoryPath, lns.iterationStats);
  writeSolutionJson(solutionJsonPath, lns, solution, finalSolutionMetrics,
                    optimizationObjective, objectiveValue);

  std::vector<std::pair<std::string, std::string>> summaryFields;
  auto addField = [&](const std::string& key, const std::string& value) {
    summaryFields.emplace_back(key, value);
  };
  auto addBoolField = [&](const std::string& key, bool value) {
    addField(key, value ? "true" : "false");
  };
  auto addIntField = [&](const std::string& key, int64_t value) {
    addField(key, std::to_string(value));
  };
  auto addDoubleField = [&](const std::string& key, double value,
                            int precision = 6) {
    addField(key, std::isfinite(value) ? formatDoubleValue(value, precision)
                                       : "n/a");
  };

  addField("timestamp", currentTimestampCompact());
  addIntField("seed", static_cast<int64_t>(lns.getSeed()));
  addBoolField("success", success);
  addField("optimization_objective", optimizationObjective);
  addIntField("objective_value", objectiveValue);
  addIntField("final_soc", finalSoc);
  addIntField("final_makespan", finalMakespan);
  addDoubleField("final_total_precedence_wait", finalPrecWait, 4);
  addDoubleField("runtime_s", lns.runtime, 6);
  addDoubleField("initial_runtime_measured_s", initialRuntimeMeasuredSec, 6);
  addDoubleField("initial_runtime_reported_s", initialRuntimeReportedSec, 6);
  addDoubleField("initial_seed_runtime_from_log_s",
                 hasInitialSeedRuntimeFromLog ? initialSeedRuntimeFromLogSec
                                              : std::numeric_limits<double>::quiet_NaN(),
                 6);
  addDoubleField("lns_loop_runtime_s", lnsLoopRuntimeSec, 6);
  addDoubleField("post_refine_runtime_s", postRefineRuntimeSec, 6);
  addDoubleField("uncovered_runtime_s", uncoveredRuntimeSec, 6);
  addField("initial_solution_requested", lns.getInitialSolutionRequested());
  addField("initial_solution_effective", lns.getInitialSolutionEffective());
  addBoolField("initial_fallback_used", lns.wasInitialSolutionFallbackUsed());
  addField("initial_fallback_reason", lns.getInitialSolutionFallbackReason());
  addField("initial_seed_log_path",
           initialSeedLogPath.empty() ? "n/a" : initialSeedLogPath);
  addField("initial_seed_solver", initialSeedSolver);
  addBoolField("post_refine_enabled", lns.isPostRefineEnabled());
  addBoolField("post_refine_attempted", lns.wasPostRefineAttempted());
  addBoolField("post_refine_accepted", lns.wasPostRefineAccepted());
  addField("post_refine_solver", lns.getPostRefineSolver());
  addDoubleField("post_refine_timeout_s",
                 static_cast<double>(lns.getPostRefineTimeoutSec()), 0);
  addIntField("iterations", lnsIterations);
  addDoubleField("iterations_per_sec", iterPerSec, 6);
  addDoubleField("avg_iteration_runtime_ms", avgIterSec * 1000.0, 4);
  addDoubleField("p95_iteration_runtime_ms", p95IterSec * 1000.0, 4);
  addDoubleField("first_feasible_runtime_s", stats.firstFeasibleRuntime, 6);
  addDoubleField("time_to_best_feasible_s", timeToBestFeasible, 6);
  addIntField("improving_feasible_updates", stats.numImprovingFeasibleUpdates);
  addIntField("total_feasible_iterations", stats.numFeasibleIterations);
  if (initialMetricsAvailable) {
    addIntField("initial_soc", initialSoc);
    addIntField("delta_soc", deltaSoc);
    addDoubleField("relative_improvement_soc_pct", relativeImprovementSoc, 4);
    addIntField("initial_makespan", initialMakespan);
    addIntField("delta_makespan", deltaMakespan);
    addDoubleField("initial_total_precedence_wait", initialPrecWait, 4);
    addDoubleField("delta_total_precedence_wait", deltaPrecWait, 4);
  } else {
    addField("initial_soc", "n/a");
    addField("delta_soc", "n/a");
    addField("relative_improvement_soc_pct", "n/a");
    addField("initial_makespan", "n/a");
    addField("delta_makespan", "n/a");
    addField("initial_total_precedence_wait", "n/a");
    addField("delta_total_precedence_wait", "n/a");
  }
  addIntField("tasks_moved_between_agents",
              assignmentDelta.tasksMovedBetweenAgents);
  addIntField("tasks_reordered_within_agent",
              assignmentDelta.tasksReorderedWithinAgent);
  addIntField("agents_touched", assignmentDelta.agentsTouched);
  addIntField("final_assignment_diff_count",
              assignmentDelta.finalAssignmentDiffCount);
  addDoubleField("final_assignment_diff_fraction",
                 assignmentDelta.finalAssignmentDiffFraction, 6);
  addIntField("accepted_better", acceptedBetter);
  addIntField("accepted_equal", acceptedEqual);
  addIntField("accepted_worse", acceptedWorse);
  addIntField("rejected_invalid", rejectedInvalid);
  addIntField("rejected_failfind", rejectedFailFind);
  addIntField("rejected_guard", rejectedGuard);
  addIntField("total_accepts", totalAccepts);
  addIntField("max_tasks_per_agent", maxTasksPerAgent);
  addDoubleField("stdev_tasks_per_agent", tasksPerAgentStddev, 6);
  addDoubleField("stdev_agent_costs", agentCostStddev, 6);
  addIntField("low_level_calls", lowLevelStats.calls);
  addIntField("low_level_expanded", lowLevelStats.expanded);
  addIntField("low_level_generated", lowLevelStats.generated);
  addIntField("low_level_found", lowLevelStats.found);
  addIntField("low_level_timeout", lowLevelStats.timeout);
  addDoubleField("candidate_insertions_tried_per_removed_task",
                 candidateTriedPerRemovedTask, 6);
  addDoubleField("low_level_calls_per_removed_task",
                 lowLevelCallsPerRemovedTask, 6);
  addDoubleField("low_level_expanded_per_removed_task",
                 lowLevelExpandedPerRemovedTask, 6);
  addIntField("regret_recompute_calls", regretStats.recomputeCalls);
  addIntField("regret_tasks_evaluated", regretStats.tasksEvaluated);
  addIntField("regret_task_agent_evaluations", regretStats.agentEvaluations);
  addIntField("regret_candidate_insertions_tried",
              regretStats.candidateInsertionsTried);
  addIntField("regret_candidate_insertions_feasible",
              regretStats.candidateInsertionsFeasible);
  addIntField("nrr_calls",
              std::max<int64_t>(nrrStatsForSummary.calls,
                                nrrStatsForSummary.attempts));
  addIntField("nrr_attempts", nrrStatsForSummary.attempts);
  addIntField("nrr_success", nrrStatsForSummary.success);
  addIntField("nrr_fallback_to_standard", nrrStatsForSummary.fallbackToStandard);
  addField("solution_json_path", solutionJsonPath.string());

  writeRunSummaryWideTsv(runSummaryPath, summaryFields);

  std::error_code existsEc;
  const bool trajectoryWritten = std::filesystem::exists(trajectoryPath, existsEc);
  existsEc.clear();
  const bool summaryWritten = std::filesystem::exists(runSummaryPath, existsEc);
  existsEc.clear();
  const bool solutionJsonWritten =
      std::filesystem::exists(solutionJsonPath, existsEc);

  std::cout << "\n=== Artifacts ===\n";
  printMetric("Run summary TSV",
              summaryWritten ? runSummaryPath.string() : "write_failed");
  printMetric("Trajectory TSV",
              trajectoryWritten ? trajectoryPath.string() : "write_failed");
  printMetric("Solution JSON",
              solutionJsonWritten ? solutionJsonPath.string() : "write_failed");
}
