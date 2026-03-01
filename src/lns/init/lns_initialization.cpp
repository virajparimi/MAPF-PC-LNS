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

namespace {
inline void resetInitialSolutionState(const Instance& instance, Solution& solution) {
  for (int& assignedAgent : solution.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }
  for (int agent = 0; agent < instance.getAgentNum(); agent++) {
    solution.agents[agent].taskPaths.clear();
    solution.agents[agent].path = AgentTaskPath();
    solution.agents[agent].terminalPath = AgentTaskPath();
    solution.agents[agent].terminalPathActive = false;
    solution.agents[agent].taskAssignments.clear();
    solution.agents[agent].intraPrecedenceConstraints.clear();
    solution.agents[agent].intraPrecedenceDirty = false;
  }
}
}  // namespace

bool LNS::buildInitialSolutionCore(bool enforceInterAgentTiming,
                                   const char* callerName) {
  resetInitialSolutionState(instance_, solution_);

  if (!greedyTaskAssignment(&instance_, &solution_)) {
    PLOGE << "Failed to compute greedy task assignment\n";
    return false;
  }

  size_t expectedIntraSize = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    vector<int> taskLocations = instance_.getTaskLocations(agentTasks);
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(agentTasks.size(), AgentTaskPath());
    if (!agentTasks.empty()) {
      expectedIntraSize += agentTasks.size() - 1;
    }
  }

  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.reserve(inputPrecedenceConstraints.size() +
                                expectedIntraSize);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
      precedenceConstraints.emplace_back(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  vector<int> planningOrder;
  if (!topologicalSort(&instance_, precedenceConstraints, planningOrder)) {
    PLOGE << "Topological sorting failed\n";
    return false;
  }

  const int taskCount = instance_.getTasksNum();
  const int agentCount = instance_.getAgentNum();
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, taskCount);

  vector<vector<int>> predecessors;
  if (enforceInterAgentTiming) {
    predecessors.assign(taskCount, {});
    for (const auto& edge : precedenceConstraints) {
      if (edge.first < 0 || edge.first >= taskCount || edge.second < 0 ||
          edge.second >= taskCount) {
        continue;
      }
      predecessors[edge.second].push_back(edge.first);
    }
  }

  initialPaths_.assign(taskCount, AgentTaskPath());
  vector<char> plannedTasks(taskCount, 0);
  const bool strictReservedPathChecks = enforceInterAgentTiming;

  for (int task : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << callerName << ": task " << task
            << " is not assigned to any agent\n";
      return false;
    }

    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    const int taskPosition =
        (task >= 0 && task < (int)assignmentIndex.pos.size())
            ? assignmentIndex.pos[task]
            : UNASSIGNED;
    if (taskPosition < 0 || taskPosition >= (int)agentTasks.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << callerName << ": invalid local task index " << taskPosition
            << " for task " << task << " on agent " << agent << "\n";
      return false;
    }

    int startTime = 0;
    if (taskPosition > 0) {
      const int previousTask = agentTasks[taskPosition - 1];
      if (previousTask < 0 || previousTask >= taskCount ||
          initialPaths_[previousTask].empty()) {
        PLOGE << callerName << ": missing path for predecessor task "
              << previousTask << "\n";
        return false;
      }
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    if (enforceInterAgentTiming) {
      int earliestGoalTime = 0;
      for (int pred : predecessors[task]) {
        if (pred < 0 || pred >= taskCount) {
          continue;
        }
        if (initialPaths_[pred].empty()) {
          PLOGE << callerName << ": predecessor task " << pred << " for task "
                << task
                << " is unexpectedly unplanned in topological order\n";
          return false;
        }
        earliestGoalTime =
            max(earliestGoalTime, initialPaths_[pred].endTimeChecked() + 1);
      }
      constraintTable.goalLocation = instance_.getTaskLocations(task);
      constraintTable.lengthMin =
          max(constraintTable.lengthMin, earliestGoalTime);
      constraintTable.latestTimestep =
          max(constraintTable.latestTimestep, constraintTable.lengthMin);
    } else {
      if (!buildConstraintTable(constraintTable, task)) {
        PLOGE << callerName << ": failed to build constraint table for task "
              << task << " (agent " << agent << ")\n";
        return false;
      }
    }

    for (int plannedTask = 0; plannedTask < taskCount; plannedTask++) {
      if (!plannedTasks[plannedTask]) {
        continue;
      }
      const int plannedAgent = solution_.getAgentWithTask(plannedTask);
      if (plannedAgent == UNASSIGNED) {
        if (strictReservedPathChecks) {
          PLOGE << callerName << ": planned task " << plannedTask
                << " has no assigned agent\n";
          return false;
        }
        continue;
      }
      const int plannedTaskPos =
          (plannedTask >= 0 && plannedTask < (int)assignmentIndex.pos.size())
              ? assignmentIndex.pos[plannedTask]
              : UNASSIGNED;
      if (plannedTaskPos < 0 || plannedTaskPos >=
                                   (int)solution_.agents[plannedAgent]
                                       .taskPaths.size()) {
        if (strictReservedPathChecks) {
          PLOGE << callerName << ": invalid reserved task index "
                << plannedTaskPos << " for task " << plannedTask << " (agent "
                << plannedAgent << ")\n";
          return false;
        }
        continue;
      }
      const auto& plannedPath =
          solution_.agents[plannedAgent].taskPaths[plannedTaskPos];
      if (plannedPath.empty()) {
        if (strictReservedPathChecks) {
          PLOGE << callerName << ": planned task " << plannedTask
                << " has empty path when reserving\n";
          return false;
        }
        continue;
      }
      const bool isFinalTask =
          (plannedTaskPos + 1 ==
           (int)solution_.agents[plannedAgent].taskAssignments.size());
      reservePathWithGoalPolicy(constraintTable, plannedPath, isFinalTask);
    }

    initialPaths_[task] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[task].empty()) {
      PLOGE << callerName << ": no path for agent " << agent << " and task "
            << task << " (ll_outcome=" << getLastLowLevelOutcomeName()
            << ", remaining_budget_sec=" << getLastLowLevelRemainingBudgetSec()
            << ", effective_timeout_sec=" << getLastLowLevelEffectiveTimeoutSec()
            << ")\n";
      logInitialSegmentFailureDiagnostics(
          instance_, solution_, constraintTable, agent, task, taskPosition,
          startTime, *solution_.agents[agent].pathPlanner,
          getLastLowLevelRemainingBudgetSec(),
          getLastLowLevelEffectiveTimeoutSec());
      return false;
    }

    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[task];
    plannedTasks[task] = 1;
  }

  vector<int> agentsToCompute(agentCount);
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << callerName << ": failed to join agent paths\n";
    return false;
  }

  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < agentCount; agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts = clampSocToInt(initialSumOfCosts, callerName);
  return true;
}

bool LNS::buildGreedySolution() {
  return buildInitialSolutionCore(false, "buildGreedySolution");
}

bool LNS::buildPrioritizedInitialSolution() {
  return buildInitialSolutionCore(true, "buildPrioritizedInitialSolution");
}

bool LNS::extractFeasibleSolution() {

  // Only update the feasible solution if the new solution has better cost!
  if (incumbentSolution_.agentPaths.empty() ||
      incumbentSolution_.sumOfCosts > solution_.sumOfCosts) {
    overwriteIncumbentFromCurrentSolution();
    return true;
  }
  return false;
}

void LNS::overwriteIncumbentFromCurrentSolution() {
  incumbentSolution_.numOfCols = instance_.numOfCols;
  incumbentSolution_.sumOfCosts = solution_.sumOfCosts;
  incumbentSolution_.agentPaths.resize(instance_.getAgentNum());
  incumbentSolution_.agentTaskAssignments.resize(instance_.getAgentNum());
  incumbentSolution_.agentTaskPaths.resize(instance_.getAgentNum());
  incumbentSolution_.taskAgentMap = solution_.taskAgentMap;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      incumbentSolution_.agentPaths[agent] = solution_.agents[agent].path;
      incumbentSolution_.agentTaskAssignments[agent] =
          solution_.agents[agent].taskAssignments;
      incumbentSolution_.agentTaskPaths[agent] =
          solution_.agents[agent].taskPaths;
    } else {
      incumbentSolution_.agentPaths[agent] = AgentTaskPath();
      incumbentSolution_.agentTaskAssignments[agent].clear();
      incumbentSolution_.agentTaskPaths[agent].clear();
    }
  }
}
