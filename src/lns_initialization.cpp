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

bool LNS::buildGreedySolution() {

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].terminalPath = AgentTaskPath();
    solution_.agents[agent].terminalPathActive = false;
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
    solution_.agents[agent].intraPrecedenceDirty = false;
  }

  // Assign tasks
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

  // Compute the precedence constraints based on current task assignments
  // Intra-agent precedence constraints
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

  // Find paths based on the task assignments
  // First we need to sort the tasks based on the precedence constraints
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return success;
  }

  // Following the topological order we find the paths for each task
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  vector<char> plannedTasks(instance_.getTasksNum(), 0);
  for (int id : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    const int task = id;
    int startTime = 0;
    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildGreedySolution: task " << task
            << " is not assigned to any agent\n";
      return false;
    }
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildGreedySolution: invalid local task index " << taskPosition
            << " for task " << task << " on agent " << agent << "\n";
      return false;
    }
    if (taskPosition != 0) {
      int previousTask =
          solution_.agents[agent].taskAssignments[taskPosition - 1];
      if (initialPaths_[previousTask].empty()) {
        PLOGE << "buildGreedySolution: missing path for predecessor task "
              << previousTask << "\n";
        return false;
      }
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    PLOGI << "Planning for agent " << agent << " and task " << task << "\n";

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    if (!buildConstraintTable(constraintTable, task)) {
      PLOGE << "buildGreedySolution: failed to build constraint table for task "
            << task << " (agent " << agent << ")\n";
      return false;
    }
    // Strengthen greedy initialization with collision constraints from tasks
    // that have already been planned in this pass. This keeps initialization
    // robust across low-level planners that may choose different but valid
    // shortest paths under precedence-only constraints.
    for (int plannedTask = 0; plannedTask < instance_.getTasksNum();
         plannedTask++) {
      if (!plannedTasks[plannedTask]) {
        continue;
      }
      const int plannedAgent = solution_.getAgentWithTask(plannedTask);
      if (plannedAgent == UNASSIGNED) {
        continue;
      }
      const int plannedTaskPos =
          (plannedTask >= 0 && plannedTask < (int)assignmentIndex.pos.size())
              ? assignmentIndex.pos[plannedTask]
              : UNASSIGNED;
      if (plannedTaskPos == UNASSIGNED ||
          plannedTaskPos >=
              (int)solution_.agents[plannedAgent].taskPaths.size()) {
        continue;
      }
      const auto& plannedPath =
          solution_.agents[plannedAgent].taskPaths[plannedTaskPos];
      if (plannedPath.empty()) {
        continue;
      }
      const bool isFinalTask =
          (plannedTaskPos + 1 ==
           (int)solution_.agents[plannedAgent].taskAssignments.size());
      reservePathWithGoalPolicy(constraintTable, plannedPath, isFinalTask);
    }

    initialPaths_[id] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[id].empty()) {
      PLOGE << "No path exists for agent " << agent << " and task " << task
            << " (ll_outcome=" << getLastLowLevelOutcomeName()
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
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[id];
    plannedTasks[task] = 1;
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolution: failed to join agent paths\n";
    return false;
  }

  // Gather the information
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts = clampSocToInt(initialSumOfCosts, "buildGreedySolution");
  return true;
}

bool LNS::buildPrioritizedInitialSolution() {

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].terminalPath = AgentTaskPath();
    solution_.agents[agent].terminalPathActive = false;
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
    solution_.agents[agent].intraPrecedenceDirty = false;
  }

  // Reuse existing greedy task assignment for now; prioritized initialization
  // focuses on robust path construction under inter-agent reservations.
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

  // Global precedence constraints = input + intra-agent ordering.
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

  // Task-level topological order used when traversing each agent's chain.
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return false;
  }
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());

  // Build predecessor adjacency to enforce precedence timing.
  vector<vector<int>> predecessors(instance_.getTasksNum());
  for (const auto& edge : precedenceConstraints) {
    if (edge.first < 0 || edge.first >= instance_.getTasksNum() ||
        edge.second < 0 || edge.second >= instance_.getTasksNum()) {
      continue;
    }
    predecessors[edge.second].push_back(edge.first);
  }

  const int agentCount = instance_.getAgentNum();
  const int taskCount = instance_.getTasksNum();
  initialPaths_.assign(taskCount, AgentTaskPath());
  vector<char> plannedTasks(taskCount, 0);

  // Plan tasks directly in global topological order. This avoids the
  // cross-agent ordering deadlocks caused by agent-level priority DAG cycles.
  for (int task : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildPrioritizedInitialSolution: task " << task
            << " is not assigned to any agent\n";
      return false;
    }

    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 || taskPosition >= (int)agentTasks.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildPrioritizedInitialSolution: invalid local task index "
            << taskPosition << " for task " << task << " on agent " << agent
            << "\n";
      return false;
    }

    int startTime = 0;
    if (taskPosition > 0) {
      const int previousTask = agentTasks[taskPosition - 1];
      if (previousTask < 0 || previousTask >= taskCount ||
          initialPaths_[previousTask].empty()) {
        PLOGE << "buildPrioritizedInitialSolution: missing prior task path "
              << "for task " << task << " on agent " << agent << "\n";
        return false;
      }
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    int earliestGoalTime = 0;
    for (int pred : predecessors[task]) {
      if (pred < 0 || pred >= taskCount) {
        continue;
      }
      if (initialPaths_[pred].empty()) {
        PLOGE << "buildPrioritizedInitialSolution: predecessor task " << pred
              << " for task " << task
              << " is unexpectedly unplanned in topological order\n";
        return false;
      }
      earliestGoalTime =
          max(earliestGoalTime, initialPaths_[pred].endTimeChecked() + 1);
    }

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    constraintTable.goalLocation = instance_.getTaskLocations(task);
    constraintTable.lengthMin = max(constraintTable.lengthMin, earliestGoalTime);
    constraintTable.latestTimestep =
        max(constraintTable.latestTimestep, constraintTable.lengthMin);

    // Reserve all already planned tasks, regardless of owning agent.
    for (int reservedTask = 0; reservedTask < taskCount; reservedTask++) {
      if (!plannedTasks[reservedTask]) {
        continue;
      }
      const int reservedAgent = solution_.getAgentWithTask(reservedTask);
      if (reservedAgent == UNASSIGNED) {
        PLOGE << "buildPrioritizedInitialSolution: planned task " << reservedTask
              << " has no assigned agent\n";
        return false;
      }
      const int reservedTaskPos =
          (reservedTask >= 0 && reservedTask < (int)assignmentIndex.pos.size())
              ? assignmentIndex.pos[reservedTask]
              : UNASSIGNED;
      if (reservedTaskPos < 0 ||
          reservedTaskPos >=
              (int)solution_.agents[reservedAgent].taskPaths.size()) {
        PLOGE << "buildPrioritizedInitialSolution: invalid reserved task index "
              << reservedTaskPos << " for task " << reservedTask << " (agent "
              << reservedAgent << ")\n";
        return false;
      }
      const auto& reservedPath =
          solution_.agents[reservedAgent].taskPaths[reservedTaskPos];
      if (reservedPath.empty()) {
        PLOGE << "buildPrioritizedInitialSolution: planned task "
              << reservedTask << " has empty path when reserving\n";
        return false;
      }
      const bool isFinalTask =
          (reservedTaskPos + 1 ==
           (int)solution_.agents[reservedAgent].taskAssignments.size());
      reservePathWithGoalPolicy(constraintTable, reservedPath, isFinalTask);
    }

    initialPaths_[task] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[task].empty()) {
      PLOGE << "buildPrioritizedInitialSolution: no path for agent " << agent
            << " and task " << task << " (ll_outcome="
            << getLastLowLevelOutcomeName()
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

  // Join per-task paths into per-agent paths.
  vector<int> agentsToCompute(agentCount);
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildPrioritizedInitialSolution: failed to join agent paths\n";
    return false;
  }

  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < agentCount; agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildPrioritizedInitialSolution");
  return true;
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
