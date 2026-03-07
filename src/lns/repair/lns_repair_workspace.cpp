#include "lns.hpp"
#include "lns_repair_internal.hpp"

#include <algorithm>

bool LNS::injectPendingAncestors(
    RegretWorkspace& workspace, int task,
    const vector<pair<int, int>>& precedenceConstraints,
    vector<char>& workspaceTouchedAgents, vector<char>* outAncestorsOfTask,
    const vector<vector<int>>* prebuiltAncestors,
    const vector<int>* previousAssignmentOwnerLookup,
    const vector<int>* previousAssignmentPosLookup) {
  AssignmentLookup previousLookup;
  const vector<int>* previousOwnerLookup = previousAssignmentOwnerLookup;
  const vector<int>* previousPosLookup = previousAssignmentPosLookup;
  if (previousOwnerLookup == nullptr || previousPosLookup == nullptr ||
      (int)previousOwnerLookup->size() != instance_.getTasksNum() ||
      (int)previousPosLookup->size() != instance_.getTasksNum()) {
    previousLookup =
        buildAssignmentLookup(previousSolution_, instance_.getTasksNum());
    previousOwnerLookup = &previousLookup.owner;
    previousPosLookup = &previousLookup.pos;
  }
  vector<vector<int>> localAncestors;
  const vector<vector<int>>* ancestors = prebuiltAncestors;
  if (ancestors == nullptr ||
      (int)ancestors->size() != instance_.getTasksNum()) {
    localAncestors.assign(instance_.getTasksNum(), {});
    for (const auto& precConstraint : precedenceConstraints) {
      if (precConstraint.first < 0 || precConstraint.second < 0 ||
          precConstraint.first >= instance_.getTasksNum() ||
          precConstraint.second >= instance_.getTasksNum()) {
        continue;
      }
      localAncestors[precConstraint.second].push_back(precConstraint.first);
    }
    ancestors = &localAncestors;
  }
  vector<char> ancestorsOfTask = reachableSet(task, *ancestors);
  if (task >= 0 && task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[task] = 0;
  }
  if (outAncestorsOfTask != nullptr) {
    *outAncestorsOfTask = ancestorsOfTask;
  }

  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }
    if (isPendingCommitState(lnsNeighborhood_, ancestorTask)) {
      // This task's path will not exist currently!
      const int ancestorTaskAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (ancestorTaskAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task " << ancestorTask
              << " in previous solution\n";
        return false;
      }
      assert(ancestorTaskAgent != UNASSIGNED);
      const int ancestorTaskLocalIndex =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousPosLookup->size() &&
           (*previousOwnerLookup)[ancestorTask] == ancestorTaskAgent)
              ? (*previousPosLookup)[ancestorTask]
              : UNASSIGNED;
      if (ancestorTaskLocalIndex == UNASSIGNED ||
          ancestorTaskLocalIndex >=
              (int)previousSolution_.agents[ancestorTaskAgent].taskPaths.size()) {
        PLOGE << "computeRegretForTask: invalid local index for ancestor task "
              << ancestorTask << " on agent " << ancestorTaskAgent << "\n";
        return false;
      }
      int ancestorTaskLocalIndexRelativeToSolution = extractOldLocalTaskIndex(
          ancestorTask,
          previousSolution_.agents[ancestorTaskAgent].taskAssignments,
          workspace.assignments(ancestorTaskAgent));
      if (ancestorTaskLocalIndexRelativeToSolution == UNASSIGNED) {
        PLOGE << "computeRegretForTask: failed to map ancestor task "
              << ancestorTask << " into current assignment order\n";
        return false;
      }
      workspace.mutableAssignments(ancestorTaskAgent).insert(
          workspace.mutableAssignments(ancestorTaskAgent).begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          ancestorTask);
      workspace.mutableTaskPaths(ancestorTaskAgent).insert(
          workspace.mutableTaskPaths(ancestorTaskAgent).begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          previousSolution_.agents[ancestorTaskAgent]
              .taskPaths[ancestorTaskLocalIndex]);
      workspaceTouchedAgents[ancestorTaskAgent] = 1;
    }
  }
  return true;
}

void LNS::evaluateAgentPositionCandidate(
    int task, int agent, int earliestTimestep, RegretWorkspace& workspace,
    vector<pair<int, int>>* precedenceConstraints,
    const TaskBaselineMetrics& baselineMetrics,
    pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
    vector<int>* candidateAgents, const vector<int>* assignmentOwnerLookup,
    const vector<int>* assignmentPosLookup,
    const vector<int>* previousAssignmentOwnerLookup,
    const vector<int>* previousAssignmentPosLookup) {
  TaskRegretPacket regretPacket = {task, agent, -1, earliestTimestep};
  const auto beforeOptions = serviceTimes->size();
  computeRegretForTaskWithAgent(regretPacket, workspace, precedenceConstraints,
                                baselineMetrics, serviceTimes,
                                assignmentOwnerLookup, assignmentPosLookup,
                                previousAssignmentOwnerLookup,
                                previousAssignmentPosLookup);
  if (candidateAgents != nullptr && serviceTimes->size() > beforeOptions) {
    candidateAgents->push_back(agent);
  }
}

bool LNS::buildRegretEntry(
    int task,
    pairing_heap<Utility, compare<Utility::CompareUtilities>>& serviceTimes) {
  if (serviceTimes.empty()) {
    // No feasible insertion exists for this task under the current temporary
    // neighborhood state.
    PLOGD << "No service time options for task " << task
          << " inside computeRegretForTask\n";
    return false;
  }
  Utility bestUtility = serviceTimes.top();
  serviceTimes.pop();
  if (serviceTimes.empty()) {
    // Exactly one feasible insertion exists; treat it as forced and prioritize
    // it ahead of ambiguous tasks.
    regretBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
    regretSecondBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
    Regret forced(task, bestUtility.agent, bestUtility.taskPosition,
                  bestUtility.pathLength, bestUtility.agentTasksLen, 0,
                  std::numeric_limits<double>::infinity(), regretStamp_[task]);
    lnsNeighborhood_.regretMaxHeap.push(forced);
    return true;
  }

  Utility secondBestUtility = serviceTimes.top();
  regretBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
  regretSecondBestOption_[task] = {secondBestUtility.agent,
                                   secondBestUtility.taskPosition};

  double value = 0;
  if (isRegretTypeAbsolute()) {
    value = secondBestUtility.value - bestUtility.value;
  } else {
    value = (secondBestUtility.value + 1) / (bestUtility.value + 1);
  }
  Regret regret(task, bestUtility.agent, bestUtility.taskPosition,
                bestUtility.pathLength, bestUtility.agentTasksLen,
                (int)serviceTimes.size(), value, regretStamp_[task]);
  lnsNeighborhood_.regretMaxHeap.push(regret);
  return true;
}

bool LNS::prepareRegretWorkspaceForTask(
    int task, RegretWorkspace& workspace,
    const vector<char>& workspaceTouchedAgents,
    vector<pair<int, int>>& precedenceConstraints,
    vector<int>& assignmentOwnerLookup, vector<int>& assignmentPosLookup,
    int& earliestTimestep, const vector<char>* precomputedAncestorsOfTask) {
  earliestTimestep = 0;

  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.assign(inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());
  size_t intraPrecedenceCount = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignmentsView = workspace.assignments(agent);
    if (assignmentsView.size() > 1) {
      intraPrecedenceCount += (assignmentsView.size() - 1);
    }
  }
  precedenceConstraints.reserve(precedenceConstraints.size() +
                                intraPrecedenceCount);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignmentsView = workspace.assignments(agent);
    for (int localTask = 1; localTask < (int)assignmentsView.size();
         localTask++) {
      precedenceConstraints.emplace_back(assignmentsView[localTask - 1],
                                         assignmentsView[localTask]);
    }
  }

  const AssignmentLookup assignmentLookup =
      buildAssignmentLookup(workspace, instance_.getTasksNum());
  assignmentOwnerLookup = assignmentLookup.owner;
  assignmentPosLookup = assignmentLookup.pos;

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    const auto& assignmentsView = workspace.assignments(agent);

    bool normalizeAgent = workspaceTouchedAgents[agent] != 0;
    if (!normalizeAgent) {
      const auto& pathsView = workspace.taskPaths(agent);
      if (pathsView.size() != assignmentsView.size()) {
        normalizeAgent = true;
      } else {
        for (int localTask = 0; localTask < (int)assignmentsView.size();
             localTask++) {
          const bool broken = pathsView[localTask].empty() ||
                              (localTask == 0 &&
                               pathsView[localTask].front().location !=
                                   instance_.getStartLocationsRef()[agent]) ||
                              (localTask > 0 &&
                               (pathsView[localTask - 1].empty() ||
                                pathsView[localTask - 1].path.back().location !=
                                    pathsView[localTask].path.front().location));
          if (broken) {
            normalizeAgent = true;
            break;
          }
          const int expectedBegin =
              (localTask == 0) ? 0 : pathsView[localTask - 1].endTime();
          if (pathsView[localTask].beginTime != expectedBegin) {
            normalizeAgent = true;
            break;
          }
        }
      }
    }

    if (!normalizeAgent) {
      continue;
    }

    auto& assignments = workspace.mutableAssignments(agent);
    auto& paths = workspace.mutableTaskPaths(agent);
    vector<int> goalLocations = instance_.getTaskLocations(assignments);
    SingleAgentSolver& localPlanner = getReusableLocalPlanner(agent);
    localPlanner.setGoalLocations(goalLocations);

    for (int localTask = 0; localTask < (int)assignments.size(); localTask++) {
      if (localTask > 0) {
        paths[localTask].beginTime = paths[localTask - 1].endTime();
      } else {
        paths[localTask].beginTime = 0;
      }

      const bool broken = paths[localTask].empty() ||
                          (localTask == 0 &&
                           paths[localTask].front().location !=
                               instance_.getStartLocationsRef()[agent]) ||
                          (localTask > 0 &&
                           (paths[localTask - 1].empty() ||
                            paths[localTask - 1].path.back().location !=
                                paths[localTask].path.front().location));
      if (!broken) {
        continue;
      }

      int startTime = 0;
      if (localTask > 0) {
        startTime = paths[localTask - 1].endTime();
      }
      ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
      TaskRegretPacket taskPacket = {assignments[localTask], agent, localTask,
                                     -1};
      if (!buildConstraintTable(constraintTable, taskPacket,
                                goalLocations[localTask], workspace,
                                &precedenceConstraints, false,
                                &assignmentOwnerLookup, &assignmentPosLookup)) {
        PLOGE << "computeRegretForTask: failed to build constraint table for "
              << "agent " << agent << ", task " << assignments[localTask]
              << " at position " << localTask << "\n";
        return false;
      }
      AgentTaskPath path = runLowLevelSearch(localPlanner, constraintTable,
                                             startTime, localTask, 0);
      if (path.empty()) {
        PLOGE << "computeRegretForTask: empty path for agent " << agent
              << ", task " << assignments[localTask] << " at position "
              << localTask << "\n";
        return false;
      }
      paths[localTask] = std::move(path);
    }
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = workspace.assignments(agent);
    const auto& paths = workspace.taskPaths(agent);
    for (int localTask = 0; localTask < (int)assignments.size(); localTask++) {
      if (localTask > 0) {
        assert(paths[localTask - 1].path.back().location ==
               paths[localTask].path.front().location);
      } else {
        assert(paths[localTask].front().location ==
               instance_.getStartLocationsRef()[agent]);
      }
    }
  }

  vector<char> ancestorsOfTask;
  if (precomputedAncestorsOfTask != nullptr &&
      (int)precomputedAncestorsOfTask->size() == instance_.getTasksNum()) {
    ancestorsOfTask = *precomputedAncestorsOfTask;
    if (task >= 0 && task < (int)ancestorsOfTask.size()) {
      ancestorsOfTask[task] = 0;
    }
  } else {
    vector<vector<int>> ancestors(instance_.getTasksNum());
    for (const auto& precConstraint : precedenceConstraints) {
      if (precConstraint.first < 0 || precConstraint.second < 0 ||
          precConstraint.first >= instance_.getTasksNum() ||
          precConstraint.second >= instance_.getTasksNum()) {
        continue;
      }
      ancestors[precConstraint.second].push_back(precConstraint.first);
    }
    ancestorsOfTask = reachableSet(task, ancestors);
    if (task >= 0 && task < (int)ancestorsOfTask.size()) {
      ancestorsOfTask[task] = 0;
    }
  }

  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }
    int ancestorTaskAgent = UNDEFINED;
    if (isPendingCommitState(lnsNeighborhood_, ancestorTask)) {
      const int prevAssignedAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (prevAssignedAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task " << ancestorTask
              << " in previous solution\n";
        return false;
      }
      ancestorTaskAgent = prevAssignedAgent;
    } else {
      const int curAssignedAgent =
          (ancestorTask >= 0 && ancestorTask < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (curAssignedAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task " << ancestorTask
              << " in current solution\n";
        return false;
      }
      ancestorTaskAgent = curAssignedAgent;
    }
    if (ancestorTaskAgent < 0 || ancestorTaskAgent >= instance_.getAgentNum()) {
      PLOGE << "Invalid ancestor agent " << ancestorTaskAgent
            << " for task " << ancestorTask << "\n";
      return false;
    }
    assert(ancestorTaskAgent != UNASSIGNED);
    if (ancestorTask < 0 || ancestorTask >= instance_.getTasksNum()) {
      continue;
    }
    const int lookupAgent = assignmentOwnerLookup[ancestorTask];
    const int ancestorTaskPosition = assignmentPosLookup[ancestorTask];
    if (lookupAgent != ancestorTaskAgent || ancestorTaskPosition == UNASSIGNED) {
      PLOGE << "Ancestor task " << ancestorTask
            << " missing from temporary assignment for agent "
            << ancestorTaskAgent << "\n";
      return false;
    }
    if (ancestorTaskPosition >=
            (int)workspace.taskPaths(ancestorTaskAgent).size() ||
        workspace.taskPaths(ancestorTaskAgent)[ancestorTaskPosition].empty()) {
      PLOGE << "Ancestor path missing for task " << ancestorTask
            << " at position " << ancestorTaskPosition << "\n";
      return false;
    }
    earliestTimestep =
        max(earliestTimestep,
            workspace.taskPaths(ancestorTaskAgent)[ancestorTaskPosition]
                    .endTimeChecked() +
                1);
  }
  return true;
}
