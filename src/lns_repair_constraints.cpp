#include "lns.hpp"
#include "lns_repair_internal.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

bool LNS::buildConstraintTable(ConstraintTable& constraintTable,
                               TaskRegretPacket taskPacket, int taskLocation,
                               RegretWorkspace& workspace,
                               vector<pair<int, int>>* precedenceConstraints,
                               bool findingNextTask,
                               const vector<int>* assignmentOwnerLookup,
                               const vector<int>* assignmentPosLookup) {
  const bool traceCtTask = shouldTraceConstraintDebugTask(taskPacket.task);
  vector<int> tracedAncestorTasks;
  vector<int> tracedAncestorFromPrevious;
  vector<int> tracedAncestorFinal;
  vector<int> tracedNonAncestorHard;
  vector<int> tracedNonAncestorSoft;
  int tracedLenMinDriverTask = UNASSIGNED;
  int tracedLenMinDriverEnd = -1;

  const int taskCount = instance_.getTasksNum();
  AssignmentLookup builtLookup;
  const vector<int>* ownerLookup = assignmentOwnerLookup;
  const vector<int>* posLookup = assignmentPosLookup;
  if (ownerLookup == nullptr || posLookup == nullptr ||
      (int)ownerLookup->size() != taskCount || (int)posLookup->size() != taskCount) {
    builtLookup = buildAssignmentLookup(workspace, taskCount);
    ownerLookup = &builtLookup.owner;
    posLookup = &builtLookup.pos;
  }

  constraintTable.goalLocation = taskLocation;

  vector<vector<int>> ancestors(taskCount);
  if (precedenceConstraints != nullptr && !precedenceConstraints->empty()) {
    for (const auto& edge : *precedenceConstraints) {
      if (edge.first < 0 || edge.second < 0 ||
          edge.first >= taskCount || edge.second >= taskCount) {
        continue;
      }
      ancestors[edge.second].push_back(edge.first);
    }
  } else {
    ancestors = instance_.getAncestorsRef();
  }
  // Dynamic intra-agent precedence induced by current assignment state.
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = workspace.assignments(agent);
    for (int pos = 1; pos < (int)assignments.size(); pos++) {
      const int pred = assignments[pos - 1];
      const int succ = assignments[pos];
      if (pred >= 0 && pred < taskCount && succ >= 0 && succ < taskCount) {
        ancestors[succ].push_back(pred);
      }
    }
  }

  vector<char> ancestorsOfTask = reachableSet(taskPacket.task, ancestors);
  if (taskPacket.task >= 0 && taskPacket.task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[taskPacket.task] = 0;
  }

  // Loop through the last task map to gather the actual final tasks of the agents
  vector<bool> finalTasks(taskCount, false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = workspace.assignments(agent);
    if ((int)assignments.size() > 0) {
      int lastTask = assignments.back();
      if (lastTask < 0 || lastTask >= taskCount) {
        PLOGE << "buildConstraintTable: invalid final task id " << lastTask
              << " for agent " << agent << "\n";
        return false;
      }
      finalTasks[lastTask] = true;
    }
  }
  vector<bool> previousFinalTasks(taskCount, false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = previousSolution_.agents[agent].taskAssignments;
    if ((int)assignments.size() > 0) {
      int lastTask = assignments.back();
      if (lastTask >= 0 && lastTask < taskCount) {
        previousFinalTasks[lastTask] = true;
      }
    }
  }

  // Add the paths of the prior tasks to the constraint table with information about whether they were their agent's final tasks or not
  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }

    const bool pendingAncestor =
        isPendingCommitState(lnsNeighborhood_, ancestorTask);
    int ancestorTaskAgent = UNDEFINED;
    int previousTaskInAgent = UNDEFINED;
    bool hasPreviousTaskInAgent = false;
    if (findingNextTask && taskPacket.agent >= 0 &&
        taskPacket.agent < workspace.numAgents() &&
        taskPacket.taskPosition > 0 &&
        taskPacket.taskPosition <
            (int)workspace.assignments(taskPacket.agent).size()) {
      previousTaskInAgent =
          workspace.assignments(taskPacket.agent)[taskPacket.taskPosition - 1];
      hasPreviousTaskInAgent = true;
    }

    const bool hasWorkspaceOwner =
        (ancestorTask >= 0 && ancestorTask < taskCount &&
         (*ownerLookup)[ancestorTask] != UNASSIGNED);

    if (hasWorkspaceOwner &&
        (*ownerLookup)[ancestorTask] >= 0 &&
        (*ownerLookup)[ancestorTask] < workspace.numAgents() &&
        (*posLookup)[ancestorTask] >= 0 &&
        (*posLookup)[ancestorTask] <
            (int)workspace.taskPaths((*ownerLookup)[ancestorTask]).size() &&
        !workspace
             .taskPaths((*ownerLookup)[ancestorTask])[(*posLookup)[ancestorTask]]
             .empty()) {
      ancestorTaskAgent = (*ownerLookup)[ancestorTask];
    } else if (hasPreviousTaskInAgent && ancestorTask == previousTaskInAgent) {
      if (taskPacket.agent >= 0 && taskPacket.agent < workspace.numAgents() &&
          taskPacket.taskPosition - 1 >= 0 &&
          taskPacket.taskPosition - 1 <
              (int)workspace.taskPaths(taskPacket.agent).size() &&
          !workspace
               .taskPaths(taskPacket.agent)[taskPacket.taskPosition - 1]
               .empty()) {
        ancestorTaskAgent = taskPacket.agent;
      }
    } else if (pendingAncestor) {
      const int prevAssignedAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (prevAssignedAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in previous solution\n";
        return false;
      }
      ancestorTaskAgent = prevAssignedAgent;
    } else {
      const int curAssignedAgent =
          (ancestorTask >= 0 && ancestorTask < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (curAssignedAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in current solution\n";
        return false;
      }
      ancestorTaskAgent = curAssignedAgent;
    }

    const bool canUseWorkspacePath =
        (ancestorTask >= 0 && ancestorTask < taskCount &&
         ancestorTaskAgent >= 0 && ancestorTaskAgent < workspace.numAgents() &&
         (*ownerLookup)[ancestorTask] == ancestorTaskAgent &&
         (*posLookup)[ancestorTask] >= 0 &&
         (*posLookup)[ancestorTask] <
             (int)workspace.taskPaths(ancestorTaskAgent).size() &&
         !workspace.taskPaths(ancestorTaskAgent)[(*posLookup)[ancestorTask]]
              .empty());

    bool usingPreviousFallback = false;
    int ancestorTaskLocalIndex = -1;
    const AgentTaskPath* ancestorPathRef = nullptr;
    if (canUseWorkspacePath) {
      ancestorTaskLocalIndex = (*posLookup)[ancestorTask];
      ancestorPathRef =
          &workspace.taskPaths(ancestorTaskAgent)[ancestorTaskLocalIndex];
    } else if (pendingAncestor) {
      const int prevAssignedAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (prevAssignedAgent == UNASSIGNED) {
        PLOGE << "buildConstraintTable: missing previous owner for pending "
                 "ancestor task "
              << ancestorTask << "\n";
        return false;
      }
      const int prevLocalIndex =
          previousSolution_.getLocalTaskIndex(prevAssignedAgent, ancestorTask);
      if (prevLocalIndex == UNASSIGNED ||
          prevLocalIndex < 0 ||
          prevLocalIndex >=
              (int)previousSolution_.agents[prevAssignedAgent].taskPaths.size() ||
          previousSolution_.agents[prevAssignedAgent]
              .taskPaths[prevLocalIndex]
              .empty()) {
        PLOGE << "buildConstraintTable: missing previous path for pending "
                 "ancestor task "
              << ancestorTask << " on agent " << prevAssignedAgent << "\n";
        return false;
      }
      ancestorTaskAgent = prevAssignedAgent;
      ancestorTaskLocalIndex = prevLocalIndex;
      ancestorPathRef =
          &previousSolution_.agents[ancestorTaskAgent]
               .taskPaths[ancestorTaskLocalIndex];
      usingPreviousFallback = true;
    } else {
      PLOGE << "buildConstraintTable: could not locate non-empty ancestor path "
               "for task "
            << ancestorTask << " (agent " << ancestorTaskAgent << ")\n";
      return false;
    }

    const bool isFinalTask =
        (ancestorTask >= 0 && ancestorTask < taskCount) &&
        (usingPreviousFallback ? previousFinalTasks[ancestorTask]
                               : finalTasks[ancestorTask]);
    if (traceCtTask) {
      tracedAncestorTasks.push_back(ancestorTask);
      if (usingPreviousFallback) {
        tracedAncestorFromPrevious.push_back(ancestorTask);
      }
      if (isFinalTask) {
        tracedAncestorFinal.push_back(ancestorTask);
      }
    }
    reservePathWithGoalPolicy(constraintTable, *ancestorPathRef, isFinalTask);
    if (isFinalTask) {
      reserveTerminalPathIfActive(constraintTable, ancestorTaskAgent);
    }

    const int contributorEnd = ancestorPathRef->endTime();
    constraintTable.lengthMin =
        max(constraintTable.lengthMin, contributorEnd + 1);
    if (traceCtTask && contributorEnd > tracedLenMinDriverEnd) {
      tracedLenMinDriverEnd = contributorEnd;
      tracedLenMinDriverTask = ancestorTask;
    }
  }

  // Reserve occupancy for non-ancestor agents as well.
  // In SIPPS mode, use these as soft-collision guidance (gold-style CAT
  // behavior); in MLA* mode keep the current hard-freeze semantics.
  const bool useSoftForNonAncestors =
      (lowLevelPlannerType_ == LowLevelPlannerType::sipps);
  vector<char> isAncestorTask(taskCount, 0);
  for (int task = 0; task < (int)ancestorsOfTask.size(); task++) {
    if (ancestorsOfTask[task]) {
      isAncestorTask[task] = 1;
    }
  }
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (agent == taskPacket.agent) {
      continue;
    }
    const auto& assignments = workspace.assignments(agent);
    const auto& paths = workspace.taskPaths(agent);
    const int localCount = min((int)assignments.size(), (int)paths.size());
    for (int localTask = 0; localTask < localCount; localTask++) {
      const int task = assignments[localTask];
      if (task < 0 || task >= taskCount) {
        continue;
      }
      if (isAncestorTask[task]) {
        continue;
      }
      if ((*ownerLookup)[task] != agent || (*posLookup)[task] != localTask) {
        continue;
      }
      const auto& pathRef = paths[localTask];
      if (pathRef.empty()) {
        continue;
      }
      if (traceCtTask) {
        if (useSoftForNonAncestors) {
          tracedNonAncestorSoft.push_back(task);
        } else {
          tracedNonAncestorHard.push_back(task);
        }
      }
      const bool isFinalTask = (localTask + 1 == (int)assignments.size());
      reservePathWithGoalPolicy(constraintTable, pathRef, isFinalTask,
                                useSoftForNonAncestors);
      if (isFinalTask) {
        reserveTerminalPathIfActive(constraintTable, agent,
                                    useSoftForNonAncestors);
      }
    }
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
  if (traceCtTask) {
    PLOGW << "CTDETAIL eval task=" << taskPacket.task
          << " agent=" << taskPacket.agent
          << " finding_next=" << (findingNextTask ? "true" : "false")
          << " len_min=" << constraintTable.lengthMin
          << " latest_ts=" << constraintTable.latestTimestep
          << " anc_count=" << tracedAncestorTasks.size()
          << " anc_prev_count=" << tracedAncestorFromPrevious.size()
          << " anc_final_count=" << tracedAncestorFinal.size()
          << " nonanc_hard_count=" << tracedNonAncestorHard.size()
          << " nonanc_soft_count=" << tracedNonAncestorSoft.size()
          << " len_min_driver_task=" << tracedLenMinDriverTask
          << " len_min_driver_end=" << tracedLenMinDriverEnd
          << " anc=" << summarizeIntList(tracedAncestorTasks)
          << " anc_prev=" << summarizeIntList(tracedAncestorFromPrevious)
          << " anc_final=" << summarizeIntList(tracedAncestorFinal)
          << " nonanc_hard=" << summarizeIntList(tracedNonAncestorHard)
          << " nonanc_soft=" << summarizeIntList(tracedNonAncestorSoft);
  }
  return true;
}

bool LNS::buildConstraintTable(ConstraintTable& constraintTable, int task) {
  const vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints(iterationStats.size() > 1);
  return buildConstraintTable(constraintTable, task, precedenceConstraints);
}

bool LNS::buildConstraintTable(
    ConstraintTable& constraintTable, int task,
    const vector<pair<int, int>>& precedenceConstraints) {
  const bool traceCtTask = shouldTraceConstraintDebugTask(task);
  vector<int> tracedAncestorTasks;
  vector<int> tracedAncestorFromPrevious;
  vector<int> tracedAncestorFinal;
  vector<int> tracedNonAncestorHard;
  vector<int> tracedNonAncestorSoft;
  int tracedLenMinDriverTask = UNASSIGNED;
  int tracedLenMinDriverEnd = -1;

  const int taskCount = instance_.getTasksNum();
  if (task < 0 || task >= taskCount) {
    PLOGE << "buildConstraintTable: invalid task id " << task << "\n";
    return false;
  }

  vector<int> taskOwner(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, UNASSIGNED);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    for (int pos = 0; pos < (int)assignments.size(); pos++) {
      const int assignedTask = assignments[pos];
      if (assignedTask < 0 || assignedTask >= taskCount) {
        continue;
      }
      taskOwner[assignedTask] = agent;
      taskToPosition[assignedTask] = pos;
    }
  }
  vector<int> previousTaskOwner(taskCount, UNASSIGNED);
  vector<int> previousTaskToPosition(taskCount, UNASSIGNED);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = previousSolution_.agents[agent].taskAssignments;
    for (int pos = 0; pos < (int)assignments.size(); pos++) {
      const int assignedTask = assignments[pos];
      if (assignedTask < 0 || assignedTask >= taskCount) {
        continue;
      }
      previousTaskOwner[assignedTask] = agent;
      previousTaskToPosition[assignedTask] = pos;
    }
  }

  int taskAgent = taskOwner[task];
  if (taskAgent == UNASSIGNED && isPendingCommitState(lnsNeighborhood_, task)) {
    taskAgent = previousTaskOwner[task];
  }
  if (taskAgent == UNASSIGNED &&
      task >= 0 && task < (int)solution_.taskAgentMap.size()) {
    taskAgent = solution_.taskAgentMap[task];
  }
  if (taskAgent == UNASSIGNED &&
      task >= 0 && task < (int)previousSolution_.taskAgentMap.size()) {
    taskAgent = previousSolution_.taskAgentMap[task];
  }
  vector<bool> finalTasks(taskCount, false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    if (!assignments.empty()) {
      const int finalTask = assignments.back();
      if (finalTask >= 0 && finalTask < taskCount) {
        finalTasks[finalTask] = true;
      }
    }
  }
  vector<bool> previousFinalTasks(taskCount, false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = previousSolution_.agents[agent].taskAssignments;
    if (!assignments.empty()) {
      const int finalTask = assignments.back();
      if (finalTask >= 0 && finalTask < taskCount) {
        previousFinalTasks[finalTask] = true;
      }
    }
  }

  constraintTable.goalLocation = instance_.getTaskLocations(task);

  vector<vector<int>> ancestors(taskCount);
  for (const auto& precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= taskCount ||
        precConstraint.second >= taskCount) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }

  vector<char> ancestorsOfTask = reachableSet(task, ancestors);
  if (task >= 0 && task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[task] = 0;
  }

  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }
    const bool pendingAncestor =
        isPendingCommitState(lnsNeighborhood_, ancestorTask);
    int ancestorTaskAgent = UNASSIGNED;
    int ancestorTaskPosition = UNASSIGNED;
    bool usingPreviousPath = false;

    const bool hasCurrentOwner =
        (ancestorTask >= 0 && ancestorTask < taskCount &&
         taskOwner[ancestorTask] != UNASSIGNED &&
         taskOwner[ancestorTask] >= 0 &&
         taskOwner[ancestorTask] < instance_.getAgentNum() &&
         taskToPosition[ancestorTask] >= 0 &&
         taskToPosition[ancestorTask] <
             (int)solution_.agents[taskOwner[ancestorTask]].taskPaths.size() &&
         !solution_.agents[taskOwner[ancestorTask]]
              .taskPaths[taskToPosition[ancestorTask]]
              .empty());
    if (hasCurrentOwner) {
      ancestorTaskAgent = taskOwner[ancestorTask];
      ancestorTaskPosition = taskToPosition[ancestorTask];
    } else if (ancestorTask >= 0 &&
               ancestorTask < (int)solution_.taskAgentMap.size()) {
      const int mappedCurrentAgent = solution_.taskAgentMap[ancestorTask];
      if (mappedCurrentAgent != UNASSIGNED && mappedCurrentAgent >= 0 &&
          mappedCurrentAgent < instance_.getAgentNum()) {
        const int mappedCurrentPos =
            solution_.getLocalTaskIndex(mappedCurrentAgent, ancestorTask);
        if (mappedCurrentPos != UNASSIGNED && mappedCurrentPos >= 0 &&
            mappedCurrentPos <
                (int)solution_.agents[mappedCurrentAgent].taskPaths.size() &&
            !solution_.agents[mappedCurrentAgent]
                 .taskPaths[mappedCurrentPos]
                 .empty()) {
          ancestorTaskAgent = mappedCurrentAgent;
          ancestorTaskPosition = mappedCurrentPos;
        }
      }
    }

    if (ancestorTaskAgent == UNASSIGNED) {
      const bool hasPreviousOwner =
          (ancestorTask >= 0 && ancestorTask < taskCount &&
           previousTaskOwner[ancestorTask] != UNASSIGNED &&
           previousTaskOwner[ancestorTask] >= 0 &&
           previousTaskOwner[ancestorTask] < instance_.getAgentNum() &&
           previousTaskToPosition[ancestorTask] >= 0 &&
           previousTaskToPosition[ancestorTask] <
               (int)previousSolution_.agents[previousTaskOwner[ancestorTask]]
                   .taskPaths.size() &&
           !previousSolution_.agents[previousTaskOwner[ancestorTask]]
                .taskPaths[previousTaskToPosition[ancestorTask]]
                .empty());
      if (hasPreviousOwner) {
        ancestorTaskAgent = previousTaskOwner[ancestorTask];
        ancestorTaskPosition = previousTaskToPosition[ancestorTask];
        usingPreviousPath = true;
      } else if (ancestorTask >= 0 &&
                 ancestorTask < (int)previousSolution_.taskAgentMap.size()) {
        const int mappedPreviousAgent = previousSolution_.taskAgentMap[ancestorTask];
        if (mappedPreviousAgent != UNASSIGNED && mappedPreviousAgent >= 0 &&
            mappedPreviousAgent < instance_.getAgentNum()) {
          const int mappedPreviousPos =
              previousSolution_.getLocalTaskIndex(mappedPreviousAgent,
                                                 ancestorTask);
          if (mappedPreviousPos != UNASSIGNED && mappedPreviousPos >= 0 &&
              mappedPreviousPos <
                  (int)previousSolution_.agents[mappedPreviousAgent]
                      .taskPaths.size() &&
              !previousSolution_.agents[mappedPreviousAgent]
                   .taskPaths[mappedPreviousPos]
                   .empty()) {
            ancestorTaskAgent = mappedPreviousAgent;
            ancestorTaskPosition = mappedPreviousPos;
            usingPreviousPath = true;
          }
        }
      }
    }

    if (ancestorTaskAgent == UNASSIGNED || ancestorTaskPosition == UNASSIGNED) {
      PLOGE << "Missing path source for ancestor task " << ancestorTask
            << " (pending=" << (pendingAncestor ? "true" : "false") << ")\n";
      return false;
    }

    const auto& pathRef =
        usingPreviousPath
            ? previousSolution_.agents[ancestorTaskAgent]
                  .taskPaths[ancestorTaskPosition]
            : solution_.agents[ancestorTaskAgent].taskPaths[ancestorTaskPosition];

    if (pathRef.empty()) {
      PLOGE << "Missing path for ancestor task " << ancestorTask
            << (usingPreviousPath ? " in previous solution\n"
                                  : " in current solution\n");
      return false;
    }
    const bool isFinalTask = (ancestorTask >= 0 && ancestorTask < taskCount) &&
                             (usingPreviousPath ? previousFinalTasks[ancestorTask]
                                                : finalTasks[ancestorTask]);
    if (traceCtTask) {
      tracedAncestorTasks.push_back(ancestorTask);
      if (usingPreviousPath) {
        tracedAncestorFromPrevious.push_back(ancestorTask);
      }
      if (isFinalTask) {
        tracedAncestorFinal.push_back(ancestorTask);
      }
    }
    reservePathWithGoalPolicy(constraintTable, pathRef, isFinalTask);
    if (isFinalTask) {
      reserveTerminalPathIfActive(constraintTable, ancestorTaskAgent);
    }
    const int contributorEnd = pathRef.endTime();
    constraintTable.lengthMin = max(constraintTable.lengthMin, contributorEnd + 1);
    if (traceCtTask && contributorEnd > tracedLenMinDriverEnd) {
      tracedLenMinDriverEnd = contributorEnd;
      tracedLenMinDriverTask = ancestorTask;
    }
  }

  // Reserve occupancy for non-ancestor agents as well.
  // In SIPPS mode, use these as soft-collision guidance (gold-style CAT
  // behavior); in MLA* mode keep the current hard-freeze semantics.
  const bool useSoftForNonAncestors =
      (lowLevelPlannerType_ == LowLevelPlannerType::sipps);
  vector<char> isAncestorTask(taskCount, 0);
  for (int t = 0; t < (int)ancestorsOfTask.size(); t++) {
    if (ancestorsOfTask[t]) {
      isAncestorTask[t] = 1;
    }
  }
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (agent == taskAgent) {
      continue;
    }
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& paths = solution_.agents[agent].taskPaths;
    const int localCount = min((int)assignments.size(), (int)paths.size());
    for (int localTask = 0; localTask < localCount; localTask++) {
      const int otherTask = assignments[localTask];
      if (otherTask < 0 || otherTask >= taskCount) {
        continue;
      }
      if (isAncestorTask[otherTask]) {
        continue;
      }
      if (taskOwner[otherTask] != agent) {
        continue;
      }
      if (taskToPosition[otherTask] != localTask) {
        continue;
      }
      const auto& pathRef = paths[localTask];
      if (pathRef.empty()) {
        continue;
      }
      if (traceCtTask) {
        if (useSoftForNonAncestors) {
          tracedNonAncestorSoft.push_back(otherTask);
        } else {
          tracedNonAncestorHard.push_back(otherTask);
        }
      }
      const bool isFinalTask = (localTask + 1 == (int)assignments.size());
      reservePathWithGoalPolicy(constraintTable, pathRef, isFinalTask,
                                useSoftForNonAncestors);
      if (isFinalTask) {
        reserveTerminalPathIfActive(constraintTable, agent,
                                    useSoftForNonAncestors);
      }
    }
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
  if (traceCtTask) {
    PLOGW << "CTDETAIL commit task=" << task
          << " len_min=" << constraintTable.lengthMin
          << " latest_ts=" << constraintTable.latestTimestep
          << " anc_count=" << tracedAncestorTasks.size()
          << " anc_prev_count=" << tracedAncestorFromPrevious.size()
          << " anc_final_count=" << tracedAncestorFinal.size()
          << " nonanc_hard_count=" << tracedNonAncestorHard.size()
          << " nonanc_soft_count=" << tracedNonAncestorSoft.size()
          << " len_min_driver_task=" << tracedLenMinDriverTask
          << " len_min_driver_end=" << tracedLenMinDriverEnd
          << " anc=" << summarizeIntList(tracedAncestorTasks)
          << " anc_prev=" << summarizeIntList(tracedAncestorFromPrevious)
          << " anc_final=" << summarizeIntList(tracedAncestorFinal)
          << " nonanc_hard=" << summarizeIntList(tracedNonAncestorHard)
          << " nonanc_soft=" << summarizeIntList(tracedNonAncestorSoft);
  }
  return true;
}

int LNS::extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue) {
  static const vector<int> kEmptyTaskQueue;
  return extractOldLocalTaskIndex(task, oldTaskQueue, kEmptyTaskQueue);
}

int LNS::extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue,
                                  const vector<int>& newTaskQueue) {
  int localTaskPositionOffset = 0;
  unordered_set<int> newTaskMembership;
  if (!newTaskQueue.empty()) {
    newTaskMembership.reserve(newTaskQueue.size());
    for (int queuedTask : newTaskQueue) {
      newTaskMembership.insert(queuedTask);
    }
  }
  // We need to compute the offset as we can invalidate multiple tasks associated with an agent. This means that simply querying the previous solution agent's task index is not enough as the it would be more than the actual task position value for the current solution
  for (int localTask : oldTaskQueue) {
    // We dont need to bother for the tasks that come after the current one since we are considering them in planning order
    if (localTask == task) {
      break;
    }

    // This local task should not be in the new task queue otherwise we have
    // accounted for it before! If not then the offset should only be
    // incremented if it was in conflict set.
    if (lnsNeighborhood_.immutableRemovedTasks.count(localTask) > 0 &&
        newTaskMembership.find(localTask) == newTaskMembership.end()) {
      localTaskPositionOffset++;
    }
  }
  int index = 0;
  for (; index < (int)oldTaskQueue.size(); index++) {
    if (oldTaskQueue[index] == task) {
      break;
    }
  }
  if (index >= (int)oldTaskQueue.size()) {
    PLOGE << "extractOldLocalTaskIndex: task " << task
          << " not found in old task queue\n";
    return UNASSIGNED;
  }
  const int relativeIndex = index - localTaskPositionOffset;
  if (relativeIndex < 0 || relativeIndex > (int)newTaskQueue.size()) {
    PLOGE << "extractOldLocalTaskIndex: computed invalid relative index "
          << relativeIndex << " for task " << task << " (oldIndex=" << index
          << ", offset=" << localTaskPositionOffset
          << ", newSize=" << newTaskQueue.size() << ")\n";
    return UNASSIGNED;
  }
  return relativeIndex;
}

vector<char> LNS::reachableSet(int source, const vector<vector<int>>& edgeList) {
  vector<char> visited(edgeList.size(), 0);
  if (source < 0 || source >= (int)edgeList.size()) {
    return visited;
  }
  stack<int> q({source});
  while (!q.empty()) {
    int current = q.top();
    q.pop();
    if (visited[current]) {
      continue;
    }
    visited[current] = 1;
    for (int sink : edgeList[current]) {
      if (sink >= 0 && sink < (int)edgeList.size() && !visited[sink]) {
        q.push(sink);
      }
    }
  }
  return visited;
}

void LNS::markResolved(int globalTask) {
  lnsNeighborhood_.removedTasks.erase(globalTask);
  if (globalTask >= 0 &&
      globalTask < (int)lnsNeighborhood_.removedTasksPathSize.size()) {
    lnsNeighborhood_.removedTasksPathSize[globalTask] = -1;
  }
  if (globalTask >= 0 &&
      globalTask < (int)lnsNeighborhood_.committedTasks.size()) {
    lnsNeighborhood_.committedTasks[globalTask] = 1;
  }
}

void LNS::patchAgentTaskPaths(int agent, int taskPosition) {
  if (taskPosition == 0) {
    // If we are the first task then ensure that we begin at 0
    solution_.agents[agent].taskPaths[taskPosition].beginTime = 0;
  }
  for (int k = taskPosition + 1;
       k < (int)solution_.agents[agent].taskAssignments.size(); k++) {
    solution_.agents[agent].taskPaths[k].beginTime =
        solution_.agents[agent].taskPaths[k - 1].endTime();
  }
}
