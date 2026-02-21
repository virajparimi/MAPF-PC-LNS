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
                               bool findingNextTask) {
  const AssignmentLookup assignmentLookup =
      buildAssignmentLookup(workspace, instance_.getTasksNum());

  constraintTable.goalLocation = taskLocation;

  vector<vector<int>> ancestors(instance_.getTasksNum());
  if (precedenceConstraints != nullptr && !precedenceConstraints->empty()) {
    for (const auto& edge : *precedenceConstraints) {
      if (edge.first < 0 || edge.second < 0 ||
          edge.first >= instance_.getTasksNum() ||
          edge.second >= instance_.getTasksNum()) {
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
      if (pred >= 0 && pred < instance_.getTasksNum() && succ >= 0 &&
          succ < instance_.getTasksNum()) {
        ancestors[succ].push_back(pred);
      }
    }
  }

  vector<char> ancestorsOfTask = reachableSet(taskPacket.task, ancestors);
  if (taskPacket.task >= 0 && taskPacket.task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[taskPacket.task] = 0;
  }

  // Loop through the last task map to gather the actual final tasks of the agents
  vector<bool> finalTasks(instance_.getTasksNum(), false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = workspace.assignments(agent);
    if ((int)assignments.size() > 0) {
      int lastTask = assignments.back();
      if (lastTask < 0 || lastTask >= instance_.getTasksNum()) {
        PLOGE << "buildConstraintTable: invalid final task id " << lastTask
              << " for agent " << agent << "\n";
        return false;
      }
      finalTasks[lastTask] = true;
    }
  }

  // Add the paths of the prior tasks to the constraint table with information about whether they were their agent's final tasks or not
  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }

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

    if (hasPreviousTaskInAgent && ancestorTask == previousTaskInAgent) {
      ancestorTaskAgent = taskPacket.agent;
    } else if (isPendingCommitState(lnsNeighborhood_, ancestorTask)) {
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

    const int ancestorTaskLocalIndex =
        (ancestorTask >= 0 && ancestorTask < instance_.getTasksNum())
            ? assignmentLookup.pos[ancestorTask]
            : -1;
    if (ancestorTaskLocalIndex < 0 ||
        ancestorTaskLocalIndex >=
            (int)workspace.taskPaths(ancestorTaskAgent).size() ||
        assignmentLookup.owner[ancestorTask] != ancestorTaskAgent) {
      PLOGE << "buildConstraintTable: could not locate ancestor task "
            << ancestorTask << " for agent " << ancestorTaskAgent << "\n";
      return false;
    }
    assert(!workspace.taskPaths(ancestorTaskAgent)[ancestorTaskLocalIndex]
                .empty());
    const bool isFinalTask = finalTasks[ancestorTask];
    reservePathWithGoalPolicy(
        constraintTable, workspace.taskPaths(ancestorTaskAgent)[ancestorTaskLocalIndex],
        isFinalTask);
    if (isFinalTask) {
      reserveTerminalPathIfActive(constraintTable, ancestorTaskAgent);
    }

    constraintTable.lengthMin = max(
        constraintTable.lengthMin,
        workspace.taskPaths(ancestorTaskAgent)[ancestorTaskLocalIndex]
                .endTime() +
            1);
  }

  // Optionally reserve occupancy for non-ancestor agents as well.
  // This tightens repair planning against cross-agent collisions by treating
  // all other agents as frozen while planning taskPacket.agent.
  if (repairIncludeNonAncestorAgents_) {
    vector<char> isAncestorTask(instance_.getTasksNum(), 0);
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
        if (task < 0 || task >= instance_.getTasksNum()) {
          continue;
        }
        if (isAncestorTask[task]) {
          continue;
        }
        if (assignmentLookup.owner[task] != agent ||
            assignmentLookup.pos[task] != localTask) {
          continue;
        }
        const auto& pathRef = paths[localTask];
        if (pathRef.empty()) {
          continue;
        }
        const bool isFinalTask = (localTask + 1 == (int)assignments.size());
        reservePathWithGoalPolicy(constraintTable, pathRef, isFinalTask);
        if (isFinalTask) {
          reserveTerminalPathIfActive(constraintTable, agent);
        }
      }
    }
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
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
  if (task < 0 || task >= instance_.getTasksNum()) {
    PLOGE << "buildConstraintTable: invalid task id " << task << "\n";
    return false;
  }
  constraintTable.goalLocation = instance_.getTaskLocations(task);

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (const auto& precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
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
    const int ancestorTaskAgent =
        (ancestorTask >= 0 && ancestorTask < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[ancestorTask]
            : UNASSIGNED;
    if (ancestorTaskAgent == UNASSIGNED) {
      PLOGE << "Missing agent assignment for ancestor task "
            << ancestorTask << " in current solution\n";
      return false;
    }
    const int ancestorTaskPosition =
        solution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
    if (ancestorTaskPosition == UNASSIGNED ||
        ancestorTaskPosition >=
            (int)solution_.agents[ancestorTaskAgent].taskPaths.size()) {
      PLOGE << "buildConstraintTable: invalid local index for ancestor task "
            << ancestorTask << " on agent " << ancestorTaskAgent << "\n";
      return false;
    }
    const auto& pathRef =
        solution_.agents[ancestorTaskAgent].taskPaths[ancestorTaskPosition];
    if (pathRef.empty()) {
      PLOGE << "Missing path for ancestor task " << ancestorTask
            << " in current solution\n";
      return false;
    }
    const bool isFinalTask =
        ancestorTask ==
        (int)solution_.agents[ancestorTaskAgent].taskAssignments.back();
    reservePathWithGoalPolicy(constraintTable, pathRef, isFinalTask);
    if (isFinalTask) {
      reserveTerminalPathIfActive(constraintTable, ancestorTaskAgent);
    }
    constraintTable.lengthMin =
        max(constraintTable.lengthMin, pathRef.endTime() + 1);
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
  return true;
}

int LNS::extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue) {
  static const vector<int> kEmptyTaskQueue;
  return extractOldLocalTaskIndex(task, oldTaskQueue, kEmptyTaskQueue);
}

int LNS::extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue,
                                  const vector<int>& newTaskQueue) {
  int localTaskPositionOffset = 0;
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
        find_if(begin(newTaskQueue), end(newTaskQueue), [localTask](int task) {
          return task == localTask;
        }) == end(newTaskQueue)) {
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
