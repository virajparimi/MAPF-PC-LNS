#include "lns.hpp"
#include "lns_repair_internal.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>

std::variant<bool, Utility> LNS::insertTask(
    TaskRegretPacket regretPacket,
    RegretWorkspace& workspace,
    vector<pair<int, int>>* precedenceConstraints,
    const TaskBaselineMetrics* baselineMetrics,
    SingleAgentSolver* reusablePlanner,
    bool rollbackAfter) {
  if (runtimeBudgetExhausted()) {
    return false;
  }

  double pathSizeChange = 0;
  int startTime = 0, previousTask = UNDEFINED, nextTask = UNDEFINED;
  int insertedTaskPosition = UNASSIGNED;
  vector<pair<int, int>> adjustedPrecedenceConstraints;
  vector<pair<int, int>>* activePrecedenceConstraints = precedenceConstraints;
  if (precedenceConstraints != nullptr) {
    adjustedPrecedenceConstraints = *precedenceConstraints;
    activePrecedenceConstraints = &adjustedPrecedenceConstraints;
  }
  std::optional<InsertTaskRollbackLog> rollbackLog;
  if (rollbackAfter) {
    rollbackLog.emplace();
  }
  ScopedInsertTaskRollback rollbackGuard = {
      rollbackLog.has_value() ? &rollbackLog.value() : nullptr,
      &workspace,
      rollbackAfter};
  auto recordInsertAssignment = [&](int agent, int index, int task) {
    if (rollbackAfter && rollbackLog.has_value()) {
      rollbackLog->ops.push_back({InsertTaskRollbackOp::Kind::insertAssignment,
                                  agent,
                                  index,
                                  0,
                                  AgentTaskPath()});
    }
    auto& assignments = workspace.mutableAssignments(agent);
    assignments.insert(assignments.begin() + index, task);
  };
  auto recordInsertTaskPath = [&](int agent, int index,
                                  const AgentTaskPath& path) {
    if (rollbackAfter && rollbackLog.has_value()) {
      rollbackLog->ops.push_back({InsertTaskRollbackOp::Kind::insertTaskPath,
                                  agent,
                                  index,
                                  0,
                                  AgentTaskPath()});
    }
    auto& taskPaths = workspace.mutableTaskPaths(agent);
    taskPaths.insert(taskPaths.begin() + index, path);
  };
  auto recordSetTaskPath = [&](int agent, int index, AgentTaskPath path) {
    auto& taskPaths = workspace.mutableTaskPaths(agent);
    if (rollbackAfter && rollbackLog.has_value()) {
      rollbackLog->ops.push_back({InsertTaskRollbackOp::Kind::setTaskPath,
                                  agent,
                                  index,
                                  0,
                                  taskPaths[index]});
    }
    taskPaths[index] = std::move(path);
  };
  auto recordSetTaskPathBeginTime = [&](int agent, int index, int beginTime) {
    auto& taskPaths = workspace.mutableTaskPaths(agent);
    if (rollbackAfter && rollbackLog.has_value()) {
      rollbackLog->ops.push_back(
          {InsertTaskRollbackOp::Kind::setTaskPathBeginTime,
           agent,
           index,
           taskPaths[index].beginTime,
           AgentTaskPath()});
    }
    taskPaths[index].beginTime = beginTime;
  };

  auto assignmentsFor = [&](int agent) -> const vector<int>& {
    return workspace.assignments(agent);
  };
  auto taskPathsFor = [&](int agent) -> const vector<AgentTaskPath>& {
    return workspace.taskPaths(agent);
  };
  auto addPrecedenceEdgeIfMissing = [&](int pred, int succ) {
    if (activePrecedenceConstraints == nullptr ||
        pred == UNDEFINED || succ == UNDEFINED) {
      return;
    }
    if (pred < 0 || succ < 0 || pred >= instance_.getTasksNum() ||
        succ >= instance_.getTasksNum()) {
      return;
    }
    for (const auto& edge : *activePrecedenceConstraints) {
      if (edge.first == pred && edge.second == succ) {
        return;
      }
    }
    activePrecedenceConstraints->emplace_back(pred, succ);
  };
  auto removePrecedenceEdge = [&](int pred, int succ) {
    if (activePrecedenceConstraints == nullptr ||
        pred == UNDEFINED || succ == UNDEFINED) {
      return;
    }
    activePrecedenceConstraints->erase(
        std::remove_if(activePrecedenceConstraints->begin(),
                       activePrecedenceConstraints->end(),
                       [&](const pair<int, int>& edge) {
                         return edge.first == pred && edge.second == succ;
                       }),
        activePrecedenceConstraints->end());
  };

  const int taskCount = instance_.getTasksNum();
  const auto& staticSuccessors = instance_.getSuccessorsRef();
  vector<int> bfsVisitStamp(taskCount, 0);
  int bfsStamp = 0;
  std::deque<int> bfsFrontier;
  vector<int> inDegreeScratch(taskCount, 0);
  std::deque<int> topoQueue;

  auto isAcyclicAfterLocalInsertion =
      [&](const AssignmentLookup& assignmentLookup, int insertedTask,
          int predecessorTask,
          int successorTask) -> bool {
    if (insertedTask < 0 || insertedTask >= taskCount) {
      return false;
    }

    auto hasPath = [&](int source, int target, int skipFrom,
                       int skipTo) -> bool {
      if (source < 0 || source >= taskCount || target < 0 ||
          target >= taskCount) {
        return false;
      }
      if (source == target) {
        return true;
      }

      if (++bfsStamp == std::numeric_limits<int>::max()) {
        std::fill(bfsVisitStamp.begin(), bfsVisitStamp.end(), 0);
        bfsStamp = 1;
      }
      bfsFrontier.clear();
      bfsFrontier.push_back(source);
      bfsVisitStamp[source] = bfsStamp;

      while (!bfsFrontier.empty()) {
        const int current = bfsFrontier.front();
        bfsFrontier.pop_front();

        if (current < 0 || current >= taskCount) {
          continue;
        }

        for (int succ : staticSuccessors[current]) {
          if (current == skipFrom && succ == skipTo) {
            continue;
          }
          if (succ < 0 || succ >= taskCount ||
              bfsVisitStamp[succ] == bfsStamp) {
            continue;
          }
          if (succ == target) {
            return true;
          }
          bfsVisitStamp[succ] = bfsStamp;
          bfsFrontier.push_back(succ);
        }

        const int owner = assignmentLookup.owner[current];
        const int pos = assignmentLookup.pos[current];
        if (owner != UNASSIGNED && owner >= 0 &&
            owner < instance_.getAgentNum() && pos >= 0 &&
            pos + 1 < (int)assignmentsFor(owner).size()) {
          const int succ = assignmentsFor(owner)[pos + 1];
          if (!(current == skipFrom && succ == skipTo) &&
              succ >= 0 && succ < taskCount &&
              bfsVisitStamp[succ] != bfsStamp) {
            if (succ == target) {
              return true;
            }
            bfsVisitStamp[succ] = bfsStamp;
            bfsFrontier.push_back(succ);
          }
        }
      }
      return false;
    };

    if (predecessorTask != UNDEFINED) {
      if (hasPath(insertedTask, predecessorTask, predecessorTask,
                  insertedTask)) {
        return false;
      }
    }
    if (successorTask != UNDEFINED) {
      if (hasPath(successorTask, insertedTask, insertedTask,
                  successorTask)) {
        return false;
      }
    }
    return true;
  };

  auto isAcyclicAssignmentState = [&](const AssignmentLookup& assignmentLookup)
      -> bool {
    std::fill(inDegreeScratch.begin(), inDegreeScratch.end(), 0);
    for (int task = 0; task < taskCount; task++) {
      for (int succ : staticSuccessors[task]) {
        if (succ < 0 || succ >= taskCount) {
          continue;
        }
        inDegreeScratch[succ]++;
      }
    }

    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      const auto& assignments = assignmentsFor(agent);
      for (int pos = 1; pos < (int)assignments.size(); pos++) {
        const int succ = assignments[pos];
        if (succ < 0 || succ >= taskCount) {
          continue;
        }
        inDegreeScratch[succ]++;
      }
    }

    topoQueue.clear();
    for (int task = 0; task < taskCount; task++) {
      if (inDegreeScratch[task] == 0) {
        topoQueue.push_back(task);
      }
    }

    int visited = 0;
    while (!topoQueue.empty()) {
      const int task = topoQueue.front();
      topoQueue.pop_front();
      visited++;

      for (int succ : staticSuccessors[task]) {
        if (succ < 0 || succ >= taskCount) {
          continue;
        }
        inDegreeScratch[succ]--;
        if (inDegreeScratch[succ] == 0) {
          topoQueue.push_back(succ);
        }
      }

      const int owner = assignmentLookup.owner[task];
      const int pos = assignmentLookup.pos[task];
      if (owner != UNASSIGNED && owner >= 0 &&
          owner < instance_.getAgentNum() && pos >= 0 &&
          pos + 1 < (int)assignmentsFor(owner).size()) {
        const int succ = assignmentsFor(owner)[pos + 1];
        if (succ >= 0 && succ < taskCount) {
          inDegreeScratch[succ]--;
          if (inDegreeScratch[succ] == 0) {
            topoQueue.push_back(succ);
          }
        }
      }
    }
    return visited == taskCount;
  };

  int agentTasksSize = (int)assignmentsFor(regretPacket.agent).size();
  double value = std::numeric_limits<double>::infinity();
  AssignmentLookup assignmentLookup;
  const vector<int>* assignmentOwnerLookup = nullptr;
  const vector<int>* assignmentPosLookup = nullptr;
  auto rebuildAssignmentLookup = [&]() {
    assignmentLookup = buildAssignmentLookup(workspace, taskCount);
    assignmentOwnerLookup = &assignmentLookup.owner;
    assignmentPosLookup = &assignmentLookup.pos;
  };

  // In this case we are inserting a task not at the last position
  if (regretPacket.taskPosition < agentTasksSize) {

    nextTask =
        assignmentsFor(regretPacket.agent)[regretPacket.taskPosition];
    pathSizeChange =
        (double)taskPathsFor(regretPacket.agent)[regretPacket.taskPosition]
            .size();

    recordSetTaskPath(regretPacket.agent, regretPacket.taskPosition,
                      AgentTaskPath());

    recordInsertAssignment(regretPacket.agent, regretPacket.taskPosition,
                           regretPacket.task);
    recordInsertTaskPath(regretPacket.agent, regretPacket.taskPosition,
                         AgentTaskPath());

    // If we are NOT inserting at the start position then we need to take care of the previous task as well
    if (regretPacket.taskPosition != 0) {
      previousTask =
          assignmentsFor(regretPacket.agent)[regretPacket.taskPosition -
                                                      1];
      // TODO: Technically the task can start being processed before the previous task ends. This is more conservative but need to check if there are better ways to tackle this.
      startTime =
          taskPathsFor(regretPacket.agent)[regretPacket.taskPosition - 1]
              .endTime();
    }
  }
  // In this case we are inserting at the very end
  else if (regretPacket.taskPosition == agentTasksSize && agentTasksSize != 0) {

    previousTask =
        assignmentsFor(regretPacket.agent)[regretPacket.taskPosition -
                                                    1];
    startTime =
        taskPathsFor(regretPacket.agent)[regretPacket.taskPosition - 1]
            .endTime();

    recordInsertAssignment(
        regretPacket.agent,
        (int)assignmentsFor(regretPacket.agent).size(),
        regretPacket.task);
    recordInsertTaskPath(regretPacket.agent,
                         (int)taskPathsFor(regretPacket.agent).size(),
                         AgentTaskPath());
  } else if (agentTasksSize == 0) {
    // This is the rare-case when the agent has no tasks assigned to it.
    assert(regretPacket.taskPosition == 0);

    startTime = 0;
    recordInsertAssignment(regretPacket.agent, 0, regretPacket.task);
    recordInsertTaskPath(regretPacket.agent, 0, AgentTaskPath());
  }

  // Keep eval-time precedence graph consistent with commit-time rewiring.
  if (previousTask != UNDEFINED && nextTask != UNDEFINED) {
    removePrecedenceEdge(previousTask, nextTask);
  }
  addPrecedenceEdgeIfMissing(previousTask, regretPacket.task);
  addPrecedenceEdgeIfMissing(regretPacket.task, nextTask);

  if (nextTask >= 0) {
    // The task paths reference does not have ancestor information about next task, so we need to add those in

    rebuildAssignmentLookup();
    const auto& staticAncestors = instance_.getAncestorsRef();
    vector<char> ancestorsOfNextTask(taskCount, 0);
    vector<char> discovered(taskCount, 0);
    vector<char> taskPresent(taskCount, 0);
    for (int task = 0; task < (int)assignmentLookup.owner.size(); task++) {
      if (assignmentLookup.owner[task] != UNASSIGNED) {
        taskPresent[task] = 1;
      }
    }
    if (nextTask >= 0 && nextTask < taskCount) {
      vector<int> frontier;
      frontier.reserve(taskCount);
      frontier.push_back(nextTask);
      discovered[nextTask] = 1;
      for (size_t frontierIdx = 0; frontierIdx < frontier.size();
           frontierIdx++) {
        const int currentTask = frontier[frontierIdx];
        if (currentTask < 0 || currentTask >= taskCount) {
          continue;
        }
        for (int predecessorTask : staticAncestors[currentTask]) {
          if (predecessorTask < 0 || predecessorTask >= taskCount) {
            continue;
          }
          ancestorsOfNextTask[predecessorTask] = 1;
          if (!discovered[predecessorTask]) {
            discovered[predecessorTask] = 1;
            frontier.push_back(predecessorTask);
          }
        }

        const int owner = assignmentLookup.owner[currentTask];
        const int localPos = assignmentLookup.pos[currentTask];
        if (owner != UNASSIGNED && localPos > 0 &&
            owner >= 0 && owner < instance_.getAgentNum()) {
          const int predecessorTask =
              assignmentsFor(owner)[localPos - 1];
          if (predecessorTask >= 0 && predecessorTask < taskCount) {
            ancestorsOfNextTask[predecessorTask] = 1;
            if (!discovered[predecessorTask]) {
              discovered[predecessorTask] = 1;
              frontier.push_back(predecessorTask);
            }
          }
        }
      }
      ancestorsOfNextTask[nextTask] = 0;
    }
    // Only these agents can become inconsistent in this insertion scenario:
    // the candidate agent plus agents that own ancestors of nextTask (including
    // pending-ancestor fallback owners).
    vector<char> affectedAgents(instance_.getAgentNum(), 0);
    if (regretPacket.agent >= 0 && regretPacket.agent < instance_.getAgentNum()) {
      affectedAgents[regretPacket.agent] = 1;
    }

    bool injectedPendingAncestor = false;
    for (int nextTaskAncestor = 0;
         nextTaskAncestor < (int)ancestorsOfNextTask.size();
         nextTaskAncestor++) {
      if (!ancestorsOfNextTask[nextTaskAncestor]) {
        continue;
      }
      const int ownerFromWorkspace =
          (nextTaskAncestor >= 0 &&
           nextTaskAncestor < (int)assignmentLookup.owner.size())
              ? assignmentLookup.owner[nextTaskAncestor]
              : UNASSIGNED;
      if (ownerFromWorkspace >= 0 &&
          ownerFromWorkspace < instance_.getAgentNum()) {
        affectedAgents[ownerFromWorkspace] = 1;
      }
      if (isPendingCommitState(lnsNeighborhood_, nextTaskAncestor) &&
          nextTaskAncestor != regretPacket.task) {
        // This task's path will not exist currently!
        const int nextTaskAncestorAgent =
            (nextTaskAncestor >= 0 &&
             nextTaskAncestor < (int)previousSolution_.taskAgentMap.size())
                ? previousSolution_.taskAgentMap[nextTaskAncestor]
                : UNASSIGNED;
        if (nextTaskAncestorAgent == UNASSIGNED) {
          PLOGE << "Missing agent assignment for ancestor task "
                << nextTaskAncestor << " in previous solution\n";
          return false;
        }
        assert(nextTaskAncestorAgent != UNASSIGNED);
        if (nextTaskAncestor < 0 || nextTaskAncestor >= taskCount ||
            !taskPresent[nextTaskAncestor]) {

          int ancestorTaskLocalIndex = previousSolution_.getLocalTaskIndex(
              nextTaskAncestorAgent, nextTaskAncestor);
          if (ancestorTaskLocalIndex == UNASSIGNED ||
              ancestorTaskLocalIndex >=
                  (int)previousSolution_.agents[nextTaskAncestorAgent]
                      .taskPaths.size()) {
            PLOGE << "insertTask: invalid local index for ancestor task "
                  << nextTaskAncestor << " on agent "
                  << nextTaskAncestorAgent << "\n";
            return false;
          }
          int ancestorTaskLocalIndexRelativeToSolution =
              extractOldLocalTaskIndex(
                  nextTaskAncestor,
                  previousSolution_.agents[nextTaskAncestorAgent]
                      .taskAssignments,
                  assignmentsFor(nextTaskAncestorAgent));
          if (ancestorTaskLocalIndexRelativeToSolution == UNASSIGNED) {
            PLOGE << "insertTask: failed to map ancestor task "
                  << nextTaskAncestor
                  << " into current assignment order for agent "
                  << nextTaskAncestorAgent << "\n";
            return false;
          }
          recordInsertAssignment(nextTaskAncestorAgent,
                                 ancestorTaskLocalIndexRelativeToSolution,
                                 nextTaskAncestor);
          recordInsertTaskPath(nextTaskAncestorAgent,
                               ancestorTaskLocalIndexRelativeToSolution,
                               previousSolution_.agents[nextTaskAncestorAgent]
                                   .taskPaths[ancestorTaskLocalIndex]);
          injectedPendingAncestor = true;
          if (nextTaskAncestorAgent >= 0 &&
              nextTaskAncestorAgent < instance_.getAgentNum()) {
            affectedAgents[nextTaskAncestorAgent] = 1;
          }
          if (nextTaskAncestor >= 0 &&
              nextTaskAncestor < taskCount) {
            taskPresent[nextTaskAncestor] = 1;
          }
        }
      }
    }
    if (injectedPendingAncestor) {
      rebuildAssignmentLookup();
      if (!isAcyclicAssignmentState(assignmentLookup)) {
        return false;
      }
    } else {
      if (!isAcyclicAfterLocalInsertion(assignmentLookup, regretPacket.task,
                                        previousTask, nextTask)) {
        return false;
      }
    }

    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      if (!affectedAgents[agent]) {
        continue;
      }
      const vector<int> goalLocations =
          instance_.getTaskLocations(assignmentsFor(agent));
      auto localPlanner = createLocalPlanner(agent);
      localPlanner->setGoalLocations(goalLocations);

      for (int localTask = 0;
           localTask < (int)assignmentsFor(agent).size();
           localTask++) {
        if (runtimeBudgetExhausted()) {
          return false;
        }
        if (localTask > 0) {
          recordSetTaskPathBeginTime(
              agent, localTask,
              taskPathsFor(agent)[localTask - 1].endTime());
        } else {
          recordSetTaskPathBeginTime(agent, localTask, 0);
        }

        const bool touchesRegretTask =
            (regretPacket.task == assignmentsFor(agent)[localTask]) ||
            (localTask > 0 &&
             regretPacket.task == assignmentsFor(agent)[localTask - 1]);
        const bool touchesNextTask =
            (nextTask == assignmentsFor(agent)[localTask]) ||
            (localTask > 0 &&
             nextTask == assignmentsFor(agent)[localTask - 1]);
        if (touchesRegretTask || touchesNextTask) {
          continue;
        }

        if ((localTask == 0 &&
             taskPathsFor(agent)[localTask].front().location !=
                 instance_.getStartLocationsRef()[agent]) ||
            (localTask > 0 &&
             taskPathsFor(agent)[localTask - 1].path.back().location !=
                 taskPathsFor(agent)[localTask].path.front().location)) {

          if (localTask < 0 || localTask >= (int)goalLocations.size()) {
            PLOGE << "insertTask: invalid local task position " << localTask
                  << " for agent " << agent << " (goal size "
                  << goalLocations.size() << ")\n";
            return false;
          }
          int startTime = 0;
          if (localTask > 0) {
            startTime = taskPathsFor(agent)[localTask - 1].endTime();
          }
          ConstraintTable constraintTable(instance_.numOfCols,
                                          instance_.mapSize);
          TaskRegretPacket taskPacket = {
              assignmentsFor(agent)[localTask], agent, localTask, -1};
          if (!buildConstraintTable(constraintTable, taskPacket,
                                    goalLocations[localTask], workspace,
                                    activePrecedenceConstraints, false,
                                    assignmentOwnerLookup,
                                    assignmentPosLookup)) {
            PLOGE << "insertTask: failed to build constraint table for agent "
                  << agent << ", task " << assignmentsFor(agent)[localTask]
                  << " at position " << localTask << "\n";
            return false;
          }
          AgentTaskPath path = runLowLevelSearch(
              *localPlanner, constraintTable, startTime, localTask, 0);
          // We must be able to find the path for the next task. If not then we cannot move forward!
          if (path.empty()) {
            PLOGE << "insertTask: empty path for agent " << agent
                  << ", task " << assignmentsFor(agent)[localTask]
                  << " at position " << localTask << "\n";
            return false;
          }
          assert(!path.empty());
          recordSetTaskPath(agent, localTask, std::move(path));
        }
      }
    }

    int taskPosition = UNASSIGNED;
    int nextTaskPosition = UNASSIGNED;
    const auto& candidateAssignments =
        assignmentsFor(regretPacket.agent);
    for (int idx = 0; idx < (int)candidateAssignments.size(); idx++) {
      const int currentTask = candidateAssignments[idx];
      if (currentTask == regretPacket.task && taskPosition == UNASSIGNED) {
        taskPosition = idx;
      }
      if (currentTask == nextTask && nextTaskPosition == UNASSIGNED) {
        nextTaskPosition = idx;
      }
      if (taskPosition != UNASSIGNED && nextTaskPosition != UNASSIGNED) {
        break;
      }
    }
    if (taskPosition == UNASSIGNED) {
      PLOGE << "insertTask: regret task " << regretPacket.task
            << " not found in agent " << regretPacket.agent << " queue\n";
      return false;
    }
    insertedTaskPosition = taskPosition;
    vector<int> goalLocations =
        instance_.getTaskLocations(assignmentsFor(regretPacket.agent));
    if (taskPosition < 0 || taskPosition >= (int)goalLocations.size()) {
      PLOGE << "insertTask: invalid task position " << taskPosition
            << " for agent " << regretPacket.agent << " goal list size "
            << goalLocations.size() << "\n";
      return false;
    }
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    std::unique_ptr<SingleAgentSolver> ownedPlanner;
    SingleAgentSolver* localPlanner = reusablePlanner;
    if (localPlanner == nullptr) {
      ownedPlanner = createLocalPlanner(regretPacket.agent);
      localPlanner = ownedPlanner.get();
    }
    localPlanner->setGoalLocations(goalLocations);

    if (!buildConstraintTable(constraintTable, regretPacket,
                              goalLocations[taskPosition], workspace,
                              activePrecedenceConstraints, false,
                              assignmentOwnerLookup,
                              assignmentPosLookup)) {
      PLOGE << "insertTask: failed to build constraint table for task "
            << regretPacket.task << " (agent " << regretPacket.agent
            << ", position " << taskPosition << ")\n";
      return false;
    }
    AgentTaskPath path = runLowLevelSearch(*localPlanner, constraintTable,
                                           startTime, taskPosition, 0);
    if (path.empty()) {
      return false;
    }
    const auto pathSize = path.size();
    recordSetTaskPath(regretPacket.agent, taskPosition, std::move(path));
    value = pathSize;
    startTime = taskPathsFor(regretPacket.agent)[taskPosition].endTime();

    // Need the current position as task order can shift when ancestor tasks are injected.
    if (nextTaskPosition == UNASSIGNED) {
      PLOGE << "insertTask: next task " << nextTask << " not found in agent "
            << regretPacket.agent << " queue\n";
      return false;
    }
    if (nextTaskPosition < 0 || nextTaskPosition >= (int)goalLocations.size()) {
      PLOGE << "insertTask: invalid next-task position " << nextTaskPosition
            << " for agent " << regretPacket.agent << " goal list size "
            << goalLocations.size() << "\n";
      return false;
    }
    TaskRegretPacket nextTaskPacket = {
        nextTask, regretPacket.agent, nextTaskPosition, {}};
    if (!buildConstraintTable(constraintTable, nextTaskPacket,
                              goalLocations[nextTaskPosition], workspace,
                              activePrecedenceConstraints, true,
                              assignmentOwnerLookup,
                              assignmentPosLookup)) {
      PLOGE << "insertTask: failed to build constraint table for next task "
            << nextTask << " (agent " << regretPacket.agent << ", position "
            << nextTaskPosition << ")\n";
      return false;
    }
    const bool traceTriple = debugImprovementDiagnostics_ &&
                             shouldTraceConstraintDebugTriple(
                                 regretPacket.task, regretPacket.agent, nextTask);
    if (traceTriple) {
      const auto digest =
          computeConstraintTableDigest(constraintTable, instance_);
      PLOGW << "CTTRACE eval-next-before-ll task=" << regretPacket.task
            << " agent=" << regretPacket.agent
            << " task_pos=" << taskPosition
            << " next_task=" << nextTask
            << " next_pos=" << nextTaskPosition
            << " start_time=" << startTime
            << " goal_loc=" << goalLocations[nextTaskPosition]
            << " len_min=" << constraintTable.lengthMin
            << " len_max=" << constraintTable.lengthMax
            << " latest_ts=" << constraintTable.latestTimestep
            << " temporal_extent=" << constraintTable.temporalExtent
            << " ct_hash=" << digest.hash
            << " ct_vertex_buckets=" << digest.vertexBuckets
            << " ct_edge_buckets=" << digest.edgeBuckets
            << " ct_intervals=" << digest.intervalCount
            << " queue="
            << summarizeTaskQueue(assignmentsFor(regretPacket.agent));
    }
    AgentTaskPath nextPath = runLowLevelSearch(
        *localPlanner, constraintTable, startTime, nextTaskPosition, 0);
    if (traceTriple) {
      PLOGW << "CTTRACE eval-next-after-ll task=" << regretPacket.task
            << " agent=" << regretPacket.agent
            << " next_task=" << nextTask
            << " next_pos=" << nextTaskPosition
            << " result=" << (nextPath.empty() ? "empty" : "ok")
            << " path_size=" << nextPath.size();
    }
    if (nextPath.empty()) {
      return false;
    }
    const auto nextPathSize = nextPath.size();
    recordSetTaskPath(regretPacket.agent, nextTaskPosition, std::move(nextPath));
    value += nextPathSize;
  } else {
    rebuildAssignmentLookup();
    if (!isAcyclicAfterLocalInsertion(assignmentLookup, regretPacket.task,
                                      previousTask, UNDEFINED)) {
      return false;
    }

    vector<int> goalLocations =
        instance_.getTaskLocations(assignmentsFor(regretPacket.agent));
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    std::unique_ptr<SingleAgentSolver> ownedPlanner;
    SingleAgentSolver* localPlanner = reusablePlanner;
    if (localPlanner == nullptr) {
      ownedPlanner = createLocalPlanner(regretPacket.agent);
      localPlanner = ownedPlanner.get();
    }
    localPlanner->setGoalLocations(goalLocations);

    if (!buildConstraintTable(constraintTable, regretPacket,
                              goalLocations[regretPacket.taskPosition],
                              workspace, activePrecedenceConstraints, false,
                              assignmentOwnerLookup,
                              assignmentPosLookup)) {
      PLOGE << "insertTask: failed to build constraint table for task "
            << regretPacket.task << " (agent " << regretPacket.agent
            << ", position " << regretPacket.taskPosition << ")\n";
      return false;
    }
    AgentTaskPath path = runLowLevelSearch(*localPlanner, constraintTable,
                                           startTime, regretPacket.taskPosition,
                                           0);
    if (path.empty()) {
      return false;
    }
    const auto pathSize = path.size();
    recordSetTaskPath(regretPacket.agent, regretPacket.taskPosition,
                      std::move(path));
    insertedTaskPosition = regretPacket.taskPosition;
    value = pathSize;
  }

  const auto pathLength = value;
  if (regretPacket.task < 0 ||
      regretPacket.task >= (int)lnsNeighborhood_.removedTasksPathSize.size() ||
      lnsNeighborhood_.removedTasksPathSize[regretPacket.task] < 0) {
    PLOGE << "computeRegretForTaskWithAgent: missing previous path size for task "
          << regretPacket.task << "\n";
    return false;
  }
  const double baseDeltaSoc =
      value - (lnsNeighborhood_.removedTasksPathSize[regretPacket.task] +
               pathSizeChange);
  double adjustedValue = baseDeltaSoc;
  double deltaExposure = 0.0;
  double deltaWait = 0.0;

  if (market_.repairTieBreak || market_.repairBlend) {
    const int task = regretPacket.task;
    const int taskLocation = instance_.getTaskLocations(task);
    const double oldExposure =
        (baselineMetrics != nullptr && baselineMetrics->valid)
            ? baselineMetrics->oldExposure
            : computeTaskMarketExposure(task, true);
    const int oldWait =
        (baselineMetrics != nullptr && baselineMetrics->valid)
            ? baselineMetrics->oldWait
            : computeTaskPrecedenceWaitInCurrentSolution(task);

    if (insertedTaskPosition >= 0 &&
        insertedTaskPosition < (int)taskPathsFor(regretPacket.agent).size()) {
      const AgentTaskPath& insertedTaskPath =
          taskPathsFor(regretPacket.agent)[insertedTaskPosition];
      const double newExposure =
          computeMarketExposureFromPath(insertedTaskPath, true);
      const int newWait =
          computeTaskPrecedenceWaitFromWorkspace(
              task, taskLocation, workspace, assignmentOwnerLookup,
              assignmentPosLookup);
      deltaExposure = newExposure - oldExposure;
      deltaWait = (double)newWait - (double)oldWait;
    }

    const bool applyTieBreak =
        std::abs(baseDeltaSoc) <= market_.tieBreakEpsSoc;
    if (applyTieBreak) {
      adjustedValue +=
          market_.lambdaPrice * deltaExposure + market_.lambdaWait * deltaWait;
    }
  }

  Utility utility(regretPacket.agent, regretPacket.taskPosition, (int)pathLength,
                  (int)assignmentsFor(regretPacket.agent).size(),
                  adjustedValue, baseDeltaSoc, deltaExposure, deltaWait);
  return utility;
}
