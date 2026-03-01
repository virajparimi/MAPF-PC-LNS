#include "lns.hpp"
#include "lns_repair_internal.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>

bool LNS::commitAncestorTaskOf(
    int globalTask, std::optional<pair<bool, int>> committingNextTask) {
  if (runtimeBudgetExhausted()) {
    return false;
  }
  // We are going to commit some ancestor of this global task. We need to ensure that the paths of all the required ancestors of this task are in order before we can commit the global task and any next task that may exist
  // If the boolean flag commitingNextTask is set then it means that the global task was the next task of some other task and we need to ensure that the ancestors of this next task are in order. This additional check is required as the first if condition changes depending on it.
  // The corresponding integer entry would be the global task id of the main task that we wanted to commit.

  vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints();
  const AssignmentLookup previousLookup =
      buildAssignmentLookup(previousSolution_, instance_.getTasksNum());

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (const auto& precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }
  vector<char> ancestorsOfTask = reachableSet(globalTask, ancestors);
  if (globalTask >= 0 && globalTask < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[globalTask] = 0;
  }

  for (int ancestorTask = 0; ancestorTask < (int)ancestorsOfTask.size();
       ancestorTask++) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    if (!ancestorsOfTask[ancestorTask]) {
      continue;
    }
    if (isPendingCommitState(lnsNeighborhood_, ancestorTask)) {
      if (committingNextTask.has_value() &&
          committingNextTask.value().second == ancestorTask) {
        continue;
      }

      const int ancestorTaskAgent =
          (ancestorTask >= 0 &&
           ancestorTask < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[ancestorTask]
              : UNASSIGNED;
      if (ancestorTaskAgent == UNASSIGNED) {
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in previous solution\n";
        return false;
      }
      assert(ancestorTaskAgent != UNASSIGNED);

      PLOGD << "Commiting ancestor task " << ancestorTask << " to agent "
            << ancestorTaskAgent << " using previous solution\n";

      int ancestorTaskPositionRelativeToSolution = extractOldLocalTaskIndex(
          ancestorTask,
          previousSolution_.agents[ancestorTaskAgent].taskAssignments,
          solution_.agents[ancestorTaskAgent].taskAssignments);
      if (ancestorTaskPositionRelativeToSolution == UNASSIGNED) {
        PLOGE << "commitAncestorTaskOf: failed to map ancestor task "
              << ancestorTask << " into current assignment order for agent "
              << ancestorTaskAgent << "\n";
        return false;
      }
      const int ancestorTaskPosition =
          (ancestorTask >= 0 && ancestorTask < (int)previousLookup.pos.size() &&
           previousLookup.owner[ancestorTask] == ancestorTaskAgent)
              ? previousLookup.pos[ancestorTask]
              : UNASSIGNED;
      if (ancestorTaskPosition == UNASSIGNED ||
          ancestorTaskPosition >=
              (int)previousSolution_.agents[ancestorTaskAgent]
                  .taskPaths.size()) {
        PLOGE << "commitAncestorTaskOf: invalid local index for ancestor task "
              << ancestorTask << " on agent " << ancestorTaskAgent << "\n";
        return false;
      }

      AgentTaskPath ancestorPath = previousSolution_.agents[ancestorTaskAgent]
                                       .taskPaths[ancestorTaskPosition];

      solution_.agents[ancestorTaskAgent].pathPlanner->goalLocations.insert(
          solution_.agents[ancestorTaskAgent]
                  .pathPlanner->goalLocations.begin() +
              ancestorTaskPositionRelativeToSolution,
          instance_.getTaskLocations(ancestorTask));
      solution_.agents[ancestorTaskAgent].pathPlanner->computeHeuristics();

      solution_.agents[ancestorTaskAgent].taskAssignments.insert(
          solution_.agents[ancestorTaskAgent].taskAssignments.begin() +
              ancestorTaskPositionRelativeToSolution,
          ancestorTask);

      solution_.agents[ancestorTaskAgent].insertIntraAgentPrecedenceConstraint(
          ancestorTask, ancestorTaskPositionRelativeToSolution);

      solution_.agents[ancestorTaskAgent].taskPaths.insert(
          solution_.agents[ancestorTaskAgent].taskPaths.begin() +
              ancestorTaskPositionRelativeToSolution,
          ancestorPath);

      solution_.taskAgentMap[ancestorTask] = ancestorTaskAgent;

      // The ancestor of the task that was in the conflict set has now been committed using its old path, hence we need to mark it as resolved now
      markResolved(ancestorTask);
    }
  }

  const vector<pair<int, int>> committedPrecedenceConstraints =
      buildFullPrecedenceConstraints();
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    for (int localTask = 0;
         localTask < (int)solution_.agents[agent].taskAssignments.size();
         localTask++) {
      if (runtimeBudgetExhausted()) {
        return false;
      }

      if (localTask > 0) {
        solution_.agents[agent].taskPaths[localTask].beginTime =
            solution_.agents[agent].taskPaths[localTask - 1].endTime();
      } else {
        solution_.agents[agent].taskPaths[localTask].beginTime = 0;
      }
      const bool touchesGlobalTask =
          (globalTask == solution_.agents[agent].taskAssignments[localTask]) ||
          (localTask > 0 &&
           globalTask ==
               solution_.agents[agent].taskAssignments[localTask - 1]);
      if (touchesGlobalTask) {
        // We have not found the path for this global task yet so its task path would be empty placeholder!
        continue;
      }

      const bool touchesCommittingNext =
          committingNextTask.has_value() &&
          ((committingNextTask.value().second ==
            solution_.agents[agent].taskAssignments[localTask]) ||
           (localTask > 0 &&
            committingNextTask.value().second ==
                solution_.agents[agent].taskAssignments[localTask - 1]));
      if (touchesCommittingNext) {
        // We wont have the path for the original commiting task here yet
        continue;
      }

      if ((localTask == 0 &&
           solution_.agents[agent].taskPaths[localTask].front().location !=
               solution_.agents[agent].pathPlanner->startLocation) ||
          (localTask > 0 && solution_.agents[agent]
                                    .taskPaths[localTask - 1]
                                    .path.back()
                                    .location != solution_.agents[agent]
                                                     .taskPaths[localTask]
                                                     .path.front()
                                                     .location)) {
        int startTime = 0;
        if (localTask > 0) {
          startTime =
              solution_.agents[agent].taskPaths[localTask - 1].endTime();
        }
        ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
        if (!buildConstraintTable(
                constraintTable,
                solution_.agents[agent].taskAssignments[localTask],
                committedPrecedenceConstraints)) {
          PLOGE << "commitAncestorTaskOf: failed to build constraint table for "
                << "agent " << agent << ", task "
                << solution_.agents[agent].taskAssignments[localTask]
                << " at position " << localTask << "\n";
          return false;
        }
        AgentTaskPath path = runLowLevelSearch(
            *solution_.agents[agent].pathPlanner, constraintTable, startTime,
            localTask, 0);
        // We must be able to find the path for the next task. If not then we cannot move forward!
        if (path.empty()) {
          PLOGE << "commitAncestorTaskOf: empty path for agent " << agent
                << ", task " << solution_.agents[agent].taskAssignments[localTask]
                << " at position " << localTask << "\n";
          return false;
        }
        assert(!path.empty());
        solution_.agents[agent].taskPaths[localTask] = path;
      }
      lnsNeighborhood_.patchedTasks.erase(
          solution_.agents[agent].taskAssignments[localTask]);
    }
  }

  // Run a validity check!
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0;
         localTask < (int)solution_.agents[agent].taskAssignments.size();
         localTask++) {
      const bool touchesGlobalTask =
          (globalTask == solution_.agents[agent].taskAssignments[localTask]) ||
          (localTask > 0 &&
           globalTask ==
               solution_.agents[agent].taskAssignments[localTask - 1]);
      if (touchesGlobalTask) {
        // We have not found the path for this global task yet so its task path would be empty placeholder!
        continue;
      }

      const bool touchesCommittingNext =
          committingNextTask.has_value() &&
          ((committingNextTask.value().second ==
            solution_.agents[agent].taskAssignments[localTask]) ||
           (localTask > 0 &&
            committingNextTask.value().second ==
                solution_.agents[agent].taskAssignments[localTask - 1]));
      if (touchesCommittingNext) {
        // We wont have the path for the original commiting task here yet
        continue;
      }

      if (localTask > 0) {
        assert(
            solution_.agents[agent]
                .taskPaths[localTask - 1]
                .path.back()
                .location ==
            solution_.agents[agent].taskPaths[localTask].path.front().location);
      } else {
        assert(solution_.agents[agent].taskPaths[localTask].front().location ==
               solution_.agents[agent].pathPlanner->startLocation);
      }
    }
  }
  return true;
}

bool LNS::commitBestRegretTask(Regret bestRegret) {
  if (runtimeBudgetExhausted()) {
    return false;
  }

  PLOGD << "Commiting for task " << bestRegret.task << " to agent "
        << bestRegret.agent << " with regret = " << bestRegret.value << "\n";

  TaskRegretPacket bestRegretPacket = {
      bestRegret.task, bestRegret.agent, bestRegret.taskPosition, {}};

  if (!commitAncestorTaskOf(bestRegret.task, std::nullopt)) {
    return false;
  }

  // At this point any previously empty paths must be resolved and we can use the insert task function to commit to the actual best regret task
  if (!insertBestRegretTask(bestRegretPacket)) {
    return false;
  }
  markResolved(bestRegret.task);
  return true;
}

bool LNS::insertBestRegretTask(TaskRegretPacket bestRegretPacket) {
  if (runtimeBudgetExhausted()) {
    return false;
  }

  int startTime = 0, previousTask = UNDEFINED, nextTask = UNDEFINED;

  vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints();

  int agentTasksSize =
      (int)solution_.agents[bestRegretPacket.agent].taskAssignments.size();

  // In this case we are inserting a task not at the last position
  if (bestRegretPacket.taskPosition < agentTasksSize) {

    nextTask = solution_.agents[bestRegretPacket.agent]
                   .taskAssignments[bestRegretPacket.taskPosition];
    solution_.agents[bestRegretPacket.agent]
        .taskPaths[bestRegretPacket.taskPosition] = AgentTaskPath();

    solution_.agents[bestRegretPacket.agent].taskAssignments.insert(
        solution_.agents[bestRegretPacket.agent].taskAssignments.begin() +
            bestRegretPacket.taskPosition,
        bestRegretPacket.task);
    precedenceConstraints.emplace_back(bestRegretPacket.task, nextTask);

    // If we are NOT inserting at the start position then we need to take care of the previous task as well
    if (bestRegretPacket.taskPosition != 0) {
      previousTask = solution_.agents[bestRegretPacket.agent]
                         .taskAssignments[bestRegretPacket.taskPosition - 1];
      // TODO: Technically the task can start being processed before the previous task ends. This is more conservative but need to check if there are better ways to tackle this.
      startTime = solution_.agents[bestRegretPacket.agent]
                      .taskPaths[bestRegretPacket.taskPosition - 1]
                      .endTime();
      precedenceConstraints.erase(
          std::remove_if(
              precedenceConstraints.begin(), precedenceConstraints.end(),
              [previousTask, nextTask](pair<int, int> x) {
                return x.first == previousTask && x.second == nextTask;
              }),
          precedenceConstraints.end());
      precedenceConstraints.emplace_back(previousTask, bestRegretPacket.task);
    }

    // Insert an empty path at that task position
    solution_.agents[bestRegretPacket.agent].taskPaths.insert(
        solution_.agents[bestRegretPacket.agent].taskPaths.begin() +
            bestRegretPacket.taskPosition,
        AgentTaskPath());
  }
  // In this case we are inserting at the very end
  else if (bestRegretPacket.taskPosition == agentTasksSize &&
           agentTasksSize != 0) {

    previousTask = solution_.agents[bestRegretPacket.agent]
                       .taskAssignments[bestRegretPacket.taskPosition - 1];
    startTime = solution_.agents[bestRegretPacket.agent]
                    .taskPaths[bestRegretPacket.taskPosition - 1]
                    .endTime();

    solution_.agents[bestRegretPacket.agent].taskAssignments.push_back(
        bestRegretPacket.task);
    precedenceConstraints.emplace_back(previousTask, bestRegretPacket.task);

    solution_.agents[bestRegretPacket.agent].taskPaths.emplace_back();
  } else if (agentTasksSize == 0) {
    // The rare-case when the agent has no tasks assigned to them
    assert(bestRegretPacket.taskPosition == 0);

    startTime = 0;
    solution_.agents[bestRegretPacket.agent].taskAssignments.push_back(
        bestRegretPacket.task);
    solution_.agents[bestRegretPacket.agent].taskPaths.emplace_back();
  }

  if (nextTask >= 0) {
    if (runtimeBudgetExhausted()) {
      return false;
    }

    if (!commitAncestorTaskOf(
            nextTask,
            std::make_optional(make_pair(true, bestRegretPacket.task)))) {
      return false;
    }

    vector<int> goalLocations = instance_.getTaskLocations(
        solution_.agents[bestRegretPacket.agent].taskAssignments);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    solution_.agents[bestRegretPacket.agent].pathPlanner->setGoalLocations(
        goalLocations);

    // Ancestor injection may shift both the committed task and next task
    // positions. Resolve both in one pass over the assignment queue.
    int taskPosition = UNASSIGNED;
    int nextTaskPosition = UNASSIGNED;
    const auto& assignmentsForAgent =
        solution_.agents[bestRegretPacket.agent].taskAssignments;
    for (int idx = 0; idx < (int)assignmentsForAgent.size(); idx++) {
      const int currentTask = assignmentsForAgent[idx];
      if (currentTask == bestRegretPacket.task && taskPosition == UNASSIGNED) {
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
      PLOGE << "insertBestRegretTask: committed task "
            << bestRegretPacket.task
            << " missing from agent assignments after ancestor injection\n";
      return false;
    }
    if (taskPosition < 0 ||
        taskPosition >=
            (int)solution_.agents[bestRegretPacket.agent].taskPaths.size()) {
      PLOGE << "insertBestRegretTask: invalid task position " << taskPosition
            << " for task " << bestRegretPacket.task << " (agent "
            << bestRegretPacket.agent << ")\n";
      return false;
    }
    if (nextTaskPosition == UNASSIGNED) {
      PLOGE << "insertBestRegretTask: next task " << nextTask
            << " missing from assignments after commit\n";
      return false;
    }

    // Ancestor injection may shift task positions and therefore invalidate the
    // precomputed start time. Recompute from the current predecessor path.
    startTime = (taskPosition > 0)
                    ? solution_.agents[bestRegretPacket.agent]
                          .taskPaths[taskPosition - 1]
                          .endTime()
                    : 0;

    if (!buildConstraintTable(constraintTable, bestRegretPacket.task,
                              precedenceConstraints)) {
      PLOGE << "insertBestRegretTask: failed to build constraint table for "
            << "task " << bestRegretPacket.task << " (agent "
            << bestRegretPacket.agent << ")\n";
      return false;
    }
    if (runtimeBudgetExhausted()) {
      return false;
    }
    AgentTaskPath path = runLowLevelSearch(
        *solution_.agents[bestRegretPacket.agent].pathPlanner, constraintTable,
        startTime, taskPosition, 0);
    // We must be able to insert this path for this task as its the best regret path and we did try it before i.e it must have succeeded then
    if (path.empty()) {
      PLOGE << "insertBestRegretTask: empty path for task "
            << bestRegretPacket.task << " (agent " << bestRegretPacket.agent
            << ", position " << taskPosition << ")\n";
      return false;
    }
    assert(!path.empty());
    solution_.agents[bestRegretPacket.agent].taskPaths[taskPosition] = path;
    solution_.agents[bestRegretPacket.agent]
        .insertIntraAgentPrecedenceConstraint(bestRegretPacket.task,
                                              taskPosition);
    solution_.taskAgentMap[bestRegretPacket.task] = bestRegretPacket.agent;

    if (!buildConstraintTable(constraintTable, nextTask,
                              precedenceConstraints)) {
      PLOGE << "insertBestRegretTask: failed to build constraint table for "
            << "next task " << nextTask << " (agent "
            << bestRegretPacket.agent << ")\n";
      return false;
    }
    if (nextTaskPosition <= 0 ||
        nextTaskPosition >=
            (int)solution_.agents[bestRegretPacket.agent].taskPaths.size()) {
      PLOGE << "insertBestRegretTask: invalid next task position "
            << nextTaskPosition << " for task " << nextTask << " (agent "
            << bestRegretPacket.agent << ")\n";
      return false;
    }
    startTime = solution_.agents[bestRegretPacket.agent]
                    .taskPaths[nextTaskPosition - 1]
                    .endTime();
    if (runtimeBudgetExhausted()) {
      return false;
    }
    const bool traceTriple = debugImprovementDiagnostics_ &&
                             shouldTraceConstraintDebugTriple(
                                 bestRegretPacket.task, bestRegretPacket.agent,
                                 nextTask);
    if (traceTriple) {
      const auto digest =
          computeConstraintTableDigest(constraintTable, instance_);
      PLOGW << "CTTRACE commit-next-before-ll task=" << bestRegretPacket.task
            << " agent=" << bestRegretPacket.agent
            << " task_pos=" << taskPosition
            << " next_task=" << nextTask
            << " next_pos=" << nextTaskPosition
            << " start_time=" << startTime
            << " goal_loc=" << constraintTable.goalLocation
            << " len_min=" << constraintTable.lengthMin
            << " len_max=" << constraintTable.lengthMax
            << " latest_ts=" << constraintTable.latestTimestep
            << " temporal_extent=" << constraintTable.temporalExtent
            << " ct_hash=" << digest.hash
            << " ct_vertex_buckets=" << digest.vertexBuckets
            << " ct_edge_buckets=" << digest.edgeBuckets
            << " ct_intervals=" << digest.intervalCount
            << " queue="
            << summarizeTaskQueue(
                   solution_.agents[bestRegretPacket.agent].taskAssignments);
    }
    AgentTaskPath nextPath = runLowLevelSearch(
        *solution_.agents[bestRegretPacket.agent].pathPlanner, constraintTable,
        startTime, nextTaskPosition, 0);
    if (traceTriple) {
      PLOGW << "CTTRACE commit-next-after-ll task=" << bestRegretPacket.task
            << " agent=" << bestRegretPacket.agent
            << " next_task=" << nextTask
            << " next_pos=" << nextTaskPosition
            << " result=" << (nextPath.empty() ? "empty" : "ok")
            << " path_size=" << nextPath.size();
    }
    // We must be able to insert this path for this task as its the best regret path and we did try it before i.e it must have succeeded then
    if (nextPath.empty()) {
      PLOGE << "insertBestRegretTask: empty path for next task "
            << nextTask << " (agent " << bestRegretPacket.agent
            << ", position " << nextTaskPosition << ")\n";
      return false;
    }
    assert(!nextPath.empty());
    solution_.agents[bestRegretPacket.agent].taskPaths[nextTaskPosition] =
        nextPath;
  } else {

    vector<int> goalLocations = instance_.getTaskLocations(
        solution_.agents[bestRegretPacket.agent].taskAssignments);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    solution_.agents[bestRegretPacket.agent].pathPlanner->setGoalLocations(
        goalLocations);

    if (!buildConstraintTable(constraintTable, bestRegretPacket.task,
                              precedenceConstraints)) {
      PLOGE << "insertBestRegretTask: failed to build constraint table for "
            << "task " << bestRegretPacket.task << " (agent "
            << bestRegretPacket.agent << ")\n";
      return false;
    }
    if (runtimeBudgetExhausted()) {
      return false;
    }
    AgentTaskPath path = runLowLevelSearch(
        *solution_.agents[bestRegretPacket.agent].pathPlanner, constraintTable,
        startTime, bestRegretPacket.taskPosition, 0);
    // We must be able to insert this path for this task as its the best regret path and we did try it before i.e it must have succeeded then
    if (path.empty()) {
      PLOGE << "insertBestRegretTask: empty path for task "
            << bestRegretPacket.task << " (agent " << bestRegretPacket.agent
            << ", position " << bestRegretPacket.taskPosition << ")\n";
      return false;
    }
    assert(!path.empty());
    solution_.agents[bestRegretPacket.agent]
        .taskPaths[bestRegretPacket.taskPosition] = path;
    solution_.agents[bestRegretPacket.agent]
        .insertIntraAgentPrecedenceConstraint(bestRegretPacket.task,
                                              bestRegretPacket.taskPosition);
    solution_.taskAgentMap[bestRegretPacket.task] = bestRegretPacket.agent;
  }

  patchAgentTaskPaths(bestRegretPacket.agent, 0);
  return true;
}
