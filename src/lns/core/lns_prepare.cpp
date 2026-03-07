#include "lns.hpp"
#include "internal/task_position_index.hpp"
#include "utils.hpp"
#include <cmath>
#include <limits>

bool LNS::prepareNextIteration() {
  PLOGI << "Preparing the solution object for the next iteration\n";
  invalidateCurrentTaskAssignmentIndexCache();
  lastPrepareAbortedByCascade_ = false;
  lastPrepareSeedTasks_ = 0;
  lastPrepareClosureTasks_ = 0;
  lastPrepareClosureAdded_ = 0;
  lastPrepareAffectedAgents_.clear();

  const int taskCount = instance_.getTasksNum();
  const vector<int> taskToPositionBefore =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);

  // Find the tasks that are following the earliest conflicting task as their paths need to be invalidated
  const auto& successors = instance_.getSuccessorsRef();

  // Include all successors of the original conflicted tasks in one multi-source
  // traversal to avoid re-traversing shared descendant subgraphs.
  const ConflictMap originalRemovedTasks = lnsNeighborhood_.removedTasks;
  ConflictMap closureRemovedTasks = originalRemovedTasks;
  vector<Conflicts> closureSeeds;
  closureSeeds.reserve(originalRemovedTasks.size());
  for (const auto& [_, conflict] : originalRemovedTasks) {
    closureSeeds.push_back(conflict);
  }
  vector<char> visitedSuccessor(taskCount, 0);
  stack<int> successorStack;
  for (const Conflicts& conflictTask : closureSeeds) {
    if (conflictTask.task >= 0 && conflictTask.task < taskCount) {
      successorStack.push(conflictTask.task);
    }
  }
  while (!successorStack.empty()) {
    const int successorTask = successorStack.top();
    successorStack.pop();
    if (successorTask < 0 || successorTask >= taskCount ||
        visitedSuccessor[successorTask]) {
      continue;
    }
    visitedSuccessor[successorTask] = 1;

    const int successorAgent =
        (successorTask >= 0 &&
         successorTask < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[successorTask]
            : UNASSIGNED;
    if (successorAgent != UNASSIGNED) {
      const int successorTaskPosition =
          (successorTask >= 0 &&
           successorTask < (int)taskToPositionBefore.size())
              ? taskToPositionBefore[successorTask]
              : UNASSIGNED;
      if (successorTaskPosition != UNASSIGNED) {
        closureRemovedTasks.emplace(
            successorTask,
            Conflicts(successorTask, successorAgent, successorTaskPosition));
      }
    }

    for (int nextTask : successors[successorTask]) {
      if (nextTask >= 0 && nextTask < taskCount &&
          !visitedSuccessor[nextTask]) {
        successorStack.push(nextTask);
      }
    }
  }
  const int closureSeedCount = (int)originalRemovedTasks.size();
  const int closureTaskCount = (int)closureRemovedTasks.size();
  const int closureAddedCount =
      max(0, closureTaskCount - closureSeedCount);
  lastPrepareSeedTasks_ = closureSeedCount;
  lastPrepareClosureTasks_ = closureTaskCount;
  lastPrepareClosureAdded_ = closureAddedCount;
  cascadeStats_.prepareCalls++;
  cascadeStats_.seedTasksSum += closureSeedCount;
  cascadeStats_.closureTasksSum += closureTaskCount;
  cascadeStats_.closureAddedSum += closureAddedCount;
  cascadeStats_.closureTasksMax =
      max(cascadeStats_.closureTasksMax, (int64_t)closureTaskCount);
  cascadeStats_.closureAddedMax =
      max(cascadeStats_.closureAddedMax, (int64_t)closureAddedCount);

  const int staticCascadeBudget = cascadeTaskBudget();
  int cascadeBudget = staticCascadeBudget;
  int adaptiveBudgetLower = 1;
  int adaptiveBudgetUpper = staticCascadeBudget;
  if (adaptiveCascadeBudget_) {
    const bool explicitHardCap = (maxCascadeTasks_ > 0);
    adaptiveBudgetUpper =
        explicitHardCap ? staticCascadeBudget
                        : max(staticCascadeBudget, instance_.getTasksNum());
    adaptiveBudgetLower = max(1, min(staticCascadeBudget, closureSeedCount + 1));
    if (adaptiveCascadeBudgetCurrent_ <= 0) {
      adaptiveCascadeBudgetCurrent_ = staticCascadeBudget;
    }
    adaptiveCascadeBudgetCurrent_ =
        min(adaptiveBudgetUpper, max(adaptiveBudgetLower,
                                     adaptiveCascadeBudgetCurrent_));
    cascadeBudget = adaptiveCascadeBudgetCurrent_;
  }

  cascadeStats_.budgetUsedSum += cascadeBudget;
  if (cascadeStats_.prepareCalls == 1) {
    cascadeStats_.budgetUsedMin = cascadeBudget;
    cascadeStats_.budgetUsedMax = cascadeBudget;
  } else {
    cascadeStats_.budgetUsedMin =
        min(cascadeStats_.budgetUsedMin, (int64_t)cascadeBudget);
    cascadeStats_.budgetUsedMax =
        max(cascadeStats_.budgetUsedMax, (int64_t)cascadeBudget);
  }
  adaptiveCascadeBudgetLastUsed_ = cascadeBudget;

  if (closureAddedCount > cascadeBudget) {
    if (adaptiveCascadeBudget_) {
      const int growthStep = max(1, adaptiveCascadeBudgetCurrent_ / 4);
      const int targetBudget =
          max(closureSeedCount + 1, adaptiveCascadeBudgetCurrent_ + growthStep);
      const int nextBudget =
          min(adaptiveBudgetUpper, max(adaptiveBudgetLower, targetBudget));
      if (nextBudget > adaptiveCascadeBudgetCurrent_) {
        cascadeStats_.adaptiveBudgetIncreases++;
      }
      adaptiveCascadeBudgetCurrent_ = nextBudget;
    }
    lastPrepareAbortedByCascade_ = true;
    cascadeStats_.budgetAborts++;
    PLOGW << "prepareNextIteration: cascade budget exceeded (seed="
          << closureSeedCount << ", closure_total=" << closureTaskCount
          << ", closure_added=" << closureAddedCount
          << ", budget=" << cascadeBudget << ", baseline="
          << staticCascadeBudget << ")\n";
    return false;
  }

  if (adaptiveCascadeBudget_) {
    int nextBudget = adaptiveCascadeBudgetCurrent_;
    const double closurePressure =
        (cascadeBudget > 0) ? ((double)closureAddedCount / (double)cascadeBudget)
                            : 1.0;
    if (closurePressure < 0.35) {
      nextBudget = max(adaptiveBudgetLower, adaptiveCascadeBudgetCurrent_ - 1);
    } else if (closurePressure > 0.85) {
      nextBudget = min(adaptiveBudgetUpper, adaptiveCascadeBudgetCurrent_ + 1);
    }

    // Avoid shrinking budget immediately after a productive iteration.
    if (!iterationStats.empty() &&
        (iterationStats.back().quality == IterationQuality::bestSolutionYet ||
         iterationStats.back().quality == IterationQuality::improvedSolution) &&
        nextBudget < adaptiveCascadeBudgetCurrent_) {
      nextBudget = adaptiveCascadeBudgetCurrent_;
    }

    if (nextBudget > adaptiveCascadeBudgetCurrent_) {
      cascadeStats_.adaptiveBudgetIncreases++;
    } else if (nextBudget < adaptiveCascadeBudgetCurrent_) {
      cascadeStats_.adaptiveBudgetDecreases++;
    }
    adaptiveCascadeBudgetCurrent_ = nextBudget;
  }

  lnsNeighborhood_.removedTasks = std::move(closureRemovedTasks);

  // If t_id is deleted then t_id + 1 task needs to be fixed
  set<int> tasksToFix, affectedAgents;
  for (const auto& [_, invalidTask] : lnsNeighborhood_.removedTasks) {

    int agent = invalidTask.agent, taskPosition = invalidTask.taskPosition;
    if (agent < 0 || agent >= instance_.getAgentNum() || taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size() ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size()) {
      PLOGE << "prepareNextIteration: invalid conflict entry (task="
            << invalidTask.task << ", agent=" << agent
            << ", position=" << taskPosition << ")\n";
      continue;
    }
    PLOGD << "Invalidating task: " << invalidTask.task << ", Agent: " << agent
          << " at position: " << taskPosition << " with path length = "
          << solution_.agents[agent].taskPaths[taskPosition].size() << "\n";

    // If the invalidated task was not the last local task of this agent then t_id + 1 exists
    // If the invalid task was not the last task
    if (invalidTask.task != solution_.getAgentGlobalTasks(agent).back()) {
      int nextTask = UNDEFINED, nextTaskPosition = taskPosition + 1;
      while (nextTask == UNDEFINED &&
             nextTaskPosition <
                 (int)solution_.getAgentGlobalTasks(agent).size()) {
        nextTask = solution_.getAgentGlobalTasks(agent)[nextTaskPosition];
        nextTaskPosition++;
      }
      PLOGD << "Found a potential next task!\n";
      // Next task can still be undefined in the case where that task was removed in the previous iteration of this loop
      if (lnsNeighborhood_.removedTasks.count(nextTask) == 0 &&
          nextTask != UNDEFINED) {
        tasksToFix.insert(nextTask);
        PLOGD << "Next task: " << nextTask << "\n";
      }
    }

    affectedAgents.insert(agent);

    // Marking past information about this conflicting task
    solution_.taskAgentMap[invalidTask.task] = UNASSIGNED;
    solution_.agents[agent].clearIntraAgentPrecedenceConstraint(
        invalidTask.task);
    // Needs to happen after clearing precedence constraints
    solution_.agents[agent].taskAssignments[taskPosition] = UNDEFINED;

    if (invalidTask.task >= 0 &&
        invalidTask.task < (int)lnsNeighborhood_.removedTasksPathSize.size()) {
      lnsNeighborhood_.removedTasksPathSize[invalidTask.task] =
          (int)solution_.agents[agent].taskPaths[taskPosition].size();
    }
    solution_.agents[agent].taskPaths[taskPosition] = AgentTaskPath();
  }
  lastPrepareAffectedAgents_.assign(affectedAgents.begin(),
                                    affectedAgents.end());

  // Marking past information about conflicting tasks
  for (int affAgent : affectedAgents) {

    // For an affected agent there can be multiple conflicting tasks so need to do it this way
    solution_.agents[affAgent].path = AgentTaskPath();
    solution_.agents[affAgent].terminalPath = AgentTaskPath();
    solution_.agents[affAgent].terminalPathActive = false;
    solution_.agents[affAgent].taskAssignments.erase(
        std::remove_if(solution_.agents[affAgent].taskAssignments.begin(),
                       solution_.agents[affAgent].taskAssignments.end(),
                       [](int task) { return task == UNDEFINED; }),
        solution_.agents[affAgent].taskAssignments.end());
    solution_.agents[affAgent].taskPaths.erase(
        std::remove_if(solution_.agents[affAgent].taskPaths.begin(),
                       solution_.agents[affAgent].taskPaths.end(),
                       [](const Path& p) { return p.empty(); }),
        solution_.agents[affAgent].taskPaths.end());

    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(affAgent));
    solution_.agents[affAgent].pathPlanner->setGoalLocations(taskLocations);
  }

  const vector<int> taskToPositionAfter =
      mapf_pc_lns::internal::buildTaskPositionIndexByMappedAgent(solution_, taskCount);

  lnsNeighborhood_.patchedTasks = tasksToFix;

  // Find the paths for the tasks whose previous tasks were removed
  for (int task : instance_.getInputPlanningOrderRef()) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    if (tasksToFix.count(task) > 0) {

      PLOGD << "Going to find path for next task: " << task << "\n";

      int startTime = 0, agent = solution_.getAgentWithTask(task);
      if (agent == UNASSIGNED) {
        PLOGE << "prepareNextIteration: patched task " << task
              << " is not assigned to any agent\n";
        return false;
      }
      int taskPosition = (task >= 0 && task < (int)taskToPositionAfter.size())
                             ? taskToPositionAfter[task]
                             : UNASSIGNED;
      const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
      if (taskPosition < 0 || taskPosition >= (int)agentTasks.size() ||
          taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
        PLOGE << "prepareNextIteration: invalid task position " << taskPosition
              << " for task " << task << " (agent " << agent << ")\n";
        return false;
      }

      if (taskPosition != 0) {
        startTime = solution_.agents[agent].taskPaths[taskPosition - 1].endTime();
      }
      assert(taskPosition <= (int)agentTasks.size() - 1);

      ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
      if (!buildConstraintTable(constraintTable, task)) {
        PLOGE << "prepareNextIteration: failed to build constraint table for "
              << "task " << task << " (agent " << agent << ")\n";
        return false;
      }
      AgentTaskPath path = runLowLevelSearch(
          *solution_.agents[agent].pathPlanner, constraintTable, startTime,
          taskPosition, 0);
      // We must be able to find the path for the next task. If not then we cannot move forward!
      if (path.empty()) {
        PLOGE << "prepareNextIteration: path finding failed for patched task "
              << task << " (agent " << agent << ", position " << taskPosition
              << ")\n";
        return false;
      }
      assert(!path.empty());
      solution_.agents[agent].taskPaths[taskPosition] = path;

      // Once the path was found fix the begin times for subsequent tasks of the agent
      patchAgentTaskPaths(agent, taskPosition);
    }
  }
  return true;
}
