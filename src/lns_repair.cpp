#include "lns.hpp"
#include "utils.hpp"
#include <limits>

namespace {
struct AssignmentLookup {
  vector<int> owner;
  vector<int> pos;
};

AssignmentLookup buildAssignmentLookup(const vector<vector<int>>& assignments,
                                       int taskCount) {
  AssignmentLookup lookup;
  lookup.owner.assign(taskCount, UNASSIGNED);
  lookup.pos.assign(taskCount, -1);
  for (int agent = 0; agent < (int)assignments.size(); agent++) {
    for (int i = 0; i < (int)assignments[agent].size(); i++) {
      const int task = assignments[agent][i];
      if (task >= 0 && task < taskCount) {
        lookup.owner[task] = agent;
        lookup.pos[task] = i;
      }
    }
  }
  return lookup;
}

bool isPendingCommitState(const Neighbor& neighborhood, int task) {
  return task >= 0 && task < (int)neighborhood.committedTasks.size() &&
         neighborhood.committedTasks[task] == 0;
}
}  // namespace

bool LNS::computeRegret() {
  regretEvalStatsCurrent_.recomputeCalls++;
  regretEvalStatsTotal_.recomputeCalls++;
  lnsNeighborhood_.regretMaxHeap.clear();
  const vector<pair<int, int>> fullPrecedenceConstraints =
      buildFullPrecedenceConstraints();
  for (const auto& [_, conflictTask] : lnsNeighborhood_.removedTasks) {
    bool enoughSpace =
        computeRegretForTask(conflictTask.task, fullPrecedenceConstraints);
    if (!enoughSpace) {
      return false;
    }
  }
  return true;
}

vector<int> LNS::collectRemainingRemovedTasks() const {
  vector<int> tasks;
  tasks.reserve(lnsNeighborhood_.removedTasks.size());
  for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
    tasks.push_back(conflict.task);
  }
  return tasks;
}

vector<int> LNS::computeCurrentTaskEndTimes() const {
  vector<int> endTimes(instance_.getTasksNum(), -1);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int pos = 0; pos < (int)solution_.agents[agent].taskAssignments.size();
         pos++) {
      const int task = solution_.agents[agent].taskAssignments[pos];
      if (pos >= (int)solution_.agents[agent].taskPaths.size()) {
        continue;
      }
      if (solution_.agents[agent].taskPaths[pos].empty()) {
        continue;
      }
      endTimes[task] = solution_.agents[agent].taskPaths[pos].endTime();
    }
  }
  return endTimes;
}

vector<int> LNS::computeCurrentLastTaskPerAgent() const {
  vector<int> lastTasks(instance_.getAgentNum(), UNASSIGNED);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      lastTasks[agent] = solution_.agents[agent].taskAssignments.back();
    }
  }
  return lastTasks;
}

bool LNS::recomputeRegretsForTasks(const vector<int>& tasks) {
  regretEvalStatsCurrent_.recomputeCalls++;
  regretEvalStatsTotal_.recomputeCalls++;
  incrementalRegretStatsCurrent_.recomputeCalls++;
  incrementalRegretStatsTotal_.recomputeCalls++;
  const vector<pair<int, int>> fullPrecedenceConstraints =
      buildFullPrecedenceConstraints();
  for (int task : tasks) {
    if (!isPendingCommitState(lnsNeighborhood_, task)) {
      continue;
    }
    regretStamp_[task]++;
    incrementalRegretStatsCurrent_.recomputedTasks++;
    incrementalRegretStatsTotal_.recomputedTasks++;
    const bool enoughSpace =
        computeRegretForTask(task, fullPrecedenceConstraints);
    if (!enoughSpace) {
      return false;
    }
  }
  return true;
}

std::optional<Regret> LNS::popNextValidRegret() {
  while (!lnsNeighborhood_.regretMaxHeap.empty()) {
    Regret r = lnsNeighborhood_.regretMaxHeap.top();
    lnsNeighborhood_.regretMaxHeap.pop();

    if (!isPendingCommitState(lnsNeighborhood_, r.task)) {
      incrementalRegretStatsCurrent_.stalePops++;
      incrementalRegretStatsTotal_.stalePops++;
      continue;
    }
    if (!incrementalRegret_ || r.stamp == regretStamp_[r.task]) {
      return r;
    }
    incrementalRegretStatsCurrent_.stalePops++;
    incrementalRegretStatsTotal_.stalePops++;
  }
  return std::nullopt;
}

vector<int> LNS::computeDirtyTasksAfterCommit(const vector<int>& endTimesBefore,
                                             const vector<int>& endTimesAfter,
                                             const vector<int>& lastTaskBefore,
                                             const vector<int>& lastTaskAfter) {
  vector<int> changedTasks;
  changedTasks.reserve(instance_.getTasksNum());
  for (int task = 0; task < instance_.getTasksNum(); task++) {
    if (endTimesAfter[task] < 0) {
      continue;
    }
    if (endTimesAfter[task] != endTimesBefore[task]) {
      changedTasks.push_back(task);
    }
  }
  incrementalRegretStatsCurrent_.changedSum += (int64_t)changedTasks.size();
  incrementalRegretStatsTotal_.changedSum += (int64_t)changedTasks.size();
  incrementalRegretStatsCurrent_.changedMax =
      max(incrementalRegretStatsCurrent_.changedMax,
          (int64_t)changedTasks.size());
  incrementalRegretStatsTotal_.changedMax =
      max(incrementalRegretStatsTotal_.changedMax, (int64_t)changedTasks.size());

  vector<bool> affectedAgents(instance_.getAgentNum(), false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (lastTaskBefore[agent] != lastTaskAfter[agent]) {
      affectedAgents[agent] = true;
    }
  }
  for (int task : changedTasks) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    if (agent >= 0 && agent < instance_.getAgentNum()) {
      affectedAgents[agent] = true;
    }
  }

  vector<bool> isDirty(instance_.getTasksNum(), false);

  const auto& successors = instance_.getSuccessorsRef();
  std::vector<int> stack;
  stack.reserve(changedTasks.size());
  for (int t : changedTasks) {
    stack.push_back(t);
  }
  while (!stack.empty()) {
    const int current = stack.back();
    stack.pop_back();
    if (current < 0 || current >= instance_.getTasksNum()) {
      continue;
    }
    if (isDirty[current]) {
      continue;
    }
    isDirty[current] = true;
    for (int succ : successors[current]) {
      if (!isDirty[succ]) {
        stack.push_back(succ);
      }
    }
  }

  if (incrementalRegretMode_ == IncrementalRegretMode::descendants_and_agent) {
    for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
      const int task = conflict.task;
      const int bestAgent = regretBestOption_[task].first;
      const int secondAgent = regretSecondBestOption_[task].first;
      if ((bestAgent != UNASSIGNED && affectedAgents[bestAgent]) ||
          (secondAgent != UNASSIGNED && affectedAgents[secondAgent])) {
        isDirty[task] = true;
      }
    }
  }

  vector<int> dirtyTasks;
  dirtyTasks.reserve(lnsNeighborhood_.removedTasks.size());
  for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
    if (isDirty[conflict.task]) {
      dirtyTasks.push_back(conflict.task);
    }
  }
  incrementalRegretStatsCurrent_.dirtySum += (int64_t)dirtyTasks.size();
  incrementalRegretStatsTotal_.dirtySum += (int64_t)dirtyTasks.size();
  incrementalRegretStatsCurrent_.dirtyMax =
      max(incrementalRegretStatsCurrent_.dirtyMax, (int64_t)dirtyTasks.size());
  incrementalRegretStatsTotal_.dirtyMax =
      max(incrementalRegretStatsTotal_.dirtyMax, (int64_t)dirtyTasks.size());
  return dirtyTasks;
}

bool LNS::computeRegretForTask(int task) {
  const vector<pair<int, int>> fullPrecedenceConstraints =
      buildFullPrecedenceConstraints();
  return computeRegretForTask(task, fullPrecedenceConstraints);
}

bool LNS::computeRegretForTask(
    int task, const vector<pair<int, int>>& fullPrecedenceConstraints) {
  regretEvalStatsCurrent_.tasksEvaluated++;
  regretEvalStatsTotal_.tasksEvaluated++;
  pairing_heap<Utility, compare<Utility::CompareUtilities>> serviceTimes;

  // The task has to start after the earliest time step but needs to finish before the latest time step. However we cannot give any guarantee on the latest timestep so we only work with the earliest timestep
  int earliestTimestep = 0;

  vector<pair<int, int>> precedenceConstraints = fullPrecedenceConstraints;

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
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

  vector<vector<int>> agentTaskAssignments(instance_.getAgentNum());
  vector<vector<AgentTaskPath>> agentTaskPaths(instance_.getAgentNum());
  vector<vector<pair<int, int>>> agentPrecedenceConstraints(
      instance_.getAgentNum());

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& sourceAssignments = solution_.agents[agent].taskAssignments;
    if (sourceAssignments.empty()) {
      continue;
    }
    agentTaskAssignments[agent] = sourceAssignments;
    agentTaskPaths[agent] = solution_.agents[agent].taskPaths;
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
        PLOGE << "Missing agent assignment for ancestor task "
              << ancestorTask << " in previous solution\n";
        return false;
      }
      assert(ancestorTaskAgent != UNASSIGNED);
      int ancestorTaskLocalIndex =
          previousSolution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
      if (ancestorTaskLocalIndex == UNASSIGNED ||
          ancestorTaskLocalIndex >=
              (int)previousSolution_.agents[ancestorTaskAgent]
                  .taskPaths.size()) {
        PLOGE << "computeRegretForTask: invalid local index for ancestor task "
              << ancestorTask << " on agent " << ancestorTaskAgent << "\n";
        return false;
      }
      int ancestorTaskLocalIndexRelativeToSolution = extractOldLocalTaskIndex(
          ancestorTask,
          previousSolution_.agents[ancestorTaskAgent].taskAssignments,
          agentTaskAssignments[ancestorTaskAgent]);
      if (ancestorTaskLocalIndexRelativeToSolution == UNASSIGNED) {
        PLOGE << "computeRegretForTask: failed to map ancestor task "
              << ancestorTask << " into current assignment order\n";
        return false;
      }
      agentTaskAssignments[ancestorTaskAgent].insert(
          agentTaskAssignments[ancestorTaskAgent].begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          ancestorTask);
      agentTaskPaths[ancestorTaskAgent].insert(
          agentTaskPaths[ancestorTaskAgent].begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          previousSolution_.agents[ancestorTaskAgent]
              .taskPaths[ancestorTaskLocalIndex]);
    }
  }

  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.assign(inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0; localTask < (int)agentTaskAssignments[agent].size();
         localTask++) {
      if (localTask > 0) {
        agentPrecedenceConstraints[agent].emplace_back(
            agentTaskAssignments[agent][localTask - 1],
            agentTaskAssignments[agent][localTask]);
        agentTaskPaths[agent][localTask].beginTime =
            agentTaskPaths[agent][localTask - 1].endTime();
      } else {
        agentTaskPaths[agent][localTask].beginTime = 0;
      }

      if ((localTask == 0 &&
           agentTaskPaths[agent][localTask].front().location !=
               instance_.getStartLocationsRef()[agent]) ||
          (localTask > 0 &&
           agentTaskPaths[agent][localTask - 1].path.back().location !=
               agentTaskPaths[agent][localTask].path.front().location)) {
        int startTime = 0;
        if (localTask > 0) {
          startTime = agentTaskPaths[agent][localTask - 1].endTime();
        }
        vector<int> goalLocations =
            instance_.getTaskLocations(agentTaskAssignments[agent]);
        auto localPlanner = createLocalPlanner(agent);
        localPlanner->setGoalLocations(goalLocations);

        ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
        TaskRegretPacket taskPacket = {agentTaskAssignments[agent][localTask],
                                       agent, localTask, -1};
        // TODO: Possible incomplete precedence constraints here!
        buildConstraintTable(constraintTable, taskPacket,
                             goalLocations[localTask], &agentTaskAssignments,
                             &agentTaskPaths, &precedenceConstraints);
        AgentTaskPath path = runLowLevelSearch(
            *localPlanner, constraintTable, startTime, localTask, 0);
        // We must be able to find the path for the next task. If not then we cannot move forward!
        if (path.empty()) {
          PLOGE << "computeRegretForTask: empty path for agent " << agent
                << ", task " << agentTaskAssignments[agent][localTask]
                << " at position " << localTask << "\n";
          return false;
        }
        assert(!path.empty());
        agentTaskPaths[agent][localTask] = path;
      }
    }
    precedenceConstraints.insert(precedenceConstraints.end(),
                                 agentPrecedenceConstraints[agent].begin(),
                                 agentPrecedenceConstraints[agent].end());
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0; localTask < (int)agentTaskAssignments[agent].size();
         localTask++) {
      if (localTask > 0) {
        assert(agentTaskPaths[agent][localTask - 1].path.back().location ==
               agentTaskPaths[agent][localTask].path.front().location);
      } else {
        assert(agentTaskPaths[agent][localTask].front().location ==
               instance_.getStartLocationsRef()[agent]);
      }
    }
  }

  ancestors.clear();
  ancestors.resize(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
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
    if (ancestorTaskAgent < 0 || ancestorTaskAgent >= instance_.getAgentNum()) {
      PLOGE << "Invalid ancestor agent " << ancestorTaskAgent
            << " for task " << ancestorTask << "\n";
      return false;
    }
    assert(ancestorTaskAgent != UNASSIGNED);
    auto ancestorTaskIt =
        find(agentTaskAssignments[ancestorTaskAgent].begin(),
             agentTaskAssignments[ancestorTaskAgent].end(), ancestorTask);
    if (ancestorTaskIt == agentTaskAssignments[ancestorTaskAgent].end()) {
      PLOGE << "Ancestor task " << ancestorTask
            << " missing from temporary assignment for agent "
            << ancestorTaskAgent << "\n";
      return false;
    }
    int ancestorTaskPosition =
        (int)distance(agentTaskAssignments[ancestorTaskAgent].begin(),
                      ancestorTaskIt);
    if (ancestorTaskPosition >=
            (int)agentTaskPaths[ancestorTaskAgent].size() ||
        agentTaskPaths[ancestorTaskAgent][ancestorTaskPosition].empty()) {
      PLOGE << "Ancestor path missing for task " << ancestorTask
            << " at position " << ancestorTaskPosition << "\n";
      return false;
    }
    earliestTimestep =
        max(earliestTimestep, agentTaskPaths[ancestorTaskAgent][ancestorTaskPosition]
                                   .endTimeChecked() +
                                   1);
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {

    TaskRegretPacket regretPacket = {task, agent, -1, earliestTimestep};
    computeRegretForTaskWithAgent(regretPacket, &agentTaskAssignments,
                                  &agentTaskPaths, &precedenceConstraints,
                                  &serviceTimes);
  }

  if ((int)serviceTimes.size() < 2) {
    // This is the case when we run out of heap i.e there are not enough options left to compute the regret for this task!
    PLOGD << "Ran out of service time options for task " << task
          << " inside the regular compute regret function\n";
    return false;
  }
  Utility bestUtility = serviceTimes.top();
  serviceTimes.pop();
  Utility secondBestUtility = serviceTimes.top();

  regretBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
  regretSecondBestOption_[task] = {secondBestUtility.agent,
                                   secondBestUtility.taskPosition};

  double value = 0;
  if (regretType == "absolute") {
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

void LNS::computeRegretForTaskWithAgent(
    TaskRegretPacket regretPacket, vector<vector<int>>* agentTaskAssignments,
    vector<vector<AgentTaskPath>>* agentTaskPaths,
    vector<pair<int, int>>* precedenceConstraints,
    pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes) {

  regretEvalStatsCurrent_.agentEvaluations++;
  regretEvalStatsTotal_.agentEvaluations++;

  // Compute the first position along the agent's task assignments where we can insert this task
  int firstValidPosition = 0;
  for (int j = (int)(*agentTaskAssignments)[regretPacket.agent].size() - 1;
       j >= 0; j--) {
    int beginTime = (*agentTaskPaths)[regretPacket.agent][j].beginTime,
        endTime = (*agentTaskPaths)[regretPacket.agent][j].endTime();
    if ((regretPacket.earliestTimestep > endTime) ||
        (regretPacket.earliestTimestep <= endTime &&
         regretPacket.earliestTimestep >= beginTime)) {
      firstValidPosition = j + 1;
      break;
    }
  }

  vector<vector<AgentTaskPath>> temporaryAgentTaskPaths;
  vector<vector<int>> temporaryAgentTaskAssignments;
  vector<pair<int, int>> temporaryPrecedenceConstraints;
  temporaryAgentTaskPaths.reserve(agentTaskPaths->size());
  temporaryAgentTaskAssignments.reserve(agentTaskAssignments->size());
  temporaryPrecedenceConstraints.reserve(precedenceConstraints->size());

  for (int j = firstValidPosition;
       j <= (int)(*agentTaskAssignments)[regretPacket.agent].size(); j++) {

    regretEvalStatsCurrent_.candidateInsertionsTried++;
    regretEvalStatsTotal_.candidateInsertionsTried++;

    const auto originalConflictIt =
        lnsNeighborhood_.removedTasks.find(regretPacket.task);
    if (originalConflictIt != end(lnsNeighborhood_.removedTasks) &&
        originalConflictIt->second.agent == regretPacket.agent &&
        originalConflictIt->second.taskPosition == j) {
      // We dont want to compute regret for the same agent, task positions that led to the original conflict!
      continue;
    }
    regretPacket.taskPosition = j;
    temporaryAgentTaskPaths = *agentTaskPaths;
    temporaryAgentTaskAssignments = *agentTaskAssignments;
    temporaryPrecedenceConstraints = *precedenceConstraints;
    std::variant<bool, Utility> insertCulmination = insertTask(
        regretPacket, &temporaryAgentTaskPaths, &temporaryAgentTaskAssignments,
        &temporaryPrecedenceConstraints);
    if (std::holds_alternative<Utility>(insertCulmination)) {
      regretEvalStatsCurrent_.candidateInsertionsFeasible++;
      regretEvalStatsTotal_.candidateInsertionsFeasible++;
      serviceTimes->push(std::get<Utility>(insertCulmination));
    }
  }
}

// Need the task paths, assignments and precedence constraints as pointers so that we can reuse this code when commiting as we can make in-place changes to these data-structures
std::variant<bool, Utility> LNS::insertTask(
    TaskRegretPacket regretPacket,
    vector<vector<AgentTaskPath>>* agentTaskPaths,
    vector<vector<int>>* agentTaskAssignments,
    vector<pair<int, int>>* precedenceConstraints) {

  double pathSizeChange = 0;
  int startTime = 0, previousTask = UNDEFINED, nextTask = UNDEFINED;

  // The task paths are all the task paths when we dont commit but if we commit they will be agent specific task paths
  vector<vector<AgentTaskPath>>& agentTaskPathsRef = *agentTaskPaths;
  vector<vector<int>>& agentTaskAssignmentsRef = *agentTaskAssignments;
  vector<pair<int, int>>& precedenceConstraintsRef = *precedenceConstraints;

  int agentTasksSize = (int)agentTaskAssignmentsRef[regretPacket.agent].size();
  double value = std::numeric_limits<double>::infinity();

  // In this case we are inserting a task not at the last position
  if (regretPacket.taskPosition < agentTasksSize) {

    nextTask =
        agentTaskAssignmentsRef[regretPacket.agent][regretPacket.taskPosition];
    pathSizeChange =
        (double)agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition]
            .size();

    agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition] =
        AgentTaskPath();

    agentTaskAssignmentsRef[regretPacket.agent].insert(
        agentTaskAssignmentsRef[regretPacket.agent].begin() +
            regretPacket.taskPosition,
        regretPacket.task);
    agentTaskPathsRef[regretPacket.agent].insert(
        agentTaskPathsRef[regretPacket.agent].begin() +
            regretPacket.taskPosition,
        AgentTaskPath());

    // Invalidate the path of the next task
    // Compute the path size of the next task before you remove it!
    precedenceConstraintsRef.emplace_back(regretPacket.task, nextTask);

    // If we are NOT inserting at the start position then we need to take care of the previous task as well
    if (regretPacket.taskPosition != 0) {
      previousTask = agentTaskAssignmentsRef[regretPacket.agent]
                                            [regretPacket.taskPosition - 1];
      // TODO: Technically the task can start being processed before the previous task ends. This is more conservative but need to check if there are better ways to tackle this.
      startTime =
          agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition - 1]
              .endTime();
      precedenceConstraintsRef.erase(
          std::remove_if(
              precedenceConstraintsRef.begin(), precedenceConstraintsRef.end(),
              [previousTask, nextTask](pair<int, int> x) {
                return x.first == previousTask && x.second == nextTask;
              }),
          precedenceConstraintsRef.end());
      precedenceConstraintsRef.emplace_back(previousTask, regretPacket.task);
    }
  }
  // In this case we are inserting at the very end
  else if (regretPacket.taskPosition == agentTasksSize && agentTasksSize != 0) {

    previousTask = agentTaskAssignmentsRef[regretPacket.agent]
                                          [regretPacket.taskPosition - 1];
    startTime =
        agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition - 1]
            .endTime();

    agentTaskAssignmentsRef[regretPacket.agent].push_back(regretPacket.task);
    precedenceConstraintsRef.emplace_back(previousTask, regretPacket.task);
    agentTaskPathsRef[regretPacket.agent].emplace_back();
  } else if (agentTasksSize == 0) {
    // This is the rare-case when the agent has no tasks assigned to it.
    assert(regretPacket.taskPosition == 0);

    startTime = 0;
    agentTaskAssignmentsRef[regretPacket.agent].push_back(regretPacket.task);
    agentTaskPathsRef[regretPacket.agent].emplace_back();
  }

  if (nextTask >= 0) {
    // The task paths reference does not have ancestor information about next task, so we need to add those in

    vector<vector<int>> ancestors(instance_.getTasksNum());
    for (pair<int, int> precConstraint : precedenceConstraintsRef) {
      ancestors[precConstraint.second].push_back(precConstraint.first);
    }
    vector<char> ancestorsOfNextTask = reachableSet(nextTask, ancestors);
    if (nextTask >= 0 && nextTask < (int)ancestorsOfNextTask.size()) {
      ancestorsOfNextTask[nextTask] = 0;
    }
    vector<char> taskPresent(instance_.getTasksNum(), 0);
    for (const auto& assignments : agentTaskAssignmentsRef) {
      for (int assignedTask : assignments) {
        if (assignedTask >= 0 && assignedTask < instance_.getTasksNum()) {
          taskPresent[assignedTask] = 1;
        }
      }
    }

    for (int nextTaskAncestor = 0;
         nextTaskAncestor < (int)ancestorsOfNextTask.size();
         nextTaskAncestor++) {
      if (!ancestorsOfNextTask[nextTaskAncestor]) {
        continue;
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
        if (nextTaskAncestor < 0 || nextTaskAncestor >= instance_.getTasksNum() ||
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
                  agentTaskAssignmentsRef[nextTaskAncestorAgent]);
          if (ancestorTaskLocalIndexRelativeToSolution == UNASSIGNED) {
            PLOGE << "insertTask: failed to map ancestor task "
                  << nextTaskAncestor
                  << " into current assignment order for agent "
                  << nextTaskAncestorAgent << "\n";
            return false;
          }
          agentTaskAssignmentsRef[nextTaskAncestorAgent].insert(
              agentTaskAssignmentsRef[nextTaskAncestorAgent].begin() +
                  ancestorTaskLocalIndexRelativeToSolution,
              nextTaskAncestor);
          agentTaskPathsRef[nextTaskAncestorAgent].insert(
              agentTaskPathsRef[nextTaskAncestorAgent].begin() +
                  ancestorTaskLocalIndexRelativeToSolution,
              previousSolution_.agents[nextTaskAncestorAgent]
                  .taskPaths[ancestorTaskLocalIndex]);
          if (nextTaskAncestor >= 0 &&
              nextTaskAncestor < instance_.getTasksNum()) {
            taskPresent[nextTaskAncestor] = 1;
          }
        }
      }
    }

    const auto& inputPrecedenceConstraints =
        instance_.getInputPrecedenceConstraintsRef();
    precedenceConstraintsRef.assign(inputPrecedenceConstraints.begin(),
                                    inputPrecedenceConstraints.end());
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      for (int localTask = 0;
           localTask < (int)agentTaskAssignmentsRef[agent].size();
           localTask++) {

        if (localTask > 0) {
          precedenceConstraintsRef.emplace_back(
              agentTaskAssignmentsRef[agent][localTask - 1],
              agentTaskAssignmentsRef[agent][localTask]);
          agentTaskPathsRef[agent][localTask].beginTime =
              agentTaskPathsRef[agent][localTask - 1].endTime();
        } else {
          agentTaskPathsRef[agent][localTask].beginTime = 0;
        }
        const bool touchesRegretTask =
            (regretPacket.task == agentTaskAssignmentsRef[agent][localTask]) ||
            (localTask > 0 &&
             regretPacket.task == agentTaskAssignmentsRef[agent][localTask - 1]);
        const bool touchesNextTask =
            (nextTask == agentTaskAssignmentsRef[agent][localTask]) ||
            (localTask > 0 &&
             nextTask == agentTaskAssignmentsRef[agent][localTask - 1]);
        if (touchesRegretTask || touchesNextTask) {
          continue;
        }

        if ((localTask == 0 &&
             agentTaskPathsRef[agent][localTask].front().location !=
                 instance_.getStartLocationsRef()[agent]) ||
            (localTask > 0 &&
             agentTaskPathsRef[agent][localTask - 1].path.back().location !=
                 agentTaskPathsRef[agent][localTask].path.front().location)) {

          int startTime = 0;
          if (localTask > 0) {
            startTime = agentTaskPathsRef[agent][localTask - 1].endTime();
          }
          vector<int> goalLocations =
              instance_.getTaskLocations(agentTaskAssignmentsRef[agent]);
          auto localPlanner = createLocalPlanner(agent);
          localPlanner->setGoalLocations(goalLocations);

          ConstraintTable constraintTable(instance_.numOfCols,
                                          instance_.mapSize);
          TaskRegretPacket taskPacket = {
              agentTaskAssignmentsRef[agent][localTask], agent, localTask, -1};
          buildConstraintTable(constraintTable, taskPacket,
                               goalLocations[localTask],
                               &agentTaskAssignmentsRef, &agentTaskPathsRef,
                               &precedenceConstraintsRef);
          AgentTaskPath path = runLowLevelSearch(
              *localPlanner, constraintTable, startTime, localTask, 0);
          // We must be able to find the path for the next task. If not then we cannot move forward!
          if (path.empty()) {
            PLOGE << "insertTask: empty path for agent " << agent
                  << ", task " << agentTaskAssignmentsRef[agent][localTask]
                  << " at position " << localTask << "\n";
            return false;
          }
          assert(!path.empty());
          agentTaskPathsRef[agent][localTask] = path;
        }
      }
    }

    vector<int> planningOrder;
    bool result =
        topologicalSort(&instance_, precedenceConstraintsRef, planningOrder);
    if (!result) {
      return false;
    }

    const auto taskIt =
        find(agentTaskAssignmentsRef[regretPacket.agent].begin(),
             agentTaskAssignmentsRef[regretPacket.agent].end(),
             regretPacket.task);
    if (taskIt == agentTaskAssignmentsRef[regretPacket.agent].end()) {
      PLOGE << "insertTask: regret task " << regretPacket.task
            << " not found in agent " << regretPacket.agent << " queue\n";
      return false;
    }
    const int taskPosition =
        (int)distance(agentTaskAssignmentsRef[regretPacket.agent].begin(), taskIt);
    vector<int> goalLocations =
        instance_.getTaskLocations(agentTaskAssignmentsRef[regretPacket.agent]);
    if (taskPosition < 0 || taskPosition >= (int)goalLocations.size()) {
      PLOGE << "insertTask: invalid task position " << taskPosition
            << " for agent " << regretPacket.agent << " goal list size "
            << goalLocations.size() << "\n";
      return false;
    }
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    auto localPlanner = createLocalPlanner(regretPacket.agent);
    localPlanner->setGoalLocations(goalLocations);

    buildConstraintTable(constraintTable, regretPacket,
                         goalLocations[taskPosition], &agentTaskAssignmentsRef,
                         &agentTaskPathsRef, &precedenceConstraintsRef);
    AgentTaskPath path = runLowLevelSearch(
        *localPlanner, constraintTable, startTime, taskPosition, 0);
    if (path.empty()) {
      return false;
    }
    agentTaskPathsRef[regretPacket.agent][taskPosition] = path;
    value = path.size();
    startTime = agentTaskPathsRef[regretPacket.agent][taskPosition].endTime();

    // Need to recompute the positions as we might add paths for parent tasks before reaching here!
    const auto nextTaskIt =
        find(agentTaskAssignmentsRef[regretPacket.agent].begin(),
             agentTaskAssignmentsRef[regretPacket.agent].end(), nextTask);
    if (nextTaskIt == agentTaskAssignmentsRef[regretPacket.agent].end()) {
      PLOGE << "insertTask: next task " << nextTask << " not found in agent "
            << regretPacket.agent << " queue\n";
      return false;
    }
    const int nextTaskPosition =
        (int)distance(agentTaskAssignmentsRef[regretPacket.agent].begin(),
                      nextTaskIt);
    if (nextTaskPosition < 0 || nextTaskPosition >= (int)goalLocations.size()) {
      PLOGE << "insertTask: invalid next-task position " << nextTaskPosition
            << " for agent " << regretPacket.agent << " goal list size "
            << goalLocations.size() << "\n";
      return false;
    }
    TaskRegretPacket nextTaskPacket = {
        nextTask, regretPacket.agent, nextTaskPosition, {}};
    buildConstraintTable(constraintTable, nextTaskPacket,
                         goalLocations[nextTaskPosition],
                         &agentTaskAssignmentsRef, &agentTaskPathsRef,
                         &precedenceConstraintsRef, true);
    AgentTaskPath nextPath = runLowLevelSearch(
        *localPlanner, constraintTable, startTime, nextTaskPosition, 0);
    if (nextPath.empty()) {
      return false;
    }
    agentTaskPathsRef[regretPacket.agent][nextTaskPosition] = nextPath;
    value += nextPath.size();
  } else {

    vector<int> planningOrder;
    bool result =
        topologicalSort(&instance_, precedenceConstraintsRef, planningOrder);
    if (!result) {
      return false;
    }

    vector<int> goalLocations =
        instance_.getTaskLocations(agentTaskAssignmentsRef[regretPacket.agent]);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    auto localPlanner = createLocalPlanner(regretPacket.agent);
    localPlanner->setGoalLocations(goalLocations);

    buildConstraintTable(constraintTable, regretPacket,
                         goalLocations[regretPacket.taskPosition],
                         &agentTaskAssignmentsRef, &agentTaskPathsRef,
                         &precedenceConstraintsRef);
    AgentTaskPath path = runLowLevelSearch(
        *localPlanner, constraintTable, startTime, regretPacket.taskPosition, 0);
    if (path.empty()) {
      return false;
    }
    agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition] = path;
    value = path.size();
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
    const double oldExposure = computeTaskMarketExposure(task, true);
    const int oldWait = computeTaskPrecedenceWaitInCurrentSolution(task);

    const auto itTaskPos = find(agentTaskAssignmentsRef[regretPacket.agent].begin(),
                                agentTaskAssignmentsRef[regretPacket.agent].end(),
                                task);
    if (itTaskPos != agentTaskAssignmentsRef[regretPacket.agent].end()) {
      const int currentTaskPos =
          (int)(itTaskPos - agentTaskAssignmentsRef[regretPacket.agent].begin());
      if (currentTaskPos >= 0 &&
          currentTaskPos < (int)agentTaskPathsRef[regretPacket.agent].size()) {
        const AgentTaskPath& insertedTaskPath =
            agentTaskPathsRef[regretPacket.agent][currentTaskPos];
        const double newExposure =
            computeMarketExposureFromPath(insertedTaskPath, true);
        const int newWait = computeTaskPrecedenceWaitFromState(
            task, taskLocation, agentTaskAssignmentsRef, agentTaskPathsRef,
            precedenceConstraintsRef);
        deltaExposure = newExposure - oldExposure;
        deltaWait = (double)newWait - (double)oldWait;
      }
    }

    const bool applyBlend = market_.repairBlend;
    const bool applyTieBreak =
        market_.repairTieBreak && std::abs(baseDeltaSoc) <= market_.tieBreakEpsSoc;
    if (applyBlend || applyTieBreak) {
      adjustedValue +=
          market_.lambdaPrice * deltaExposure + market_.lambdaWait * deltaWait;
    }
  }

  Utility utility(regretPacket.agent, regretPacket.taskPosition, (int)pathLength,
                  (int)agentTaskAssignmentsRef[regretPacket.agent].size(),
                  adjustedValue, baseDeltaSoc, deltaExposure, deltaWait);
  return utility;
}

bool LNS::commitAncestorTaskOf(
    int globalTask, std::optional<pair<bool, int>> committingNextTask) {
  // We are going to commit some ancestor of this global task. We need to ensure that the paths of all the required ancestors of this task are in order before we can commit the global task and any next task that may exist
  // If the boolean flag commitingNextTask is set then it means that the global task was the next task of some other task and we need to ensure that the ancestors of this next task are in order. This additional check is required as the first if condition changes depending on it.
  // The corresponding integer entry would be the global task id of the main task that we wanted to commit.

  vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints();

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
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
      int ancestorTaskPosition =
          previousSolution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
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

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0;
         localTask < (int)solution_.agents[agent].taskAssignments.size();
         localTask++) {

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
        buildConstraintTable(
            constraintTable,
            solution_.agents[agent].taskAssignments[localTask]);
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

    // Need to recompute this task position as we may have added tasks in the agent's task queue when trying to account for the next task parents
    int taskPosition =
        find(solution_.agents[bestRegretPacket.agent].taskAssignments.begin(),
             solution_.agents[bestRegretPacket.agent].taskAssignments.end(),
             bestRegretPacket.task) -
        solution_.agents[bestRegretPacket.agent].taskAssignments.begin();

    buildConstraintTable(constraintTable, bestRegretPacket.task);
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

    buildConstraintTable(constraintTable, nextTask);
    int nextTaskPosition =
        find(solution_.agents[bestRegretPacket.agent].taskAssignments.begin(),
             solution_.agents[bestRegretPacket.agent].taskAssignments.end(),
             nextTask) -
        solution_.agents[bestRegretPacket.agent].taskAssignments.begin();
    assert(nextTaskPosition - 1 >= 0);
    startTime = solution_.agents[bestRegretPacket.agent]
                    .taskPaths[nextTaskPosition - 1]
                    .endTime();
    AgentTaskPath nextPath = runLowLevelSearch(
        *solution_.agents[bestRegretPacket.agent].pathPlanner, constraintTable,
        startTime, nextTaskPosition, 0);
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

    buildConstraintTable(constraintTable, bestRegretPacket.task);
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

void LNS::buildConstraintTable(ConstraintTable& constraintTable,
                               TaskRegretPacket taskPacket, int taskLocation,
                               vector<vector<int>>* agentTaskAssignments,
                               vector<vector<AgentTaskPath>>* agentTaskPaths,
                               vector<pair<int, int>>* precedenceConstraints,
                               bool findingNextTask) {

  vector<vector<AgentTaskPath>>& agentTaskPathsRef = *agentTaskPaths;
  vector<vector<int>>& agentTaskAssignmentsRef = *agentTaskAssignments;
  vector<pair<int, int>>& precedenceConstraintsRef = *precedenceConstraints;
  const AssignmentLookup assignmentLookup =
      buildAssignmentLookup(agentTaskAssignmentsRef, instance_.getTasksNum());

  constraintTable.goalLocation = taskLocation;

  vector<vector<int>> ancestors(instance_.getAncestorsRef());
  // TODO: We used input precedence constraints here but to me it seems like the input precedence constraints should be augmented by the precedence constraints of the agent we are considering here as well!
  for (const pair<int, int>& precedenceConstraint : precedenceConstraintsRef) {
    ancestors[precedenceConstraint.second].push_back(
        precedenceConstraint.first);
  }

  vector<char> ancestorsOfTask = reachableSet(taskPacket.task, ancestors);
  if (taskPacket.task >= 0 && taskPacket.task < (int)ancestorsOfTask.size()) {
    ancestorsOfTask[taskPacket.task] = 0;
  }

  // Loop through the last task map to gather the actual final tasks of the agents
  vector<bool> finalTasks(instance_.getTasksNum(), false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if ((int)agentTaskAssignmentsRef[agent].size() > 0) {
      int lastTask = agentTaskAssignmentsRef[agent].back();
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
        taskPacket.agent < (int)agentTaskAssignmentsRef.size() &&
        taskPacket.taskPosition > 0 &&
        taskPacket.taskPosition <
            (int)agentTaskAssignmentsRef[taskPacket.agent].size()) {
      previousTaskInAgent =
          agentTaskAssignmentsRef[taskPacket.agent][taskPacket.taskPosition - 1];
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
        return;
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
        return;
      }
      ancestorTaskAgent = curAssignedAgent;
    }

    const int ancestorTaskLocalIndex =
        (ancestorTask >= 0 && ancestorTask < instance_.getTasksNum())
            ? assignmentLookup.pos[ancestorTask]
            : -1;
    if (ancestorTaskLocalIndex < 0 ||
        ancestorTaskLocalIndex >=
            (int)agentTaskPathsRef[ancestorTaskAgent].size() ||
        assignmentLookup.owner[ancestorTask] != ancestorTaskAgent) {
      PLOGE << "buildConstraintTable: could not locate ancestor task "
            << ancestorTask << " for agent " << ancestorTaskAgent << "\n";
      return;
    }
    assert(
        !agentTaskPathsRef[ancestorTaskAgent][ancestorTaskLocalIndex].empty());
    bool waitAtGoal = finalTasks[ancestorTask];
    constraintTable.addPath(
        agentTaskPathsRef[ancestorTaskAgent][ancestorTaskLocalIndex],
        waitAtGoal);

    constraintTable.lengthMin = max(
        constraintTable.lengthMin,
        agentTaskPathsRef[ancestorTaskAgent][ancestorTaskLocalIndex].endTime() +
            1);
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
}

void LNS::buildConstraintTable(ConstraintTable& constraintTable, int task) {

  constraintTable.goalLocation = instance_.getTaskLocations(task);

  vector<pair<int, int>> precedenceConstraints =
      buildFullPrecedenceConstraints(iterationStats.size() > 1);

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
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
      return;
    }
    const int ancestorTaskPosition =
        solution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
    if (ancestorTaskPosition == UNASSIGNED ||
        ancestorTaskPosition >=
            (int)solution_.agents[ancestorTaskAgent].taskPaths.size()) {
      PLOGE << "buildConstraintTable: invalid local index for ancestor task "
            << ancestorTask << " on agent " << ancestorTaskAgent << "\n";
      return;
    }
    const auto& pathRef =
        solution_.agents[ancestorTaskAgent].taskPaths[ancestorTaskPosition];
    if (pathRef.empty()) {
      PLOGE << "Missing path for ancestor task " << ancestorTask
            << " in current solution\n";
      return;
    }
    const bool waitAtGoal =
        ancestorTask ==
        (int)solution_.agents[ancestorTaskAgent].taskAssignments.back();
    constraintTable.addPath(pathRef, waitAtGoal);
    constraintTable.lengthMin =
        max(constraintTable.lengthMin, pathRef.endTime() + 1);
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
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
