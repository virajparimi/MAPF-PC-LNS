#include "utils.hpp"
#include <climits>
#include <cmath>

void greedyTaskAssignment(const Instance* instance, Solution* solution) {
  assert(instance != nullptr);
  assert(solution != nullptr);

  const int numAgents = instance->getAgentNum();
  const int numTasks = instance->getTasksNum();
  const auto taskDependencies = instance->getTaskDependencies();

  ppqg q;
  vector<int> agentLastTimesteps(numAgents, 0);
  vector<int> agentLastLocations = instance->getStartLocations();
  vector<int> taskCompleteTimesteps(numTasks, -1);

  // We first compute the heuristic value for all the tasks irrespective of the agents
  unique_ptr<SingleAgentSolver> searchEngine =
      make_unique<MultiLabelSpaceTimeAStar>((*instance), 0);

  for (int agent = 0; agent < numAgents; agent++) {
    q.emplace(0, agent);  // (key, value) - (timestep, agent)
  }

  int taskCounter = 0;
  while (taskCounter < numTasks) {
    int timestep, agent;
    tie(timestep, agent) = q.top();

    PLOGD << "Planning for agent " << agent << " at timestep " << timestep
          << endl;
    int lastLocationOfAgent = agentLastLocations[agent];
    q.pop();

    int bestTaskToService = -1, bestTaskToServiceTimestep = INT_MAX;
    for (int task = 0; task < numTasks; task++) {
      if (taskCompleteTimesteps[task] != -1) {  // Task has been assigned before
        continue;
      }

      bool taskReady = true;
      // The time this agent can service this task and estimated cost of completing that
      // task from the agent's location
      int taskTimestep =
          agentLastTimesteps[agent] +
          (*searchEngine->heuristic[task])[lastLocationOfAgent];

      // Check for temporal dependencies
      const auto depsIt = taskDependencies.find(task);
      if (depsIt != taskDependencies.end()) {
        for (int dependentTask : depsIt->second) {
          if (taskCompleteTimesteps[dependentTask] < 0) {
            // The dependent tasks need to be completed before this task can be serviced
            taskReady = false;
            break;
          }
          taskTimestep =
              max(taskCompleteTimesteps[dependentTask], taskTimestep);
        }
      }

      if (taskReady && taskTimestep < bestTaskToServiceTimestep) {
        bestTaskToService = task;
        bestTaskToServiceTimestep = taskTimestep;
      }
    }

    if (bestTaskToService == -1) {
      PLOGE << "greedyTaskAssignment: no feasible task found (cycle or invalid "
               "dependencies?)\n";
      assert(false);
      return;
    }

    // Assign the best task found to the agent
    PLOGD << "Assign task " << bestTaskToService << " to agent " << agent
          << " with distance "
          << (*searchEngine->heuristic[bestTaskToService])[lastLocationOfAgent]
          << endl;
    solution->assignTaskToAgent(agent, bestTaskToService);
    agentLastTimesteps[agent] = bestTaskToServiceTimestep;
    taskCompleteTimesteps[bestTaskToService] = bestTaskToServiceTimestep;
    agentLastLocations[agent] = instance->getTaskLocations(bestTaskToService);
    taskCounter++;

    q.emplace(agentLastTimesteps[agent], agent);
  }
}

bool topologicalSort(const Instance* instance,
                     vector<pair<int, int>>* precedenceConstraints,
                     vector<int>& planningOrder) {
  assert(precedenceConstraints != nullptr);
  return topologicalSort(instance, *precedenceConstraints, planningOrder);
}

bool topologicalSort(const Instance* instance,
                     const vector<pair<int, int>>& precedenceConstraints,
                     vector<int>& planningOrder) {
  assert(instance != nullptr);
  const int numTasks = instance->getTasksNum();

  planningOrder.clear();
  planningOrder.reserve(numTasks);
  vector<bool> closed(numTasks, false);
  vector<bool> expanded(numTasks, false);

  vector<vector<int>> successors;
  successors.resize(numTasks);
  for (const auto& precedenceConstraint : precedenceConstraints) {
    assert(precedenceConstraint.first >= 0 &&
           precedenceConstraint.first < numTasks);
    assert(precedenceConstraint.second >= 0 &&
           precedenceConstraint.second < numTasks);
    successors[precedenceConstraint.first].push_back(
        precedenceConstraint.second);
  }

  for (int task = 0; task < numTasks; task++) {
    if (closed[task]) {
      continue;
    }

    stack<int> dfsStack;
    dfsStack.push(task);

    while (!dfsStack.empty()) {

      int currentTask = dfsStack.top();
      dfsStack.pop();
      if (closed[currentTask]) {
        continue;
      }
      if (expanded[currentTask]) {
        closed[currentTask] = true;
        planningOrder.push_back(currentTask);
      } else {
        expanded[currentTask] = true;
        dfsStack.push(currentTask);
        for (int dependentTask : successors[currentTask]) {
          if (closed[dependentTask]) {
            continue;
          }
          if (expanded[dependentTask]) {
            PLOGE << "Detected a cycle while running topological sort\n";
            return false;
          }
          dfsStack.push(dependentTask);
        }
      }
    }
  }

  reverse(planningOrder.begin(), planningOrder.end());

  vector<bool> tasksOrder(numTasks, false);
  for (int task : planningOrder) {
    for (int dependentTask : successors[task]) {
      if (tasksOrder[dependentTask]) {
        PLOGE << "The topological sort violated a precedence constraint\n";
        return false;
      }
    }
    tasksOrder[task] = true;
  }

  assert((int)planningOrder.size() == numTasks);
  return true;
}

bool isSamePath(const Path& p1, const Path& p2) {
  if (p1.size() != p2.size()) {
    return false;
  }
  for (int i = 0; i < (int)p1.size(); i++) {
    if (p1.path[i].location != p2.path[i].location) {
      return false;
    }
  }
  return true;
}

set<Conflicts> extractNConflicts(int size, const set<Conflicts>& conflicts) {
  int i = 0;
  set<Conflicts> result;
  for (const auto& conflict : conflicts) {
    if (i >= size) {
      break;
    }
    result.insert(conflict);
    i++;
  }
  return result;
}

double MovingMetrics::computeMovingMetrics(int numberOfConflicts,
                                           int sumOfCosts) {

  // Compute the utility of this solution
  // Compute the new sample
  const double numConflictsSquare =
      (double)numberOfConflicts * (double)numberOfConflicts;
  const double numCostSquare = (double)sumOfCosts * (double)sumOfCosts;
  // Extract the oldest sample
  double oldestNumConflicts = conflictNum[oldestValue],
         oldestNumConflictsSquare = conflictSquareNum[oldestValue],
         oldestNumCost = costNum[oldestValue],
         oldestNumCostSquare = costSquareNum[oldestValue];

  // Update the oldest sample
  conflictNum[oldestValue] = numberOfConflicts;
  conflictSquareNum[oldestValue] = numConflictsSquare;
  costNum[oldestValue] = sumOfCosts;
  costSquareNum[oldestValue] = numCostSquare;

  // Update the oldest sample location
  oldestValue++;
  oldestValue %= size;

  // Compute the new sum based on the adding the new sample and removing the oldest sample
  sumOfNumConflicts += numberOfConflicts - oldestNumConflicts;
  sumOfNumConflictsSquare += numConflictsSquare - oldestNumConflictsSquare;
  sumOfNumCosts += sumOfCosts - oldestNumCost;
  sumOfNumCostsSquare += numCostSquare - oldestNumCostSquare;

  // Compute the moving average and variance of the number of conflicts and sum of costs variables
  assert(size > 0);
  const double windowSize = (double)size;
  const double movingNumConflictAverage = sumOfNumConflicts / windowSize;
  const double movingNumCostAverage = sumOfNumCosts / windowSize;

  double movingNumConflictVar = 0.0;
  double movingNumCostVar = 0.0;
  if (size > 1) {
    movingNumConflictVar =
        (windowSize * sumOfNumConflictsSquare - (sumOfNumConflicts * sumOfNumConflicts)) /
        (windowSize * (windowSize - 1.0));
    movingNumCostVar =
        (windowSize * sumOfNumCostsSquare - (sumOfNumCosts * sumOfNumCosts)) /
        (windowSize * (windowSize - 1.0));
    movingNumConflictVar = max(0.0, movingNumConflictVar);
    movingNumCostVar = max(0.0, movingNumCostVar);
  }

  PLOGD << "Moving average of conflicts = " << movingNumConflictAverage
        << ", Moving average of costs = " << movingNumCostAverage << "\n";

  double utility =
      lnsConflictWeight * ((numberOfConflicts - movingNumConflictAverage) /
                           std::sqrt(movingNumConflictVar + 1)) +
      lnsCostWeight *
          ((sumOfCosts - movingNumCostAverage) / std::sqrt(movingNumCostVar + 1));
  return utility;
}
