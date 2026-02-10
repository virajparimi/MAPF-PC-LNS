#include "utils.hpp"
#include <climits>
#include <cmath>

bool greedyTaskAssignment(const Instance* instance, Solution* solution) {
  if (instance == nullptr || solution == nullptr) {
    PLOGE << "greedyTaskAssignment: received null instance/solution\n";
    return false;
  }

  const int numAgents = instance->getAgentNum();
  const int numTasks = instance->getTasksNum();
  if (numAgents <= 0 || numTasks < 0) {
    PLOGE << "greedyTaskAssignment: invalid problem dimensions (agents="
          << numAgents << ", tasks=" << numTasks << ")\n";
    return false;
  }
  const auto& ancestors = instance->getAncestorsRef();
  const auto& successors = instance->getSuccessorsRef();

  ppqg q;
  vector<int> agentLastTimesteps(numAgents, 0);
  vector<int> agentLastLocations(instance->getStartLocationsRef());
  vector<int> taskCompleteTimesteps(numTasks, -1);
  vector<int> remainingPredecessors(numTasks, 0);
  vector<int> releaseTime(numTasks, 0);
  vector<int> readyTasks;
  vector<int> readyTaskIndex(numTasks, -1);
  readyTasks.reserve(numTasks);
  // Use instance-level precomputed task-centric heuristics directly.
  // These distances are independent of agent identity in this model.
  const auto& heuristics = instance->getHeuristicsRef();

  auto pushReadyTask = [&](int task) {
    if (task < 0 || task >= numTasks || readyTaskIndex[task] != -1 ||
        taskCompleteTimesteps[task] != -1) {
      return;
    }
    readyTaskIndex[task] = (int)readyTasks.size();
    readyTasks.push_back(task);
  };

  auto removeReadyTask = [&](int task) {
    if (task < 0 || task >= numTasks) {
      return;
    }
    const int index = readyTaskIndex[task];
    if (index == -1) {
      return;
    }
    const int lastTask = readyTasks.back();
    readyTasks[index] = lastTask;
    readyTaskIndex[lastTask] = index;
    readyTasks.pop_back();
    readyTaskIndex[task] = -1;
  };

  for (int task = 0; task < numTasks; task++) {
    remainingPredecessors[task] = (int)ancestors[task].size();
    if (remainingPredecessors[task] == 0) {
      pushReadyTask(task);
    }
  }

  for (int agent = 0; agent < numAgents; agent++) {
    q.emplace(0, agent);  // (key, value) - (timestep, agent)
  }

  int taskCounter = 0;
  while (taskCounter < numTasks) {
    int timestep, agent;
    tie(timestep, agent) = q.top();

    PLOGD << "Planning for agent " << agent << " at timestep " << timestep
          << "\n";
    int lastLocationOfAgent = agentLastLocations[agent];
    q.pop();

    if (readyTasks.empty()) {
      PLOGE << "greedyTaskAssignment: no ready tasks remain (cycle or invalid "
               "dependencies?)\n";
      return false;
    }

    int bestTaskToService = -1, bestTaskToServiceTimestep = INT_MAX;
    for (int task : readyTasks) {
      // Earliest completion for this task by this agent under current state.
      const int earliestArrival =
          agentLastTimesteps[agent] + heuristics[task][lastLocationOfAgent];
      const int taskTimestep = max(earliestArrival, releaseTime[task]);
      if (taskTimestep < bestTaskToServiceTimestep) {
        bestTaskToService = task;
        bestTaskToServiceTimestep = taskTimestep;
      }
    }

    if (bestTaskToService == -1) {
      PLOGE << "greedyTaskAssignment: no feasible task found (cycle or invalid "
               "dependencies?)\n";
      return false;
    }

    // Assign the best task found to the agent
    PLOGD << "Assign task " << bestTaskToService << " to agent " << agent
          << " with distance "
          << heuristics[bestTaskToService][lastLocationOfAgent]
          << "\n";
    solution->assignTaskToAgent(agent, bestTaskToService);
    agentLastTimesteps[agent] = bestTaskToServiceTimestep;
    taskCompleteTimesteps[bestTaskToService] = bestTaskToServiceTimestep;
    agentLastLocations[agent] = instance->getTaskLocations(bestTaskToService);
    removeReadyTask(bestTaskToService);

    for (int successorTask : successors[bestTaskToService]) {
      if (successorTask < 0 || successorTask >= numTasks ||
          taskCompleteTimesteps[successorTask] != -1) {
        continue;
      }
      releaseTime[successorTask] =
          max(releaseTime[successorTask], bestTaskToServiceTimestep);
      if (remainingPredecessors[successorTask] <= 0) {
        PLOGE << "greedyTaskAssignment: invalid predecessor bookkeeping for "
              << successorTask << "\n";
        return false;
      }
      remainingPredecessors[successorTask]--;
      if (remainingPredecessors[successorTask] == 0) {
        pushReadyTask(successorTask);
      }
    }
    taskCounter++;

    q.emplace(agentLastTimesteps[agent], agent);
  }
  return true;
}

bool topologicalSort(const Instance* instance,
                     const vector<pair<int, int>>& precedenceConstraints,
                     vector<int>& planningOrder) {
  if (instance == nullptr) {
    PLOGE << "topologicalSort: received null instance\n";
    return false;
  }
  assert(instance != nullptr);
  const int numTasks = instance->getTasksNum();

  planningOrder.clear();
  planningOrder.reserve(numTasks);
  vector<bool> closed(numTasks, false);
  vector<bool> expanded(numTasks, false);

  vector<vector<int>> successors;
  successors.resize(numTasks);
  for (const auto& precedenceConstraint : precedenceConstraints) {
    if (precedenceConstraint.first < 0 || precedenceConstraint.first >= numTasks ||
        precedenceConstraint.second < 0 ||
        precedenceConstraint.second >= numTasks) {
      PLOGE << "topologicalSort: constraint index out of bounds ("
            << precedenceConstraint.first << " -> "
            << precedenceConstraint.second << "), numTasks=" << numTasks
            << "\n";
      return false;
    }
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

  if ((int)planningOrder.size() != numTasks) {
    PLOGE << "Topological sort produced an incomplete order (expected "
          << numTasks << ", got " << planningOrder.size() << ")\n";
    return false;
  }

  vector<bool> tasksOrder(numTasks, false);
  for (int task : planningOrder) {
    if (task < 0 || task >= numTasks) {
      PLOGE << "Topological sort produced out-of-range task id " << task
            << "\n";
      return false;
    }
    if (tasksOrder[task]) {
      PLOGE << "Topological sort produced duplicate task id " << task << "\n";
      return false;
    }
    for (int dependentTask : successors[task]) {
      if (tasksOrder[dependentTask]) {
        PLOGE << "The topological sort violated a precedence constraint\n";
        return false;
      }
    }
    tasksOrder[task] = true;
  }

  if (std::find(tasksOrder.begin(), tasksOrder.end(), false) !=
      tasksOrder.end()) {
    PLOGE << "Topological sort order does not cover all tasks\n";
    return false;
  }

  assert((int)planningOrder.size() == numTasks);
  return true;
}

std::ostream& operator<<(std::ostream& os, const Path& path) {
  os << "[begin=" << path.beginTime << ", size=" << path.size() << ", path=";
  for (int i = 0; i < (int)path.size(); i++) {
    os << path.path[i].location;
    if (path.path[i].isGoal) {
      os << "*";
    }
    if (i + 1 < (int)path.size()) {
      os << "->";
    }
  }
  os << "]";
  return os;
}

ConflictMap extractNConflicts(int size, const ConflictMap& conflicts) {
  ConflictMap result;
  if (size <= 0 || conflicts.empty()) {
    return result;
  }

  const size_t count =
      std::min<size_t>(static_cast<size_t>(size), conflicts.size());
  auto it = conflicts.begin();
  for (size_t i = 0; i < count && it != conflicts.end(); ++i, ++it) {
    result.emplace(it->first, it->second);
  }
  return result;
}

double MovingMetrics::computeMovingMetrics(int numberOfConflicts,
                                           int sumOfCosts) {
  if (size <= 0 || conflictNum.empty() || costNum.empty()) {
    PLOGE << "MovingMetrics: invalid internal window state (size=" << size
          << "). Returning neutral utility.\n";
    return 0.0;
  }

  // Compute the utility of this solution
  // Extract the oldest sample
  double oldestNumConflicts = conflictNum[oldestValue],
         oldestNumCost = costNum[oldestValue];

  // Update the oldest sample
  conflictNum[oldestValue] = numberOfConflicts;
  costNum[oldestValue] = sumOfCosts;

  // Update the oldest sample location
  oldestValue++;
  oldestValue %= size;

  // Compute the new sum based on the adding the new sample and removing the oldest sample
  sumOfNumConflicts += numberOfConflicts - oldestNumConflicts;
  sumOfNumCosts += sumOfCosts - oldestNumCost;

  // Compute the moving average and variance of the number of conflicts and sum of costs variables
  const double windowSize = (double)size;
  const double movingNumConflictAverage = sumOfNumConflicts / windowSize;
  const double movingNumCostAverage = sumOfNumCosts / windowSize;

  double movingNumConflictVar = 0.0;
  double movingNumCostVar = 0.0;
  if (size > 1) {
    auto sampleVariance = [](const vector<double>& values, double mean) {
      long double sqDeviationSum = 0.0L;
      for (double value : values) {
        const long double delta = (long double)value - (long double)mean;
        sqDeviationSum += delta * delta;
      }
      return (double)(sqDeviationSum / (long double)(values.size() - 1));
    };
    movingNumConflictVar = sampleVariance(conflictNum, movingNumConflictAverage);
    movingNumCostVar = sampleVariance(costNum, movingNumCostAverage);
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
