#include "lns.hpp"
#include "utils.hpp"
#include <cmath>
#include <limits>

bool LNS::simulatedAnnealing() {

  bool accepted = false;
  // Guard against degenerate temperatures to avoid NaN/inf behavior.
  if (!std::isfinite(temperature_) ||
      temperature_ <= std::numeric_limits<double>::epsilon()) {
    accepted = solution_.utility <= previousSolution_.utility;
    if (!accepted) {
      solution_ = previousSolution_;
      PLOGD << "Rejecting this solution!\n";
    }
    temperature_ *= coolingCoefficient_;
    return accepted;
  }

  const double acceptanceProb =
      exp((previousSolution_.utility - solution_.utility) / temperature_);
  std::uniform_real_distribution<double> unit01(0.0, 1.0);
  if (unit01(rng_) < acceptanceProb) {
    // Use simulated annealing to potentially accept worse solutions!
    accepted = true;
  } else {
    // Reject this solution
    solution_ = previousSolution_;
    PLOGD << "Rejecting this solution!\n";
  }
  temperature_ *= coolingCoefficient_;
  return accepted;
}

bool LNS::thresholdAcceptance() {

  bool accepted = false;
  // In this case we are worse than the previous solution but within some threshold so we can accept this one
  if (solution_.utility - previousSolution_.utility <= temperature_) {
    accepted = true;
  } else {
    // Reject this solution
    solution_ = previousSolution_;
    PLOGD << "Rejecting this solution!\n";
  }
  temperature_ *= coolingCoefficient_;
  return accepted;
}

bool LNS::oldBachelorsAcceptance() {

  bool accepted = false;
  if (solution_.utility - previousSolution_.utility < temperature_) {
    // Accept this solution and reduce the temperature
    temperature_ *= coolingCoefficient_;
    accepted = true;
  } else {
    // Reject this solution and increase the temperature
    solution_ = previousSolution_;
    temperature_ *= heatingCoefficient_;
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}

bool LNS::greatDelugeAlgorithm() {

  bool accepted = false;
  if (solution_.utility - previousSolution_.utility < temperature_) {
    // This temperature acts as a water level and we want to accept solutions that fall within some water level and corresponding increase it further for future iterations
    // Since we are effectively doing a minimization problem we need to decrease the temperature ONLY if we accept
    temperature_ *= coolingCoefficient_;
    accepted = true;
  } else {
    // Reject this solution but dont change the temperature'
    solution_ = previousSolution_;
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}

bool LNS::run() {

  bool success = false;
  if (initialSolutionStrategy == "greedy") {
    // Run the greedy task assignment and subsequent path finding algorithm
    success = buildGreedySolution();
  } else if (initialSolutionStrategy == "greedy_precedence_only") {
    // Precedence-feasible, collision-infeasible warm start.
    success = buildGreedySolutionPrecedenceOnly();
  } else if (initialSolutionStrategy.find("sota") != string::npos) {
    // Run the greedy task assignment and use CBS-PC for finding the paths of agents
    success = buildGreedySolutionWithMAPFPC(initialSolutionStrategy);
  }

  if (!success && initialSolutionStrategy != "greedy") {
    success = buildGreedySolution();
  }

  // If the initial solution strategy failed then we cannot do anything!
  if (!success) {
    return success;
  }

  initialSolutionRuntime_ = ((fsec)(Time::now() - plannerStartTime_)).count();
  runtime = initialSolutionRuntime_;

  PLOGD << "Initial solution cost = " << solution_.sumOfCosts
        << ", Runtime = " << initialSolutionRuntime_ << "\n";

  ConflictMap potentialNeighborhood;  // Need for the conflict removal case
  bool valid = validateSolution(&potentialNeighborhood);

  bool feasibleSolutionUpdated = false;
  if (valid) {
    feasibleSolutionUpdated = true;
    extractFeasibleSolution();
  }

  if (market_.heuristics) {
    updateMarketStateFromCurrentSolution();
    if (valid) {
      market_.bestPressure = computeSolutionMarketPressure();
      market_.bestWait = computeSolutionPrecedenceWait();
    }
  }

  iterationStats.emplace_back(initialSolutionRuntime_, initialSolutionStrategy,
                              instance_.getAgentNum(), instance_.getTasksNum(),
                              solution_.sumOfCosts, feasibleSolutionUpdated,
                              bestSolutionYet);

  ConflictMap oldNeighborhood;

  // Needed to maintain a running mean and standard deviation which can then be used to standardize the sum of costs and conflicts in accepting criteria functions
  // Decouple moving-metrics stability from run-loop bounds:
  // when maxIterations is unbounded/unspecified (0), keep a reasonably sized
  // window so utility doesn't collapse to zero and freeze TA acceptance.
  constexpr int kDefaultMetricsWindowSize = 200;
  const int metricsWindowSize =
      (numOfIterations_ > 0) ? max(2, numOfIterations_)
                             : kDefaultMetricsWindowSize;
  MovingMetrics metrics(metricsWindowSize, lnsConflictWeight_, lnsCostWeight_,
                        (int)potentialNeighborhood.size(),
                        solution_.sumOfCosts);
  solution_.utility = metrics.computeMovingMetrics(
      (int)potentialNeighborhood.size(), solution_.sumOfCosts);

  temperature_ = solution_.utility * (tolerance_ / 100);
  if (acceptanceCriteria == "SA") {
    temperature_ /= log(2);
  }

  previousSolution_ = solution_;

  const int64_t iterationLimit =
      (numOfIterations_ > 0)
          ? static_cast<int64_t>(numOfIterations_) * 2
          : std::numeric_limits<int64_t>::max();

  // LNS loop
  while (runtime < timeLimit_ &&
         static_cast<int64_t>(iterationStats.size()) < iterationLimit) {
    const int previousSocForIter = previousSolution_.sumOfCosts;
    int alnsHeuristicForIter = -1;

    // These functions populate the LNS neighborhoods' removedTask parameter
    if (destroyHeuristic == "conflict") {
      conflictRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "worst") {
      worstRemoval();
    } else if (destroyHeuristic == "random") {
      randomRemoval();
    } else if (destroyHeuristic == "shaw") {
      shawRemoval(neighborSize_ * 3);
    } else if (destroyHeuristic == "precedence_wait") {
      precedenceWaitRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "low_slack") {
      lowSlackRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "market_tatonnement") {
      marketTatonnementRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "alns") {
      alnsRemoval(std::make_optional(potentialNeighborhood));
      alnsHeuristicForIter = adaptiveLNS_.recentDestroyHeuristic;
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.selections[alnsHeuristicForIter]++;
      }
    } else {
      PLOGE << "Unknown destroy heuristic: " << destroyHeuristic << "\n";
      return false;
    }

    oldNeighborhood = lnsNeighborhood_.removedTasks;

    PLOGD << "Printing neighborhood conflict tasks\n";
    PLOGD << "Size: " << lnsNeighborhood_.removedTasks.size() << "\n";
    for (const auto& [_, conflictTask] : lnsNeighborhood_.removedTasks) {
      PLOGD << "Conflicted Task : " << conflictTask.task << "\n";
    }

    prepareNextIteration();

    // This needs to happen after prepare iteration since we updated the conflictedTasks variable in the prepare next iteration function
    for (const auto& [_, conflictedTask] : lnsNeighborhood_.removedTasks) {
      if (conflictedTask.task >= 0 &&
          conflictedTask.task < (int)lnsNeighborhood_.committedTasks.size()) {
        lnsNeighborhood_.committedTasks[conflictedTask.task] = 0;
      }
    }
    lnsNeighborhood_.immutableRemovedTasks = lnsNeighborhood_.removedTasks;

    // Repair: commit removed tasks back using regret.
    //
    // Default behavior recomputes regrets from scratch every commit.
    // Optional incremental mode recomputes regrets only for a dirty subset.
    lnsNeighborhood_.regretMaxHeap.clear();
    bool repairFailed = false;
    regretEvalStatsCurrent_.reset();
    {
      const int removedCount = (int)lnsNeighborhood_.removedTasks.size();
      regretEvalStatsCurrent_.neighborhoods++;
      regretEvalStatsTotal_.neighborhoods++;
      regretEvalStatsCurrent_.removedTasksSum += removedCount;
      regretEvalStatsTotal_.removedTasksSum += removedCount;
      regretEvalStatsCurrent_.removedTasksMax =
          max(regretEvalStatsCurrent_.removedTasksMax, (int64_t)removedCount);
      regretEvalStatsTotal_.removedTasksMax =
          max(regretEvalStatsTotal_.removedTasksMax, (int64_t)removedCount);
    }
    if (!incrementalRegret_) {
      // Compute regret for each of the tasks that are in the conflicting set
      // Pick the best one and repeat the whole process again
      while (!lnsNeighborhood_.removedTasks.empty()) {
        bool enoughSpace = computeRegret();
        if (!enoughSpace) {
          // We could not compute enough regrets for each task so we need to try
          // and reset the solution and try potentially with a different
          // neighborhood!
          repairFailed = true;
          break;
        }
        if (lnsNeighborhood_.regretMaxHeap.empty()) {
          PLOGE << "regretMaxHeap is empty after computeRegret\n";
          repairFailed = true;
          break;
        }
        assert(!lnsNeighborhood_.regretMaxHeap.empty());
        Regret bestRegret = lnsNeighborhood_.regretMaxHeap.top();
        // Use the best regret task and insert it in its correct location
        if (!commitBestRegretTask(bestRegret)) {
          PLOGE << "run: failed to commit best-regret task "
                << bestRegret.task << "\n";
          repairFailed = true;
          break;
        }
      }
    } else {
      // Initial regret computation for the neighborhood.
      incrementalRegretStatsCurrent_.reset();
      if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
        repairFailed = true;
      }

      int64_t stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
      int64_t stalePopsSinceRefresh = 0;
      int64_t commitsSinceRefresh = 0;

      while (!repairFailed && !lnsNeighborhood_.removedTasks.empty()) {
        // If we are spending too much effort discarding stale heap entries,
        // do a full refresh of all remaining regrets.
        //
        // This helps when dirty rules miss some dependencies (regret drift),
        // and also prevents the heap from filling up with stale entries.
        if (stalePopsSinceRefresh >= 100 &&
            stalePopsSinceRefresh > 2 * commitsSinceRefresh + 50) {
          lnsNeighborhood_.regretMaxHeap.clear();
          incrementalRegretStatsCurrent_.fullRefreshes++;
          incrementalRegretStatsTotal_.fullRefreshes++;
          if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
            repairFailed = true;
            break;
          }
          stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
          stalePopsSinceRefresh = 0;
          commitsSinceRefresh = 0;
        }

        const auto bestRegret = popNextValidRegret();
        const int64_t staleDelta =
            incrementalRegretStatsCurrent_.stalePops - stalePopsAtLastCheck;
        stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
        stalePopsSinceRefresh += staleDelta;

        if (!bestRegret.has_value()) {
          // Heap may have been exhausted by stale entries; rebuild for whatever
          // is left.
          incrementalRegretStatsCurrent_.heapRebuilds++;
          incrementalRegretStatsTotal_.heapRebuilds++;
          if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
            repairFailed = true;
          }
          continue;
        }

        const vector<int> endTimesBefore = computeCurrentTaskEndTimes();
        const vector<int> lastTaskBefore = computeCurrentLastTaskPerAgent();
        if (!commitBestRegretTask(*bestRegret)) {
          PLOGE << "run: failed to commit best-regret task "
                << bestRegret->task << "\n";
          repairFailed = true;
          break;
        }
        incrementalRegretStatsCurrent_.commits++;
        incrementalRegretStatsTotal_.commits++;
        commitsSinceRefresh++;
        const vector<int> endTimesAfter = computeCurrentTaskEndTimes();
        const vector<int> lastTaskAfter = computeCurrentLastTaskPerAgent();

        const vector<int> dirtyTasks = computeDirtyTasksAfterCommit(
            endTimesBefore, endTimesAfter, lastTaskBefore, lastTaskAfter);
        if (!recomputeRegretsForTasks(dirtyTasks)) {
          repairFailed = true;
        }
      }

      // Stats are collected and printed in a dedicated summary section.
    }

    IterationQuality quality = IterationQuality::none;

    // If we could not successfully compute the regrets and commit to all the tasks in the neighborhood then we need to reset this neighborhood!
    if (repairFailed || !lnsNeighborhood_.removedTasks.empty()) {
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
      }
      // Reject whatever we done till now
      solution_ = previousSolution_;
      feasibleSolutionUpdated = false;
      quality = IterationQuality::couldNotFind;
      runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
      iterationStats.emplace_back(runtime, "LNS", instance_.getAgentNum(),
                                  instance_.getTasksNum(), solution_.sumOfCosts,
                                  feasibleSolutionUpdated, quality);
      // Skip everything after this statement
      PLOGD << "Could not find paths for the neighborhood! Attempting a new "
               "neighborhood computation\n";
      maybeUpdateMarketState(false);
      continue;
    }

    // Join only agents impacted by this neighborhood (removed tasks and their
    // precedence neighborhood). This avoids rebuilding all agent paths every
    // iteration when only a small subset changed.
    vector<int> agentsToCompute;
    vector<char> agentMarked(instance_.getAgentNum(), 0);
    auto markAgent = [&](int agent) {
      if (agent >= 0 && agent < instance_.getAgentNum() && !agentMarked[agent]) {
        agentMarked[agent] = 1;
        agentsToCompute.push_back(agent);
      }
    };
    vector<char> visitedTask(instance_.getTasksNum(), 0);
    stack<int> taskStack;
    for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
      taskStack.push(conflict.task);
      markAgent(conflict.agent);
    }
    const auto& ancestors = instance_.getAncestorsRef();
    const auto& successors = instance_.getSuccessorsRef();
    while (!taskStack.empty()) {
      const int task = taskStack.top();
      taskStack.pop();
      if (task < 0 || task >= instance_.getTasksNum() || visitedTask[task]) {
        continue;
      }
      visitedTask[task] = 1;
      const int curAgent =
          (task >= 0 && task < (int)solution_.taskAgentMap.size())
              ? solution_.taskAgentMap[task]
              : UNASSIGNED;
      if (curAgent != UNASSIGNED) {
        markAgent(curAgent);
      }
      const int prevAgent =
          (task >= 0 && task < (int)previousSolution_.taskAgentMap.size())
              ? previousSolution_.taskAgentMap[task]
              : UNASSIGNED;
      if (prevAgent != UNASSIGNED) {
        markAgent(prevAgent);
      }
      for (int parent : ancestors[task]) {
        if (parent >= 0 && parent < instance_.getTasksNum() &&
            !visitedTask[parent]) {
          taskStack.push(parent);
        }
      }
      for (int child : successors[task]) {
        if (child >= 0 && child < instance_.getTasksNum() &&
            !visitedTask[child]) {
          taskStack.push(child);
        }
      }
    }
    if (agentsToCompute.empty()) {
      agentsToCompute.resize(instance_.getAgentNum());
      std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
    }
    if (!solution_.joinPaths(agentsToCompute)) {
      PLOGE << "run: failed to join agent paths for candidate solution\n";
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
      }
      solution_ = previousSolution_;
      feasibleSolutionUpdated = false;
      quality = IterationQuality::couldNotFind;
      runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
      iterationStats.emplace_back(runtime, "LNS", instance_.getAgentNum(),
                                  instance_.getTasksNum(), solution_.sumOfCosts,
                                  feasibleSolutionUpdated, quality);
      maybeUpdateMarketState(false);
      continue;
    }

    // Compute the updated sum of costs
    solution_.sumOfCosts = 0;
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      solution_.sumOfCosts += solution_.agents[agent].path.endTimeOrZero();
    }

    PLOGD << "Old sum of costs = " << previousSolution_.sumOfCosts << "\n";
    PLOGD << "New sum of costs = " << solution_.sumOfCosts << "\n";

    PLOGD << "Number of conflicts in old solution: "
          << (int)potentialNeighborhood.size() << "\n";

    // Extract the set of conflicting tasks
    potentialNeighborhood.clear();
    valid = validateSolution(&potentialNeighborhood);

    PLOGD << "Number of conflicts in new solution: "
          << potentialNeighborhood.size() << "\n";

    // Accept the solution only if the new one has higher utility compared to the old solution where utility is a weighted combination of the number of conflicts and sum of costs.
    // Compute the utility of this solution
    solution_.utility = metrics.computeMovingMetrics(
        (int)potentialNeighborhood.size(), solution_.sumOfCosts);

    if (!valid) {
      // Solution was not valid as we found some conflicts!
      feasibleSolutionUpdated = false;
      PLOGE << "The solution was not valid!\n";
    } else {
      if (extractFeasibleSolution()) {
        // This is the case when the feasible solution was updated!
        quality = IterationQuality::bestSolutionYet;
        feasibleSolutionUpdated = true;
      } else {
        feasibleSolutionUpdated = false;
      }
    }

    // Ensure that we are either accepting or rejecting a solution here!
    const int proposedSocForIter = solution_.sumOfCosts;
    const double candidatePressure =
        market_.heuristics ? computeSolutionMarketPressure() : 0.0;
    const double candidateWait =
        market_.heuristics ? computeSolutionPrecedenceWait() : 0.0;
    bool accepted = false;
    bool guardRejected = false;
    if (market_.heuristics && market_.acceptanceGuards &&
        !passMarketAcceptanceGuards(candidatePressure, candidateWait)) {
      solution_ = previousSolution_;
      accepted = false;
      guardRejected = true;
      PLOGD << "Rejecting this solution due to market acceptance guards\n";
    } else {
      if (acceptanceCriteria == "SA") {
        accepted = simulatedAnnealing();
      } else if (acceptanceCriteria == "TA") {
        accepted = thresholdAcceptance();
      } else if (acceptanceCriteria == "OBA") {
        accepted = oldBachelorsAcceptance();
      } else if (acceptanceCriteria == "GDA") {
        accepted = greatDelugeAlgorithm();
      } else {
        PLOGE << "Unknown acceptance criteria: " << acceptanceCriteria << "\n";
        return false;
      }
    }

    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      const double deltaSoc = (double)(previousSocForIter - proposedSocForIter);
      adaptiveLNS_.deltaSocAll[alnsHeuristicForIter] += deltaSoc;
      if (valid) {
        adaptiveLNS_.feasible[alnsHeuristicForIter]++;
      }
      if (!accepted) {
        adaptiveLNS_.rejected[alnsHeuristicForIter]++;
      } else {
        adaptiveLNS_.accepted[alnsHeuristicForIter]++;
        adaptiveLNS_.deltaSocAccepted[alnsHeuristicForIter] += deltaSoc;
        if (previousSolution_.utility < solution_.utility) {
          adaptiveLNS_.downgradedAccepted[alnsHeuristicForIter]++;
        } else {
          adaptiveLNS_.improvedAccepted[alnsHeuristicForIter]++;
        }
        if (feasibleSolutionUpdated) {
          adaptiveLNS_.bestUpdates[alnsHeuristicForIter]++;
        }
      }
    }

    if (!accepted) {
      quality = IterationQuality::none;
      potentialNeighborhood = oldNeighborhood;
      if (guardRejected && market_.heuristics) {
        // Candidate utility was discarded; keep previous temperature behavior
        // untouched and preserve previous solution.
      }
    } else {
      if (previousSolution_.utility < solution_.utility) {
        // We accepted a potentially worse solution to get out of local minima
        quality = IterationQuality::downgradedButAccepted;
      } else {
        // We accepted a strictly better solution!
        quality = IterationQuality::improvedSolution;
      }
      previousSolution_ = solution_;
      if (market_.heuristics) {
        market_.bestPressure = min(market_.bestPressure, candidatePressure);
        market_.bestWait = min(market_.bestWait, candidateWait);
      }
    }

    maybeUpdateMarketState(accepted);

    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    double costToLog = (feasibleSolutionUpdated) ? incumbentSolution_.sumOfCosts
                                                 : solution_.sumOfCosts;
    iterationStats.emplace_back(runtime, "LNS", instance_.getAgentNum(),
                                instance_.getTasksNum(), costToLog,
                                feasibleSolutionUpdated, quality);
  }

  // printPaths();
  return !incumbentSolution_.agentPaths.empty();
}

void LNS::prepareNextIteration() {
  PLOGI << "Preparing the solution object for the next iteration\n";

  // Find the tasks that are following the earliest conflicting task as their paths need to be invalidated
  const auto& successors = instance_.getSuccessorsRef();

  // Include all successors of the original conflicted tasks in one multi-source
  // traversal to avoid re-traversing shared descendant subgraphs.
  vector<Conflicts> originalRemovedTasks;
  originalRemovedTasks.reserve(lnsNeighborhood_.removedTasks.size());
  for (const auto& [_, conflict] : lnsNeighborhood_.removedTasks) {
    originalRemovedTasks.push_back(conflict);
  }
  vector<char> visitedSuccessor(instance_.getTasksNum(), 0);
  stack<int> successorStack;
  for (const Conflicts& conflictTask : originalRemovedTasks) {
    if (conflictTask.task >= 0 && conflictTask.task < instance_.getTasksNum()) {
      successorStack.push(conflictTask.task);
    }
  }
  while (!successorStack.empty()) {
    const int successorTask = successorStack.top();
    successorStack.pop();
    if (successorTask < 0 || successorTask >= instance_.getTasksNum() ||
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
          solution_.getLocalTaskIndex(successorAgent, successorTask);
      if (successorTaskPosition != UNASSIGNED) {
        lnsNeighborhood_.removedTasks.emplace(
            successorTask,
            Conflicts(successorTask, successorAgent, successorTaskPosition));
      }
    }

    for (int nextTask : successors[successorTask]) {
      if (nextTask >= 0 && nextTask < instance_.getTasksNum() &&
          !visitedSuccessor[nextTask]) {
        successorStack.push(nextTask);
      }
    }
  }

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

  // Marking past information about conflicting tasks
  for (int affAgent : affectedAgents) {

    // For an affected agent there can be multiple conflicting tasks so need to do it this way
    solution_.agents[affAgent].path = AgentTaskPath();
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

  lnsNeighborhood_.patchedTasks = tasksToFix;

  // Find the paths for the tasks whose previous tasks were removed
  for (int task : instance_.getInputPlanningOrderRef()) {
    if (tasksToFix.count(task) > 0) {

      PLOGD << "Going to find path for next task: " << task << "\n";

      int startTime = 0, agent = solution_.getAgentWithTask(task);
      if (agent == UNASSIGNED) {
        PLOGE << "prepareNextIteration: patched task " << task
              << " is not assigned to any agent\n";
        continue;
      }
      int taskPosition = solution_.getLocalTaskIndex(agent, task);
      const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
      if (taskPosition < 0 || taskPosition >= (int)agentTasks.size() ||
          taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
        PLOGE << "prepareNextIteration: invalid task position " << taskPosition
              << " for task " << task << " (agent " << agent << ")\n";
        continue;
      }

      if (taskPosition != 0) {
        startTime = solution_.agents[agent].taskPaths[taskPosition - 1].endTime();
      }
      assert(taskPosition <= (int)agentTasks.size() - 1);

      ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
      buildConstraintTable(constraintTable, task);
      AgentTaskPath path = runLowLevelSearch(
          *solution_.agents[agent].pathPlanner, constraintTable, startTime,
          taskPosition, 0);
      // We must be able to find the path for the next task. If not then we cannot move forward!
      if (path.empty()) {
        PLOGE << "prepareNextIteration: path finding failed for patched task "
              << task << " (agent " << agent << ", position " << taskPosition
              << ")\n";
        continue;
      }
      assert(!path.empty());
      solution_.agents[agent].taskPaths[taskPosition] = path;

      // Once the path was found fix the begin times for subsequent tasks of the agent
      patchAgentTaskPaths(agent, taskPosition);
    }
  }
}
