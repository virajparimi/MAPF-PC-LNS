#pragma once

#include <utility>

#include "common.hpp"
#include "utils.hpp"

struct TaskAssignmentIndex {
  vector<int> owner;
  vector<int> pos;
};

int clampSocToInt(long long soc, const char* context);

TaskAssignmentIndex buildTaskAssignmentIndex(
    const vector<vector<int>>& assignments, int numTasks);

TaskAssignmentIndex buildCurrentTaskAssignmentIndex(
    const Solution& solution, int numTasks);

string summarizeIntervals(const vector<pair<int, int>>* intervals,
                          int maxIntervals = 6);

int permanentOccupancyStart(const vector<pair<int, int>>* intervals);

int firstFreeTimeAtOrAfter(const vector<pair<int, int>>* intervals,
                           int timestep);

bool reachableWithPermanentBlocksByTime(
    const Instance& instance, const ConstraintTable& constraintTable, int start,
    int goal, int byTime);

void logInitialSegmentFailureDiagnostics(
    const Instance& instance, const Solution& solution,
    const ConstraintTable& constraintTable, int agent, int task,
    int taskPosition, int startTime, SingleAgentSolver& solver,
    double remainingBudgetSec, double effectiveTimeoutSec,
    bool enableSippsCrossCheck);
