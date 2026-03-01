#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "run_reporting.hpp"

namespace run_reporting_internal {

std::string formatDouble(double value, int precision = 4);
void printMetric(const string& key, const string& value);
void printMetric(const string& key, double value, int precision = 4);
void printMetric(const string& key, int64_t value);
void printMetric(const string& key, int value);
void printIntListLine(const string& label, const vector<int>& values);
void printDoubleListLine(const string& label, const vector<double>& values,
                         int precision = 4);
double percentileValue(std::vector<double> values, double percentile);
vector<double> computeIterationDurations(const vector<IterationStats>& stats);
int countLnsIterations(const vector<IterationStats>& stats);
double computeTimeToBestFeasible(const vector<IterationStats>& stats);

struct FinalSolutionScheduleMetrics {
  int makespan = 0;
  int maxIndividualCost = 0;
  bool hasTaskScheduleData = false;
  int totalPrecedenceWait = 0;
  int maxPrecedenceWait = 0;
  vector<double> precedenceSlacks;
};

FinalSolutionScheduleMetrics computeFinalSolutionScheduleMetrics(
    const FeasibleSolution& solution, const Instance& instance);

}  // namespace run_reporting_internal
