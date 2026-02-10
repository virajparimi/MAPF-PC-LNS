#pragma once

#include "common.hpp"
#include "instance.hpp"
#include "lns.hpp"

bool greedyTaskAssignment(const Instance* instance, Solution* solution);

bool topologicalSort(const Instance* instance,
                     const vector<pair<int, int>>& precedenceConstraints,
                     vector<int>& planningOrder);

ConflictMap extractNConflicts(int size, const ConflictMap& conflicts);

struct MovingMetrics {
 private:
  int size{}, oldestValue = 0;
  vector<double> conflictNum{}, costNum{};
  double lnsConflictWeight{}, lnsCostWeight{}, sumOfNumConflicts{},
      sumOfNumCosts{};

 public:
  MovingMetrics(int windowSize, double conflictWeight, double costWeight,
                int numberOfConflicts, int sumOfCosts) {
    if (windowSize <= 0) {
      PLOGE << "MovingMetrics: windowSize must be positive; received "
            << windowSize << ". Falling back to 1.\n";
      windowSize = 1;
    }
    this->size = windowSize;
    this->lnsConflictWeight = conflictWeight;
    this->lnsCostWeight = costWeight;
    const double conflicts = static_cast<double>(numberOfConflicts);
    const double costs = static_cast<double>(sumOfCosts);
    // Prefill the full window to avoid early-iteration dilution by zeros.
    conflictNum.assign(windowSize, conflicts);
    costNum.assign(windowSize, costs);
    sumOfNumConflicts = conflicts * static_cast<double>(windowSize);
    sumOfNumCosts = costs * static_cast<double>(windowSize);
  }
  double computeMovingMetrics(int numberOfConflicts, int sumOfCosts);
};
