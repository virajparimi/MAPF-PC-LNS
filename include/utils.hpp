#pragma once

#include "common.hpp"
#include "instance.hpp"
#include "lns.hpp"

void greedyTaskAssignment(const Instance* instance, Solution* solution);

bool topologicalSort(const Instance* instance,
                     vector<pair<int, int>>* precedenceConstraints,
                     vector<int>& planningOrder);
bool topologicalSort(const Instance* instance,
                     const vector<pair<int, int>>& precedenceConstraints,
                     vector<int>& planningOrder);

set<Conflicts> extractNConflicts(int size, const set<Conflicts>& conflicts);

struct MovingMetrics {
  int size{}, oldestValue = 0;
  vector<double> conflictNum{}, conflictSquareNum{}, costNum{}, costSquareNum{};
  double lnsConflictWeight{}, lnsCostWeight{}, sumOfNumConflicts{},
      sumOfNumConflictsSquare{}, sumOfNumCosts{}, sumOfNumCostsSquare{};

  MovingMetrics(int size, double conflictWeight, double costWeight,
                int numberOfConflicts, int sumOfCosts) {
    assert(size > 0);
    this->size = size;
    this->lnsConflictWeight = conflictWeight;
    this->lnsCostWeight = costWeight;
    conflictNum.resize(size);
    conflictNum[oldestValue] = numberOfConflicts;
    conflictSquareNum.resize(size);
    conflictSquareNum[oldestValue] =
        (double)numberOfConflicts * (double)numberOfConflicts;
    costNum.resize(size);
    costNum[oldestValue] = sumOfCosts;
    costSquareNum.resize(size);
    costSquareNum[oldestValue] = (double)sumOfCosts * (double)sumOfCosts;
    sumOfNumConflicts =
        std::accumulate(begin(conflictNum), end(conflictNum), 0.0);
    sumOfNumConflictsSquare =
        std::accumulate(begin(conflictSquareNum), end(conflictSquareNum), 0.0);
    sumOfNumCosts = std::accumulate(begin(costNum), end(costNum), 0.0);
    sumOfNumCostsSquare =
        std::accumulate(begin(costSquareNum), end(costSquareNum), 0.0);
  }
  double computeMovingMetrics(int numberOfConflicts, int sumOfCosts);
};
