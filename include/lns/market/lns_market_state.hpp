#pragma once

#include <limits>
#include "lns_params.hpp"
#include "lns_types.hpp"

struct LNSMarketState : LNSParams::Market {
  int64_t acceptedCounter = 0;
  int64_t updateCounter = 0;
  double bestPressure = std::numeric_limits<double>::infinity();
  double bestWait = std::numeric_limits<double>::infinity();

  unordered_map<uint64_t, double> vertexPrices;
  unordered_map<uint64_t, double> edgePrices;
  unordered_map<uint64_t, double> vertexExcessHat;
  unordered_map<uint64_t, double> edgeExcessHat;
  unordered_set<uint64_t> prevContendedVertices;
  unordered_set<uint64_t> prevContendedEdges;
  bool hasStabilityBaseline = false;
  bool candidateUpdateConsumed = false;
  vector<int> taskCooldownUntilIter;
  MarketStats stats;
};
