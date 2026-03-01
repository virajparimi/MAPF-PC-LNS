#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>

#include "lns_types.hpp"

class LNS;

struct DestroySamplingContext {
  ALNS& adaptiveLns;
  std::mt19937& rng;
  bool alnsEnablePrecedenceAwareDestroy;

  bool marketHeuristicsEnabled;
  bool marketDestroySoftGate;
  bool marketWarmupReady;
  bool marketStableReady;
  double marketDestroyWarmupWeightScale;
  double marketDestroyUnstableWeightScale;
  double marketDestroyMinAlnsWeight;
  int64_t& marketDestroyWarmupSkipped;
  int64_t& marketDestroyUnstableSkipped;
};

class DestroyOrchestrator {
 public:
  // Returns sampled destroy heuristic id, or std::nullopt if no eligible
  // heuristic exists under the current config/gating constraints.
  static std::optional<int> sampleAlnsHeuristic(DestroySamplingContext& ctx);

  // Maps CLI/string destroy heuristic to numeric id.
  static std::optional<int> heuristicIdFromName(const std::string& name);

  // Execute one destroy heuristic by id.
  static bool executeHeuristic(LNS& lns, int destroyHeuristicId,
                               const ConflictMap* potentialNeighborhood);

  // Run destroy phase for one iteration (ALNS/manual dispatch + validation).
  static bool runPhase(LNS& lns, const ConflictMap* potentialNeighborhood,
                       int& alnsHeuristicForIter, std::string& errorMessage);
};
