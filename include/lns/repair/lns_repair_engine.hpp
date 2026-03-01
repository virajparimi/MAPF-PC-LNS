#pragma once

#include <string>

class LNS;

enum class RepairStrategy {
  mapfpc_fixed,
  mapfpc_neighborhood_fixed,
  mapfpc_neighborhood_reassign_greedy,
  full_regret,
  incremental_regret
};

struct RepairPlan {
  bool tryNrr = false;
  bool allowFallbackToStandard = true;
  RepairStrategy strategy = RepairStrategy::full_regret;
};

class RepairEngine {
 public:
  static RepairPlan buildPlan(bool enableNrrRepair,
                              bool nrrFallbackToStandard,
                              bool incrementalRegret,
                              const std::string& repairHeuristic);

  static bool run(LNS& lns, bool& repairFailed, bool& nrrRepairSucceeded);
};
