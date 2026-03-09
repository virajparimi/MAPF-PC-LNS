#pragma once

#include <string>

class LNS;

enum class RepairStrategy {
  full_regret
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
                              int repairHeuristicMode);

  static bool run(LNS& lns, bool& repairFailed, bool& nrrRepairSucceeded);
};
