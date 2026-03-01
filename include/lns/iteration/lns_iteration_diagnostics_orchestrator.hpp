#pragma once

#include <cstdint>
#include <string>

class LNS;

struct NeighborhoodDiagnosticsResult {
  int removedTasks = 0;
  uint64_t neighborhoodFingerprint = 0;
  bool neighborhoodFingerprintSeenBefore = false;
  int neighborhoodRepeatStreak = 0;
  std::string removedTaskIdsCsv;
  double neighborhoodJaccardPrev = 0.0;
};

class IterationDiagnosticsOrchestrator {
 public:
  static NeighborhoodDiagnosticsResult analyzeNeighborhood(LNS& lns);
};
