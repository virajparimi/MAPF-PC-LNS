#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class CandidatePhaseStatus { success, join_failed, terminal_failed };

struct CandidatePhaseResult {
  CandidatePhaseStatus status = CandidatePhaseStatus::success;
  double timeJoinPathsSec = 0.0;
  double timeTerminalReplanSec = 0.0;
  double timeRecomputeSocSec = 0.0;
  double timeValidationSec = 0.0;
  int removedTasksChangedAgent = 0;
  int removedTasksChangedOrder = 0;
  int removedTasksUnchanged = 0;
  int proposedSoc = 0;
  bool candidateValid = false;
  bool feasibleBestUpdate = false;
  int candidateConflictSignal = 0;
  int precedenceViolations = 0;
  int vertexCollisions = 0;
  int edgeSwapCollisions = 0;
  int structuralViolations = 0;
  int64_t precedenceDebt = 0;
  int64_t precedencePairsChecked = 0;
  std::vector<int> candidateTouchedAgents;
};

struct AcceptanceDecisionResult {
  bool ok = true;
  bool accepted = false;
  bool guardRejected = false;
  bool invalidGuardRejected = false;
  bool acceptedAsWorse = false;
  bool usedSoftRecoveryOverride = false;
  bool softRecoveryModeBefore = false;
  bool softRecoveryModeAfter = false;
  std::string softRecoveryDecisionReason = "none";
  double candidatePressure = 0.0;
  double candidateWait = 0.0;
  double timeAcceptanceSec = 0.0;
};

struct IterationMarketContext {
  double previousPressure = 0.0;
  double previousWait = 0.0;
};

struct MarketGuardDecision {
  bool allowed = true;
  bool guardRejected = false;
  double candidatePressure = 0.0;
  double candidateWait = 0.0;
};
