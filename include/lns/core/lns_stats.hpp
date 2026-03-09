#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace lns_stats {

struct LowLevelSearchStats {
  uint64_t calls = 0;
  uint64_t expanded = 0;
  uint64_t generated = 0;
  uint64_t found = 0;
  uint64_t timeout = 0;
  uint64_t searchExhausted = 0;
  uint64_t invalidInput = 0;
  uint64_t budgetExhausted = 0;
  uint64_t unknown = 0;
  uint64_t timeoutGoalPermanentBeforeArrivalLb = 0;
  uint64_t timeoutStartTrappedAtTPlus1 = 0;
  uint64_t timeoutStaticDisconnectedPermanent = 0;
  uint64_t timeoutOther = 0;
  uint64_t timeoutReducedByGlobalBudget = 0;
  uint64_t timeoutMultiCertificate = 0;
  uint64_t structuralPrePruned = 0;
  uint64_t structuralPrePrunedGoalPermanentBeforeArrivalLb = 0;
  uint64_t structuralPrePrunedStartTrappedAtTPlus1 = 0;
  uint64_t structuralPrePrunedStaticDisconnectedPermanent = 0;
  uint64_t structuralPrePrunedMultiCertificate = 0;
};

struct RegretEvalStats {
  int64_t recomputeCalls = 0;
  int64_t tasksEvaluated = 0;
  int64_t agentEvaluations = 0;
  int64_t candidateInsertionsTried = 0;
  int64_t candidateInsertionsFeasible = 0;
  int64_t workspaceAgentsCloned = 0;
  int64_t workspaceMaxClonedPerTask = 0;
  int64_t neighborhoods = 0;
  int64_t removedTasksSum = 0;
  int64_t removedTasksMax = 0;

  void reset() { *this = RegretEvalStats(); }
};

struct NrrStats {
  int64_t calls = 0;
  int64_t attempts = 0;
  int64_t miniSolverCalls = 0;
  int64_t success = 0;
  int64_t fallbackToStandard = 0;
  std::map<std::string, int64_t> failureReasonHistogram;
  int64_t stitchedInvalidAttempts = 0;
  int64_t stitchedInvalidPrecedenceViolations = 0;
  int64_t stitchedInvalidVertexCollisions = 0;
  int64_t stitchedInvalidEdgeSwapCollisions = 0;
  int64_t stitchedInvalidStructuralViolations = 0;
  int64_t solverPbs = 0;
  int64_t solverCbs = 0;
  double runtimeSecSum = 0.0;
  double timeSetupSecSum = 0.0;
  double timeBoundaryWindowsSecSum = 0.0;
  double timeProposalBuildSecSum = 0.0;
  double timeTemporalPrecheckSecSum = 0.0;
  double timeInputBuildSecSum = 0.0;
  double timeMiniSolverSecSum = 0.0;
  double timeStitchSecSum = 0.0;
  double timeTaskMapRebuildSecSum = 0.0;
  double timeTerminalReplanSecSum = 0.0;
  double timeRecomputeObjectiveSecSum = 0.0;
  double timeValidationSecSum = 0.0;
  int64_t removedTasksSum = 0;
  int64_t removedTasksMax = 0;
  int64_t neighborhoodAgentsSum = 0;
  int64_t neighborhoodAgentsMax = 0;
  int64_t globalTouchedAgentsSum = 0;
  int64_t globalTouchedAgentsMax = 0;
  int64_t globalDemotionsCount = 0;
  int64_t globalPatchedTasksSum = 0;
  int64_t globalSlotEvalCount = 0;
  int64_t globalNoFeasibleSlotCount = 0;
  std::map<int, int64_t> removedTasksHistogram;
  std::map<int, int64_t> neighborhoodAgentsHistogram;
  std::map<int, int64_t> globalTouchedAgentsHistogram;
  double acceptedDeltaSocSum = 0.0;
  int64_t acceptedDeltaSocCount = 0;
  int64_t softCandidatesProduced = 0;
  int64_t softCandidatesAccepted = 0;
  int64_t softCandidatesRejected = 0;
  int64_t softRecoveryEntries = 0;
  int64_t softRecoveryExits = 0;
  int64_t softRecoveryAcceptedEntry = 0;
  int64_t softRecoveryAcceptedDescent = 0;
  int64_t softRecoveryRejected = 0;
  // Heuristic-attributed soft-recovery stats (indexed by DestroyHeuristic id).
  std::vector<int64_t> softModeSelectionsByDestroy;
  std::vector<int64_t> softModeAcceptedByDestroy;
  std::vector<int64_t> softModeBestUpdatesByDestroy;
  int64_t softModeEntryConflictSum = 0;
  int64_t softModeExitConflictSum = 0;
  int64_t softModeResolvedEntries = 0;
  double softCandidateConflictSum = 0.0;
  int64_t softCandidateConflictSamples = 0;

  void reset() { *this = NrrStats(); }
};

struct CascadeStats {
  int64_t prepareCalls = 0;
  int64_t budgetAborts = 0;
  int64_t budgetUsedSum = 0;
  int64_t budgetUsedMin = std::numeric_limits<int64_t>::max();
  int64_t budgetUsedMax = 0;
  int64_t seedTasksSum = 0;
  int64_t closureTasksSum = 0;
  int64_t closureAddedSum = 0;
  int64_t closureTasksMax = 0;
  int64_t closureAddedMax = 0;

  void reset() { *this = CascadeStats(); }
};

struct SolutionRestoreStats {
  int64_t restoreCalls = 0;
  int64_t fullRestores = 0;

  void reset() { *this = SolutionRestoreStats(); }
};

struct ImprovementDiagnosticsStats {
  int64_t iterationsStarted = 0;
  int64_t earlyAbortPrepare = 0;
  int64_t earlyAbortRepair = 0;
  int64_t earlyAbortJoin = 0;

  int64_t candidateValid = 0;
  int64_t candidateInvalid = 0;

  int64_t accepted = 0;
  int64_t rejected = 0;
  int64_t guardRejected = 0;
  int64_t acceptedAsWorseUtility = 0;
  int64_t acceptedAsBetterOrEqualUtility = 0;

  int64_t acceptedSocBetterVsPrevious = 0;
  int64_t acceptedSocEqualVsPrevious = 0;
  int64_t acceptedSocWorseVsPrevious = 0;

  int64_t acceptedSocBetterVsIncumbent = 0;
  int64_t acceptedSocEqualVsIncumbent = 0;
  int64_t acceptedSocWorseVsIncumbent = 0;

  int64_t feasibleBestUpdates = 0;
  int64_t feasibleNoBestUpdate = 0;
  int64_t acceptedFeasibleNoBestUpdate = 0;
  int64_t acceptedInvalid = 0;

  int64_t conflictSignalBetter = 0;
  int64_t conflictSignalEqual = 0;
  int64_t conflictSignalWorse = 0;
  int64_t acceptedConflictSignalBetter = 0;
  int64_t acceptedConflictSignalEqual = 0;
  int64_t acceptedConflictSignalWorse = 0;
  int64_t acceptedFingerprintUnique = 0;
  int64_t acceptedFingerprintRepeat = 0;
  int64_t softCandidateProduced = 0;
  int64_t softCandidateAccepted = 0;
  int64_t softCandidateRejected = 0;
  int64_t softRecoveryEntries = 0;
  int64_t softRecoveryExits = 0;
  int64_t softRecoveryAcceptedEntry = 0;
  int64_t softRecoveryAcceptedDescent = 0;
  int64_t softRecoveryRejected = 0;
  std::vector<int64_t> softModeSelectionsByDestroy;
  std::vector<int64_t> softModeAcceptedByDestroy;
  std::vector<int64_t> softModeBestUpdatesByDestroy;
  int64_t softModeEntryConflictSum = 0;
  int64_t softModeExitConflictSum = 0;
  int64_t softModeResolvedEntries = 0;
  double softCandidateConflictSum = 0.0;
  int64_t softCandidateConflictSamples = 0;

  int64_t neighborhoodsCount = 0;
  int64_t neighborhoodsWithChanges = 0;
  int64_t neighborhoodsWithoutChanges = 0;
  int64_t neighborhoodFingerprintUnique = 0;
  int64_t neighborhoodFingerprintRepeat = 0;
  int64_t neighborhoodRepeatStreakMax = 0;
  double neighborhoodJaccardPrevSum = 0.0;
  int64_t neighborhoodJaccardPrevSamples = 0;
  double neighborhoodJaccardPrevRepeatSum = 0.0;
  int64_t neighborhoodJaccardPrevRepeatSamples = 0;
  int64_t removedTasksTotal = 0;
  int64_t removedTasksChangedAgentTotal = 0;
  int64_t removedTasksChangedOrderTotal = 0;
  int64_t removedTasksUnchangedTotal = 0;
  int64_t removedTasksMax = 0;
  int64_t changedTasksPerNeighborhoodMax = 0;

  int64_t acceptedRemovedTasksTotal = 0;
  int64_t acceptedRemovedTasksChangedAgentTotal = 0;
  int64_t acceptedRemovedTasksChangedOrderTotal = 0;
  int64_t acceptedRemovedTasksUnchangedTotal = 0;
  int64_t acceptedRemovedTasksMax = 0;
  int64_t acceptedChangedTasksPerNeighborhoodMax = 0;

  double sumPreviousSoc = 0.0;
  double sumCandidateSoc = 0.0;
  double sumIncumbentSoc = 0.0;
  int64_t incumbentSocSamples = 0;
  double sumPreviousConflictSignal = 0.0;
  double sumCandidateConflictSignal = 0.0;

  double timeDestroyAndPrepareSec = 0.0;
  double timeRepairAndCommitSec = 0.0;
  double timeRegretCandidateEvalSec = 0.0;
  double timeRegretCommitSec = 0.0;
  double timeRegretLowLevelSec = 0.0;
  double timeJoinPathsSec = 0.0;
  double timeRecomputeSocSec = 0.0;
  double timeValidationSec = 0.0;
  double timeAcceptanceSec = 0.0;
  double timeBookkeepingSec = 0.0;
  double timeInitialSolutionSec = 0.0;
  double timeInitialSolutionReportedSec = 0.0;
  double timeLnsLoopSec = 0.0;
  double timePostRefineSec = 0.0;

  void reset() { *this = ImprovementDiagnosticsStats(); }
};

struct IterationDebugRecord {
  int64_t iteration = -1;
  double runtimeSec = 0.0;
  int previousSoc = 0;
  int candidateSoc = 0;
  int incumbentSocBefore = std::numeric_limits<int>::max();
  int previousConflictSignal = -1;
  int candidateConflictSignal = -1;
  int removedTasks = 0;
  std::string removedTaskIdsCsv;
  int removedTasksChangedAgent = -1;
  int removedTasksChangedOrder = -1;
  int removedTasksUnchanged = -1;
  bool neighborhoodFingerprintSeenBefore = false;
  uint64_t neighborhoodFingerprint = 0;
  double neighborhoodJaccardPrev = -1.0;
  int neighborhoodRepeatStreak = 0;
  bool candidateValid = false;
  bool accepted = false;
  bool guardRejected = false;
  bool acceptedAsWorseUtility = false;
  bool feasibleBestUpdate = false;
  bool fingerprintSeenBefore = false;
  uint64_t fingerprint = 0;
  bool nrrSoftCandidate = false;
  bool nrrSoftOnlyInvalid = false;
  bool softRecoveryModeBefore = false;
  bool softRecoveryModeAfter = false;
  int64_t nrrSoftConflictCount = -1;
  std::string softRecoveryDecisionReason = "none";
  int destroyHeuristicId = -1;
  std::string destroyHeuristicName = "unknown";
  bool destroySelectedInSoftMode = false;
  std::string earlyAbortReason;
  std::string quality;
  double timeDestroyAndPrepareSec = 0.0;
  double timeRepairAndCommitSec = 0.0;
  double timeRegretCandidateEvalSec = 0.0;
  double timeRegretCommitSec = 0.0;
  double timeRegretLowLevelSec = 0.0;
  double timeJoinPathsSec = 0.0;
  double timeRecomputeSocSec = 0.0;
  double timeValidationSec = 0.0;
  double timeAcceptanceSec = 0.0;
  double timeBookkeepingSec = 0.0;
};

struct ValidationStats {
  int precedenceViolations = 0;
  int vertexCollisions = 0;
  int edgeSwapCollisions = 0;
  int structuralViolations = 0;
  int64_t precedenceDebt = 0;
  int64_t precedencePairsChecked = 0;

  int totalConflictEvents() const {
    return precedenceViolations + vertexCollisions + edgeSwapCollisions +
           structuralViolations;
  }
  int spatialConflictEvents() const {
    return vertexCollisions + edgeSwapCollisions + structuralViolations;
  }
};

}  // namespace lns_stats
