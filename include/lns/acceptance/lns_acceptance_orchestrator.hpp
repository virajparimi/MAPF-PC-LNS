#pragma once

#include "lns.hpp"
#include "lns_iteration_phase_types.hpp"

class AcceptanceOrchestrator {
 public:
  static AcceptanceDecisionResult runDecision(
      LNS& lns, bool candidateValid, double previousPressureForIter,
      double previousWaitForIter, LNS::IterationDebugRecord& debugRow);

  static void updateAlnsStats(LNS& lns, int alnsHeuristicForIter,
                              int previousSocForIter, int proposedSocForIter,
                              bool candidateValid, bool accepted,
                              bool acceptedAsWorse,
                              bool feasibleSolutionUpdated);

  static void applyOutcome(
      LNS& lns, const AcceptanceDecisionResult& decisionResult,
      bool candidateValid, bool feasibleSolutionUpdated,
      bool nrrRepairSucceeded, int previousSocForIter, int proposedSocForIter,
      int incumbentSocBeforeIter, int previousConflictSignalForIter,
      int candidateConflictSignal,
      const LNS::ValidationStats& candidateValidationStats,
      ConflictMap& potentialNeighborhood, const ConflictMap& oldNeighborhood,
      bool& currentSolutionValid, LNS::ValidationStats& currentValidationStats,
      IterationQuality& quality, LNS::IterationDebugRecord& debugRow);
};
