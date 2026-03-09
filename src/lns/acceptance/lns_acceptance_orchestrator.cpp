#include "lns_acceptance_orchestrator.hpp"

#include <algorithm>
#include <limits>

AcceptanceDecisionResult AcceptanceOrchestrator::runDecision(
    LNS& lns, bool candidateValid, int previousConflictSignalForIter,
    const std::vector<int>& candidateTouchedAgents,
    LNS::IterationDebugRecord& debugRow) {
  return lns.runAcceptanceDecision(
      candidateValid, previousConflictSignalForIter, candidateTouchedAgents,
      debugRow);
}

void AcceptanceOrchestrator::updateAlnsStats(
    LNS& lns, int alnsHeuristicForIter, int previousSocForIter,
    int proposedSocForIter, bool candidateValid, bool accepted,
    bool acceptedAsWorse, bool feasibleSolutionUpdated) {
  lns.updateAcceptanceAlnsStats(
      alnsHeuristicForIter, previousSocForIter, proposedSocForIter,
      candidateValid, accepted, acceptedAsWorse, feasibleSolutionUpdated);
}

void AcceptanceOrchestrator::applyOutcome(
    LNS& lns, const AcceptanceDecisionResult& decisionResult,
    bool candidateValid, bool feasibleSolutionUpdated,
    bool nrrRepairSucceeded, int previousSocForIter, int proposedSocForIter,
    int incumbentSocBeforeIter, int previousConflictSignalForIter,
    int candidateConflictSignal,
    const LNS::ValidationStats& candidateValidationStats,
    ConflictMap& potentialNeighborhood, const ConflictMap& oldNeighborhood,
    bool& currentSolutionValid, LNS::ValidationStats& currentValidationStats,
    IterationQuality& quality, const std::vector<int>& candidateTouchedAgents,
    LNS::IterationDebugRecord& debugRow) {
  lns.applyAcceptanceOutcome(
      decisionResult, candidateValid, feasibleSolutionUpdated,
      nrrRepairSucceeded, previousSocForIter, proposedSocForIter,
      incumbentSocBeforeIter, previousConflictSignalForIter,
      candidateConflictSignal, candidateValidationStats, potentialNeighborhood,
      oldNeighborhood, currentSolutionValid, currentValidationStats, quality,
      candidateTouchedAgents, debugRow);
}

AcceptanceDecisionResult LNS::runAcceptanceDecision(
    bool candidateValid, int previousConflictSignalForIter,
    const std::vector<int>& candidateTouchedAgents,
    LNS::IterationDebugRecord& debugRow) {
  AcceptanceDecisionResult result;
  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };
  auto advanceTemperatureOnGuardReject = [&]() {
    if (acceptanceCriteria == "SA" || acceptanceCriteria == "TA") {
      acceptanceState_.temperature *= acceptanceState_.coolingCoefficient;
    } else if (acceptanceCriteria == "OBA") {
      const double reheated = acceptanceState_.temperature * acceptanceState_.heatingCoefficient;
      acceptanceState_.temperature = std::min(reheated, acceptanceState_.maxTemperature);
    } else if (acceptanceCriteria == "GDA") {
      acceptanceState_.temperature =
          std::max(0.0, acceptanceState_.temperature - acceptanceState_.greatDelugeDecay);
    }
  };
  auto finalizeDecision = [&](const Time::time_point& acceptanceStart) {
    result.softRecoveryModeAfter = acceptanceState_.softRecoveryActive;
    result.timeAcceptanceSec += elapsedSecSince(acceptanceStart);
    return result;
  };

  const Time::time_point acceptanceStart = Time::now();
  result.softRecoveryModeBefore = acceptanceState_.softRecoveryActive;
  result.softRecoveryModeAfter = acceptanceState_.softRecoveryActive;
  const bool candidateIsNrrSoftCandidate =
      (!candidateValid && lastNrrSoftCandidate_ &&
       lastNrrSoftOnlyInvalid_ && lastNrrSoftConflictCount_ > 0);
  if (!acceptanceState_.acceptOnlyValidCandidates &&
      (acceptanceState_.softRecoveryActive || candidateIsNrrSoftCandidate)) {
    if (!acceptanceState_.softRecoveryActive && candidateIsNrrSoftCandidate &&
        previousConflictSignalForIter == 0) {
      acceptanceState_.softRecoveryActive = true;
      acceptanceState_.softRecoveryCurrentConflicts = lastNrrSoftConflictCount_;
      softRecoveryState_.lastSoftFailureConflictAgents.clear();
      softRecoveryState_.lastSoftFailureConflictTasks.clear();
      nrrStats_.softRecoveryEntries++;
      improvementDiagnosticsStats_.softRecoveryEntries++;
      nrrStats_.softModeEntryConflictSum += lastNrrSoftConflictCount_;
      improvementDiagnosticsStats_.softModeEntryConflictSum +=
          lastNrrSoftConflictCount_;
      result.accepted = true;
      result.acceptedAsWorse =
          previousSolution_.utility < solution_.utility;
      result.usedSoftRecoveryOverride = true;
      result.softRecoveryDecisionReason = "soft_entry_accept";
      nrrStats_.softCandidatesAccepted++;
      improvementDiagnosticsStats_.softCandidateAccepted++;
      nrrStats_.softRecoveryAcceptedEntry++;
      improvementDiagnosticsStats_.softRecoveryAcceptedEntry++;
      return finalizeDecision(acceptanceStart);
    }

    if (acceptanceState_.softRecoveryActive) {
      if (candidateValid) {
        acceptanceState_.softRecoveryActive = false;
        acceptanceState_.softRecoveryCurrentConflicts = -1;
        nrrStats_.softRecoveryExits++;
        improvementDiagnosticsStats_.softRecoveryExits++;
        nrrStats_.softModeExitConflictSum += 0;
        improvementDiagnosticsStats_.softModeExitConflictSum += 0;
        nrrStats_.softModeResolvedEntries++;
        improvementDiagnosticsStats_.softModeResolvedEntries++;
        result.accepted = true;
        result.acceptedAsWorse =
            previousSolution_.utility < solution_.utility;
        result.usedSoftRecoveryOverride = true;
        result.softRecoveryDecisionReason = "soft_exit_on_valid_accept";
        return finalizeDecision(acceptanceStart);
      }

      if (candidateIsNrrSoftCandidate) {
        const int64_t candidateSoftConflicts = lastNrrSoftConflictCount_;
        if (candidateSoftConflicts <= acceptanceState_.softRecoveryCurrentConflicts) {
          acceptanceState_.softRecoveryCurrentConflicts = candidateSoftConflicts;
          result.accepted = true;
          result.acceptedAsWorse =
              previousSolution_.utility < solution_.utility;
          result.usedSoftRecoveryOverride = true;
          result.softRecoveryDecisionReason = "soft_descent_accept";
          nrrStats_.softCandidatesAccepted++;
          improvementDiagnosticsStats_.softCandidateAccepted++;
          nrrStats_.softRecoveryAcceptedDescent++;
          improvementDiagnosticsStats_.softRecoveryAcceptedDescent++;
          return finalizeDecision(acceptanceStart);
        }
        restoreSolutionFromPrevious(candidateTouchedAgents);
        result.accepted = false;
        result.usedSoftRecoveryOverride = true;
        result.softRecoveryDecisionReason = "soft_descent_reject_non_improving";
        softRecoveryState_.lastSoftFailureConflictAgents = softRecoveryState_.lastValidationConflictAgents;
        softRecoveryState_.lastSoftFailureConflictTasks = softRecoveryState_.lastValidationConflictTasks;
        nrrStats_.softCandidatesRejected++;
        improvementDiagnosticsStats_.softCandidateRejected++;
        nrrStats_.softRecoveryRejected++;
        improvementDiagnosticsStats_.softRecoveryRejected++;
        advanceTemperatureOnGuardReject();
        return finalizeDecision(acceptanceStart);
      }
      restoreSolutionFromPrevious(candidateTouchedAgents);
      result.accepted = false;
      result.usedSoftRecoveryOverride = true;
      result.softRecoveryDecisionReason = "soft_mode_reject_nonsoft_invalid";
      nrrStats_.softRecoveryRejected++;
      improvementDiagnosticsStats_.softRecoveryRejected++;
      advanceTemperatureOnGuardReject();
      return finalizeDecision(acceptanceStart);
    } else if (candidateIsNrrSoftCandidate) {
      restoreSolutionFromPrevious(candidateTouchedAgents);
      result.accepted = false;
      result.usedSoftRecoveryOverride = true;
      result.softRecoveryDecisionReason = "soft_entry_reject_prev_nonzero";
      nrrStats_.softCandidatesRejected++;
      improvementDiagnosticsStats_.softCandidateRejected++;
      advanceTemperatureOnGuardReject();
      return finalizeDecision(acceptanceStart);
    }
  }

  if (!candidateValid && !acceptanceState_.acceptOnlyValidCandidates) {
    PLOGD << "Invalid candidate forwarded to acceptance criteria\n";
  }
  if (acceptanceState_.acceptOnlyValidCandidates && !candidateValid) {
    restoreSolutionFromPrevious(candidateTouchedAgents);
    result.accepted = false;
    result.invalidGuardRejected = true;
    if (candidateIsNrrSoftCandidate) {
      nrrStats_.softCandidatesRejected++;
      improvementDiagnosticsStats_.softCandidateRejected++;
      if (result.softRecoveryDecisionReason == "none") {
        result.softRecoveryDecisionReason = "soft_reject_strict_valid_guard";
      }
    }
    advanceTemperatureOnGuardReject();
    PLOGD << "Rejecting this solution due to strict-valid acceptance guard\n";
    return finalizeDecision(acceptanceStart);
  }

  if (acceptanceCriteria == "SA") {
    result.accepted = simulatedAnnealing(candidateTouchedAgents);
  } else if (acceptanceCriteria == "TA") {
    result.accepted = thresholdAcceptance(candidateTouchedAgents);
  } else if (acceptanceCriteria == "OBA") {
    result.accepted = oldBachelorsAcceptance(candidateTouchedAgents);
  } else if (acceptanceCriteria == "GDA") {
    result.accepted = greatDelugeAlgorithm(candidateTouchedAgents);
  } else {
    PLOGE << "Unknown acceptance criteria: " << acceptanceCriteria << "\n";
    debugRow.earlyAbortReason = "acceptance_criteria_error";
    result.ok = false;
    return finalizeDecision(acceptanceStart);
  }
  if (result.accepted) {
    result.acceptedAsWorse =
        previousSolution_.utility < solution_.utility;
    if (acceptanceState_.softRecoveryActive && candidateValid) {
      acceptanceState_.softRecoveryActive = false;
      acceptanceState_.softRecoveryCurrentConflicts = -1;
      nrrStats_.softRecoveryExits++;
      improvementDiagnosticsStats_.softRecoveryExits++;
      nrrStats_.softModeExitConflictSum += 0;
      improvementDiagnosticsStats_.softModeExitConflictSum += 0;
      nrrStats_.softModeResolvedEntries++;
      improvementDiagnosticsStats_.softModeResolvedEntries++;
      if (result.softRecoveryDecisionReason == "none") {
        result.softRecoveryDecisionReason = "soft_exit_on_valid_accept";
      }
    }
  }
  return finalizeDecision(acceptanceStart);
}

void LNS::updateAcceptanceAlnsStats(int alnsHeuristicForIter, int previousSocForIter,
    int proposedSocForIter, bool candidateValid, bool accepted,
    bool acceptedAsWorse, bool feasibleSolutionUpdated) {
  if (alnsHeuristicForIter < 0 ||
      alnsHeuristicForIter >= adaptiveLNS_.numDestroyHeuristics) {
    return;
  }
  const double deltaSoc = (double)(previousSocForIter - proposedSocForIter);
  adaptiveLNS_.deltaSocAll[alnsHeuristicForIter] += deltaSoc;
  if (deltaSoc > 0.0) {
    adaptiveLNS_.proposedBetter[alnsHeuristicForIter]++;
  } else if (deltaSoc < 0.0) {
    adaptiveLNS_.proposedWorse[alnsHeuristicForIter]++;
  } else {
    adaptiveLNS_.proposedEqual[alnsHeuristicForIter]++;
  }
  if (candidateValid) {
    adaptiveLNS_.feasible[alnsHeuristicForIter]++;
  }
  if (!accepted) {
    adaptiveLNS_.rejected[alnsHeuristicForIter]++;
    return;
  }
  adaptiveLNS_.accepted[alnsHeuristicForIter]++;
  adaptiveLNS_.deltaSocAccepted[alnsHeuristicForIter] += deltaSoc;
  if (deltaSoc < 0.0) {
    adaptiveLNS_.acceptedWorse[alnsHeuristicForIter]++;
  }
  if (acceptedAsWorse) {
    adaptiveLNS_.downgradedAccepted[alnsHeuristicForIter]++;
  } else {
    adaptiveLNS_.improvedAccepted[alnsHeuristicForIter]++;
  }
  if (feasibleSolutionUpdated) {
    adaptiveLNS_.bestUpdates[alnsHeuristicForIter]++;
    adaptiveLNS_.bestUpdateDeltaSocSum[alnsHeuristicForIter] += deltaSoc;
  }
}

void LNS::applyAcceptanceOutcome(const AcceptanceDecisionResult& decisionResult, bool candidateValid,
    bool feasibleSolutionUpdated, bool nrrRepairSucceeded,
    int previousSocForIter, int proposedSocForIter, int incumbentSocBeforeIter,
    int previousConflictSignalForIter, int candidateConflictSignal,
    const LNS::ValidationStats& candidateValidationStats,
    ConflictMap& potentialNeighborhood, const ConflictMap& oldNeighborhood,
    bool& currentSolutionValid, LNS::ValidationStats& currentValidationStats,
    IterationQuality& quality, const std::vector<int>& candidateTouchedAgents,
    LNS::IterationDebugRecord& debugRow) {
  const bool accepted = decisionResult.accepted;
  const bool acceptedAsWorse = decisionResult.acceptedAsWorse;
  debugRow.softRecoveryModeBefore = decisionResult.softRecoveryModeBefore;
  debugRow.softRecoveryModeAfter = decisionResult.softRecoveryModeAfter;
  if (!decisionResult.softRecoveryDecisionReason.empty()) {
    debugRow.softRecoveryDecisionReason =
        decisionResult.softRecoveryDecisionReason;
  }
  if (accepted) {
    debugRow.accepted = true;
    debugRow.acceptedAsWorseUtility = acceptedAsWorse;
    if (debugRow.destroySelectedInSoftMode && debugRow.destroyHeuristicId >= 0 &&
        debugRow.destroyHeuristicId <
            (int)nrrStats_.softModeAcceptedByDestroy.size()) {
      nrrStats_.softModeAcceptedByDestroy[debugRow.destroyHeuristicId]++;
      improvementDiagnosticsStats_
          .softModeAcceptedByDestroy[debugRow.destroyHeuristicId]++;
      if (feasibleSolutionUpdated) {
        nrrStats_.softModeBestUpdatesByDestroy[debugRow.destroyHeuristicId]++;
        improvementDiagnosticsStats_
            .softModeBestUpdatesByDestroy[debugRow.destroyHeuristicId]++;
      }
    }
    improvementDiagnosticsStats_.accepted++;
    if (acceptedAsWorse) {
      improvementDiagnosticsStats_.acceptedAsWorseUtility++;
    } else {
      improvementDiagnosticsStats_.acceptedAsBetterOrEqualUtility++;
    }
    if (proposedSocForIter < previousSocForIter) {
      improvementDiagnosticsStats_.acceptedSocBetterVsPrevious++;
    } else if (proposedSocForIter > previousSocForIter) {
      improvementDiagnosticsStats_.acceptedSocWorseVsPrevious++;
    } else {
      improvementDiagnosticsStats_.acceptedSocEqualVsPrevious++;
    }
    if (incumbentSocBeforeIter != std::numeric_limits<int>::max()) {
      if (proposedSocForIter < incumbentSocBeforeIter) {
        improvementDiagnosticsStats_.acceptedSocBetterVsIncumbent++;
      } else if (proposedSocForIter > incumbentSocBeforeIter) {
        improvementDiagnosticsStats_.acceptedSocWorseVsIncumbent++;
      } else {
        improvementDiagnosticsStats_.acceptedSocEqualVsIncumbent++;
      }
    }
    if (candidateValid && !feasibleSolutionUpdated) {
      improvementDiagnosticsStats_.acceptedFeasibleNoBestUpdate++;
    }
    if (!candidateValid) {
      improvementDiagnosticsStats_.acceptedInvalid++;
    }
    if (candidateConflictSignal < previousConflictSignalForIter) {
      improvementDiagnosticsStats_.acceptedConflictSignalBetter++;
    } else if (candidateConflictSignal > previousConflictSignalForIter) {
      improvementDiagnosticsStats_.acceptedConflictSignalWorse++;
    } else {
      improvementDiagnosticsStats_.acceptedConflictSignalEqual++;
    }

    const uint64_t fingerprint = computeSolutionFingerprint(solution_);
    debugRow.fingerprint = fingerprint;
    const auto [_, inserted] =
        acceptanceState_.acceptedSolutionFingerprints.insert(fingerprint);
    debugRow.fingerprintSeenBefore = !inserted;
    if (inserted) {
      improvementDiagnosticsStats_.acceptedFingerprintUnique++;
    } else {
      improvementDiagnosticsStats_.acceptedFingerprintRepeat++;
    }

    improvementDiagnosticsStats_.acceptedRemovedTasksTotal +=
        debugRow.removedTasks;
    if (debugRow.removedTasksChangedAgent >= 0) {
      improvementDiagnosticsStats_.acceptedRemovedTasksChangedAgentTotal +=
          debugRow.removedTasksChangedAgent;
      improvementDiagnosticsStats_.acceptedRemovedTasksChangedOrderTotal +=
          debugRow.removedTasksChangedOrder;
      improvementDiagnosticsStats_.acceptedRemovedTasksUnchangedTotal +=
          debugRow.removedTasksUnchanged;
      const int acceptedChangedRemovedTasks =
          debugRow.removedTasksChangedAgent + debugRow.removedTasksChangedOrder;
      improvementDiagnosticsStats_.acceptedChangedTasksPerNeighborhoodMax =
          std::max(
              improvementDiagnosticsStats_.acceptedChangedTasksPerNeighborhoodMax,
              (int64_t)acceptedChangedRemovedTasks);
    }
    improvementDiagnosticsStats_.acceptedRemovedTasksMax =
        std::max(improvementDiagnosticsStats_.acceptedRemovedTasksMax,
                 (int64_t)debugRow.removedTasks);
    if (nrrRepairSucceeded) {
      nrrStats_.acceptedDeltaSocSum +=
          static_cast<double>(proposedSocForIter - previousSocForIter);
      nrrStats_.acceptedDeltaSocCount++;
    }
  } else {
    debugRow.accepted = false;
    debugRow.acceptedAsWorseUtility = false;
    improvementDiagnosticsStats_.rejected++;
    if (decisionResult.usedSoftRecoveryOverride &&
        decisionResult.softRecoveryDecisionReason != "none") {
      debugRow.earlyAbortReason = decisionResult.softRecoveryDecisionReason;
    }
    if (decisionResult.invalidGuardRejected) {
      debugRow.earlyAbortReason = "invalid_candidate_guard_reject";
    }
    if (decisionResult.guardRejected) {
      debugRow.guardRejected = true;
      improvementDiagnosticsStats_.guardRejected++;
    }
  }

  if (!accepted) {
    quality = IterationQuality::none;
    if (acceptanceState_.forceNeighborhoodChangeOnReject) {
      potentialNeighborhood.clear();
    } else {
      potentialNeighborhood = oldNeighborhood;
    }
    return;
  }

  if (acceptedAsWorse) {
    quality = IterationQuality::downgradedButAccepted;
  } else {
    quality = IterationQuality::improvedSolution;
  }
  snapshotPreviousFromCurrent(candidateTouchedAgents);
  if (softRecoveryState_.persistentConflictGraph) {
    softRecoveryState_.persistentConflictPairs = softRecoveryState_.lastValidationCollisionPairs;
    softRecoveryState_.persistentConflictAgents = softRecoveryState_.lastValidationConflictAgents;
  }
  currentSolutionValid = candidateValid;
  currentValidationStats = candidateValidationStats;
}
