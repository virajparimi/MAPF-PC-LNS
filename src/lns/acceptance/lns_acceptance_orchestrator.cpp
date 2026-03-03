#include "lns_acceptance_orchestrator.hpp"
#include "lns_market_iteration_orchestrator.hpp"

#include <algorithm>
#include <limits>

AcceptanceDecisionResult AcceptanceOrchestrator::runDecision(
    LNS& lns, bool candidateValid, double previousPressureForIter,
    double previousWaitForIter, int previousConflictSignalForIter,
    LNS::IterationDebugRecord& debugRow) {
  AcceptanceDecisionResult result;
  auto elapsedSecSince = [](const Time::time_point& startTimePoint) -> double {
    return ((fsec)(Time::now() - startTimePoint)).count();
  };
  auto advanceTemperatureOnGuardReject = [&]() {
    if (lns.acceptanceCriteria == "SA" || lns.acceptanceCriteria == "TA") {
      lns.temperature_ *= lns.coolingCoefficient_;
    } else if (lns.acceptanceCriteria == "OBA") {
      const double reheated = lns.temperature_ * lns.heatingCoefficient_;
      lns.temperature_ = std::min(reheated, lns.maxTemperature_);
    } else if (lns.acceptanceCriteria == "GDA") {
      lns.temperature_ =
          std::max(0.0, lns.temperature_ - lns.greatDelugeDecay_);
    }
  };
  auto finalizeDecision = [&](const Time::time_point& acceptanceStart) {
    result.softRecoveryModeAfter = lns.softRecoveryActive_;
    result.timeAcceptanceSec += elapsedSecSince(acceptanceStart);
    return result;
  };

  const Time::time_point acceptanceStart = Time::now();
  result.softRecoveryModeBefore = lns.softRecoveryActive_;
  result.softRecoveryModeAfter = lns.softRecoveryActive_;
  const bool candidateIsNrrSoftCandidate =
      (!candidateValid && lns.lastNrrSoftCandidate_ &&
       lns.lastNrrSoftOnlyInvalid_ && lns.lastNrrSoftConflictCount_ > 0);
  if (!lns.acceptOnlyValidCandidates_ &&
      (lns.softRecoveryActive_ || candidateIsNrrSoftCandidate)) {
    if (!lns.softRecoveryActive_ && candidateIsNrrSoftCandidate &&
        previousConflictSignalForIter == 0) {
      lns.softRecoveryActive_ = true;
      lns.softRecoveryCurrentConflicts_ = lns.lastNrrSoftConflictCount_;
      lns.lastSoftFailureConflictAgents_.clear();
      lns.lastSoftFailureConflictTasks_.clear();
      lns.nrrStats_.softRecoveryEntries++;
      lns.improvementDiagnosticsStats_.softRecoveryEntries++;
      lns.nrrStats_.softModeEntryConflictSum += lns.lastNrrSoftConflictCount_;
      lns.improvementDiagnosticsStats_.softModeEntryConflictSum +=
          lns.lastNrrSoftConflictCount_;
      result.accepted = true;
      result.acceptedAsWorse =
          lns.previousSolution_.utility < lns.solution_.utility;
      result.usedSoftRecoveryOverride = true;
      result.softRecoveryDecisionReason = "soft_entry_accept";
      lns.nrrStats_.softCandidatesAccepted++;
      lns.improvementDiagnosticsStats_.softCandidateAccepted++;
      lns.nrrStats_.softRecoveryAcceptedEntry++;
      lns.improvementDiagnosticsStats_.softRecoveryAcceptedEntry++;
      return finalizeDecision(acceptanceStart);
    }

    if (lns.softRecoveryActive_) {
      if (candidateValid) {
        lns.softRecoveryActive_ = false;
        lns.softRecoveryCurrentConflicts_ = -1;
        lns.nrrStats_.softRecoveryExits++;
        lns.improvementDiagnosticsStats_.softRecoveryExits++;
        lns.nrrStats_.softModeExitConflictSum += 0;
        lns.improvementDiagnosticsStats_.softModeExitConflictSum += 0;
        lns.nrrStats_.softModeResolvedEntries++;
        lns.improvementDiagnosticsStats_.softModeResolvedEntries++;
        result.accepted = true;
        result.acceptedAsWorse =
            lns.previousSolution_.utility < lns.solution_.utility;
        result.usedSoftRecoveryOverride = true;
        result.softRecoveryDecisionReason = "soft_exit_on_valid_accept";
        return finalizeDecision(acceptanceStart);
      }

      if (candidateIsNrrSoftCandidate) {
        const int64_t candidateSoftConflicts = lns.lastNrrSoftConflictCount_;
        if (candidateSoftConflicts <= lns.softRecoveryCurrentConflicts_) {
          lns.softRecoveryCurrentConflicts_ = candidateSoftConflicts;
          result.accepted = true;
          result.acceptedAsWorse =
              lns.previousSolution_.utility < lns.solution_.utility;
          result.usedSoftRecoveryOverride = true;
          result.softRecoveryDecisionReason = "soft_descent_accept";
          lns.nrrStats_.softCandidatesAccepted++;
          lns.improvementDiagnosticsStats_.softCandidateAccepted++;
          lns.nrrStats_.softRecoveryAcceptedDescent++;
          lns.improvementDiagnosticsStats_.softRecoveryAcceptedDescent++;
          return finalizeDecision(acceptanceStart);
        }
        lns.restoreSolutionFromPrevious();
        result.accepted = false;
        result.usedSoftRecoveryOverride = true;
        result.softRecoveryDecisionReason = "soft_descent_reject_non_improving";
        lns.lastSoftFailureConflictAgents_ = lns.lastValidationConflictAgents_;
        lns.lastSoftFailureConflictTasks_ = lns.lastValidationConflictTasks_;
        lns.nrrStats_.softCandidatesRejected++;
        lns.improvementDiagnosticsStats_.softCandidateRejected++;
        lns.nrrStats_.softRecoveryRejected++;
        lns.improvementDiagnosticsStats_.softRecoveryRejected++;
        advanceTemperatureOnGuardReject();
        return finalizeDecision(acceptanceStart);
      }
      lns.restoreSolutionFromPrevious();
      result.accepted = false;
      result.usedSoftRecoveryOverride = true;
      result.softRecoveryDecisionReason = "soft_mode_reject_nonsoft_invalid";
      lns.nrrStats_.softRecoveryRejected++;
      lns.improvementDiagnosticsStats_.softRecoveryRejected++;
      advanceTemperatureOnGuardReject();
      return finalizeDecision(acceptanceStart);
    } else if (candidateIsNrrSoftCandidate) {
      lns.restoreSolutionFromPrevious();
      result.accepted = false;
      result.usedSoftRecoveryOverride = true;
      result.softRecoveryDecisionReason = "soft_entry_reject_prev_nonzero";
      lns.nrrStats_.softCandidatesRejected++;
      lns.improvementDiagnosticsStats_.softCandidateRejected++;
      advanceTemperatureOnGuardReject();
      return finalizeDecision(acceptanceStart);
    }
  }

  if (!candidateValid && !lns.acceptOnlyValidCandidates_) {
    PLOGD << "Invalid candidate forwarded to acceptance criteria\n";
  }
  if (lns.acceptOnlyValidCandidates_ && !candidateValid) {
    lns.restoreSolutionFromPrevious();
    result.accepted = false;
    result.invalidGuardRejected = true;
    if (candidateIsNrrSoftCandidate) {
      lns.nrrStats_.softCandidatesRejected++;
      lns.improvementDiagnosticsStats_.softCandidateRejected++;
      if (result.softRecoveryDecisionReason == "none") {
        result.softRecoveryDecisionReason = "soft_reject_strict_valid_guard";
      }
    }
    advanceTemperatureOnGuardReject();
    PLOGD << "Rejecting this solution due to strict-valid acceptance guard\n";
    return finalizeDecision(acceptanceStart);
  }

  const MarketGuardDecision guardDecision =
      MarketIterationOrchestrator::evaluateGuards(
          lns, previousPressureForIter, previousWaitForIter);
  result.candidatePressure = guardDecision.candidatePressure;
  result.candidateWait = guardDecision.candidateWait;
  if (!guardDecision.allowed) {
    lns.restoreSolutionFromPrevious();
    result.accepted = false;
    result.guardRejected = guardDecision.guardRejected;
    advanceTemperatureOnGuardReject();
    PLOGD << "Rejecting this solution due to market acceptance guards\n";
    return finalizeDecision(acceptanceStart);
  }

  if (lns.acceptanceCriteria == "SA") {
    result.accepted = lns.simulatedAnnealing();
  } else if (lns.acceptanceCriteria == "TA") {
    result.accepted = lns.thresholdAcceptance();
  } else if (lns.acceptanceCriteria == "OBA") {
    result.accepted = lns.oldBachelorsAcceptance();
  } else if (lns.acceptanceCriteria == "GDA") {
    result.accepted = lns.greatDelugeAlgorithm();
  } else {
    PLOGE << "Unknown acceptance criteria: " << lns.acceptanceCriteria << "\n";
    debugRow.earlyAbortReason = "acceptance_criteria_error";
    result.ok = false;
    return finalizeDecision(acceptanceStart);
  }
  if (result.accepted) {
    result.acceptedAsWorse =
        lns.previousSolution_.utility < lns.solution_.utility;
    if (lns.softRecoveryActive_ && candidateValid) {
      lns.softRecoveryActive_ = false;
      lns.softRecoveryCurrentConflicts_ = -1;
      lns.nrrStats_.softRecoveryExits++;
      lns.improvementDiagnosticsStats_.softRecoveryExits++;
      lns.nrrStats_.softModeExitConflictSum += 0;
      lns.improvementDiagnosticsStats_.softModeExitConflictSum += 0;
      lns.nrrStats_.softModeResolvedEntries++;
      lns.improvementDiagnosticsStats_.softModeResolvedEntries++;
      if (result.softRecoveryDecisionReason == "none") {
        result.softRecoveryDecisionReason = "soft_exit_on_valid_accept";
      }
    }
  }
  return finalizeDecision(acceptanceStart);
}

void AcceptanceOrchestrator::updateAlnsStats(
    LNS& lns, int alnsHeuristicForIter, int previousSocForIter,
    int proposedSocForIter, bool candidateValid, bool accepted,
    bool acceptedAsWorse, bool feasibleSolutionUpdated) {
  if (alnsHeuristicForIter < 0 ||
      alnsHeuristicForIter >= lns.adaptiveLNS_.numDestroyHeuristics) {
    return;
  }
  const double deltaSoc = (double)(previousSocForIter - proposedSocForIter);
  lns.adaptiveLNS_.deltaSocAll[alnsHeuristicForIter] += deltaSoc;
  if (deltaSoc > 0.0) {
    lns.adaptiveLNS_.proposedBetter[alnsHeuristicForIter]++;
  } else if (deltaSoc < 0.0) {
    lns.adaptiveLNS_.proposedWorse[alnsHeuristicForIter]++;
  } else {
    lns.adaptiveLNS_.proposedEqual[alnsHeuristicForIter]++;
  }
  if (candidateValid) {
    lns.adaptiveLNS_.feasible[alnsHeuristicForIter]++;
  }
  if (!accepted) {
    lns.adaptiveLNS_.rejected[alnsHeuristicForIter]++;
    return;
  }
  lns.adaptiveLNS_.accepted[alnsHeuristicForIter]++;
  lns.adaptiveLNS_.deltaSocAccepted[alnsHeuristicForIter] += deltaSoc;
  if (deltaSoc < 0.0) {
    lns.adaptiveLNS_.acceptedWorse[alnsHeuristicForIter]++;
  }
  if (acceptedAsWorse) {
    lns.adaptiveLNS_.downgradedAccepted[alnsHeuristicForIter]++;
  } else {
    lns.adaptiveLNS_.improvedAccepted[alnsHeuristicForIter]++;
  }
  if (feasibleSolutionUpdated) {
    lns.adaptiveLNS_.bestUpdates[alnsHeuristicForIter]++;
  }
}

void AcceptanceOrchestrator::applyOutcome(
    LNS& lns, const AcceptanceDecisionResult& decisionResult, bool candidateValid,
    bool feasibleSolutionUpdated, bool nrrRepairSucceeded,
    int previousSocForIter, int proposedSocForIter, int incumbentSocBeforeIter,
    int previousConflictSignalForIter, int candidateConflictSignal,
    const LNS::ValidationStats& candidateValidationStats,
    ConflictMap& potentialNeighborhood, const ConflictMap& oldNeighborhood,
    bool& currentSolutionValid, LNS::ValidationStats& currentValidationStats,
    IterationQuality& quality, LNS::IterationDebugRecord& debugRow) {
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
            (int)lns.nrrStats_.softModeAcceptedByDestroy.size()) {
      lns.nrrStats_.softModeAcceptedByDestroy[debugRow.destroyHeuristicId]++;
      lns.improvementDiagnosticsStats_
          .softModeAcceptedByDestroy[debugRow.destroyHeuristicId]++;
      if (feasibleSolutionUpdated) {
        lns.nrrStats_.softModeBestUpdatesByDestroy[debugRow.destroyHeuristicId]++;
        lns.improvementDiagnosticsStats_
            .softModeBestUpdatesByDestroy[debugRow.destroyHeuristicId]++;
      }
    }
    lns.improvementDiagnosticsStats_.accepted++;
    if (acceptedAsWorse) {
      lns.improvementDiagnosticsStats_.acceptedAsWorseUtility++;
    } else {
      lns.improvementDiagnosticsStats_.acceptedAsBetterOrEqualUtility++;
    }
    if (proposedSocForIter < previousSocForIter) {
      lns.improvementDiagnosticsStats_.acceptedSocBetterVsPrevious++;
    } else if (proposedSocForIter > previousSocForIter) {
      lns.improvementDiagnosticsStats_.acceptedSocWorseVsPrevious++;
    } else {
      lns.improvementDiagnosticsStats_.acceptedSocEqualVsPrevious++;
    }
    if (incumbentSocBeforeIter != std::numeric_limits<int>::max()) {
      if (proposedSocForIter < incumbentSocBeforeIter) {
        lns.improvementDiagnosticsStats_.acceptedSocBetterVsIncumbent++;
      } else if (proposedSocForIter > incumbentSocBeforeIter) {
        lns.improvementDiagnosticsStats_.acceptedSocWorseVsIncumbent++;
      } else {
        lns.improvementDiagnosticsStats_.acceptedSocEqualVsIncumbent++;
      }
    }
    if (candidateValid && !feasibleSolutionUpdated) {
      lns.improvementDiagnosticsStats_.acceptedFeasibleNoBestUpdate++;
    }
    if (!candidateValid) {
      lns.improvementDiagnosticsStats_.acceptedInvalid++;
    }
    if (candidateConflictSignal < previousConflictSignalForIter) {
      lns.improvementDiagnosticsStats_.acceptedConflictSignalBetter++;
    } else if (candidateConflictSignal > previousConflictSignalForIter) {
      lns.improvementDiagnosticsStats_.acceptedConflictSignalWorse++;
    } else {
      lns.improvementDiagnosticsStats_.acceptedConflictSignalEqual++;
    }

    const uint64_t fingerprint = lns.computeSolutionFingerprint(lns.solution_);
    debugRow.fingerprint = fingerprint;
    const auto [_, inserted] =
        lns.acceptedSolutionFingerprints_.insert(fingerprint);
    debugRow.fingerprintSeenBefore = !inserted;
    if (inserted) {
      lns.improvementDiagnosticsStats_.acceptedFingerprintUnique++;
    } else {
      lns.improvementDiagnosticsStats_.acceptedFingerprintRepeat++;
    }

    lns.improvementDiagnosticsStats_.acceptedRemovedTasksTotal +=
        debugRow.removedTasks;
    if (debugRow.removedTasksChangedAgent >= 0) {
      lns.improvementDiagnosticsStats_.acceptedRemovedTasksChangedAgentTotal +=
          debugRow.removedTasksChangedAgent;
      lns.improvementDiagnosticsStats_.acceptedRemovedTasksChangedOrderTotal +=
          debugRow.removedTasksChangedOrder;
      lns.improvementDiagnosticsStats_.acceptedRemovedTasksUnchangedTotal +=
          debugRow.removedTasksUnchanged;
      const int acceptedChangedRemovedTasks =
          debugRow.removedTasksChangedAgent + debugRow.removedTasksChangedOrder;
      lns.improvementDiagnosticsStats_.acceptedChangedTasksPerNeighborhoodMax =
          std::max(
              lns.improvementDiagnosticsStats_.acceptedChangedTasksPerNeighborhoodMax,
              (int64_t)acceptedChangedRemovedTasks);
    }
    lns.improvementDiagnosticsStats_.acceptedRemovedTasksMax =
        std::max(lns.improvementDiagnosticsStats_.acceptedRemovedTasksMax,
                 (int64_t)debugRow.removedTasks);
    if (nrrRepairSucceeded) {
      lns.nrrStats_.acceptedDeltaSocSum +=
          static_cast<double>(proposedSocForIter - previousSocForIter);
      lns.nrrStats_.acceptedDeltaSocCount++;
    }
  } else {
    debugRow.accepted = false;
    debugRow.acceptedAsWorseUtility = false;
    lns.improvementDiagnosticsStats_.rejected++;
    if (decisionResult.usedSoftRecoveryOverride &&
        decisionResult.softRecoveryDecisionReason != "none") {
      debugRow.earlyAbortReason = decisionResult.softRecoveryDecisionReason;
    }
    if (decisionResult.invalidGuardRejected) {
      debugRow.earlyAbortReason = "invalid_candidate_guard_reject";
    }
    if (decisionResult.guardRejected) {
      debugRow.guardRejected = true;
      lns.improvementDiagnosticsStats_.guardRejected++;
    }
  }

  if (!accepted) {
    quality = IterationQuality::none;
    if (lns.forceNeighborhoodChangeOnReject_) {
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
  lns.previousSolution_ = lns.solution_;
  currentSolutionValid = candidateValid;
  currentValidationStats = candidateValidationStats;
  MarketIterationOrchestrator::updateBestOnAccepted(
      lns, decisionResult.candidatePressure, decisionResult.candidateWait);
}
