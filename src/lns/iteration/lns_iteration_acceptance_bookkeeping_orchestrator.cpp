#include "lns_iteration_acceptance_bookkeeping_orchestrator.hpp"

#include "lns_acceptance_orchestrator.hpp"
#include <algorithm>
#include <limits>

bool IterationAcceptanceBookkeepingOrchestrator::run(
    LNS& lns, bool nrrRepairSucceeded, ConflictMap& potentialNeighborhood,
    const ConflictMap& oldNeighborhood, bool& currentSolutionValid,
    LNS::ValidationStats& currentValidationStats,
    IterationExecutionContext& context) {
  return lns.runAcceptanceAndBookkeeping(
      nrrRepairSucceeded, potentialNeighborhood, oldNeighborhood,
      currentSolutionValid, currentValidationStats, context);
}

bool LNS::runAcceptanceAndBookkeeping(
    bool nrrRepairSucceeded, ConflictMap& potentialNeighborhood,
    const ConflictMap& oldNeighborhood, bool& currentSolutionValid,
    ValidationStats& currentValidationStats,
    IterationExecutionContext& context) {
  const AcceptanceDecisionResult acceptanceDecision =
      AcceptanceOrchestrator::runDecision(
          *this, context.candidateValid, context.previousConflictSignalForIter,
          context.candidateTouchedAgents, context.debugRow);
  context.softRecoveryModeBefore = acceptanceDecision.softRecoveryModeBefore;
  context.softRecoveryModeAfter = acceptanceDecision.softRecoveryModeAfter;
  context.softRecoveryDecisionReason =
      acceptanceDecision.softRecoveryDecisionReason;
  context.timeAcceptanceSec += acceptanceDecision.timeAcceptanceSec;
  if (!acceptanceDecision.ok) {
    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    return false;
  }
  AcceptanceOrchestrator::updateAlnsStats(
      *this, context.alnsHeuristicForIter, context.previousSocForIter,
      context.proposedSocForIter, context.candidateValid,
      acceptanceDecision.accepted, acceptanceDecision.acceptedAsWorse,
      context.feasibleSolutionUpdated);
  AcceptanceOrchestrator::applyOutcome(
      *this, acceptanceDecision, context.candidateValid,
      context.feasibleSolutionUpdated, nrrRepairSucceeded,
      context.previousSocForIter, context.proposedSocForIter,
      context.incumbentSocBeforeIter, context.previousConflictSignalForIter,
      context.candidateConflictSignal, context.candidateValidationStats,
      potentialNeighborhood, oldNeighborhood, currentSolutionValid,
      currentValidationStats, context.quality, context.candidateTouchedAgents,
      context.debugRow);

  if (context.debugRow.earlyAbortReason.empty()) {
    context.debugRow.earlyAbortReason = "none";
  }
  const Time::time_point bookkeepingStart = Time::now();
  runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
  double costToLog = context.feasibleSolutionUpdated
                         ? incumbentObjectiveValueOrMax()
                         : currentObjectiveValue();
  int makespanToLog = -1;
  if (context.feasibleSolutionUpdated) {
    long long makespanLl = 0;
    for (const auto& agent : solution_.agents) {
      makespanLl = std::max(
          makespanLl, static_cast<long long>(agent.path.endTimeOrZero()));
    }
    makespanToLog =
        makespanLl > std::numeric_limits<int>::max()
            ? std::numeric_limits<int>::max()
            : static_cast<int>(makespanLl);
  }
  const double precedenceWaitToLog =
      context.feasibleSolutionUpdated ? computeSolutionPrecedenceWait()
                                      : std::numeric_limits<double>::quiet_NaN();
  appendIterationStatBounded(IterationStats(
      runtime, "LNS", instance_.getAgentNum(), instance_.getTasksNum(),
      costToLog, context.feasibleSolutionUpdated, context.quality, 0, 0,
      makespanToLog, precedenceWaitToLog));
  context.timeBookkeepingSec +=
      ((fsec)(Time::now() - bookkeepingStart)).count();
  return true;
}
