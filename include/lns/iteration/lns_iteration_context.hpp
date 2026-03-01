#pragma once

#include "lns.hpp"
#include "lns_iteration_phase_types.hpp"

// Mutable per-iteration state shared across phase orchestrators.
struct IterationExecutionContext {
  int previousSocForIter = 0;
  LNS::ValidationStats previousValidationStatsForIter;
  double previousPressureForIter = 0.0;
  double previousWaitForIter = 0.0;
  int incumbentSocBeforeIter = 0;
  int alnsHeuristicForIter = -1;
  IterationQuality quality = IterationQuality::none;
  bool collectIterationDebug = false;
  LNS::IterationDebugRecord debugRow;
  bool feasibleSolutionUpdated = false;

  double timeDestroyAndPrepareSec = 0.0;
  double timeRepairAndCommitSec = 0.0;
  double timeJoinPathsSec = 0.0;
  double timeTerminalReplanSec = 0.0;
  double timeRecomputeSocSec = 0.0;
  double timeValidationSec = 0.0;
  double timeAcceptanceSec = 0.0;
  double timeBookkeepingSec = 0.0;
  bool iterationTimingCommitted = false;
  bool iterationRowCommitted = false;

  LNS::ValidationStats candidateValidationStats;
  bool candidateValid = false;
  int candidateConflictSignal = 0;
  int previousConflictSignalForIter = 0;
  int proposedSocForIter = 0;
};
