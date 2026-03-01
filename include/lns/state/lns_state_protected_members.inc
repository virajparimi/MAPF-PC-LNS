  ALNS adaptiveLNS_;
  int neighborSize_;
  int regretCandidateTopK_ = 0;
  bool regretShortlistDiagnostics_ = false;
  struct SuccessorPressureStaticSignal {
    int successorTask = UNASSIGNED;
    int depth = 1;
  };
  vector<vector<SuccessorPressureStaticSignal>>
      successorPressureStaticSignalsByTask_;
  double maxCascadeFactor_ = 0.0;
  int maxCascadeTasks_ = 0;
  bool adaptiveCascadeBudget_ = false;
  int adaptiveCascadeBudgetCurrent_ = 0;
  int adaptiveCascadeBudgetLastUsed_ = 0;
  bool alnsEnablePrecedenceAwareDestroy_ = true;
  bool useTerminalPathsInValidation_ = false;
  bool lastPrepareAbortedByCascade_ = false;
  int lastPrepareSeedTasks_ = 0;
  int lastPrepareClosureTasks_ = 0;
  int lastPrepareClosureAdded_ = 0;
  vector<int> lastPrepareAffectedAgents_;
  CascadeStats cascadeStats_;
  SolutionRestoreStats solutionRestoreStats_;
  ImprovementDiagnosticsStats improvementDiagnosticsStats_;
  vector<pair<int, int>> fullPrecedenceConstraintsScratch_;
  Neighbor lnsNeighborhood_;
  const Instance& instance_;
  unsigned int seed_ = 0;
  std::mt19937 rng_;
  vector<AgentTaskPath> initialPaths_;
  FeasibleSolution incumbentSolution_;
  Solution solution_, previousSolution_;
  uint64_t lowLevelCalls_ = 0;
  uint64_t lowLevelExpanded_ = 0;
  uint64_t lowLevelGenerated_ = 0;
  uint64_t lowLevelFound_ = 0;
  uint64_t lowLevelTimeout_ = 0;
  uint64_t lowLevelSearchExhausted_ = 0;
  uint64_t lowLevelInvalidInput_ = 0;
  uint64_t lowLevelBudgetExhausted_ = 0;
  uint64_t lowLevelUnknown_ = 0;
  uint64_t lowLevelTimeoutGoalPermanentBeforeArrivalLb_ = 0;
  uint64_t lowLevelTimeoutStartTrappedAtTPlus1_ = 0;
  uint64_t lowLevelTimeoutStaticDisconnectedPermanent_ = 0;
  uint64_t lowLevelTimeoutOther_ = 0;
  uint64_t lowLevelTimeoutReducedByGlobalBudget_ = 0;
  uint64_t lowLevelTimeoutMultiCertificate_ = 0;
  uint64_t lowLevelTimeoutDiagnosticsLogsEmitted_ = 0;
  uint64_t lowLevelStructuralPrePruned_ = 0;
  uint64_t lowLevelStructuralPrePrunedGoalPermanentBeforeArrivalLb_ = 0;
  uint64_t lowLevelStructuralPrePrunedStartTrappedAtTPlus1_ = 0;
  uint64_t lowLevelStructuralPrePrunedStaticDisconnectedPermanent_ = 0;
  uint64_t lowLevelStructuralPrePrunedMultiCertificate_ = 0;
  SingleAgentSolver::SearchOutcome lastLowLevelOutcome_ =
      SingleAgentSolver::SearchOutcome::unknown;
  double lastLowLevelRemainingBudgetSec_ = 0.0;
  double lastLowLevelEffectiveTimeoutSec_ = 0.0;
  // Reusable per-agent scratch planners for hot regret/insert loops.
  // Single-threaded use only.
  vector<std::unique_ptr<SingleAgentSolver>> reusableLocalPlanners_;
  double initialTemperature_ = 0.0;
  bool lowLevelStructuralPrePrune_ = false;
  bool acceptOnlyValidCandidates_ = false;
  double maxTemperature_ = std::numeric_limits<double>::infinity();
  double greatDelugeDecay_ = 0.0;
  double timeLimit_, initialSolutionRuntime_ = 0, temperature_ = 100,
                     coolingCoefficient_ = 0.99975,
                     heatingCoefficient_ = 1.00025, tolerance_ = 5,
                     shawDistanceWeight_ = 9, shawTemporalWeight_ = 3,
                     lnsConflictWeight_ = 0.75, lnsCostWeight_ = 0.25;
  Time::time_point plannerStartTime_;
  string goalOccupationMode_ = "reposition_true";
  mutable unordered_map<int, vector<int>> parkingCandidatesCache_;
  TerminalRepositionStats terminalRepositionStats_;
  string initialSolutionRequested_;
  string initialSolutionEffective_;
  bool initialSolutionFallbackUsed_ = false;
  string initialSolutionFallbackReason_;
  double initialPortfolioTimeFraction_ = 0.10;
  bool adaptiveInitialPortfolioBudget_ = false;
  string initialSeedFromPbsLog_;
  bool postRefineWithMapfpc_ = false;
  string postRefineAssignmentSource_ = "solution";
  string postRefineAssignmentLog_;
  string postRefineSolver_ = "pbs";
  int postRefineTimeoutSec_ = 120;
  bool postRefineAcceptOnlyIfBetter_ = true;
  bool enableNrrRepair_ = false;
  bool nrrFallbackToStandard_ = true;
  bool forceNeighborhoodChangeOnReject_ = false;
  string nrrMiniSolver_ = "cbs";
  NrrStats nrrStats_;
  string repairMapfpcSolver_ = "cbs";
  int repairMapfpcTimeoutSec_ = 30;
