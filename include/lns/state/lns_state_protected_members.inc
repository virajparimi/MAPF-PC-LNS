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
  struct CascadeState {
    double maxFactor = 0.0;
    int maxTasks = 0;
    bool adaptiveBudgetEnabled = false;
    int adaptiveBudgetCurrent = 0;
    int adaptiveBudgetLastUsed = 0;
    bool lastPrepareAborted = false;
    int lastPrepareSeedTasks = 0;
    int lastPrepareClosureTasks = 0;
    int lastPrepareClosureAdded = 0;
    vector<int> lastPrepareAffectedAgents;
    CascadeStats stats;
  };
  CascadeState cascadeState_;
  bool alnsEnablePrecedenceAwareDestroy_ = true;
  bool useTerminalPathsInValidation_ = false;
  SolutionRestoreStats solutionRestoreStats_;
  ImprovementDiagnosticsStats improvementDiagnosticsStats_;
  vector<pair<int, int>> fullPrecedenceConstraintsScratch_;
  vector<vector<int>> fullPrecedenceAssignmentsSnapshot_;
  bool fullPrecedenceConstraintsScratchValid_ = false;
  Neighbor lnsNeighborhood_;
  const Instance& instance_;
  unsigned int seed_ = 0;
  std::mt19937 rng_;
  vector<AgentTaskPath> initialPaths_;
  FeasibleSolution incumbentSolution_;
  Solution solution_, previousSolution_;
  struct LowLevelState {
    LowLevelSearchStats counters;
    uint64_t timeoutDiagnosticsLogsEmitted = 0;
    SingleAgentSolver::SearchOutcome lastOutcome =
        SingleAgentSolver::SearchOutcome::unknown;
    double lastRemainingBudgetSec = 0.0;
    double lastEffectiveTimeoutSec = 0.0;
    // Reusable per-agent scratch planners for hot regret/insert loops.
    // Single-threaded use only.
    vector<std::unique_ptr<SingleAgentSolver>> reusableLocalPlanners;
  };
  LowLevelState lowLevelState_;
  double initialTemperature_ = 0.0;
  bool lowLevelStructuralPrePrune_ = false;
  struct AcceptanceState {
    bool acceptOnlyValidCandidates = false;
    bool forceNeighborhoodChangeOnReject = false;
    bool softRecoveryActive = false;
    int64_t softRecoveryCurrentConflicts = -1;
    double temperature = 100.0;
    double coolingCoefficient = 0.99975;
    double heatingCoefficient = 1.00025;
    double maxTemperature = std::numeric_limits<double>::infinity();
    double greatDelugeDecay = 0.0;
    unordered_set<uint64_t> acceptedSolutionFingerprints;
  };
  AcceptanceState acceptanceState_;
  double timeLimit_, initialSolutionRuntime_ = 0,
                     initialSolutionRuntimeReported_ = 0,
                     lnsLoopRuntimeSec_ = 0, postRefineRuntimeSec_ = 0,
                     cumulativeRegretCandidateEvalSec_ = 0,
                     cumulativeRegretCommitSec_ = 0,
                     cumulativeLowLevelSearchSec_ = 0, tolerance_ = 5,
                     shawDistanceWeight_ = 9, shawTemporalWeight_ = 3,
                     lnsConflictWeight_ = 0.75, lnsCostWeight_ = 0.25;
  Time::time_point plannerStartTime_;
  string optimizationObjective_ = "soc";
  OptimizationObjectiveMode optimizationObjectiveMode_ =
      OptimizationObjectiveMode::soc;
  string goalOccupationMode_ = "reposition_true";
  GoalOccupationMode goalOccupationModeMode_ = GoalOccupationMode::reposition_true;
  // SINGLE-THREADED ONLY: mutable cache updated from const accessors.
  // Not safe for concurrent reads/writes without external synchronization.
  mutable unordered_map<int, vector<int>> parkingCandidatesCache_;
  TerminalRepositionStats terminalRepositionStats_;
  string initialSolutionRequested_;
  string initialSolutionEffective_;
  bool initialSolutionFallbackUsed_ = false;
  string initialSolutionFallbackReason_;
  double initialSeedRuntimeFromLogSec_ = -1.0;
  bool initialAssignmentSnapshotAvailable_ = false;
  bool initialMetricsAvailable_ = false;
  int initialObjectiveValue_ = std::numeric_limits<int>::max();
  int initialSoc_ = std::numeric_limits<int>::max();
  int initialMakespan_ = -1;
  double initialPrecedenceWait_ = std::numeric_limits<double>::quiet_NaN();
  vector<vector<int>> initialAssignmentsByAgent_;
  vector<int> initialTaskOwnerByTask_;
  vector<int> initialTaskPosByTask_;
  double initialPortfolioTimeFraction_ = 0.10;
  bool adaptiveInitialPortfolioBudget_ = false;
  string initialSeedFromMapfpcLog_;
  // SINGLE-THREADED ONLY: mutable assignment-index caches updated from const
  // query paths. Not thread-safe without external synchronization.
  mutable vector<int> currentTaskOwnerByTaskCache_;
  mutable vector<int> currentTaskPosByTaskCache_;
  mutable bool currentTaskAssignmentIndexCacheValid_ = false;
  bool postRefineWithMapfpc_ = false;
  bool postRefineAttempted_ = false;
  bool postRefineAccepted_ = false;
  string postRefineAssignmentSource_ = "solution";
  string postRefineAssignmentLog_;
  string postRefineSolver_ = "pbs";
  int postRefineTimeoutSec_ = 120;
  bool postRefineAcceptOnlyIfBetter_ = true;
  bool enableNrrRepair_ = false;
  bool nrrFallbackToStandard_ = true;
  bool nrrGlobalReassign_ = false;
  struct SoftRecoveryState {
    bool destroyMode = false;
    bool persistentConflictGraph = false;
    bool lastDestroySampledInSoftMode = false;
    vector<pair<int, int>> persistentConflictPairs;
    vector<int> persistentConflictAgents;
    vector<pair<int, int>> lastValidationCollisionPairs;
    vector<int> lastValidationConflictAgents;
    vector<int> lastValidationConflictTasks;
    vector<int> lastSoftFailureConflictAgents;
    vector<int> lastSoftFailureConflictTasks;
  };
  SoftRecoveryState softRecoveryState_;
  string nrrMiniSolver_ = "cbs";
  NrrMiniSolverMode nrrMiniSolverMode_ = NrrMiniSolverMode::cbs;
  string nrrCatBackend_ = "pathtablewc";
  NrrStats nrrStats_;
  bool lastNrrSoftCandidate_ = false;
  int64_t lastNrrSoftConflictCount_ = -1;
  bool lastNrrSoftOnlyInvalid_ = false;
  string repairMapfpcSolver_ = "cbs";
  int repairMapfpcTimeoutSec_ = 30;
  RepairHeuristicMode repairHeuristicMode_ = RepairHeuristicMode::regret;
  RegretTypeMode regretTypeMode_ = RegretTypeMode::absolute;
