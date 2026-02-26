#pragma once

#include <iosfwd>

#include "lns_params.hpp"
#include "lns_types.hpp"

class MovingMetrics;

class LNS {
 public:
  struct RegretWorkspace;
  struct ValidationStats;
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
    int64_t shortlistAgentEvaluations = 0;
    int64_t shortlistFallbackEvaluations = 0;
    int64_t shortlistFallbackRecovered = 0;
    int64_t waitProxyDiagEvaluations = 0;
    int64_t waitProxyDiagFiniteCandidates = 0;
    int64_t waitProxyDiagNonZeroCandidates = 0;
    int64_t waitProxyDiagPositiveEvals = 0;
    int64_t waitProxyDiagVaryingEvals = 0;
    int64_t waitProxyDiagNormalizedActiveEvals = 0;
    int64_t waitProxyDiagTop1Changed = 0;
    int64_t waitProxyDiagTopKChanged = 0;
    double waitProxyDiagTopKOverlapFracSum = 0.0;
    double waitProxyDiagMeanAbsWaitZSum = 0.0;
    double waitProxyDiagMeanAbsDistanceZSum = 0.0;
    int64_t successorPressureDiagEvaluations = 0;
    int64_t successorPressureDiagFiniteCandidates = 0;
    int64_t successorPressureDiagNonZeroCandidates = 0;
    int64_t successorPressureDiagPositiveEvals = 0;
    int64_t successorPressureDiagVaryingEvals = 0;
    int64_t successorPressureDiagNormalizedActiveEvals = 0;
    int64_t successorPressureDiagTop1Changed = 0;
    int64_t successorPressureDiagTopKChanged = 0;
    double successorPressureDiagTopKOverlapFracSum = 0.0;
    double successorPressureDiagMeanAbsPressureZSum = 0.0;
    double successorPressureDiagMeanAbsDistanceZSum = 0.0;
    int64_t successorPressureDiagPrevFallbackCount = 0;
    int64_t successorPressureDiagPrecedenceClampCount = 0;
    double successorPressureDiagPrecedenceClampDeltaSum = 0.0;
    int64_t successorPressureDiagDepth1Signals = 0;
    int64_t successorPressureDiagDepthGt1Signals = 0;
    int64_t successorPressureDiagDescendantActiveEvals = 0;
    double successorPressureDiagMeanDepth1ContributionSum = 0.0;
    double successorPressureDiagMeanDepthGt1ContributionSum = 0.0;
    int64_t workspaceAgentsCloned = 0;
    int64_t workspaceMaxClonedPerTask = 0;
    int64_t neighborhoods = 0;
    int64_t removedTasksSum = 0;
    int64_t removedTasksMax = 0;

    void reset() { *this = RegretEvalStats(); }
  };

  struct IncrementalRegretStats {
    int64_t commits = 0;
    int64_t heapRebuilds = 0;
    int64_t fullRefreshes = 0;
    int64_t stalePops = 0;
    int64_t recomputeCalls = 0;
    int64_t recomputedTasks = 0;
    int64_t dirtySum = 0;
    int64_t dirtyMax = 0;
    int64_t changedSum = 0;
    int64_t changedMax = 0;
    int64_t changedAgentsSum = 0;
    int64_t changedAgentsMax = 0;
    int64_t dirtyByDescendants = 0;
    int64_t dirtyByCandidateAgent = 0;
    int64_t dirtyByAncestors = 0;
    int64_t refreshByHighStale = 0;
    int64_t refreshByStaleGrowth = 0;
    int64_t refreshByPeriodic = 0;
    int64_t endgameFullRecomputes = 0;

    void reset() { *this = IncrementalRegretStats(); }
  };

  struct CascadeStats {
    int64_t prepareCalls = 0;
    int64_t budgetAborts = 0;
    int64_t budgetUsedSum = 0;
    int64_t budgetUsedMin = std::numeric_limits<int64_t>::max();
    int64_t budgetUsedMax = 0;
    int64_t adaptiveBudgetIncreases = 0;
    int64_t adaptiveBudgetDecreases = 0;
    int64_t seedTasksSum = 0;
    int64_t closureTasksSum = 0;
    int64_t closureAddedSum = 0;
    int64_t closureTasksMax = 0;
    int64_t closureAddedMax = 0;

    void reset() { *this = CascadeStats(); }
  };

  struct TerminalRepositionStats {
    int64_t replansRequested = 0;
    int64_t agentsEvaluated = 0;
    int64_t agentsPlanned = 0;
    int64_t skippedNoDemand = 0;
    int64_t planningFailures = 0;
    int64_t candidateCacheHits = 0;
    int64_t candidateCacheMisses = 0;

    void reset() { *this = TerminalRepositionStats(); }
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
    int64_t earlyAbortTerminal = 0;

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
    double timeJoinPathsSec = 0.0;
    double timeTerminalReplanSec = 0.0;
    double timeRecomputeSocSec = 0.0;
    double timeValidationSec = 0.0;
    double timeAcceptanceSec = 0.0;
    double timeBookkeepingSec = 0.0;

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
    string removedTaskIdsCsv;
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
    string earlyAbortReason;
    string quality;
    double timeDestroyAndPrepareSec = 0.0;
    double timeRepairAndCommitSec = 0.0;
    double timeJoinPathsSec = 0.0;
    double timeTerminalReplanSec = 0.0;
    double timeRecomputeSocSec = 0.0;
    double timeValidationSec = 0.0;
    double timeAcceptanceSec = 0.0;
    double timeBookkeepingSec = 0.0;
  };

 private:
  int numOfIterations_;
  bool debugImprovementDiagnostics_ = false;
  string debugIterationTsvPath_;
  unordered_set<uint64_t> acceptedSolutionFingerprints_;
  unordered_set<uint64_t> seenNeighborhoodFingerprints_;
  vector<int> previousNeighborhoodTasksSorted_;
  int currentNeighborhoodRepeatStreak_ = 0;
  vector<IterationDebugRecord> iterationDebugRecords_;
  bool incrementalRegret_ = false;
  LowLevelPlannerType lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  double lowLevelSegmentTimeout_ = 600.0;
  bool plannerParityCheck_ = false;
  struct MarketState : LNSParams::Market {
    // Runtime-only market state. Configuration fields are inherited from
    // LNSParams::Market to avoid duplicated declarations.
    int64_t acceptedCounter = 0;
    int64_t updateCounter = 0;
    double bestPressure = std::numeric_limits<double>::infinity();
    double bestWait = std::numeric_limits<double>::infinity();

    unordered_map<uint64_t, double> vertexPrices;
    unordered_map<uint64_t, double> edgePrices;
    unordered_map<uint64_t, double> vertexExcessHat;
    unordered_map<uint64_t, double> edgeExcessHat;
    unordered_set<uint64_t> prevContendedVertices;
    unordered_set<uint64_t> prevContendedEdges;
    bool hasStabilityBaseline = false;
    bool candidateUpdateConsumed = false;
    vector<int> taskCooldownUntilIter;
    MarketStats stats;
  };
  MarketState market_;

  struct TaskScheduleMetrics {
    bool valid = false;
    int arrive = 0;
    int release = 0;
    int start = 0;
    int end = 0;
    int waitPrec = 0;
    int blocker = UNASSIGNED;
  };

  enum class IncrementalRegretMode { descendants, descendants_and_agent };
  IncrementalRegretMode incrementalRegretMode_ =
      IncrementalRegretMode::descendants_and_agent;
  vector<uint32_t> regretStamp_;
  vector<pair<int, int>> regretBestOption_;
  vector<pair<int, int>> regretSecondBestOption_;
  vector<vector<int>> regretCandidateAgents_;

  RegretEvalStats regretEvalStatsCurrent_;
  RegretEvalStats regretEvalStatsTotal_;

  IncrementalRegretStats incrementalRegretStatsCurrent_;
  IncrementalRegretStats incrementalRegretStatsTotal_;

  void buildFullPrecedenceConstraints(
      vector<pair<int, int>>& out,
      bool includeIntraConstraints = true) const;
  vector<pair<int, int>> buildFullPrecedenceConstraints(
      bool includeIntraConstraints = true) const;
  void computeTaskScheduleMetricsFromIndex(
      const vector<int>& taskPosByTask, vector<TaskScheduleMetrics>& perTask,
      vector<double>* blockedWaitSum = nullptr) const;
  double computeSolutionPrecedenceWaitFromIndex(
      const vector<int>& taskPosByTask) const;
  AgentTaskPath runLowLevelSearch(SingleAgentSolver& solver,
                                  ConstraintTable& constraintTable,
                                  int startTime, int stage, int lowerBound);
  void recordLowLevelTimeoutDiagnostics(const SingleAgentSolver& solver,
                                        const ConstraintTable& constraintTable,
                                        int startTime, int stage, int lowerBound,
                                        double configuredTimeout,
                                        double effectiveTimeout);
  double elapsedRuntimeSec() const;
  double remainingRuntimeBudgetSec() const;
  bool runtimeBudgetExhausted() const;
  int cascadeTaskBudget() const;
  void clearNeighborhood();
  void restoreSolutionFromPrevious();
  uint64_t computeSolutionFingerprint(const Solution& solution) const;
  uint64_t computeNeighborhoodFingerprint(
      const ConflictMap& removedTasks) const;
  string iterationQualityName(IterationQuality quality) const;
  bool writeIterationDebugTsv(const string& outputPath) const;
  bool parseMAPFPCStreamIntoSolution(std::istream& inputStream,
                                     const string& sourceLabel);
  bool parseMAPFPCAssignmentsFromStream(
      std::istream& inputStream, const string& sourceLabel,
      vector<vector<int>>& outAssignments) const;
  bool writeMAPFPCAssignmentFile(const vector<vector<int>>& assignments,
                                 const string& outputPath,
                                 string& errorMessage) const;
  bool runMAPFPCForAgentFile(const string& agentFilePath,
                             const string& solverVariant,
                             int solverTimeoutSec,
                             const string& sourceLabel,
                             const string& fixedAssignmentFilePath = "");
  bool runMAPFPCOnAssignments(const vector<vector<int>>& assignments,
                              const string& solverVariant,
                              int solverTimeoutSec,
                              const string& sourceLabel);
  bool runPostMAPFPCRefinement();
  void overwriteIncumbentFromCurrentSolution();

 protected:
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
  void buildSuccessorPressureStaticSignals();
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
 public:
  double runtime = 0;
  int numOfFailures = 0, sumOfCosts = 0;
  int64_t invalidCandidateRejections = 0;
  int64_t marketGuardRejections = 0;
  vector<IterationStats> iterationStats;
  string initialSolutionStrategy, destroyHeuristic, acceptanceCriteria,
      repairHeuristic,
      regretType;

 private:
  void appendIterationStatBounded(const IterationStats& stat);
  bool runOneIteration(ConflictMap& potentialNeighborhood,
                       ConflictMap& oldNeighborhood, MovingMetrics& metrics,
                       bool& currentSolutionValid,
                       ValidationStats& currentValidationStats,
                       bool& feasibleSolutionUpdated);
  void reservePathWithGoalPolicy(ConstraintTable& constraintTable,
                                 const AgentTaskPath& path,
                                 bool isFinalTask,
                                 bool softOnly = false) const;
  int getServiceOccupancyEndExclusive(int agent) const;
  void reserveTerminalPathIfActive(ConstraintTable& constraintTable,
                                   int agent,
                                   bool softOnly = false) const;
  bool didAgentServicePathChange(int agent) const;
  vector<int> selectTerminalReplanAgents(
      const vector<int>& candidateAgents) const;
  const vector<int>& getParkingCandidatesForGoal(int finalGoal);

 public:

  LNS(int numOfIterations, const Instance& instance,
      const LNSParams& parameters);

  inline const Instance& getInstance() const { return instance_; }

  bool run();

  bool buildGreedySolution();
  bool buildPrioritizedInitialSolution();
  bool buildGreedySolutionWithMAPFPC(const string& variant,
                                     int solverTimeoutSec = 120);
  bool buildSeededSolutionFromMAPFPCLog(const string& logFilePath);
  bool planTerminalReposition(const vector<int>& agentsToPlan,
                              bool fullRebuild);

  bool prepareNextIteration();
  void markResolved(int globalTask);
  // Patches the task paths of an agent such that the begin times and end times match up
  void patchAgentTaskPaths(int agent, int taskPosition);

  void printPaths() const;
  enum class OccupancySource { undefined, service, terminal };
  OccupancySource getAgentOccupancySourceAt(
      int agent, int timestep, bool includeTerminal = true) const;
  // Returns an agent's occupied location at timestep.
  // If includeTerminal is false, occupancy follows service + goal policy
  // (stay/reposition_true), excluding explicit terminalPath.
  int getAgentLocationAt(int agent, int timestep,
                         bool includeTerminal = true) const;
  // Returns the occupancy horizon (exclusive upper bound) for collision checks.
  // Horizon is policy-aware for service occupancy and optionally includes
  // explicit terminalPath when includeTerminal is true.
  int getAgentOccupancyHorizon(int agent,
                               bool includeTerminal = true) const;
  struct ValidationStats {
    int precedenceViolations = 0;
    int vertexCollisions = 0;
    int edgeSwapCollisions = 0;
    int structuralViolations = 0;
    int64_t precedenceDebt = 0;
    int64_t precedencePairsChecked = 0;

    inline int totalConflictEvents() const {
      return precedenceViolations + vertexCollisions + edgeSwapCollisions +
             structuralViolations;
    }
    inline int spatialConflictEvents() const {
      return vertexCollisions + edgeSwapCollisions + structuralViolations;
    }
  };
  bool validateSolution(ConflictMap* conflictedTasks = nullptr,
                        ValidationStats* stats = nullptr);
  void addConflictingTask(int agent, int timestep, ConflictMap* out) const;

  bool buildConstraintTable(ConstraintTable& constraintTable, int task);
  bool buildConstraintTable(
      ConstraintTable& constraintTable, int task,
      const vector<pair<int, int>>& precedenceConstraints);

  bool buildConstraintTable(ConstraintTable& constraintTable,
                            TaskRegretPacket taskPacket, int taskLocation,
                            RegretWorkspace& workspace,
                            vector<pair<int, int>>* precedenceConstraints,
                            bool findingNextTask = false,
                            const vector<int>* assignmentOwnerLookup = nullptr,
                            const vector<int>* assignmentPosLookup = nullptr);

  int extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue);
  int extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue,
                               const vector<int>& newTaskQueue);
  vector<char> reachableSet(int source, const vector<vector<int>>& edgeList);

  bool computeRegret();
  bool computeRegretForTask(int task);
  bool computeRegretForTask(
      int task,
      const vector<pair<int, int>>& fullPrecedenceConstraints);
  void computeRegretForTaskWithAgent(
      TaskRegretPacket regretPacket, RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics& baselineMetrics,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
      const vector<int>* assignmentOwnerLookup = nullptr,
      const vector<int>* assignmentPosLookup = nullptr);

  bool recomputeRegretsForTasks(const vector<int>& tasks);
  std::optional<Regret> popNextValidRegret();
  vector<int> collectRemainingRemovedTasks() const;
  vector<int> computeCurrentTaskEndTimes() const;
  vector<int> computeCurrentLastTaskPerAgent() const;
  vector<uint64_t> computeCurrentAgentScheduleSignatures() const;
  vector<int> computeDirtyTasksAfterCommit(const vector<int>& endTimesBefore,
                                          const vector<int>& endTimesAfter,
                                          const vector<uint64_t>& agentSignaturesBefore,
                                          const vector<uint64_t>& agentSignaturesAfter);
  std::shared_ptr<SingleAgentSolver> createSharedPlanner(int agent) const;
  std::unique_ptr<SingleAgentSolver> createLocalPlanner(int agent) const;

  bool commitBestRegretTask(Regret bestRegret);
  bool commitAncestorTaskOf(
      int globalTask, std::optional<pair<bool, int>> committingNextTask);

  std::variant<bool, Utility> insertTask(
      TaskRegretPacket regretPacket,
      RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics* baselineMetrics = nullptr,
      SingleAgentSolver* reusablePlanner = nullptr,
      bool rollbackAfter = false);
  bool insertBestRegretTask(TaskRegretPacket bestRegretPacket);

  const Solution& getSolution() const { return solution_; }
  const string& getInitialSolutionRequested() const {
    return initialSolutionRequested_;
  }
  const string& getInitialSolutionEffective() const {
    return initialSolutionEffective_;
  }
  bool wasInitialSolutionFallbackUsed() const {
    return initialSolutionFallbackUsed_;
  }
  const string& getInitialSolutionFallbackReason() const {
    return initialSolutionFallbackReason_;
  }
  const ALNS& getAdaptiveLNSRef() const { return adaptiveLNS_; }
  ALNS getAdaptiveLNS() const { return adaptiveLNS_; }
  bool lastPrepareAbortedByCascade() const {
    return lastPrepareAbortedByCascade_;
  }
  int lastPrepareClosureAdded() const { return lastPrepareClosureAdded_; }
  const CascadeStats& getCascadeStatsRef() const { return cascadeStats_; }
  int getCascadeTaskBudget() const { return cascadeTaskBudget(); }
  bool isAdaptiveCascadeBudgetEnabled() const { return adaptiveCascadeBudget_; }
  int getAdaptiveCascadeBudgetCurrent() const {
    return adaptiveCascadeBudget_ ? adaptiveCascadeBudgetCurrent_
                                  : cascadeTaskBudget();
  }
  const SolutionRestoreStats& getSolutionRestoreStats() const {
    return solutionRestoreStats_;
  }
  bool isDebugImprovementDiagnosticsEnabled() const {
    return debugImprovementDiagnostics_;
  }
  const ImprovementDiagnosticsStats& getImprovementDiagnosticsStats() const {
    return improvementDiagnosticsStats_;
  }
  const TerminalRepositionStats& getTerminalRepositionStats() const {
    return terminalRepositionStats_;
  }
  LowLevelSearchStats getLowLevelSearchStats() const {
    return {lowLevelCalls_,         lowLevelExpanded_,      lowLevelGenerated_,
            lowLevelFound_,         lowLevelTimeout_,       lowLevelSearchExhausted_,
            lowLevelInvalidInput_,  lowLevelBudgetExhausted_,
            lowLevelUnknown_,        lowLevelTimeoutGoalPermanentBeforeArrivalLb_,
            lowLevelTimeoutStartTrappedAtTPlus1_,
            lowLevelTimeoutStaticDisconnectedPermanent_, lowLevelTimeoutOther_,
            lowLevelTimeoutReducedByGlobalBudget_,
            lowLevelTimeoutMultiCertificate_, lowLevelStructuralPrePruned_,
            lowLevelStructuralPrePrunedGoalPermanentBeforeArrivalLb_,
            lowLevelStructuralPrePrunedStartTrappedAtTPlus1_,
            lowLevelStructuralPrePrunedStaticDisconnectedPermanent_,
            lowLevelStructuralPrePrunedMultiCertificate_};
  }
  const char* getLastLowLevelOutcomeName() const {
    return SingleAgentSolver::searchOutcomeName(lastLowLevelOutcome_);
  }
  double getLastLowLevelRemainingBudgetSec() const {
    return lastLowLevelRemainingBudgetSec_;
  }
  double getLastLowLevelEffectiveTimeoutSec() const {
    return lastLowLevelEffectiveTimeoutSec_;
  }
  std::optional<IncrementalRegretStats> getIncrementalRegretStats() const {
    if (!incrementalRegret_) {
      return std::nullopt;
    }
    return incrementalRegretStatsTotal_;
  }
  const RegretEvalStats& getRegretEvalStatsRef() const {
    return regretEvalStatsTotal_;
  }
  RegretEvalStats getRegretEvalStats() const { return regretEvalStatsTotal_; }
  string getIncrementalRegretMode() const {
    return incrementalRegretMode_ == IncrementalRegretMode::descendants
               ? "descendants"
               : "descendants+agent";
  }

  bool extractFeasibleSolution();
  const FeasibleSolution& getFeasibleSolution() const {
    return incumbentSolution_;
  }
  MarketStats getMarketStats() const { return market_.stats; }

  void randomRemoval();
  void worstRemoval();
  void conflictRemoval(const ConflictMap* potentialNeighborhood);
  void shawRemoval(int prioritySize);
  void precedenceWaitRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void lowSlackRemoval(const ConflictMap* potentialNeighborhood = nullptr);
  void marketTatonnementRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void alnsRemoval(const ConflictMap* potentialNeighborhood);

  bool simulatedAnnealing();
  bool thresholdAcceptance();
  bool oldBachelorsAcceptance();
  bool greatDelugeAlgorithm();

  int marketTimeBucket(int timestep) const;
  uint64_t makeMarketVertexKey(int location, int bucket) const;
  uint64_t makeMarketEdgeKey(int from, int to, int bucket) const;
  void computeTaskScheduleMetrics(
      vector<TaskScheduleMetrics>& perTask,
      vector<double>* blockedWaitSum = nullptr) const;
  double computeTaskMarketExposure(int task, bool normalized) const;
  double computeMarketMarginalReliefFromPath(
      const AgentTaskPath& taskPath,
      const unordered_map<uint64_t, int>& vertexDemand,
      const unordered_map<uint64_t, int>& edgeDemand,
      bool normalized) const;
  double computeSolutionMarketPressure() const;
  double computeSolutionPrecedenceWait() const;
  bool passMarketAcceptanceGuards(double previousPressure,
                                  double candidatePressure,
                                  double previousWait,
                                  double candidateWait,
                                  bool candidateIsWorse) const;
  bool marketDestroyStabilityReady() const;
  void updateMarketStateFromCurrentSolution();
  void maybeUpdateMarketState(bool accepted,
                              bool candidateStateUpdate = false);
  double computeMarketExposureFromPath(const AgentTaskPath& taskPath,
                                       bool normalized) const;
  void buildMarketDemandFromCurrentOccupancy(
      unordered_map<uint64_t, int>& vertexDemand,
      unordered_map<uint64_t, int>& edgeDemand) const;
  int computeTaskPrecedenceWaitFromState(
      int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
      const vector<vector<AgentTaskPath>>& agentTaskPaths) const;
  int computeTaskPrecedenceWaitFromWorkspace(int task, int taskLocation,
                                             const RegretWorkspace& workspace,
                                             const vector<int>* assignmentOwnerLookup = nullptr,
                                             const vector<int>* assignmentPosLookup = nullptr) const;
  int computeTaskPrecedenceWaitInCurrentSolution(int task) const;

  void computeMovingMetrics(int numberOfConflicts, int sumOfCosts);

  void printAgents() const {
    for (int i = 0; i < instance_.getAgentNum(); i++) {
      pair<int, int> startLoc =
          instance_.getCoordinate(instance_.getStartLocationsRef()[i]);
      PLOGI << "Agent " << i << " : S = (" << startLoc.first << ", "
            << startLoc.second << ") ;\nGoals : \n";
      for (int j = 0; j < (int)solution_.agents[i].taskAssignments.size();
           j++) {
        const int globalTask = solution_.agents[i].taskAssignments[j];
        if (globalTask < 0 || globalTask >= instance_.getTasksNum()) {
          PLOGE << "\t" << j << " : invalid task id " << globalTask << "\n";
          continue;
        }
        const int goalLocation = instance_.getTaskLocations(globalTask);
        pair<int, int> goalLoc =
            instance_.getCoordinate(goalLocation);
        PLOGI << "\t" << j << " : (" << goalLoc.first << " , " << goalLoc.second
              << ")\n";
      }
    }
  }
};
