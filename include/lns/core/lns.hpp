#pragma once

#include <iosfwd>
#include <functional>
#include <map>
#include <string>

#include "lns_iteration_phase_types.hpp"
#include "lns_market_state.hpp"
#include "lns_params.hpp"
#include "lns_stats.hpp"
#include "lns_types.hpp"

class MovingMetrics;
class CandidatePhaseOrchestrator;
class AcceptanceOrchestrator;
class MarketIterationOrchestrator;
class IterationDiagnosticsOrchestrator;
class IterationOutcomeOrchestrator;
class IterationLifecycleOrchestrator;
class IterationSetupOrchestrator;
class IterationCandidatePostprocessOrchestrator;
class IterationAcceptanceBookkeepingOrchestrator;
class RepairEngine;
class DestroyOrchestrator;

class LNS {
 public:
  struct RegretWorkspace;
  using LowLevelSearchStats = lns_stats::LowLevelSearchStats;
  using RegretEvalStats = lns_stats::RegretEvalStats;
  using IncrementalRegretStats = lns_stats::IncrementalRegretStats;
  using NrrStats = lns_stats::NrrStats;
  using CascadeStats = lns_stats::CascadeStats;
  using TerminalRepositionStats = lns_stats::TerminalRepositionStats;
  using SolutionRestoreStats = lns_stats::SolutionRestoreStats;
  using ImprovementDiagnosticsStats = lns_stats::ImprovementDiagnosticsStats;
  using IterationDebugRecord = lns_stats::IterationDebugRecord;
  using ValidationStats = lns_stats::ValidationStats;

 private:
  friend class CandidatePhaseOrchestrator;
  friend class AcceptanceOrchestrator;
  friend class MarketIterationOrchestrator;
  friend class IterationDiagnosticsOrchestrator;
  friend class IterationOutcomeOrchestrator;
  friend class IterationLifecycleOrchestrator;
  friend class IterationSetupOrchestrator;
  friend class IterationCandidatePostprocessOrchestrator;
  friend class IterationAcceptanceBookkeepingOrchestrator;
  friend class RepairEngine;
  friend class DestroyOrchestrator;

  #include "state/lns_state_private_members.inc"

  struct TaskScheduleMetrics {
    bool valid = false;
    int arrive = 0;
    int release = 0;
    int start = 0;
    int end = 0;
    int waitPrec = 0;
    int blocker = UNASSIGNED;
  };

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
                             const string& fixedAssignmentFilePath = "",
                             const string& mutableAgentsFilePath = "",
                             const string& initialPathsFilePath = "",
                             const string& mutableTasksFilePath = "",
                             const string& lowLevelPlannerOverride = "");
  bool runMAPFPCOnAssignments(const vector<vector<int>>& assignments,
                              const string& solverVariant,
                              int solverTimeoutSec,
                              const string& sourceLabel);
  bool runNeighborhoodReoptimizationRepair();
  bool runFixedAssignmentMapfpcRepair();
  bool runNeighborhoodFixedMapfpcRepair();
  bool runNeighborhoodReassignGreedyMapfpcRepair();
  bool runPostMAPFPCRefinement();
  void overwriteIncumbentFromCurrentSolution();

 protected:
  #include "state/lns_state_protected_members.inc"
  void buildSuccessorPressureStaticSignals();
 public:
  #include "state/lns_state_public_members.inc"

 private:
  struct ConstraintTableAncestorReservation {
    const AgentTaskPath* path = nullptr;
    int agent = UNASSIGNED;
    bool usedPreviousSource = false;
    bool isFinalTask = false;
  };
  struct ConstraintTableTraceState {
    vector<int> ancestorTasks;
    vector<int> ancestorFromPrevious;
    vector<int> ancestorFinal;
    vector<int> nonAncestorHard;
    vector<int> nonAncestorSoft;
    int lenMinDriverTask = UNASSIGNED;
    int lenMinDriverEnd = -1;
  };
  using ResolveAncestorReservationFn = std::function<bool(
      int ancestorTask, bool pendingAncestor,
      ConstraintTableAncestorReservation& out)>;
  using ForEachNonAncestorReservationFn = std::function<void(const std::function<void(
      int task, int agent, const AgentTaskPath& path,
      bool isFinalTask)>& visitor)>;
  bool buildConstraintTableCore(
      ConstraintTable& constraintTable, int task, int skipAgent,
      const vector<vector<int>>& ancestors, bool traceCtTask,
      const string& traceTag,
      const ResolveAncestorReservationFn& resolveAncestorReservation,
      const ForEachNonAncestorReservationFn& forEachNonAncestorReservation);

  void appendIterationStatBounded(const IterationStats& stat);
  bool runOneIteration(ConflictMap& potentialNeighborhood,
                       ConflictMap& oldNeighborhood, MovingMetrics& metrics,
                       bool& currentSolutionValid,
                       ValidationStats& currentValidationStats,
                       bool& feasibleSolutionUpdated);
  bool injectPendingAncestors(
      RegretWorkspace& workspace, int task,
      const vector<pair<int, int>>& precedenceConstraints,
      vector<char>& workspaceTouchedAgents,
      vector<char>* outAncestorsOfTask = nullptr);
  void evaluateAgentPositionCandidate(
      int task, int agent, int earliestTimestep, RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics& baselineMetrics,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
      vector<int>* candidateAgents, const vector<int>* assignmentOwnerLookup,
      const vector<int>* assignmentPosLookup);
  bool buildRegretEntry(
      int task,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>& serviceTimes);
  bool prepareRegretWorkspaceForTask(
      int task, RegretWorkspace& workspace,
      const vector<char>& workspaceTouchedAgents,
      vector<pair<int, int>>& precedenceConstraints,
      vector<int>& assignmentOwnerLookup, vector<int>& assignmentPosLookup,
      int& earliestTimestep);
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
  bool buildInitialSolutionCore(bool enforceInterAgentTiming,
                                const char* callerName);

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

 private:
  bool prepareNextIteration();
  void markResolved(int globalTask);
  // Patches the task paths of an agent such that the begin times and end times match up
  void patchAgentTaskPaths(int agent, int taskPosition);

 public:
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
  bool validateSolution(ConflictMap* conflictedTasks = nullptr,
                        ValidationStats* stats = nullptr);
  void addConflictingTask(int agent, int timestep, ConflictMap* out) const;

 private:
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
  SingleAgentSolver& getReusableLocalPlanner(int agent);

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

 public:
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
  const NrrStats& getNrrStats() const { return nrrStats_; }

  bool extractFeasibleSolution();
  const FeasibleSolution& getFeasibleSolution() const {
    return incumbentSolution_;
  }
  MarketStats getMarketStats() const { return market_.stats; }

 private:
  void randomRemoval();
  void worstRemoval();
  void conflictRemoval(const ConflictMap* potentialNeighborhood);
  void shawRemoval(int prioritySize);
  void precedenceWaitRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void lowSlackRemoval(const ConflictMap* potentialNeighborhood = nullptr);
  void marketTatonnementRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  bool alnsRemoval(const ConflictMap* potentialNeighborhood);

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

 public:
  void printAgents() const;
};
