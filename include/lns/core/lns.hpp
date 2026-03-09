#pragma once

#include <iosfwd>
#include <functional>
#include <map>
#include <string>

#include "lns_iteration_phase_types.hpp"
#include "lns_params.hpp"
#include "lns_stats.hpp"
#include "lns_types.hpp"

class MovingMetrics;
class CandidatePhaseOrchestrator;
class AcceptanceOrchestrator;
class IterationDiagnosticsOrchestrator;
class IterationOutcomeOrchestrator;
class IterationLifecycleOrchestrator;
class IterationSetupOrchestrator;
class IterationCandidatePostprocessOrchestrator;
class IterationAcceptanceBookkeepingOrchestrator;
class RepairEngine;
class DestroyOrchestrator;
struct NeighborhoodDiagnosticsResult;
struct IterationExecutionContext;

class LNS {
 public:
  enum class OptimizationObjectiveMode { soc, makespan };
  enum class RepairHeuristicMode { regret };
  enum class NrrMiniSolverMode { pbs, cbs, auto_mode };
  enum class RegretTypeMode { absolute, relative };

  struct RegretWorkspace;
  using LowLevelSearchStats = lns_stats::LowLevelSearchStats;
  using RegretEvalStats = lns_stats::RegretEvalStats;
  using NrrStats = lns_stats::NrrStats;
  using CascadeStats = lns_stats::CascadeStats;
  using SolutionRestoreStats = lns_stats::SolutionRestoreStats;
  using ImprovementDiagnosticsStats = lns_stats::ImprovementDiagnosticsStats;
  using IterationDebugRecord = lns_stats::IterationDebugRecord;
  using ValidationStats = lns_stats::ValidationStats;

 private:
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
  void clearNeighborhood();
  void restoreSolutionFromPrevious();
  void restoreSolutionFromPrevious(const vector<int>& agentSubset);
  void snapshotPreviousFromCurrent(const vector<int>& agentSubset);
  uint64_t computeSolutionFingerprint(const Solution& solution) const;
  uint64_t computeNeighborhoodFingerprint(
      const ConflictMap& removedTasks) const;
  string iterationQualityName(IterationQuality quality) const;
  int computeObjectiveValue(const Solution& solution) const;
  int computeObjectiveValue(const FeasibleSolution& solution) const;
  int currentObjectiveValue() const;
  int previousObjectiveValue() const;
  int incumbentObjectiveValueOrMax() const;
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
                             const string& lowLevelPlannerOverride = "",
                             const string& catBackendOverride = "",
                             int catBackendSmallMapsOverride = -1);
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
      vector<char>* outAncestorsOfTask = nullptr,
      const vector<vector<int>>* prebuiltAncestors = nullptr,
      const vector<int>* previousAssignmentOwnerLookup = nullptr,
      const vector<int>* previousAssignmentPosLookup = nullptr);
  void evaluateAgentPositionCandidate(
      int task, int agent, int earliestTimestep, RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics& baselineMetrics,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
      vector<int>* candidateAgents, const vector<int>* assignmentOwnerLookup,
      const vector<int>* assignmentPosLookup,
      const vector<int>* previousAssignmentOwnerLookup = nullptr,
      const vector<int>* previousAssignmentPosLookup = nullptr);
  bool buildRegretEntry(
      int task,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>& serviceTimes);
  bool prepareRegretWorkspaceForTask(
      int task, RegretWorkspace& workspace,
      const vector<char>& workspaceTouchedAgents,
      vector<pair<int, int>>& precedenceConstraints,
      vector<int>& assignmentOwnerLookup, vector<int>& assignmentPosLookup,
      int& earliestTimestep,
      const vector<char>* precomputedAncestorsOfTask = nullptr);
  void reservePathWithGoalPolicy(ConstraintTable& constraintTable,
                                 const AgentTaskPath& path,
                                 bool isFinalTask,
                                 bool softOnly = false) const;
  int getServiceOccupancyEndExclusive(int agent) const;
  bool buildInitialSolutionCore(bool enforceInterAgentTiming,
                                const char* callerName);

 public:

  LNS(int numOfIterations, const Instance& instance,
      const LNSParams& parameters);

  inline const Instance& getInstance() const { return instance_; }

  bool run();
  NeighborhoodDiagnosticsResult analyzeNeighborhoodDiagnostics();
  bool finalizeCouldNotFindAbort(
      const std::string& reason, bool restorePrevious,
      ConflictMap& potentialNeighborhood, IterationExecutionContext& context);
  void initializeIterationContext(
      const ValidationStats& currentValidationStats,
      IterationExecutionContext& context);
  void processCandidatePhasePost(
      const CandidatePhaseResult& candidatePhase,
      IterationExecutionContext& context);
  bool runAcceptanceAndBookkeeping(
      bool nrrRepairSucceeded, ConflictMap& potentialNeighborhood,
      const ConflictMap& oldNeighborhood, bool& currentSolutionValid,
      ValidationStats& currentValidationStats,
      IterationExecutionContext& context);
  void commitIterationTimingAndMaybeRecord(IterationExecutionContext& context);
  CandidatePhaseResult runCandidatePhase(
      const std::vector<int>& initialAgentsToCompute,
      ConflictMap& potentialNeighborhood, MovingMetrics& metrics);
  bool executeDestroyHeuristic(int destroyHeuristicId,
                               const ConflictMap* potentialNeighborhood);
  bool runDestroyPhase(const ConflictMap* potentialNeighborhood,
                       int& alnsHeuristicForIter, std::string& errorMessage);
  bool runRepairEngine(bool& repairFailed, bool& nrrRepairSucceeded);

  AcceptanceDecisionResult runAcceptanceDecision(
      bool candidateValid, int previousConflictSignalForIter,
      const std::vector<int>& candidateTouchedAgents,
      IterationDebugRecord& debugRow);
  void updateAcceptanceAlnsStats(int alnsHeuristicForIter,
                                 int previousSocForIter, int proposedSocForIter,
                                 bool candidateValid, bool accepted,
                                 bool acceptedAsWorse,
                                 bool feasibleSolutionUpdated);
  void applyAcceptanceOutcome(
      const AcceptanceDecisionResult& decisionResult, bool candidateValid,
      bool feasibleSolutionUpdated, bool nrrRepairSucceeded,
      int previousSocForIter, int proposedSocForIter,
      int incumbentSocBeforeIter, int previousConflictSignalForIter,
      int candidateConflictSignal,
      const ValidationStats& candidateValidationStats,
      ConflictMap& potentialNeighborhood, const ConflictMap& oldNeighborhood,
      bool& currentSolutionValid, ValidationStats& currentValidationStats,
      IterationQuality& quality,
      const std::vector<int>& candidateTouchedAgents,
      IterationDebugRecord& debugRow);

  bool buildGreedySolution();
  bool buildPrioritizedInitialSolution();
  bool buildGreedySolutionWithMAPFPC(const string& variant,
                                     int solverTimeoutSec = 120);
  bool buildSeededSolutionFromMAPFPCLog(const string& logFilePath);

 private:
  bool prepareNextIteration();
  void markResolved(int globalTask);
  // Patches the task paths of an agent such that the begin times and end times match up
  void patchAgentTaskPaths(int agent, int taskPosition);

 public:
  void printPaths() const;
  enum class OccupancySource { undefined, service };
  OccupancySource getAgentOccupancySourceAt(int agent, int timestep) const;
  // Returns an agent's occupied location at timestep under stay-goal occupancy.
  int getAgentLocationAt(int agent, int timestep) const;
  // Returns the occupancy horizon (exclusive upper bound) for collision checks.
  int getAgentOccupancyHorizon(int agent) const;
  bool validateSolution(ConflictMap* conflictedTasks = nullptr,
                        ValidationStats* stats = nullptr,
                        vector<pair<int, int>>* collisionPairs = nullptr);
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
                            const vector<int>* assignmentPosLookup = nullptr,
                            const vector<int>* previousAssignmentOwnerLookup = nullptr,
                            const vector<int>* previousAssignmentPosLookup = nullptr);

  int extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue);
  int extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue,
                               const vector<int>& newTaskQueue);
  vector<char> reachableSet(int source, const vector<vector<int>>& edgeList);

  bool computeRegret();
  bool computeRegretForTask(int task);
  bool computeRegretForTask(
      int task,
      const vector<pair<int, int>>& fullPrecedenceConstraints);
  bool computeRegretForTask(
      int task, const vector<pair<int, int>>& fullPrecedenceConstraints,
      const vector<vector<int>>& fullAncestors,
      const vector<int>* previousAssignmentOwnerLookup = nullptr,
      const vector<int>* previousAssignmentPosLookup = nullptr);
  void computeRegretForTaskWithAgent(
      TaskRegretPacket regretPacket, RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics& baselineMetrics,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
      const vector<int>* assignmentOwnerLookup = nullptr,
      const vector<int>* assignmentPosLookup = nullptr,
      const vector<int>* previousAssignmentOwnerLookup = nullptr,
      const vector<int>* previousAssignmentPosLookup = nullptr);

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
      bool rollbackAfter = false,
      const vector<int>* assignmentOwnerLookup = nullptr,
      const vector<int>* assignmentPosLookup = nullptr,
      const vector<int>* previousAssignmentOwnerLookup = nullptr,
      const vector<int>* previousAssignmentPosLookup = nullptr);
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
  const string& getInitialSeedFromMapfpcLog() const {
    return initialSeedFromMapfpcLog_;
  }
  bool hasInitialAssignmentSnapshot() const {
    return initialAssignmentSnapshotAvailable_;
  }
  const vector<vector<int>>& getInitialAssignmentsByAgent() const {
    return initialAssignmentsByAgent_;
  }
  const vector<int>& getInitialTaskOwnerByTask() const {
    return initialTaskOwnerByTask_;
  }
  const vector<int>& getInitialTaskPosByTask() const {
    return initialTaskPosByTask_;
  }
  bool hasInitialMetrics() const {
    return initialMetricsAvailable_;
  }
  int getInitialObjectiveValue() const {
    return initialObjectiveValue_;
  }
  int getInitialSoc() const {
    return initialSoc_;
  }
  int getInitialMakespan() const {
    return initialMakespan_;
  }
  double getInitialPrecedenceWait() const {
    return initialPrecedenceWait_;
  }
  double getInitialSolutionRuntimeSec() const {
    return initialSolutionRuntime_;
  }
  double getInitialSolutionRuntimeReportedSec() const {
    return initialSolutionRuntimeReported_;
  }
  double getInitialSeedRuntimeFromLogSec() const {
    return initialSeedRuntimeFromLogSec_;
  }
  double getLnsLoopRuntimeSec() const {
    return lnsLoopRuntimeSec_;
  }
  double getPostRefineRuntimeSec() const {
    return postRefineRuntimeSec_;
  }
  bool isPostRefineEnabled() const {
    return postRefineWithMapfpc_;
  }
  bool wasPostRefineAttempted() const {
    return postRefineAttempted_;
  }
  bool wasPostRefineAccepted() const {
    return postRefineAccepted_;
  }
  const string& getPostRefineAssignmentSource() const {
    return postRefineAssignmentSource_;
  }
  const string& getPostRefineAssignmentLog() const {
    return postRefineAssignmentLog_;
  }
  const string& getPostRefineSolver() const {
    return postRefineSolver_;
  }
  int getPostRefineTimeoutSec() const {
    return postRefineTimeoutSec_;
  }
  bool isPostRefineAcceptOnlyIfBetter() const {
    return postRefineAcceptOnlyIfBetter_;
  }
  const string& getOptimizationObjective() const {
    return optimizationObjective_;
  }
  OptimizationObjectiveMode getOptimizationObjectiveMode() const {
    return optimizationObjectiveMode_;
  }
  RepairHeuristicMode getRepairHeuristicMode() const {
    return repairHeuristicMode_;
  }
  NrrMiniSolverMode getNrrMiniSolverMode() const {
    return nrrMiniSolverMode_;
  }
  RegretTypeMode getRegretTypeMode() const {
    return regretTypeMode_;
  }
  bool isOptimizationObjectiveSoc() const {
    return optimizationObjectiveMode_ == OptimizationObjectiveMode::soc;
  }
  bool isRepairHeuristicRegret() const {
    return repairHeuristicMode_ == RepairHeuristicMode::regret;
  }
  bool isNrrMiniSolverPbs() const {
    return nrrMiniSolverMode_ == NrrMiniSolverMode::pbs;
  }
  bool isRegretTypeAbsolute() const {
    return regretTypeMode_ == RegretTypeMode::absolute;
  }
  int evaluateObjective(const FeasibleSolution& solution) const {
    return computeObjectiveValue(solution);
  }
  unsigned int getSeed() const { return seed_; }
  const ALNS& getAdaptiveLNSRef() const { return adaptiveLNS_; }
  ALNS getAdaptiveLNS() const { return adaptiveLNS_; }
  bool lastPrepareAbortedByCascade() const {
    return cascadeState_.lastPrepareAborted;
  }
  int lastPrepareClosureAdded() const { return cascadeState_.lastPrepareClosureAdded; }
  const CascadeStats& getCascadeStatsRef() const { return cascadeState_.stats; }
  const SolutionRestoreStats& getSolutionRestoreStats() const {
    return solutionRestoreStats_;
  }
  bool isDebugImprovementDiagnosticsEnabled() const {
    return debugImprovementDiagnostics_;
  }
  const ImprovementDiagnosticsStats& getImprovementDiagnosticsStats() const {
    return improvementDiagnosticsStats_;
  }
  LowLevelSearchStats getLowLevelSearchStats() const {
    return lowLevelState_.counters;
  }
  const char* getLastLowLevelOutcomeName() const {
    return SingleAgentSolver::searchOutcomeName(lowLevelState_.lastOutcome);
  }
  double getLastLowLevelRemainingBudgetSec() const {
    return lowLevelState_.lastRemainingBudgetSec;
  }
  double getLastLowLevelEffectiveTimeoutSec() const {
    return lowLevelState_.lastEffectiveTimeoutSec;
  }
  const RegretEvalStats& getRegretEvalStatsRef() const {
    return regretEvalStatsTotal_;
  }
  RegretEvalStats getRegretEvalStats() const { return regretEvalStatsTotal_; }
  const NrrStats& getNrrStats() const { return nrrStats_; }
  bool isNrrGlobalReassignEnabled() const { return nrrGlobalReassign_; }

  bool extractFeasibleSolution();
  const FeasibleSolution& getFeasibleSolution() const {
    return incumbentSolution_;
  }

 private:
  void randomRemoval();
  void worstRemoval();
  void conflictRemoval(const ConflictMap* potentialNeighborhood);
  void shawRemoval(int prioritySize);
  void precedenceWaitRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void lowSlackRemoval(const ConflictMap* potentialNeighborhood = nullptr);
  void collisionSoftRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void failureSoftRemoval(const ConflictMap* potentialNeighborhood = nullptr);
  bool alnsRemoval(const ConflictMap* potentialNeighborhood);

  bool simulatedAnnealing(const std::vector<int>& candidateTouchedAgents);
  bool thresholdAcceptance(const std::vector<int>& candidateTouchedAgents);
  bool oldBachelorsAcceptance(const std::vector<int>& candidateTouchedAgents);
  bool greatDelugeAlgorithm(const std::vector<int>& candidateTouchedAgents);

  void invalidateCurrentTaskAssignmentIndexCache();
  const vector<int>& getCurrentTaskPositionIndexByTask() const;
  void computeTaskScheduleMetrics(
      vector<TaskScheduleMetrics>& perTask,
      vector<double>* blockedWaitSum = nullptr) const;
  double computeSolutionPrecedenceWait() const;
  int computeTaskPrecedenceWaitFromState(
      int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
      const vector<vector<AgentTaskPath>>& agentTaskPaths) const;
  int computeTaskPrecedenceWaitInCurrentSolution(int task) const;

  void computeMovingMetrics(int numberOfConflicts, int sumOfCosts);

 public:
  void printAgents() const;
};
