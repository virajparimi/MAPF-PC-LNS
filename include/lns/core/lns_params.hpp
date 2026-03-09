#pragma once

#include "common.hpp"

struct LNSParams {
  struct Core {
    int neighborhoodSize = 0;
    double timeLimit = 0.0;
    double temperature = 100.0;
    double coolingCoefficient = 0.99975;
    double heatingCoefficient = 1.00025;
    double tolerance = 5.0;
    double shawDistanceWeight = 9.0;
    double shawTemporalWeight = 3.0;
    double lnsConflictWeight = 0.75;
    double lnsCostWeight = 0.25;
    // Optimization objective used for incumbent/best-update comparisons:
    // - "soc": minimize sum of costs
    // - "makespan": minimize maximum individual completion time
    string optimizationObjective = "soc";
    string initialSolutionStrategy;
    // Optional path to a MAPF-PC log (PBS/CBS) containing TASK ASSIGNMENTS and
    // TASK PATHS sections. If set, initialization can be seeded directly from
    // this log instead of running an initializer arm.
    string initialSeedFromMapfpcLog;
    // Optional post-LNS refinement stage: rerun MAPF-PC (PBS/CBS) on a fixed
    // task assignment to improve final routed paths.
    bool postRefineWithMapfpc = false;
    // Assignment source for post-refinement:
    // - "solution": use the current LNS assignment
    // - "log": parse TASK ASSIGNMENTS from postRefineAssignmentLog
    string postRefineAssignmentSource = "solution";
    // Used when postRefineAssignmentSource == "log".
    string postRefineAssignmentLog;
    // Solver used by MAPF-PC post-refinement: "pbs" or "cbs".
    string postRefineSolver = "pbs";
    // MAPF-PC solver cutoff in seconds for post-refinement.
    int postRefineTimeoutSec = 120;
    // If true, adopt post-refinement only when the selected optimization
    // objective is strictly better.
    bool postRefineAcceptOnlyIfBetter = true;
    // Emit additional end-of-run diagnostics to explain why accepted
    // iterations may fail to improve global best SoC and where time is spent.
    bool debugImprovementDiagnostics = false;
    // Optional output TSV path for per-iteration debug records.
    // Empty means disabled.
    string debugIterationTsvPath;
    string destroyHeuristic;
    string acceptanceCriteria;
    // If true, reject any candidate that fails full solution validation
    // before running acceptance criteria.
    bool acceptOnlyValidCandidates = false;
    // Repair heuristic:
    // - "regret": classic regret repair
    string repairHeuristic = "regret";
    // If true, attempt Neighborhood Reoptimization Repair (NRR) first.
    // On NRR failure, fallback to the configured repairHeuristic.
    bool enableNrrRepair = false;
    // If true, failed NRR attempts fallback to the configured repairHeuristic.
    // If false, failed NRR attempts terminate repair for that iteration.
    bool nrrFallbackToStandard = true;
    // If true, NRR iterative proposal considers all agents as insertion
    // candidates and maintains a dynamic frozen occupancy index by demoting
    // newly touched agents during proposal construction.
    bool nrrGlobalReassign = false;
    // If true, rejected iterations do not reuse the previous removed-task
    // neighborhood seed for the next destroy step.
    bool forceNeighborhoodChangeOnReject = false;
    // NRR mini-solver mode:
    // - "pbs": always use PBS
    // - "cbs": always use CBS
    // - "auto": use CBS only for tiny neighborhoods, otherwise PBS
    string nrrMiniSolver = "cbs";
    // Solver used when repairHeuristic is MAPF-PC-based: "pbs" or "cbs".
    string repairMapfpcSolver = "cbs";
    // MAPF-PC timeout in seconds per neighborhood repair attempt.
    int repairMapfpcTimeoutSec = 30;
    string regretType;
    // If false, ALNS excludes precedence_wait and low_slack destroy operators.
    bool alnsEnablePrecedenceAwareDestroy = true;
    // If true, ALNS keeps the full destroy pool in normal mode, but when
    // soft-recovery is active it restricts sampling to
    // collision_soft/failure_soft destroy heuristics.
    bool softRecoveryDestroyMode = true;
    // If true, soft destroy heuristics use a persistent accepted-solution
    // conflict structure (pairs/agents) instead of seeding only from the
    // current potentialNeighborhood.
    bool softPersistentConflictGraph = false;
    unsigned int seed = 0;
  } core;

  struct LowLevel {
    string planner = "mlastar";
    // Suboptimality bound for embedded SIPPS low-level solver (>= 1.0).
    // Ignored when planner == "mlastar".
    double sippsSuboptimality = 1.0;
    double segmentTimeout = 600.0;
    bool parityCheck = false;
    bool structuralPrePrune = false;
  } lowLevel;

};
