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
    string initialSolutionStrategy;
    // Portfolio warm-start budget as a fraction of total cutoff.
    // Used only when initialSolutionStrategy == "portfolio".
    double initialPortfolioTimeFraction = 0.10;
    // If true, scale initialPortfolioTimeFraction by instance difficulty
    // (agents/tasks/precedence), then clamp to [0.05, 0.35].
    bool adaptiveInitialPortfolioBudget = false;
    // Optional path to a MAPF-PC/PBS log containing TASK ASSIGNMENTS and
    // TASK PATHS sections. If set, initialization can be seeded directly from
    // this log instead of running an initializer arm.
    string initialSeedFromPbsLog;
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
    // If true, adopt post-refinement only when SoC is strictly better.
    bool postRefineAcceptOnlyIfBetter = true;
    // Emit additional end-of-run diagnostics to explain why accepted
    // iterations may fail to improve global best SoC and where time is spent.
    bool debugImprovementDiagnostics = false;
    // Optional output TSV path for per-iteration debug records.
    // Empty means disabled.
    string debugIterationTsvPath;
    // Final-goal occupancy policy used when reserving completed task paths in
    // constraint tables:
    // - "stay": reserve final goal indefinitely (legacy behavior)
    // - "reposition_true": explicit terminal move-out/return path (Phase B+)
    string goalOccupationMode = "reposition_true";
    // Phase-D knobs for true reposition performance.
    string destroyHeuristic;
    string acceptanceCriteria;
    // If true, reject any candidate that fails full solution validation
    // before running acceptance criteria.
    bool acceptOnlyValidCandidates = false;
    // Repair heuristic:
    // - "regret": classic regret repair
    // - "market_shortlist_regret": market-aware shortlist for candidate
    //   insertion ranking with exhaustive fallback when shortlist finds none
    // - "mapfpc_fixed": call MAPF-PC using a fixed assignment for repair
    // - "mapfpc_neighborhood_fixed": call MAPF-PC on fixed assignment while
    //   freezing non-neighborhood agents to their seeded paths
    // - "mapfpc_neighborhood_reassign_greedy": greedily reassign removed
    //   tasks among neighborhood agents (topological order), then run MAPF-PC
    string repairHeuristic = "regret";
    // If true, attempt Neighborhood Reoptimization Repair (NRR) first.
    // On NRR failure, fallback to the configured repairHeuristic.
    bool enableNrrRepair = false;
    // If true, failed NRR attempts fallback to the configured repairHeuristic.
    // If false, failed NRR attempts terminate repair for that iteration.
    bool nrrFallbackToStandard = true;
    // If true, rejected iterations do not reuse the previous removed-task
    // neighborhood seed for the next destroy step.
    bool forceNeighborhoodChangeOnReject = false;
    // NRR mini-solver mode:
    // - "pbs": always use PBS
    // - "cbs": always use CBS
    // - "auto": use CBS only for tiny neighborhoods, otherwise PBS
    string nrrMiniSolver = "cbs";
    // CAT backend for NRR mini-solver MAPF-PC subprocesses:
    // - "legacy": existing large-map CAT lists
    // - "pathtablewc": sparse PathTableWC backend on large maps
    string nrrCatBackend = "pathtablewc";
    // Solver used when repairHeuristic is MAPF-PC-based: "pbs" or "cbs".
    string repairMapfpcSolver = "cbs";
    // MAPF-PC timeout in seconds per neighborhood repair attempt.
    int repairMapfpcTimeoutSec = 30;
    string regretType;
    bool incrementalRegret = false;
    // If false, ALNS excludes precedence_wait and low_slack destroy operators.
    bool alnsEnablePrecedenceAwareDestroy = true;
    // If true, ALNS keeps the full destroy pool in normal mode, but when
    // soft-recovery is active it restricts sampling to
    // collision_soft/failure_soft destroy heuristics.
    bool softRecoveryDestroyMode = true;
    // Candidate insertion budget per (task, agent) regret evaluation.
    // Tuned default is 8; set 0 to evaluate all candidate positions.
    int regretCandidateTopK = 8;
    // With repairHeuristic='regret', shortlist ranking always uses normalized
    // distance + precedence lateness proxy + successor pressure.
    // Emit regret shortlist diagnostics for wait-proxy signal strength and
    // ranking impact (top-1/top-K changes under normalized wait-proxy blend).
    bool regretShortlistDiagnostics = false;
    // Hard cap on precedence successor-closure growth in prepareNextIteration.
    // If maxCascadeTasks == 0, use:
    //   max(maxCascadeFactor * neighborSize, neighborSize + 10)
    // If maxCascadeFactor <= 0 and maxCascadeTasks == 0, cap is disabled.
    double maxCascadeFactor = 0.0;
    int maxCascadeTasks = 0;
    // If true, adapt cascade budget online using closure pressure and abort
    // feedback. No additional tuning knobs are required.
    bool adaptiveCascadeBudget = false;
    // Supported: "descendants", "descendants+agent".
    string incrementalRegretMode = "descendants+agent";
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

  struct Market {
    bool heuristics = false;
    int bucketDt = 3;
    int vertexBucketCapacity = 2;
    int edgeBucketCapacity = 2;
    bool updateOnAcceptedOnly = true;
    // When updateOnAcceptedOnly=false, update prices from the current
    // candidate solution before accept/reject restores.
    bool updateFromCandidate = false;
    int updatePeriodAccepted = 1;
    double eta = 0.05;
    double rho = 0.9;
    double priceCap = 50.0;
    // Seed price used when a newly contended resource has no prior price.
    double priceInit = 0.05;
    double gamma = 0.01;
    bool acceptanceGuards = false;
    double tauP = 0.0;
    double tauW = 0.0;
    double destroyWeightPrice = 1.0;
    double destroyWeightWait = 2.0;
    double destroyWeightRoot = 1.5;
    // ALNS-only warmup gate: MarketTatonnement is ineligible until at least
    // this many market price updates have been performed.
    // 0 disables warmup gating.
    int destroyWarmupUpdates = 3;
    // Optional stability gate for ALNS selection of market destroy.
    bool destroyRequireStable = false;
    // When true, keep market destroy eligible during warmup/instability but
    // downweight it instead of excluding it.
    bool destroySoftGate = false;
    double destroyWarmupWeightScale = 0.20;
    double destroyUnstableWeightScale = 0.20;
    double destroyMinAlnsWeight = 0.05;
    // EMA alpha for stability diagnostics in [0, 1].
    double stabilityEmaAlpha = 0.25;
    // Stability thresholds (active when destroyRequireStable=true).
    double stabilityMaxRelPriceDelta = 0.35;
    double stabilityMaxTopMassDelta = 0.08;
    double stabilityMinContendedJaccard = 0.50;
    double seedTopFrac = 0.2;
    double randomDestroyQuota = 0.15;
    int cooldownIters = 3;
    int dUp = 1;
    int dDown = 1;
    int closureCap = 0;
    bool repairTieBreak = false;
    bool repairBlend = false;
    // Normalize market shortlist price term by observed live price scale
    // (instead of static priceCap) to avoid near-zero signals.
    bool repairNormalizeByObservedPrice = true;
    double tieBreakEpsSoc = 0.0;
    double lambdaPrice = 0.0;
    double lambdaWait = 0.0;
  } market;
};
