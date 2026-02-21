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
    // If true, reject invalid candidates before SA/TA/OBA/GDA acceptance.
    bool rejectInvalidCandidates = false;
    // If true, utility uses raw conflict event count (vertex + edge-swap +
    // precedence violations). If false, utility uses conflicting-task count.
    bool utilityUseConflictEventCount = true;
    // If true, acceptance uses feasibility-first ordering:
    // valid dominates invalid, invalid-vs-invalid compares a precedence-debt
    // score instead of utility that mixes potentially noisy invalid SoC.
    bool acceptanceFeasibilityFirstPrecedenceDebt = false;
    // Weight of spatial invalidity magnitude:
    // vertex collisions + edge swaps + structural violations.
    double acceptanceInvalidSpatialWeight = 1.0;
    // Weight of temporal precedence debt magnitude.
    double acceptanceInvalidPrecedenceDebtWeight = 1.0;
    // Optional tie-break term for invalid-vs-invalid scoring.
    // Default 0 keeps invalid scoring independent of SoC.
    double acceptanceInvalidSocTieBreakWeight = 0.0;
    // If true, feasibility-first invalid-score transitions (valid->invalid and
    // invalid->invalid) use a dedicated acceptance temperature state instead of
    // sharing the utility-based SA/TA/OBA/GDA temperature.
    bool acceptanceUseDedicatedInvalidTemperature = false;
    // Scale used to initialize dedicated invalid temperature from score
    // magnitude: T0 = scale * max(1, |prevScore|, |candScore|, |delta|).
    double acceptanceInvalidTemperatureScale = 0.25;
    // Lower bound for dedicated invalid temperature initialization/recovery.
    double acceptanceInvalidTemperatureFloor = 1e-3;
    string initialSolutionStrategy;
    // Fallback strategy when initialSolutionStrategy fails.
    // Supported: "greedy", "none".
    string initialSolutionFallback = "none";
    // Portfolio warm-start budget as a fraction of total cutoff.
    // Used only when initialSolutionStrategy == "portfolio".
    double initialPortfolioTimeFraction = 0.10;
    // Minimum per-arm budget (seconds) in portfolio mode.
    double initialPortfolioMinArmTimeSec = 1.0;
    // If true, portfolio stops after the first feasible arm.
    bool initialPortfolioStopOnFirstFeasible = false;
    // Final-goal occupancy policy used when reserving completed task paths in
    // constraint tables:
    // - "stay": reserve final goal indefinitely (legacy behavior)
    // - "tail": reserve final goal for goalTailSteps and then release
    // - "reposition": MVP alias of tail-release (explicit move-out/return is
    //   not yet modeled in task paths)
    // - "reposition_true": explicit terminal move-out/return path (Phase B+)
    string goalOccupationMode = "stay";
    int goalTailSteps = 0;
    // Phase-D knobs for true reposition performance.
    int repositionMaxCandidates = 12;
    // 0 = scan full path horizon when detecting demand at final goals.
    int repositionDemandLookahead = 0;
    // Additional timesteps beyond active service horizon to reserve
    // terminalPath occupancy in constraint tables.
    int repositionReservationSlack = 64;
    // Emit per-(agent,task) greedy low-level segment diagnostics.
    bool greedySegmentDiagnostics = false;
    int greedySegmentDiagnosticsTopK = 10;
    string destroyHeuristic;
    string acceptanceCriteria;
    // Repair heuristic:
    // - "regret": classic regret repair
    // - "market_shortlist_regret": market-aware shortlist for candidate
    //   insertion ranking with exhaustive fallback when shortlist finds none
    string repairHeuristic = "regret";
    string regretType;
    bool incrementalRegret = false;
    // Add occupancy constraints from non-ancestor agents during repair
    // planning. This reduces collision-heavy candidates at the cost of extra
    // low-level constraint processing.
    bool repairIncludeNonAncestorAgents = true;
    // If false, ALNS excludes precedence_wait and low_slack destroy operators.
    bool alnsEnablePrecedenceAwareDestroy = true;
    // Candidate insertion budget per (task, agent) regret evaluation.
    // 0 means evaluate all candidate positions.
    int regretCandidateTopK = 0;
    // If true, adapt regret shortlist Top-K online from feasibility/fallback
    // feedback. Requires regretCandidateTopK > 0.
    bool adaptiveRegretTopK = false;
    // When shortlist is active with repairHeuristic='regret', add a normalized
    // precedence wait-proxy term to shortlist ranking. This only affects
    // shortlist ordering, not full low-level feasibility checks.
    bool regretShortlistUseNormalizedWaitProxy = false;
    // When shortlist is active with repairHeuristic='regret', add a normalized
    // successor-pressure term that estimates downstream precedence-delay impact
    // of each candidate insertion position.
    bool regretShortlistUseNormalizedSuccessorPressure = false;
    // When computing successor-pressure from partially removed states, clamp
    // successor start estimates to precedence release lower bounds derived from
    // the mixed workspace/previous assignment state.
    bool regretShortlistClampSuccessorToPrecedenceRelease = true;
    // When true, successor-pressure uses all reachable descendants with
    // depth-decayed weights (instead of immediate successors only).
    bool regretShortlistUseDescendantWeightedSuccessorPressure = false;
    // Depth-decay factor used in descendant-weighted successor-pressure:
    // weight(depth) = decay^(depth-1), with depth=1 for immediate successors.
    double regretShortlistSuccessorPressureDepthDecay = 0.5;
    // Maximum descendant depth considered by descendant-weighted
    // successor-pressure (0 = unlimited).
    int regretShortlistSuccessorPressureMaxDepth = 0;
    // Emit regret shortlist diagnostics for wait-proxy signal strength and
    // ranking impact (top-1/top-K changes under normalized wait-proxy blend).
    bool regretShortlistDiagnostics = false;
    // Hard cap on precedence successor-closure growth in prepareNextIteration.
    // If maxCascadeTasks == 0, use:
    //   max(maxCascadeFactor * neighborSize, neighborSize + 10)
    // If maxCascadeFactor <= 0 and maxCascadeTasks == 0, cap is disabled.
    double maxCascadeFactor = 3.0;
    int maxCascadeTasks = 0;
    // If true, adapt cascade budget online using closure pressure and abort
    // feedback. No additional tuning knobs are required.
    bool adaptiveCascadeBudget = false;
    // Enable touched-agent-scoped rollback restore.
    // Full-copy restore remains the default/fallback behavior.
    bool partialSolutionRestore = false;
    // Supported: "descendants", "descendants+agent".
    string incrementalRegretMode = "descendants+agent";
    unsigned int seed = 0;
  } core;

  struct LowLevel {
    string planner = "mlastar";
    double segmentTimeout = 600.0;
    bool parityCheck = false;
    int parityMaxLogs = 10;
    // If true, MLA* refreshes focal from f-buckets instead of scanning the
    // whole open list when the focal bound increases.
    bool mlastarIncrementalFocalRefresh = true;
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
