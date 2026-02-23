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
    // Final-goal occupancy policy used when reserving completed task paths in
    // constraint tables:
    // - "stay": reserve final goal indefinitely (legacy behavior)
    // - "reposition_true": explicit terminal move-out/return path (Phase B+)
    string goalOccupationMode = "reposition_true";
    // Phase-D knobs for true reposition performance.
    string destroyHeuristic;
    string acceptanceCriteria;
    // Repair heuristic:
    // - "regret": classic regret repair
    // - "market_shortlist_regret": market-aware shortlist for candidate
    //   insertion ranking with exhaustive fallback when shortlist finds none
    string repairHeuristic = "regret";
    string regretType;
    bool incrementalRegret = false;
    // If false, ALNS excludes precedence_wait and low_slack destroy operators.
    bool alnsEnablePrecedenceAwareDestroy = true;
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
    double segmentTimeout = 600.0;
    bool parityCheck = false;
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
