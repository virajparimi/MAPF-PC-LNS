#include <plog/Log.h>
#include <plog/Severity.h>
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"

#include <chrono>
#include <cstdint>
#include <cmath>
#include <boost/program_options.hpp>
#include "common.hpp"
#include "instance.hpp"
#include "lns.hpp"
#include "run_reporting.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {

  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::error, &consoleAppender);

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message");
  desc.add_options()("map,m", po::value<string>()->required(),
                     "Input file for map");
  desc.add_options()("agents,a", po::value<string>()->required(),
                     "Input file for agents");
  desc.add_options()("output,o", po::value<string>(),
                     "Output file for schedule");
  desc.add_options()("cutoffTime,t", po::value<double>()->default_value(7200),
                     "Cutoff time (seconds)");
  desc.add_options()("agentNum,k", po::value<int>()->default_value(0),
                     "Number of agents to plan for");
  desc.add_options()("taskNum,l", po::value<int>()->default_value(0),
                     "Number of tasks to plan for");
  desc.add_options()("neighborSize,n", po::value<int>()->default_value(10),
                     "Size of the neighborhood");
  desc.add_options()("maxIterations,i", po::value<int>()->default_value(0),
                     "Maximum number of iterations");
  desc.add_options()(
      "regretCandidateTopK",
      po::value<int>()->default_value(0),
      "Top-K insertion positions per (task,agent) during regret evaluation "
      "(0 = evaluate all positions)");
  desc.add_options()(
      "adaptiveRegretTopK",
      po::value<bool>()->default_value(false),
      "Enable adaptive regret shortlist Top-K (requires regretCandidateTopK > 0)");
  desc.add_options()(
      "regretShortlistUseNormalizedWaitProxy",
      po::value<bool>()->default_value(false),
      "For repairHeuristic='regret' shortlist mode, add normalized wait-proxy "
      "to shortlist ranking");
  desc.add_options()(
      "regretShortlistUseNormalizedSuccessorPressure",
      po::value<bool>()->default_value(false),
      "For repairHeuristic='regret' shortlist mode, add normalized successor-pressure "
      "to shortlist ranking");
  desc.add_options()(
      "regretShortlistClampSuccessorToPrecedenceRelease",
      po::value<bool>()->default_value(true),
      "Clamp successor-pressure successor timing estimates to precedence release lower bounds");
  desc.add_options()(
      "regretShortlistUseDescendantWeightedSuccessorPressure",
      po::value<bool>()->default_value(false),
      "Use descendant-weighted successor-pressure (depth-decayed over precedence descendants)");
  desc.add_options()(
      "regretShortlistSuccessorPressureDepthDecay",
      po::value<double>()->default_value(0.5),
      "Depth-decay factor for descendant-weighted successor-pressure, weight(d)=decay^(d-1)");
  desc.add_options()(
      "regretShortlistSuccessorPressureMaxDepth",
      po::value<int>()->default_value(0),
      "Max descendant depth used in descendant-weighted successor-pressure (0 = unlimited)");
  desc.add_options()(
      "regretShortlistDiagnostics",
      po::value<bool>()->default_value(false),
      "Emit regret shortlist diagnostics for wait-proxy ranking impact");
  desc.add_options()(
      "maxCascadeFactor",
      po::value<double>()->default_value(3.0),
      "Multiplier for successor-closure cap in prepareNextIteration "
      "(<=0 with maxCascadeTasks=0 disables cap)");
  desc.add_options()(
      "maxCascadeTasks",
      po::value<int>()->default_value(0),
      "Absolute cap for successor-closure added tasks in "
      "prepareNextIteration (0 = use maxCascadeFactor formula)");
  desc.add_options()(
      "adaptiveCascadeBudget",
      po::value<bool>()->default_value(false),
      "Enable adaptive successor-closure budget in prepareNextIteration");
  desc.add_options()(
      "partialSolutionRestore",
      po::value<bool>()->default_value(false),
      "Use touched-agent-scoped rollback restore instead of always copying "
      "the full solution snapshot");
  desc.add_options()(
      "repairIncludeNonAncestorAgents",
      po::value<bool>()->default_value(true),
      "Include non-ancestor agent paths in repair constraint tables");
  desc.add_options()(
      "alnsEnablePrecedenceAwareDestroy",
      po::value<bool>()->default_value(true),
      "Allow ALNS to sample precedence_wait and low_slack destroy operators");
  desc.add_options()("severity,d", po::value<int>()->default_value(0),
                     "Debugging level");
  desc.add_options()("initialSolution,s",
                     po::value<string>()->default_value("portfolio"),
                     "Strategy for the initial solution (portfolio default for "
                     "best final SoC; use greedy/prioritized for faster time-to-best)");
  desc.add_options()(
      "initialFallback",
      po::value<string>()->default_value("none"),
      "Fallback when initial solution fails: 'greedy' or 'none'");
  desc.add_options()(
      "initialPortfolioTimeFraction",
      po::value<double>()->default_value(0.10),
      "Portfolio warm-start budget fraction in [0,1] "
      "(used when initialSolution='portfolio')");
  desc.add_options()(
      "initialPortfolioMinArmTimeSec",
      po::value<double>()->default_value(1.0),
      "Minimum per-arm budget in portfolio warm-start (seconds)");
  desc.add_options()(
      "initialPortfolioStopOnFirstFeasible",
      po::value<bool>()->default_value(false),
      "Stop portfolio warm-start after first feasible arm");
  desc.add_options()(
      "goalOccupationMode",
      po::value<string>()->default_value("stay"),
      "Final-goal reservation policy: 'stay', 'tail', 'reposition', or "
      "'reposition_true'");
  desc.add_options()(
      "goalTailSteps",
      po::value<int>()->default_value(0),
      "Additional timesteps to reserve final goals in 'tail'/'reposition' modes");
  desc.add_options()(
      "repositionMaxCandidates",
      po::value<int>()->default_value(12),
      "Maximum parking candidates evaluated per final goal in "
      "'reposition_true' mode");
  desc.add_options()(
      "repositionDemandLookahead",
      po::value<int>()->default_value(0),
      "Demand scan horizon after completion in 'reposition_true' "
      "(0 = full path horizon)");
  desc.add_options()(
      "repositionReservationSlack",
      po::value<int>()->default_value(64),
      "Additional timesteps beyond active service horizon to reserve "
      "terminal occupancy in 'reposition_true'");
  desc.add_options()(
      "greedySegmentDiagnostics",
      po::bool_switch()->default_value(false),
      "Emit per-segment low-level diagnostics for greedy initialization");
  desc.add_options()(
      "greedySegmentDiagnosticsTopK",
      po::value<int>()->default_value(10),
      "Top-K expensive greedy segments to print in diagnostics summary");
  desc.add_options()(
      "destroyHeuristic,H", po::value<string>()->default_value("conflict"),
      "Destroy heuristic to use for creating the LNS neighborhood");
  desc.add_options()("acceptanceCriteria,c",
                     po::value<string>()->default_value("SA"),
                     "Acceptance criteria for new solutions");
  desc.add_options()(
      "repairHeuristic",
      po::value<string>()->default_value("regret"),
      "Repair heuristic to use: 'regret' or 'market_shortlist_regret'");
  desc.add_options()(
      "rejectInvalidCandidates",
      po::value<bool>()->default_value(false),
      "Hard-reject invalid candidates before acceptance criteria");
  desc.add_options()(
      "utilityUseConflictEventCount",
      po::value<bool>()->default_value(true),
      "Use conflict-event magnitude (vertex/swap/precedence) in utility instead of conflicting-task count");
  desc.add_options()(
      "acceptanceFeasibilityFirstPrecedenceDebt",
      po::value<bool>()->default_value(false),
      "Feasibility-first acceptance: valid dominates invalid; invalid-vs-invalid uses precedence-debt score");
  desc.add_options()(
      "acceptanceInvalidSpatialWeight",
      po::value<double>()->default_value(1.0),
      "Invalid-score weight for spatial conflicts (vertex + swap + structural)");
  desc.add_options()(
      "acceptanceInvalidPrecedenceDebtWeight",
      po::value<double>()->default_value(1.0),
      "Invalid-score weight for precedence-debt magnitude");
  desc.add_options()(
      "acceptanceInvalidSocTieBreakWeight",
      po::value<double>()->default_value(0.0),
      "Invalid-score SoC tie-break weight (default 0 keeps invalid scoring SoC-independent)");
  desc.add_options()(
      "acceptanceUseDedicatedInvalidTemperature",
      po::value<bool>()->default_value(false),
      "Use a dedicated SA/TA/OBA/GDA temperature state for feasibility-first invalid-score transitions");
  desc.add_options()(
      "acceptanceInvalidTemperatureScale",
      po::value<double>()->default_value(0.25),
      "Dedicated invalid-temperature scale: T0 = scale * max(1, |prev|, |cand|, |delta|)");
  desc.add_options()(
      "acceptanceInvalidTemperatureFloor",
      po::value<double>()->default_value(1e-3),
      "Lower bound for dedicated invalid-temperature initialization/recovery");
  desc.add_options()("seed",
                     po::value<unsigned int>()->default_value(0),
                     "Random seed (0 = time-based)");
  desc.add_options()("regretType,r",
                     po::value<string>()->default_value("absolute"),
                     "Type of regret metric to use i.e relative or absolute");
  desc.add_options()(
      "incrementalRegret",
      po::bool_switch()->default_value(false),
      "Use incremental regret recomputation during LNS repair");
  desc.add_options()(
      "incrementalRegretMode",
      po::value<string>()->default_value("descendants+agent"),
      "Dirty-set strategy for incremental regret: 'descendants' or "
      "'descendants+agent'");
  desc.add_options()(
      "lowLevelPlanner",
      po::value<string>()->default_value("mlastar"),
      "Low-level planner to use: 'mlastar' or 'sipps'");
  desc.add_options()(
      "plannerParityCheck",
      po::bool_switch()->default_value(false),
      "Debug mode: in SIPPS runs, shadow each low-level segment with MLA* and "
      "log parity mismatches");
  desc.add_options()(
      "plannerParityMaxLogs",
      po::value<int>()->default_value(10),
      "Maximum number of SIPPS-vs-MLA* parity mismatch logs");
  desc.add_options()(
      "lowLevelSegmentTimeout",
      po::value<double>()->default_value(600.0),
      "Per-segment timeout (seconds) for low-level planner searches");
  desc.add_options()(
      "mlastarIncrementalFocalRefresh",
      po::value<bool>()->default_value(true),
      "Enable incremental MLA* focal refresh using f-value buckets");

  struct MarketIntOptionSpec {
    const char* name;
    int defaultValue;
    const char* description;
    int LNSParams::Market::*field;
  };
  struct MarketDoubleOptionSpec {
    const char* name;
    double defaultValue;
    const char* description;
    double LNSParams::Market::*field;
  };
  static constexpr MarketIntOptionSpec kMarketIntOptions[] = {
      {"marketBucketDt", 3, "Time bucket size used by market resources",
       &LNSParams::Market::bucketDt},
      {"marketVertexBucketCapacity", 2,
       "Capacity used for vertex-bucket market resources",
       &LNSParams::Market::vertexBucketCapacity},
      {"marketEdgeBucketCapacity", 2,
       "Capacity used for edge-bucket market resources",
       &LNSParams::Market::edgeBucketCapacity},
      {"marketUpdatePeriodAccepted", 1,
       "Number of accepted iterations between market updates",
       &LNSParams::Market::updatePeriodAccepted},
      {"marketCooldownIters", 3,
       "Cooldown iterations before re-selecting a task in market destroy",
       &LNSParams::Market::cooldownIters},
      {"marketDestroyWarmupUpdates", 3,
       "Minimum market update count before ALNS can sample market_tatonnement (0 disables warmup gate)",
       &LNSParams::Market::destroyWarmupUpdates},
      {"marketDUp", 1, "Ancestor expansion depth for market destroy",
       &LNSParams::Market::dUp},
      {"marketDDown", 1, "Successor expansion depth for market destroy",
       &LNSParams::Market::dDown},
      {"marketClosureCap", 0,
       "Maximum tasks expanded by market closure (0 = neighbor size cap)",
       &LNSParams::Market::closureCap},
  };
  static constexpr MarketDoubleOptionSpec kMarketDoubleOptions[] = {
      {"marketEta", 0.05, "Market tatonnement base step size",
       &LNSParams::Market::eta},
      {"marketRho", 0.9, "EMA smoothing factor for excess demand",
       &LNSParams::Market::rho},
      {"marketPriceCap", 50.0, "Maximum market resource price",
       &LNSParams::Market::priceCap},
      {"marketPriceInit", 0.05,
       "Initial price for newly contended resources",
       &LNSParams::Market::priceInit},
      {"marketGamma", 0.01, "Price evaporation factor per market update",
       &LNSParams::Market::gamma},
      {"marketStabilityEmaAlpha", 0.25,
       "EMA alpha for market stability diagnostics",
       &LNSParams::Market::stabilityEmaAlpha},
      {"marketStabilityMaxRelPriceDelta", 0.35,
       "Max EMA-relative price L1 delta for ALNS market readiness",
       &LNSParams::Market::stabilityMaxRelPriceDelta},
      {"marketStabilityMaxTopMassDelta", 0.08,
       "Max EMA top-price-mass delta for ALNS market readiness",
       &LNSParams::Market::stabilityMaxTopMassDelta},
      {"marketStabilityMinContendedJaccard", 0.50,
       "Min EMA contended-resource Jaccard for ALNS market readiness",
       &LNSParams::Market::stabilityMinContendedJaccard},
      {"marketDestroyWarmupWeightScale", 0.20,
       "ALNS weight multiplier applied to market destroy during warmup when soft gate is enabled",
       &LNSParams::Market::destroyWarmupWeightScale},
      {"marketDestroyUnstableWeightScale", 0.20,
       "ALNS weight multiplier applied to market destroy during instability when soft gate is enabled",
       &LNSParams::Market::destroyUnstableWeightScale},
      {"marketDestroyMinAlnsWeight", 0.05,
       "Minimum ALNS sampling weight for market destroy when soft gate is enabled",
       &LNSParams::Market::destroyMinAlnsWeight},
      {"marketTauP", 0.0, "Acceptance guard threshold for market pressure",
       &LNSParams::Market::tauP},
      {"marketTauW", 0.0, "Acceptance guard threshold for precedence wait",
       &LNSParams::Market::tauW},
      {"marketDestroyWeightPrice", 1.0,
       "Weight of market exposure in market destroy burden",
       &LNSParams::Market::destroyWeightPrice},
      {"marketDestroyWeightWait", 2.0,
       "Weight of precedence wait in market destroy burden",
       &LNSParams::Market::destroyWeightWait},
      {"marketDestroyWeightRoot", 1.5,
       "Weight of blocker-root wait in market destroy burden",
       &LNSParams::Market::destroyWeightRoot},
      {"marketSeedTopFrac", 0.2,
       "Top fraction of burden-ranked tasks used as market destroy seed pool",
       &LNSParams::Market::seedTopFrac},
      {"marketRandomDestroyQuota", 0.15,
       "Random sampling quota for market destroy neighborhoods",
       &LNSParams::Market::randomDestroyQuota},
      {"marketTieBreakEpsSoc", 0.0,
       "Tie-break epsilon on |deltaSoC| for market-aware repair",
       &LNSParams::Market::tieBreakEpsSoc},
      {"marketLambdaPrice", 0.0, "Repair weight for market exposure delta",
       &LNSParams::Market::lambdaPrice},
      {"marketLambdaWait", 0.0, "Repair weight for precedence-wait delta",
       &LNSParams::Market::lambdaWait},
  };

  desc.add_options()("marketHeuristics",
                     po::bool_switch()->default_value(false),
                     "Enable tatonnement-style market heuristics");
  desc.add_options()("marketUpdateOnAcceptedOnly",
                     po::value<bool>()->default_value(true),
                     "Update market prices only after accepted iterations");
  desc.add_options()("marketUpdateFromCandidate",
                     po::bool_switch()->default_value(false),
                     "When updateOnAcceptedOnly=false, update prices from the candidate solution before accept/reject");
  desc.add_options()("marketAcceptanceGuards",
                     po::bool_switch()->default_value(false),
                     "Enable market pressure / precedence-wait acceptance guards");
  desc.add_options()("marketDestroyRequireStable",
                     po::bool_switch()->default_value(false),
                     "Require market stability diagnostics before ALNS can sample market_tatonnement");
  desc.add_options()("marketDestroySoftGate",
                     po::bool_switch()->default_value(false),
                     "Use soft ALNS downweighting (instead of exclusion) for market_tatonnement during warmup/instability");
  desc.add_options()("marketRepairTieBreak",
                     po::bool_switch()->default_value(false),
                     "Enable market-aware tie-break in repair when deltaSoC is near zero");
  desc.add_options()("marketRepairBlend",
                     po::bool_switch()->default_value(false),
                     "Enable blended market-aware repair score");
  desc.add_options()("marketRepairNormalizeByObservedPrice",
                     po::value<bool>()->default_value(true),
                     "Normalize market shortlist prices by observed live price scale (fallback: price cap)");
  for (const auto& spec : kMarketIntOptions) {
    desc.add_options()(spec.name, po::value<int>()->default_value(spec.defaultValue),
                       spec.description);
  }
  for (const auto& spec : kMarketDoubleOptions) {
    desc.add_options()(spec.name,
                       po::value<double>()->default_value(spec.defaultValue),
                       spec.description);
  }

  // Backward compatibility: historically "-h <heuristic>" was used for
  // destroy heuristic. We now reserve -h for help and remap legacy usage.
  std::vector<std::string> normalizedArgs;
  normalizedArgs.reserve((size_t)std::max(0, argc - 1));
  for (int i = 1; i < argc; ++i) {
    normalizedArgs.emplace_back(argv[i]);
  }
  bool usedLegacyDestroyHeuristicFlag = false;
  for (size_t i = 0; i + 1 < normalizedArgs.size(); ++i) {
    if (normalizedArgs[i] == "-h" && !normalizedArgs[i + 1].empty() &&
        normalizedArgs[i + 1][0] != '-') {
      normalizedArgs[i] = "--destroyHeuristic";
      usedLegacyDestroyHeuristicFlag = true;
    }
  }

  std::vector<const char*> parsedArgv;
  parsedArgv.reserve(normalizedArgs.size() + 1);
  parsedArgv.push_back(argv[0]);
  for (const auto& arg : normalizedArgs) {
    parsedArgv.push_back(arg.c_str());
  }

  po::variables_map vm;
  try {
    po::store(
        po::parse_command_line((int)parsedArgv.size(), parsedArgv.data(), desc),
        vm);
  } catch (const std::exception& e) {
    PLOGE << e.what() << "\n" << desc << "\n";
    return 1;
  }

  if (vm.count("help") != 0u) {
    plog::get()->setMaxSeverity(plog::debug);
    PLOGD << desc << "\n";
    return 0;
  }
  if (usedLegacyDestroyHeuristicFlag) {
    PLOGW << "Using deprecated '-h <heuristic>' syntax for destroy heuristic. "
             "Use '--destroyHeuristic' (or '-H') and reserve '-h' for help.\n";
  }

  try {
    po::notify(vm);
  } catch (const std::exception& e) {
    PLOGE << e.what() << "\n" << desc << "\n";
    return 1;
  }
  int severity = vm["severity"].as<int>();
  if (severity < (int)plog::none) {
    severity = (int)plog::none;
  } else if (severity > (int)plog::verbose) {
    severity = (int)plog::verbose;
  }
  plog::get()->setMaxSeverity(static_cast<plog::Severity>(severity));

  string initialSolutionStrategy = vm["initialSolution"].as<string>();
  if (initialSolutionStrategy != "greedy" &&
      initialSolutionStrategy != "prioritized" &&
      initialSolutionStrategy != "portfolio" &&
      initialSolutionStrategy.find("sota") == string::npos) {
    PLOGE << "Incorrect initial solution strategy provided. Please choose from "
             "'greedy', 'prioritized', 'portfolio', "
             "'sota_cbs' or 'sota_pbs' options"
          << "\n";
    return 1;
  }
  const string initialSolutionFallback = vm["initialFallback"].as<string>();
  const double initialPortfolioTimeFraction =
      vm["initialPortfolioTimeFraction"].as<double>();
  const double initialPortfolioMinArmTimeSec =
      vm["initialPortfolioMinArmTimeSec"].as<double>();
  const bool initialPortfolioStopOnFirstFeasible =
      vm["initialPortfolioStopOnFirstFeasible"].as<bool>();
  if (initialSolutionFallback != "greedy" &&
      initialSolutionFallback != "none") {
    PLOGE << "Incorrect initial fallback strategy provided. Please choose "
             "from 'greedy' and 'none'\n";
    return 1;
  }
  if (!std::isfinite(initialPortfolioTimeFraction) ||
      initialPortfolioTimeFraction < 0.0 ||
      initialPortfolioTimeFraction > 1.0) {
    PLOGE << "initialPortfolioTimeFraction must be finite and in [0, 1]\n";
    return 1;
  }
  if (!std::isfinite(initialPortfolioMinArmTimeSec) ||
      initialPortfolioMinArmTimeSec <= 0.0) {
    PLOGE << "initialPortfolioMinArmTimeSec must be finite and > 0\n";
    return 1;
  }
  if (initialSolutionStrategy != "portfolio" &&
      initialPortfolioStopOnFirstFeasible) {
    PLOGW << "initialPortfolioStopOnFirstFeasible is ignored unless "
             "initialSolution='portfolio'\n";
  }
  string goalOccupationMode = vm["goalOccupationMode"].as<string>();
  if (goalOccupationMode != "stay" && goalOccupationMode != "tail" &&
      goalOccupationMode != "reposition" &&
      goalOccupationMode != "reposition_true") {
    PLOGE << "Incorrect goal occupation mode provided. Please choose from "
             "'stay', 'tail', 'reposition', and 'reposition_true'\n";
    return 1;
  }
  const int goalTailSteps = vm["goalTailSteps"].as<int>();
  const int repositionMaxCandidates =
      vm["repositionMaxCandidates"].as<int>();
  const int repositionDemandLookahead =
      vm["repositionDemandLookahead"].as<int>();
  const int repositionReservationSlack =
      vm["repositionReservationSlack"].as<int>();
  const bool greedySegmentDiagnostics =
      vm["greedySegmentDiagnostics"].as<bool>();
  const int greedySegmentDiagnosticsTopK =
      vm["greedySegmentDiagnosticsTopK"].as<int>();
  if (goalTailSteps < 0) {
    PLOGE << "goalTailSteps must be non-negative\n";
    return 1;
  }
  if (repositionMaxCandidates <= 0) {
    PLOGE << "repositionMaxCandidates must be positive\n";
    return 1;
  }
  if (repositionDemandLookahead < 0) {
    PLOGE << "repositionDemandLookahead must be non-negative\n";
    return 1;
  }
  if (repositionReservationSlack < 0) {
    PLOGE << "repositionReservationSlack must be non-negative\n";
    return 1;
  }
  if (greedySegmentDiagnosticsTopK <= 0) {
    PLOGE << "greedySegmentDiagnosticsTopK must be positive\n";
    return 1;
  }
  if (goalOccupationMode == "stay" && goalTailSteps > 0) {
    PLOGW << "goalTailSteps is ignored when goalOccupationMode='stay'\n";
  }

  string destroyHeuristic = vm["destroyHeuristic"].as<string>();
  if (destroyHeuristic != "conflict" && destroyHeuristic != "worst" &&
      destroyHeuristic != "random" && destroyHeuristic != "shaw" &&
      destroyHeuristic != "precedence_wait" &&
      destroyHeuristic != "low_slack" &&
      destroyHeuristic != "market_tatonnement" &&
      destroyHeuristic != "alns") {
    PLOGE << "The destroy heuristic provided is not supported! Please choose "
             "from 'conflict', 'worst', 'random', 'shaw', 'precedence_wait', "
             "'low_slack', 'market_tatonnement' and 'alns' removal operators\n";
    return 1;
  }

  string acceptanceCriteria = vm["acceptanceCriteria"].as<string>();
  if (acceptanceCriteria != "SA" && acceptanceCriteria != "TA" &&
      acceptanceCriteria != "OBA" && acceptanceCriteria != "GDA") {
    PLOGE << "The acceptance criteria provided is not supported! Please choose "
             "from SA(Simulated Annealing), TA(Threshold Acceptance), OBA(Old "
             "Bachelor's Acceptance and GDA(Great Deluge Algorithm)\n";
    return 1;
  }
  const string repairHeuristic = vm["repairHeuristic"].as<string>();
  if (repairHeuristic != "regret" &&
      repairHeuristic != "market_shortlist_regret") {
    PLOGE << "The repair heuristic provided is not supported! Please choose "
             "from 'regret' and 'market_shortlist_regret'\n";
    return 1;
  }
  const bool rejectInvalidCandidates =
      vm["rejectInvalidCandidates"].as<bool>();
  const bool utilityUseConflictEventCount =
      vm["utilityUseConflictEventCount"].as<bool>();
  const bool acceptanceFeasibilityFirstPrecedenceDebt =
      vm["acceptanceFeasibilityFirstPrecedenceDebt"].as<bool>();
  const double acceptanceInvalidSpatialWeight =
      vm["acceptanceInvalidSpatialWeight"].as<double>();
  const double acceptanceInvalidPrecedenceDebtWeight =
      vm["acceptanceInvalidPrecedenceDebtWeight"].as<double>();
  const double acceptanceInvalidSocTieBreakWeight =
      vm["acceptanceInvalidSocTieBreakWeight"].as<double>();
  const bool acceptanceUseDedicatedInvalidTemperature =
      vm["acceptanceUseDedicatedInvalidTemperature"].as<bool>();
  const double acceptanceInvalidTemperatureScale =
      vm["acceptanceInvalidTemperatureScale"].as<double>();
  const double acceptanceInvalidTemperatureFloor =
      vm["acceptanceInvalidTemperatureFloor"].as<double>();
  if (!std::isfinite(acceptanceInvalidSpatialWeight) ||
      acceptanceInvalidSpatialWeight < 0.0) {
    PLOGE << "acceptanceInvalidSpatialWeight must be finite and non-negative\n";
    return 1;
  }
  if (!std::isfinite(acceptanceInvalidPrecedenceDebtWeight) ||
      acceptanceInvalidPrecedenceDebtWeight < 0.0) {
    PLOGE << "acceptanceInvalidPrecedenceDebtWeight must be finite and "
             "non-negative\n";
    return 1;
  }
  if (!std::isfinite(acceptanceInvalidSocTieBreakWeight) ||
      acceptanceInvalidSocTieBreakWeight < 0.0) {
    PLOGE << "acceptanceInvalidSocTieBreakWeight must be finite and "
             "non-negative\n";
    return 1;
  }
  if (!std::isfinite(acceptanceInvalidTemperatureScale) ||
      acceptanceInvalidTemperatureScale <= 0.0) {
    PLOGE << "acceptanceInvalidTemperatureScale must be finite and > 0\n";
    return 1;
  }
  if (!std::isfinite(acceptanceInvalidTemperatureFloor) ||
      acceptanceInvalidTemperatureFloor <= 0.0) {
    PLOGE << "acceptanceInvalidTemperatureFloor must be finite and > 0\n";
    return 1;
  }
  if (acceptanceUseDedicatedInvalidTemperature &&
      !acceptanceFeasibilityFirstPrecedenceDebt) {
    PLOGW << "acceptanceUseDedicatedInvalidTemperature has no effect unless "
             "acceptanceFeasibilityFirstPrecedenceDebt is enabled\n";
  }
  if (acceptanceFeasibilityFirstPrecedenceDebt &&
      acceptanceInvalidSpatialWeight == 0.0 &&
      acceptanceInvalidPrecedenceDebtWeight == 0.0 &&
      acceptanceInvalidSocTieBreakWeight == 0.0) {
    PLOGE << "With acceptanceFeasibilityFirstPrecedenceDebt enabled, at least "
             "one invalid-score weight must be > 0\n";
    return 1;
  }
  const bool regretShortlistUseNormalizedWaitProxy =
      vm["regretShortlistUseNormalizedWaitProxy"].as<bool>();
  const bool regretShortlistUseNormalizedSuccessorPressure =
      vm["regretShortlistUseNormalizedSuccessorPressure"].as<bool>();
  const bool regretShortlistClampSuccessorToPrecedenceRelease =
      vm["regretShortlistClampSuccessorToPrecedenceRelease"].as<bool>();
  const bool regretShortlistUseDescendantWeightedSuccessorPressure =
      vm["regretShortlistUseDescendantWeightedSuccessorPressure"].as<bool>();
  const double regretShortlistSuccessorPressureDepthDecay =
      vm["regretShortlistSuccessorPressureDepthDecay"].as<double>();
  const int regretShortlistSuccessorPressureMaxDepth =
      vm["regretShortlistSuccessorPressureMaxDepth"].as<int>();
  const bool regretShortlistDiagnostics =
      vm["regretShortlistDiagnostics"].as<bool>();
  if (!std::isfinite(regretShortlistSuccessorPressureDepthDecay) ||
      regretShortlistSuccessorPressureDepthDecay < 0.0 ||
      regretShortlistSuccessorPressureDepthDecay > 1.0) {
    PLOGE << "regretShortlistSuccessorPressureDepthDecay must be in [0, 1]\n";
    return 1;
  }
  if (regretShortlistSuccessorPressureMaxDepth < 0) {
    PLOGE << "regretShortlistSuccessorPressureMaxDepth must be non-negative\n";
    return 1;
  }

  string regretType = vm["regretType"].as<string>();
  if (regretType != "absolute" && regretType != "relative") {
    PLOGE << "The regret type provided is not supported! Please choose from "
             "'absolute' and 'relative' regret\n";
    return 1;
  }

  const bool incrementalRegret = vm["incrementalRegret"].as<bool>();
  const string incrementalRegretMode = vm["incrementalRegretMode"].as<string>();
  if (incrementalRegret &&
      incrementalRegretMode != "descendants" &&
      incrementalRegretMode != "descendants+agent") {
    PLOGE << "The incremental regret mode provided is not supported! Please "
             "choose from 'descendants' and 'descendants+agent'\n";
    return 1;
  }

  const string lowLevelPlanner = vm["lowLevelPlanner"].as<string>();
  if (lowLevelPlanner != "mlastar" && lowLevelPlanner != "sipps") {
    PLOGE << "The low-level planner provided is not supported! Please choose "
             "from 'mlastar' and 'sipps'\n";
    return 1;
  }
  const bool plannerParityCheck = vm["plannerParityCheck"].as<bool>();
  const int plannerParityMaxLogs = vm["plannerParityMaxLogs"].as<int>();
  const double lowLevelSegmentTimeout =
      vm["lowLevelSegmentTimeout"].as<double>();
  const bool mlastarIncrementalFocalRefresh =
      vm["mlastarIncrementalFocalRefresh"].as<bool>();
  if (plannerParityMaxLogs <= 0) {
    PLOGE << "plannerParityMaxLogs must be a positive integer\n";
    return 1;
  }
  if (lowLevelSegmentTimeout <= 0.0) {
    PLOGE << "lowLevelSegmentTimeout must be positive\n";
    return 1;
  }
  if (plannerParityCheck) {
    if (lowLevelPlanner != "sipps") {
      PLOGW << "plannerParityCheck is enabled but lowLevelPlanner is '"
            << lowLevelPlanner
            << "'. Parity checks are only executed in SIPPS mode.\n";
    } else {
      PLOGW << "plannerParityCheck is a debug-only mode and can significantly "
               "increase runtime.\n";
    }
  }

  LNSParams::Market marketCli;
  marketCli.heuristics = vm["marketHeuristics"].as<bool>();
  marketCli.updateOnAcceptedOnly = vm["marketUpdateOnAcceptedOnly"].as<bool>();
  marketCli.updateFromCandidate = vm["marketUpdateFromCandidate"].as<bool>();
  marketCli.acceptanceGuards = vm["marketAcceptanceGuards"].as<bool>();
  marketCli.destroyRequireStable = vm["marketDestroyRequireStable"].as<bool>();
  marketCli.destroySoftGate = vm["marketDestroySoftGate"].as<bool>();
  marketCli.repairTieBreak = vm["marketRepairTieBreak"].as<bool>();
  marketCli.repairBlend = vm["marketRepairBlend"].as<bool>();
  marketCli.repairNormalizeByObservedPrice =
      vm["marketRepairNormalizeByObservedPrice"].as<bool>();
  for (const auto& spec : kMarketIntOptions) {
    marketCli.*(spec.field) = vm[spec.name].as<int>();
  }
  for (const auto& spec : kMarketDoubleOptions) {
    marketCli.*(spec.field) = vm[spec.name].as<double>();
  }

  if (marketCli.bucketDt <= 0) {
    PLOGE << "marketBucketDt must be a positive integer\n";
    return 1;
  }
  if (marketCli.vertexBucketCapacity <= 0 ||
      marketCli.edgeBucketCapacity <= 0) {
    PLOGE << "marketVertexBucketCapacity and marketEdgeBucketCapacity must be positive integers\n";
    return 1;
  }
  if (marketCli.updatePeriodAccepted <= 0) {
    PLOGE << "marketUpdatePeriodAccepted must be a positive integer\n";
    return 1;
  }
  if (marketCli.eta < 0.0 || marketCli.priceCap < 0.0 ||
      marketCli.priceInit < 0.0 ||
      marketCli.gamma < 0.0) {
    PLOGE << "marketEta, marketPriceCap, marketPriceInit and marketGamma "
             "must be non-negative\n";
    return 1;
  }
  if (marketCli.stabilityEmaAlpha < 0.0 || marketCli.stabilityEmaAlpha > 1.0) {
    PLOGE << "marketStabilityEmaAlpha must be in [0, 1]\n";
    return 1;
  }
  if (marketCli.stabilityMaxRelPriceDelta < 0.0 ||
      marketCli.stabilityMaxTopMassDelta < 0.0) {
    PLOGE << "marketStabilityMaxRelPriceDelta and marketStabilityMaxTopMassDelta must be non-negative\n";
    return 1;
  }
  if (marketCli.destroyWarmupWeightScale < 0.0 ||
      marketCli.destroyUnstableWeightScale < 0.0 ||
      marketCli.destroyMinAlnsWeight < 0.0) {
    PLOGE << "marketDestroyWarmupWeightScale, marketDestroyUnstableWeightScale and marketDestroyMinAlnsWeight must be non-negative\n";
    return 1;
  }
  if (marketCli.stabilityMinContendedJaccard < 0.0 ||
      marketCli.stabilityMinContendedJaccard > 1.0) {
    PLOGE << "marketStabilityMinContendedJaccard must be in [0, 1]\n";
    return 1;
  }
  if (marketCli.rho <= 0.0 || marketCli.rho >= 1.0) {
    PLOGE << "marketRho must be strictly between 0 and 1\n";
    return 1;
  }
  if (marketCli.seedTopFrac <= 0.0 || marketCli.seedTopFrac > 1.0) {
    PLOGE << "marketSeedTopFrac must be in (0, 1]\n";
    return 1;
  }
  if (marketCli.randomDestroyQuota < 0.0 ||
      marketCli.randomDestroyQuota > 1.0) {
    PLOGE << "marketRandomDestroyQuota must be in [0, 1]\n";
    return 1;
  }
  if (marketCli.cooldownIters < 0 || marketCli.destroyWarmupUpdates < 0 ||
      marketCli.dUp < 0 || marketCli.dDown < 0 || marketCli.closureCap < 0) {
    PLOGE << "marketCooldownIters, marketDestroyWarmupUpdates, marketDUp, "
             "marketDDown and marketClosureCap must be non-negative\n";
    return 1;
  }
  if (marketCli.tieBreakEpsSoc < 0.0 || marketCli.lambdaPrice < 0.0 ||
      marketCli.lambdaWait < 0.0) {
    PLOGE << "marketTieBreakEpsSoc, marketLambdaPrice and marketLambdaWait must be non-negative\n";
    return 1;
  }
  if (marketCli.destroyWeightPrice < 0.0 || marketCli.destroyWeightWait < 0.0 ||
      marketCli.destroyWeightRoot < 0.0) {
    PLOGE << "marketDestroyWeightPrice, marketDestroyWeightWait and "
             "marketDestroyWeightRoot must be non-negative\n";
    return 1;
  }
  if (destroyHeuristic == "market_tatonnement" && !marketCli.heuristics) {
    PLOGE << "destroyHeuristic='market_tatonnement' requires "
             "--marketHeuristics\n";
    return 1;
  }

  const int agentNum = vm["agentNum"].as<int>();
  const int taskNum = vm["taskNum"].as<int>();
  const int neighborSize = vm["neighborSize"].as<int>();
  const int maxIterations = vm["maxIterations"].as<int>();
  const int regretCandidateTopK = vm["regretCandidateTopK"].as<int>();
  const bool adaptiveRegretTopK = vm["adaptiveRegretTopK"].as<bool>();
  const double maxCascadeFactor = vm["maxCascadeFactor"].as<double>();
  const int maxCascadeTasks = vm["maxCascadeTasks"].as<int>();
  const bool adaptiveCascadeBudget =
      vm["adaptiveCascadeBudget"].as<bool>();
  const bool partialSolutionRestore = vm["partialSolutionRestore"].as<bool>();
  const bool repairIncludeNonAncestorAgents =
      vm["repairIncludeNonAncestorAgents"].as<bool>();
  const bool alnsEnablePrecedenceAwareDestroy =
      vm["alnsEnablePrecedenceAwareDestroy"].as<bool>();
  if (agentNum < 0) {
    PLOGE << "agentNum must be non-negative (0 means all agents from file)\n";
    return 1;
  }
  if (taskNum < 0) {
    PLOGE << "taskNum must be non-negative (0 means all tasks from file)\n";
    return 1;
  }
  if (neighborSize < 0) {
    PLOGE << "neighborSize must be non-negative\n";
    return 1;
  }
  if (maxIterations < 0) {
    PLOGE << "maxIterations must be non-negative\n";
    return 1;
  }
  if (regretCandidateTopK < 0) {
    PLOGE << "regretCandidateTopK must be non-negative (0 means all positions)\n";
    return 1;
  }
  if (!std::isfinite(maxCascadeFactor) || maxCascadeFactor < 0.0) {
    PLOGE << "maxCascadeFactor must be finite and non-negative\n";
    return 1;
  }
  if (maxCascadeTasks < 0) {
    PLOGE << "maxCascadeTasks must be non-negative (0 uses factor formula)\n";
    return 1;
  }

  const double cutoffTime = vm["cutoffTime"].as<double>();
  if (!std::isfinite(cutoffTime) || cutoffTime < 0.0) {
    PLOGE << "cutoffTime must be finite and non-negative\n";
    return 1;
  }
  if (cutoffTime == 0.0) {
    PLOGW << "cutoffTime is 0: planner will stop immediately after start\n";
  }

  // Need to store the seed for debugging.
  unsigned int seed = vm["seed"].as<unsigned int>();
  if (seed == 0) {
    std::uint64_t ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    // Mix entropy before narrowing to 32-bit to avoid low-quality truncation.
    ticks ^= ticks >> 30;
    ticks *= 0xbf58476d1ce4e5b9ULL;
    ticks ^= ticks >> 27;
    ticks *= 0x94d049bb133111ebULL;
    ticks ^= ticks >> 31;
    seed = static_cast<unsigned int>(ticks ^ (ticks >> 32));
  }
  // Keep seed on stdout for easy script parsing.
  std::cout << "seed = " << seed << '\n';
  std::unique_ptr<Instance> instancePtr;
  try {
    instancePtr = std::make_unique<Instance>(
        vm["map"].as<string>(), vm["agents"].as<string>(), agentNum, taskNum);
  } catch (const std::exception& e) {
    PLOGE << "Initialization failed: " << e.what() << "\n";
    return 1;
  }
  Instance& instance = *instancePtr;

  for (int i = 0; i < instance.getAgentNum(); i++) {
    pair<int, int> location =
        instance.getCoordinate(instance.getStartLocationsRef()[i]);
    PLOGD << "Agent " << i << " starts at :(" << location.first << ", "
          << location.second << ")\n";
  }
  for (int i = 0; i < instance.getTasksNum(); i++) {
    pair<int, int> location =
        instance.getCoordinate(instance.getTaskLocations(i));
    PLOGD << "Task " << i << " starts at :(" << location.first << ", "
          << location.second << ")\n";
  }
  const auto& ancestors = instance.getAncestorsRef();
  for (int task = 0; task < (int)ancestors.size(); ++task) {
    if (ancestors[task].empty()) {
      continue;
    }
    PLOGD << "Task  " << task << " has the following predecessor tasks\n";
    for (int predecessor : ancestors[task]) {
      PLOGD << "\t Task : " << predecessor << "\n";
    }
  }

  LNSParams parameters{};
  parameters.core.neighborhoodSize = neighborSize;
  parameters.core.timeLimit = cutoffTime;
  parameters.core.initialSolutionStrategy = initialSolutionStrategy;
  parameters.core.initialSolutionFallback = initialSolutionFallback;
  parameters.core.initialPortfolioTimeFraction =
      initialPortfolioTimeFraction;
  parameters.core.initialPortfolioMinArmTimeSec =
      initialPortfolioMinArmTimeSec;
  parameters.core.initialPortfolioStopOnFirstFeasible =
      initialPortfolioStopOnFirstFeasible;
  parameters.core.goalOccupationMode = goalOccupationMode;
  parameters.core.goalTailSteps = goalTailSteps;
  parameters.core.repositionMaxCandidates = repositionMaxCandidates;
  parameters.core.repositionDemandLookahead = repositionDemandLookahead;
  parameters.core.repositionReservationSlack = repositionReservationSlack;
  parameters.core.greedySegmentDiagnostics = greedySegmentDiagnostics;
  parameters.core.greedySegmentDiagnosticsTopK = greedySegmentDiagnosticsTopK;
  parameters.core.destroyHeuristic = destroyHeuristic;
  parameters.core.acceptanceCriteria = acceptanceCriteria;
  parameters.core.repairHeuristic = repairHeuristic;
  parameters.core.rejectInvalidCandidates = rejectInvalidCandidates;
  parameters.core.utilityUseConflictEventCount = utilityUseConflictEventCount;
  parameters.core.acceptanceFeasibilityFirstPrecedenceDebt =
      acceptanceFeasibilityFirstPrecedenceDebt;
  parameters.core.acceptanceInvalidSpatialWeight =
      acceptanceInvalidSpatialWeight;
  parameters.core.acceptanceInvalidPrecedenceDebtWeight =
      acceptanceInvalidPrecedenceDebtWeight;
  parameters.core.acceptanceInvalidSocTieBreakWeight =
      acceptanceInvalidSocTieBreakWeight;
  parameters.core.acceptanceUseDedicatedInvalidTemperature =
      acceptanceUseDedicatedInvalidTemperature;
  parameters.core.acceptanceInvalidTemperatureScale =
      acceptanceInvalidTemperatureScale;
  parameters.core.acceptanceInvalidTemperatureFloor =
      acceptanceInvalidTemperatureFloor;
  parameters.core.regretType = regretType;
  parameters.core.incrementalRegret = incrementalRegret;
  parameters.core.regretCandidateTopK = regretCandidateTopK;
  parameters.core.adaptiveRegretTopK = adaptiveRegretTopK;
  parameters.core.regretShortlistUseNormalizedWaitProxy =
      regretShortlistUseNormalizedWaitProxy;
  parameters.core.regretShortlistUseNormalizedSuccessorPressure =
      regretShortlistUseNormalizedSuccessorPressure;
  parameters.core.regretShortlistClampSuccessorToPrecedenceRelease =
      regretShortlistClampSuccessorToPrecedenceRelease;
  parameters.core.regretShortlistUseDescendantWeightedSuccessorPressure =
      regretShortlistUseDescendantWeightedSuccessorPressure;
  parameters.core.regretShortlistSuccessorPressureDepthDecay =
      regretShortlistSuccessorPressureDepthDecay;
  parameters.core.regretShortlistSuccessorPressureMaxDepth =
      regretShortlistSuccessorPressureMaxDepth;
  parameters.core.regretShortlistDiagnostics = regretShortlistDiagnostics;
  parameters.core.maxCascadeFactor = maxCascadeFactor;
  parameters.core.maxCascadeTasks = maxCascadeTasks;
  parameters.core.adaptiveCascadeBudget = adaptiveCascadeBudget;
  parameters.core.partialSolutionRestore = partialSolutionRestore;
  parameters.core.repairIncludeNonAncestorAgents =
      repairIncludeNonAncestorAgents;
  parameters.core.alnsEnablePrecedenceAwareDestroy =
      alnsEnablePrecedenceAwareDestroy;
  parameters.core.incrementalRegretMode = incrementalRegretMode;
  parameters.core.seed = seed;

  parameters.lowLevel.parityCheck = plannerParityCheck;
  parameters.lowLevel.parityMaxLogs = plannerParityMaxLogs;
  parameters.lowLevel.planner = lowLevelPlanner;
  parameters.lowLevel.segmentTimeout = lowLevelSegmentTimeout;
  parameters.lowLevel.mlastarIncrementalFocalRefresh =
      mlastarIncrementalFocalRefresh;

  parameters.market = marketCli;
  auto lnsInstance =
      std::make_unique<LNS>(maxIterations, instance, parameters);
  bool success = lnsInstance->run();

  const FeasibleSolution& anytimeSolution = lnsInstance->getFeasibleSolution();
  if (success) {
    PLOGI << "Anytime solution found!\n";
    // Keep anytime solution on stdout for easy script parsing.
    std::cout << anytimeSolution.toString() << '\n';
  } else {
    PLOGE << "Anytime solution was not found!\n";
  }

  const FeasibleTrajectoryStats feasibleStats =
      collectFeasibleTrajectoryStats(*lnsInstance, success);
  printAdaptiveLNSPerformance(*lnsInstance, destroyHeuristic);
  printFeasibleTrajectoryReport(feasibleStats);
  printRunSummaryReport(*lnsInstance, anytimeSolution, success,
                        parameters.market.heuristics, incrementalRegret,
                        feasibleStats);
  return 0;
}
