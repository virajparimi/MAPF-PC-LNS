#include <plog/Log.h>
#include <plog/Severity.h>
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"

#include <chrono>
#include <cstdint>
#include <boost/program_options.hpp>
#include "common.hpp"
#include "report_exporter.hpp"
#include "instance.hpp"
#include "lns.hpp"
#include "run_reporting.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {

  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::error, &consoleAppender);

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message");
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
  desc.add_options()("neighborSize,n", po::value<int>()->default_value(8),
                     "Size of the neighborhood");
  desc.add_options()("maxIterations,i", po::value<int>()->default_value(0),
                     "Maximum number of iterations");
  desc.add_options()("severity,d", po::value<int>()->default_value(0),
                     "Debugging level");
  desc.add_options()("initialSolution,s",
                     po::value<string>()->default_value("greedy"),
                     "Strategy for the initial solution");
  desc.add_options()(
      "destroyHeuristic,h", po::value<string>()->default_value("conflict"),
      "Destroy heuristic to use for creating the LNS neighborhood");
  desc.add_options()("acceptanceCriteria,c",
                     po::value<string>()->default_value("SA"),
                     "Acceptance criteria for new solutions");
  desc.add_options()("genReport,g", po::bool_switch()->default_value(false),
                     "Whether to generate the report file that can be fed to "
                     "CBS-PC for verification");
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
      {"marketGamma", 0.01, "Price evaporation factor per market update",
       &LNSParams::Market::gamma},
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
  desc.add_options()("marketAcceptanceGuards",
                     po::bool_switch()->default_value(false),
                     "Enable market pressure / precedence-wait acceptance guards");
  desc.add_options()("marketRepairTieBreak",
                     po::bool_switch()->default_value(false),
                     "Enable market-aware tie-break in repair when deltaSoC is near zero");
  desc.add_options()("marketRepairBlend",
                     po::bool_switch()->default_value(false),
                     "Enable blended market-aware repair score");
  for (const auto& spec : kMarketIntOptions) {
    desc.add_options()(spec.name, po::value<int>()->default_value(spec.defaultValue),
                       spec.description);
  }
  for (const auto& spec : kMarketDoubleOptions) {
    desc.add_options()(spec.name,
                       po::value<double>()->default_value(spec.defaultValue),
                       spec.description);
  }

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (const std::exception& e) {
    PLOGE << e.what() << "\n" << desc << "\n";
    return 1;
  }

  if (vm.count("help") != 0u) {
    plog::get()->setMaxSeverity(plog::debug);
    PLOGD << desc << "\n";
    return 0;
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
      initialSolutionStrategy != "greedy_precedence_only" &&
      initialSolutionStrategy.find("sota") == string::npos) {
    PLOGE << "Incorrect initial solution strategy provided. Please choose from "
             "'greedy', 'greedy_precedence_only', 'sota_cbs' or 'sota_pbs' options"
          << "\n";
    return 1;
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
  marketCli.acceptanceGuards = vm["marketAcceptanceGuards"].as<bool>();
  marketCli.repairTieBreak = vm["marketRepairTieBreak"].as<bool>();
  marketCli.repairBlend = vm["marketRepairBlend"].as<bool>();
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
      marketCli.gamma < 0.0) {
    PLOGE << "marketEta, marketPriceCap and marketGamma must be non-negative\n";
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
  if (marketCli.cooldownIters < 0 || marketCli.dUp < 0 || marketCli.dDown < 0 ||
      marketCli.closureCap < 0) {
    PLOGE << "marketCooldownIters, marketDUp, marketDDown and marketClosureCap must be non-negative\n";
    return 1;
  }
  if (marketCli.tieBreakEpsSoc < 0.0 || marketCli.lambdaPrice < 0.0 ||
      marketCli.lambdaWait < 0.0) {
    PLOGE << "marketTieBreakEpsSoc, marketLambdaPrice and marketLambdaWait must be non-negative\n";
    return 1;
  }

  const int agentNum = vm["agentNum"].as<int>();
  const int taskNum = vm["taskNum"].as<int>();
  const int neighborSize = vm["neighborSize"].as<int>();
  const int maxIterations = vm["maxIterations"].as<int>();
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
        vm["map"].as<string>(), vm["agents"].as<string>(),
        agentNum, taskNum);
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
  parameters.core.timeLimit = vm["cutoffTime"].as<double>();
  parameters.core.initialSolutionStrategy = initialSolutionStrategy;
  parameters.core.destroyHeuristic = destroyHeuristic;
  parameters.core.acceptanceCriteria = acceptanceCriteria;
  parameters.core.regretType = regretType;
  parameters.core.incrementalRegret = incrementalRegret;
  parameters.core.incrementalRegretMode = incrementalRegretMode;
  parameters.core.seed = seed;

  parameters.lowLevel.parityCheck = plannerParityCheck;
  parameters.lowLevel.parityMaxLogs = plannerParityMaxLogs;
  parameters.lowLevel.planner = lowLevelPlanner;
  parameters.lowLevel.segmentTimeout = lowLevelSegmentTimeout;

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

  if (vm["genReport"].as<bool>()) {
    CBSReportExporter reportExporter;
    reportExporter.printStart();
    const Solution& finalSolution = lnsInstance->getSolution();
    reportExporter.writeReport(&instance, &finalSolution);
    reportExporter.printSaveStatus();
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
