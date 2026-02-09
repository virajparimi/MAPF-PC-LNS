#include <plog/Log.h>
#include <plog/Severity.h>
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"

#include <chrono>
#include <iomanip>
#include <limits>
#include <boost/program_options.hpp>
#include "common.hpp"
#include "costchecker.hpp"
#include "instance.hpp"
#include "lns.hpp"
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
  desc.add_options()("marketHeuristics",
                     po::bool_switch()->default_value(false),
                     "Enable tatonnement-style market heuristics");
  desc.add_options()("marketBucketDt", po::value<int>()->default_value(3),
                     "Time bucket size used by market resources");
  desc.add_options()(
      "marketVertexBucketCapacity",
      po::value<int>()->default_value(2),
      "Capacity used for vertex-bucket market resources");
  desc.add_options()(
      "marketEdgeBucketCapacity",
      po::value<int>()->default_value(2),
      "Capacity used for edge-bucket market resources");
  desc.add_options()("marketUpdateOnAcceptedOnly",
                     po::value<bool>()->default_value(true),
                     "Update market prices only after accepted iterations");
  desc.add_options()("marketUpdatePeriodAccepted",
                     po::value<int>()->default_value(1),
                     "Number of accepted iterations between market updates");
  desc.add_options()("marketEta", po::value<double>()->default_value(0.05),
                     "Market tatonnement base step size");
  desc.add_options()("marketRho", po::value<double>()->default_value(0.9),
                     "EMA smoothing factor for excess demand");
  desc.add_options()("marketPriceCap", po::value<double>()->default_value(50.0),
                     "Maximum market resource price");
  desc.add_options()("marketGamma", po::value<double>()->default_value(0.01),
                     "Price evaporation factor per market update");
  desc.add_options()("marketAcceptanceGuards",
                     po::bool_switch()->default_value(false),
                     "Enable market pressure / precedence-wait acceptance guards");
  desc.add_options()("marketTauP", po::value<double>()->default_value(0.0),
                     "Acceptance guard threshold for market pressure");
  desc.add_options()("marketTauW", po::value<double>()->default_value(0.0),
                     "Acceptance guard threshold for precedence wait");
  desc.add_options()("marketDestroyWeightPrice",
                     po::value<double>()->default_value(1.0),
                     "Weight of market exposure in market destroy burden");
  desc.add_options()("marketDestroyWeightWait",
                     po::value<double>()->default_value(2.0),
                     "Weight of precedence wait in market destroy burden");
  desc.add_options()("marketDestroyWeightRoot",
                     po::value<double>()->default_value(1.5),
                     "Weight of blocker-root wait in market destroy burden");
  desc.add_options()("marketSeedTopFrac",
                     po::value<double>()->default_value(0.2),
                     "Top fraction of burden-ranked tasks used as market destroy seed pool");
  desc.add_options()("marketRandomDestroyQuota",
                     po::value<double>()->default_value(0.15),
                     "Random sampling quota for market destroy neighborhoods");
  desc.add_options()("marketCooldownIters",
                     po::value<int>()->default_value(3),
                     "Cooldown iterations before re-selecting a task in market destroy");
  desc.add_options()("marketDUp", po::value<int>()->default_value(1),
                     "Ancestor expansion depth for market destroy");
  desc.add_options()("marketDDown", po::value<int>()->default_value(1),
                     "Successor expansion depth for market destroy");
  desc.add_options()("marketClosureCap",
                     po::value<int>()->default_value(0),
                     "Maximum tasks expanded by market closure (0 = neighbor size cap)");
  desc.add_options()("marketRepairTieBreak",
                     po::bool_switch()->default_value(false),
                     "Enable market-aware tie-break in repair when deltaSoC is near zero");
  desc.add_options()("marketRepairBlend",
                     po::bool_switch()->default_value(false),
                     "Enable blended market-aware repair score");
  desc.add_options()("marketTieBreakEpsSoc",
                     po::value<double>()->default_value(0.0),
                     "Tie-break epsilon on |deltaSoC| for market-aware repair");
  desc.add_options()("marketLambdaPrice",
                     po::value<double>()->default_value(0.0),
                     "Repair weight for market exposure delta");
  desc.add_options()("marketLambdaWait",
                     po::value<double>()->default_value(0.0),
                     "Repair weight for precedence-wait delta");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (const std::exception& e) {
    PLOGE << e.what() << "\n" << desc << endl;
    return 1;
  }

  if (vm.count("help") != 0u) {
    plog::get()->setMaxSeverity(plog::debug);
    PLOGD << desc << endl;
    return 0;
  }

  try {
    po::notify(vm);
  } catch (const std::exception& e) {
    PLOGE << e.what() << "\n" << desc << endl;
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
          << endl;
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
  if (plannerParityMaxLogs <= 0) {
    PLOGE << "plannerParityMaxLogs must be a positive integer\n";
    return 1;
  }

  const bool marketHeuristics = vm["marketHeuristics"].as<bool>();
  const int marketBucketDt = vm["marketBucketDt"].as<int>();
  const int marketVertexBucketCapacity =
      vm["marketVertexBucketCapacity"].as<int>();
  const int marketEdgeBucketCapacity =
      vm["marketEdgeBucketCapacity"].as<int>();
  const bool marketUpdateOnAcceptedOnly =
      vm["marketUpdateOnAcceptedOnly"].as<bool>();
  const int marketUpdatePeriodAccepted =
      vm["marketUpdatePeriodAccepted"].as<int>();
  const double marketEta = vm["marketEta"].as<double>();
  const double marketRho = vm["marketRho"].as<double>();
  const double marketPriceCap = vm["marketPriceCap"].as<double>();
  const double marketGamma = vm["marketGamma"].as<double>();
  const bool marketAcceptanceGuards = vm["marketAcceptanceGuards"].as<bool>();
  const double marketTauP = vm["marketTauP"].as<double>();
  const double marketTauW = vm["marketTauW"].as<double>();
  const double marketDestroyWeightPrice =
      vm["marketDestroyWeightPrice"].as<double>();
  const double marketDestroyWeightWait =
      vm["marketDestroyWeightWait"].as<double>();
  const double marketDestroyWeightRoot =
      vm["marketDestroyWeightRoot"].as<double>();
  const double marketSeedTopFrac = vm["marketSeedTopFrac"].as<double>();
  const double marketRandomDestroyQuota =
      vm["marketRandomDestroyQuota"].as<double>();
  const int marketCooldownIters = vm["marketCooldownIters"].as<int>();
  const int marketDUp = vm["marketDUp"].as<int>();
  const int marketDDown = vm["marketDDown"].as<int>();
  const int marketClosureCap = vm["marketClosureCap"].as<int>();
  const bool marketRepairTieBreak = vm["marketRepairTieBreak"].as<bool>();
  const bool marketRepairBlend = vm["marketRepairBlend"].as<bool>();
  const double marketTieBreakEpsSoc = vm["marketTieBreakEpsSoc"].as<double>();
  const double marketLambdaPrice = vm["marketLambdaPrice"].as<double>();
  const double marketLambdaWait = vm["marketLambdaWait"].as<double>();

  if (marketBucketDt <= 0) {
    PLOGE << "marketBucketDt must be a positive integer\n";
    return 1;
  }
  if (marketVertexBucketCapacity <= 0 || marketEdgeBucketCapacity <= 0) {
    PLOGE << "marketVertexBucketCapacity and marketEdgeBucketCapacity must be positive integers\n";
    return 1;
  }
  if (marketUpdatePeriodAccepted <= 0) {
    PLOGE << "marketUpdatePeriodAccepted must be a positive integer\n";
    return 1;
  }
  if (marketEta < 0.0 || marketPriceCap < 0.0 || marketGamma < 0.0) {
    PLOGE << "marketEta, marketPriceCap and marketGamma must be non-negative\n";
    return 1;
  }
  if (marketRho <= 0.0 || marketRho >= 1.0) {
    PLOGE << "marketRho must be strictly between 0 and 1\n";
    return 1;
  }
  if (marketSeedTopFrac <= 0.0 || marketSeedTopFrac > 1.0) {
    PLOGE << "marketSeedTopFrac must be in (0, 1]\n";
    return 1;
  }
  if (marketRandomDestroyQuota < 0.0 || marketRandomDestroyQuota > 1.0) {
    PLOGE << "marketRandomDestroyQuota must be in [0, 1]\n";
    return 1;
  }
  if (marketCooldownIters < 0 || marketDUp < 0 || marketDDown < 0 ||
      marketClosureCap < 0) {
    PLOGE << "marketCooldownIters, marketDUp, marketDDown and marketClosureCap must be non-negative\n";
    return 1;
  }
  if (marketTieBreakEpsSoc < 0.0 || marketLambdaPrice < 0.0 ||
      marketLambdaWait < 0.0) {
    PLOGE << "marketTieBreakEpsSoc, marketLambdaPrice and marketLambdaWait must be non-negative\n";
    return 1;
  }

  // Need to store the seed for debugging.
  unsigned int seed = vm["seed"].as<unsigned int>();
  if (seed == 0) {
    seed = (unsigned int)std::chrono::high_resolution_clock::now()
               .time_since_epoch()
               .count();
  }
  std::cout << "seed = " << seed << std::endl;
  srand(seed);

  Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
                    vm["agentNum"].as<int>(), vm["taskNum"].as<int>());

  for (int i = 0; i < instance.getAgentNum(); i++) {
    pair<int, int> location =
        instance.getCoordinate(instance.getStartLocations()[i]);
    PLOGD << "Agent " << i << " starts at :(" << location.first << ", "
          << location.second << ")\n";
  }
  for (int i = 0; i < instance.getTasksNum(); i++) {
    pair<int, int> location =
        instance.getCoordinate(instance.getTaskLocations(i));
    PLOGD << "Task " << i << " starts at :(" << location.first << ", "
          << location.second << ")\n";
  }
  for (const auto& dependencies : instance.getTaskDependencies()) {
    PLOGD << "Task  " << dependencies.first
          << " has the following dependent tasks\n";
    for (int task : dependencies.second) {
      PLOGD << "\t Task : " << task << "\n";
    }
  }

  LNSParams parameters(vm["neighborSize"].as<int>(),
                       vm["cutoffTime"].as<double>(), 100, 0.99975, 1.00025, 5,
                       9, 3, 0.75, 0.25, initialSolutionStrategy,
                       destroyHeuristic, acceptanceCriteria, regretType,
                       incrementalRegret, incrementalRegretMode, seed,
                       plannerParityCheck, plannerParityMaxLogs, marketHeuristics,
                       marketBucketDt, marketVertexBucketCapacity,
                       marketEdgeBucketCapacity, marketUpdateOnAcceptedOnly,
                       marketUpdatePeriodAccepted, marketEta, marketRho,
                       marketPriceCap, marketGamma, marketAcceptanceGuards,
                       marketTauP, marketTauW, marketDestroyWeightPrice,
                       marketDestroyWeightWait, marketDestroyWeightRoot,
                       marketSeedTopFrac, marketRandomDestroyQuota,
                       marketCooldownIters, marketDUp, marketDDown,
                       marketClosureCap, marketRepairTieBreak,
                       marketRepairBlend, marketTieBreakEpsSoc,
                       marketLambdaPrice, marketLambdaWait,
                       lowLevelPlanner);
  auto lnsInstance =
      std::make_unique<LNS>(vm["maxIterations"].as<int>(), instance, parameters);
  bool success = lnsInstance->run();

  FeasibleSolution anytimeSolution = lnsInstance->getFeasibleSolution();
  if (success) {
    PLOGI << "Anytime solution found!\n";
    std::cout << anytimeSolution.toString() << std::endl;
  } else {
    PLOGE << "Anytime solution was not found!\n";
  }

  if (vm["genReport"].as<bool>()) {
    SaveToTxt saveFileForCBSPC;
    saveFileForCBSPC.printStart();
    Solution finalSolution = lnsInstance->getSolution();
    saveFileForCBSPC.runData(&instance, &finalSolution);
    saveFileForCBSPC.fileSave();
  }

  vector<double> feasibleSolutionIterations, feasibleSolutionRuntimes,
      feasibleSolutionValues;
  double firstFeasibleSolutionTime = 0, numFeasibleSolutionUpdate = 0,
         couldNotFindCounter = 0, counter = 0;
  if (success) {
    for (const IterationStats& iter : lnsInstance->iterationStats) {
      if (iter.feasibleSolutionFound) {
        firstFeasibleSolutionTime = iter.runtime;
        break;
      }
    }
    for (const IterationStats& iter : lnsInstance->iterationStats) {
      if (iter.feasibleSolutionFound) {
        numFeasibleSolutionUpdate += 1;
        feasibleSolutionIterations.push_back(counter);
        feasibleSolutionRuntimes.push_back(iter.runtime);
        feasibleSolutionValues.push_back(iter.sumOfCosts);
      }
      if (iter.quality == IterationQuality::couldNotFind) {
        couldNotFindCounter++;
      }
      counter++;
    }
  } else {
    firstFeasibleSolutionTime = std::numeric_limits<double>::infinity();
  }

  // Compute the results from ALNS
  if (destroyHeuristic == "alns") {
    ALNS adaptiveLNS = lnsInstance->getAdaptiveLNS();
    constexpr int kNumDestroyHeuristics =
        (int)DestroyHeuristic::destroyHeuristicCount;
    vector<int> destroyHeuristicFrequency(kNumDestroyHeuristics, 0);
    std::cout << "Size of destroy heuristic history -> "
              << adaptiveLNS.destroyHeuristicHistory.size() << std::endl;
    for (int destroyHeuristicUsed : adaptiveLNS.destroyHeuristicHistory) {
      if (destroyHeuristicUsed >= 0 &&
          destroyHeuristicUsed < kNumDestroyHeuristics) {
        destroyHeuristicFrequency[destroyHeuristicUsed] += 1;
      }
    }
    auto heuristicName = [](DestroyHeuristic h) -> const char* {
      switch (h) {
      case DestroyHeuristic::randomRemoval:
        return "Random";
      case DestroyHeuristic::worstRemoval:
        return "Worst";
      case DestroyHeuristic::conflictRemoval:
        return "Conflict";
      case DestroyHeuristic::shawRemoval:
        return "Shaw";
      case DestroyHeuristic::precedenceWaitRemoval:
        return "PrecedenceWait";
      case DestroyHeuristic::lowSlackRemoval:
        return "LowSlack";
      case DestroyHeuristic::marketTatonnementRemoval:
        return "MarketTatonnement";
      default:
        return "Unknown";
      }
    };

    if (!adaptiveLNS.destroyHeuristicHistory.empty()) {
      const std::ios::fmtflags oldFlags = std::cout.flags();
      const std::streamsize oldPrecision = std::cout.precision();

      std::cout << "Adaptive LNS performance:\n";
      std::cout << std::left << std::setw(18) << "Heuristic"
                << std::right << std::setw(9) << "Share%"
                << std::setw(10) << "Selected"
                << std::setw(10) << "Accept%"
                << std::setw(12) << "BestUpd%"
                << std::setw(12) << "FailFind%"
                << std::setw(15) << "AvgDelta(all)"
                << std::setw(15) << "AvgDelta(acc)" << '\n';
      std::cout << std::string(101, '-') << '\n';

      std::cout << std::fixed << std::setprecision(2);
      for (int i = 0; i < kNumDestroyHeuristics; i++) {
        const double share =
            (double)destroyHeuristicFrequency[i] /
            (int)adaptiveLNS.destroyHeuristicHistory.size();
        const int64_t selected = adaptiveLNS.selections[i];
        const int64_t accepted = adaptiveLNS.accepted[i];
        const int64_t bestUpdates = adaptiveLNS.bestUpdates[i];
        const int64_t couldNotFind = adaptiveLNS.couldNotFind[i];
        const double acceptRate =
            selected > 0 ? (double)accepted / (double)selected : 0.0;
        const double bestRate =
            selected > 0 ? (double)bestUpdates / (double)selected : 0.0;
        const double couldNotFindRate =
            selected > 0 ? (double)couldNotFind / (double)selected : 0.0;
        const double avgDeltaSocAll =
            selected > 0 ? adaptiveLNS.deltaSocAll[i] / (double)selected : 0.0;
        const double avgDeltaSocAccepted =
            accepted > 0 ? adaptiveLNS.deltaSocAccepted[i] / (double)accepted
                         : 0.0;
        std::cout << std::left << std::setw(18)
                  << heuristicName((DestroyHeuristic)i)
                  << std::right << std::setw(9) << (share * 100.0)
                  << std::setw(10) << selected
                  << std::setw(10) << (acceptRate * 100.0)
                  << std::setw(12) << (bestRate * 100.0)
                  << std::setw(12) << (couldNotFindRate * 100.0)
                  << std::setw(15) << avgDeltaSocAll
                  << std::setw(15) << avgDeltaSocAccepted << '\n';
      }
      std::cout.flags(oldFlags);
      std::cout.precision(oldPrecision);
    } else {
      std::cout << "Adaptive LNS performance: no heuristic samples collected.\n";
    }
  }

  std::cout << "\n\nCould not find solution for " << couldNotFindCounter
            << " iterations!\n";

  std::cout << "\n\nFeasible Solution Iterations: \n";
  for (double iter : feasibleSolutionIterations) {
    std::cout << iter << ",\t";
  }
  std::cout << "\nFeasible Solution Runtimes: \n";
  for (double runtimes : feasibleSolutionRuntimes) {
    std::cout << runtimes << ",\t";
  }
  std::cout << "\nFeasible Solution Values: \n";
  for (double values : feasibleSolutionValues) {
    std::cout << values << ",\t";
  }

  std::cout << "\n\nMAPF-PC-LNS: "
            << "\n\tRuntime = " << lnsInstance->runtime
            << "\n\tIterations = " << lnsInstance->iterationStats.size()
            << "\n\tFirst Feasible Solution Runtime = "
            << firstFeasibleSolutionTime
            << "\n\tAverage Update of Feasible Solution = "
            << numFeasibleSolutionUpdate
            << "\n\tSolution Cost = " << anytimeSolution.sumOfCosts
            << "\n\tNumber of failures = " << lnsInstance->numOfFailures
            << "\n\tSuccess = " << success << endl;

  if (marketHeuristics) {
    const MarketStats marketStats = lnsInstance->getMarketStats();
    std::cout << "\n\nMarket Stats: "
              << "\n\tUpdates = " << marketStats.updates
              << "\n\tContended Resources = " << marketStats.contendedResources
              << "\n\tMean Price (Contended) = "
              << marketStats.meanPriceContended
              << "\n\tMax Price = " << marketStats.maxPrice
              << "\n\tTop Price Mass Fraction = "
              << marketStats.topPriceMassFrac
              << "\n\tTotal Precedence Wait = "
              << marketStats.totalPrecedenceWait
              << "\n\tMax Precedence Wait = "
              << marketStats.maxPrecedenceWait << endl;
  }

  const auto regretStats = lnsInstance->getRegretEvalStats();
  const double feasibleRate =
      regretStats.candidateInsertionsTried > 0
          ? (double)regretStats.candidateInsertionsFeasible /
                (double)regretStats.candidateInsertionsTried
          : 0.0;
  const double avgRemovedTasks =
      regretStats.neighborhoods > 0
          ? (double)regretStats.removedTasksSum /
                (double)regretStats.neighborhoods
          : 0.0;
  std::cout << "\n\nRegret Evaluation Stats: "
            << "\n\tRecompute Calls = " << regretStats.recomputeCalls
            << "\n\tTasks Evaluated = " << regretStats.tasksEvaluated
            << "\n\tTask-Agent Evaluations = " << regretStats.agentEvaluations
            << "\n\tCandidate Insertions Tried = "
            << regretStats.candidateInsertionsTried
            << "\n\tCandidate Insertions Feasible = "
            << regretStats.candidateInsertionsFeasible
            << "\n\tFeasible Rate = " << feasibleRate
            << "\n\tRepair Neighborhoods = " << regretStats.neighborhoods
            << "\n\tAvg Removed Tasks/Neighborhood = " << avgRemovedTasks
            << "\n\tMax Removed Tasks/Neighborhood = "
            << regretStats.removedTasksMax << endl;

  if (incrementalRegret) {
    const auto statsOpt = lnsInstance->getIncrementalRegretStats();
    if (statsOpt.has_value()) {
      const auto& stats = statsOpt.value();
      const double avgDirty =
          stats.commits > 0 ? (double)stats.dirtySum / (double)stats.commits
                            : 0.0;
      const double avgChanged =
          stats.commits > 0 ? (double)stats.changedSum / (double)stats.commits
                            : 0.0;
      const double avgRecomputedTasksPerCommit =
          stats.commits > 0
              ? (double)stats.recomputedTasks / (double)stats.commits
              : 0.0;

      std::cout << "\n\nIncremental Regret Stats: "
                << "\n\tMode = " << lnsInstance->getIncrementalRegretMode()
                << "\n\tCommits = " << stats.commits
                << "\n\tRecompute Calls = " << stats.recomputeCalls
                << "\n\tRecomputed Tasks = " << stats.recomputedTasks
                << "\n\tRecomputed Tasks/Commit = "
                << avgRecomputedTasksPerCommit
                << "\n\tStale Heap Pops = " << stats.stalePops
                << "\n\tHeap Rebuilds = " << stats.heapRebuilds
                << "\n\tFull Refreshes = " << stats.fullRefreshes
                << "\n\tAvg Dirty Tasks/Commit = " << avgDirty
                << "\n\tMax Dirty Tasks = " << stats.dirtyMax
                << "\n\tAvg Changed EndTimes/Commit = " << avgChanged
                << "\n\tMax Changed EndTimes = " << stats.changedMax << endl;
    }
  }
  return 0;
}
