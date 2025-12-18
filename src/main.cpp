#include <plog/Log.h>
#include <plog/Severity.h>
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"

#include <chrono>
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
      destroyHeuristic != "alns") {
    PLOGE << "The destroy heuristic provided is not supported! Please choose "
             "from 'conflict', 'worst', 'random', 'shaw' and 'alns' removal "
             "operators\n";
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
                       incrementalRegret, incrementalRegretMode, seed);
  LNS lnsInstance = LNS(vm["maxIterations"].as<int>(), instance, parameters);
  bool success = lnsInstance.run();

  FeasibleSolution anytimeSolution = lnsInstance.getFeasibleSolution();
  if (success) {
    PLOGI << "Anytime solution found!\n";
    std::cout << anytimeSolution.toString() << std::endl;
  } else {
    PLOGE << "Anytime solution was not found!\n";
  }

  if (vm["genReport"].as<bool>()) {
    SaveToTxt saveFileForCBSPC;
    saveFileForCBSPC.printStart();
    Solution finalSolution = lnsInstance.getSolution();
    saveFileForCBSPC.runData(&instance, &finalSolution);
    saveFileForCBSPC.fileSave();
  }

  vector<double> feasibleSolutionIterations, feasibleSolutionRuntimes,
      feasibleSolutionValues;
  double firstFeasibleSolutionTime = 0, numFeasibleSolutionUpdate = 0,
         couldNotFindCounter = 0, counter = 0;
  if (success) {
    for (const IterationStats& iter : lnsInstance.iterationStats) {
      if (iter.feasibleSolutionFound) {
        firstFeasibleSolutionTime = iter.runtime;
        break;
      }
    }
    for (const IterationStats& iter : lnsInstance.iterationStats) {
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
    ALNS adaptiveLNS = lnsInstance.getAdaptiveLNS();
    constexpr int kNumDestroyHeuristics =
        (int)DestroyHeuristic::shawRemoval + 1;
    vector<int> destroyHeuristicFrequency(kNumDestroyHeuristics, 0);
    std::cout << "Size of destroy heuristic history -> "
              << adaptiveLNS.destroyHeuristicHistory.size() << std::endl;
    for (int destroyHeuristicUsed : adaptiveLNS.destroyHeuristicHistory) {
      if (destroyHeuristicUsed >= 0 &&
          destroyHeuristicUsed < kNumDestroyHeuristics) {
        destroyHeuristicFrequency[destroyHeuristicUsed] += 1;
      }
    }
    std::cout << "Adaptive LNS performance: \n\t";
    if (!adaptiveLNS.destroyHeuristicHistory.empty()) {
      for (int i = 0; i < kNumDestroyHeuristics; i++) {
        double averageUsed = (double)destroyHeuristicFrequency[i] /
                             (int)adaptiveLNS.destroyHeuristicHistory.size();
        switch ((DestroyHeuristic)i) {
        case DestroyHeuristic::randomRemoval:
          std::cout << "Random Removal: " << averageUsed << "\n\t";
          break;
        case DestroyHeuristic::worstRemoval:
          std::cout << "Worst Removal: " << averageUsed << "\n\t";
          break;
        case DestroyHeuristic::conflictRemoval:
          std::cout << "Conflict Removal: " << averageUsed << "\n\t";
          break;
        case DestroyHeuristic::shawRemoval:
          std::cout << "Shaw Removal: " << averageUsed << "\n\t";
          break;
        default:
          break;
        }
      }
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
            << "\n\tRuntime = " << lnsInstance.runtime
            << "\n\tIterations = " << lnsInstance.iterationStats.size()
            << "\n\tFirst Feasible Solution Runtime = "
            << firstFeasibleSolutionTime
            << "\n\tAverage Update of Feasible Solution = "
            << numFeasibleSolutionUpdate
            << "\n\tSolution Cost = " << anytimeSolution.sumOfCosts
            << "\n\tNumber of failures = " << lnsInstance.numOfFailures
            << "\n\tSuccess = " << success << endl;

  const auto regretStats = lnsInstance.getRegretEvalStats();
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
    const auto statsOpt = lnsInstance.getIncrementalRegretStats();
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
                << "\n\tMode = " << lnsInstance.getIncrementalRegretMode()
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
