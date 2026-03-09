#include <plog/Log.h>
#include <plog/Severity.h>
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"

#include <chrono>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <boost/program_options.hpp>
#include "common.hpp"
#include "instance.hpp"
#include "lns.hpp"
#include "run_reporting.hpp"
#include "utils.hpp"

namespace {
struct CoreParameterInputs {
  int neighborhoodSize = 0;
  double timeLimit = 0.0;
  double lnsConflictWeight = 0.0;
  double lnsCostWeight = 0.0;
  string initialSolutionStrategy;
  string initialSeedFromMapfpcLog;
  bool postRefineWithMapfpc = false;
  string postRefineAssignmentSource;
  string postRefineAssignmentLog;
  string postRefineSolver;
  int postRefineTimeoutSec = 0;
  bool postRefineAcceptOnlyIfBetter = false;
  bool debugImprovementDiagnostics = false;
  string debugIterationTsvPath;
  string destroyHeuristic;
  string acceptanceCriteria;
  string optimizationObjective;
  bool acceptOnlyValidCandidates = false;
  string repairHeuristic;
  bool enableNrrRepair = false;
  bool nrrFallbackToStandard = false;
  bool nrrGlobalReassign = false;
  bool forceNeighborhoodChangeOnReject = false;
  string nrrMiniSolver;
  string repairMapfpcSolver;
  int repairMapfpcTimeoutSec = 0;
  string regretType;
  bool alnsEnablePrecedenceAwareDestroy = false;
  bool softRecoveryDestroyMode = false;
  bool softPersistentConflictGraph = false;
  unsigned int seed = 0;
};

LNSParams buildLnsParams(const CoreParameterInputs& coreInputs,
                         const LNSParams::LowLevel& lowLevelInputs) {
  LNSParams parameters{};
  parameters.core.neighborhoodSize = coreInputs.neighborhoodSize;
  parameters.core.timeLimit = coreInputs.timeLimit;
  parameters.core.lnsConflictWeight = coreInputs.lnsConflictWeight;
  parameters.core.lnsCostWeight = coreInputs.lnsCostWeight;
  parameters.core.initialSolutionStrategy = coreInputs.initialSolutionStrategy;
  parameters.core.initialSeedFromMapfpcLog =
      coreInputs.initialSeedFromMapfpcLog;
  parameters.core.postRefineWithMapfpc = coreInputs.postRefineWithMapfpc;
  parameters.core.postRefineAssignmentSource =
      coreInputs.postRefineAssignmentSource;
  parameters.core.postRefineAssignmentLog = coreInputs.postRefineAssignmentLog;
  parameters.core.postRefineSolver = coreInputs.postRefineSolver;
  parameters.core.postRefineTimeoutSec = coreInputs.postRefineTimeoutSec;
  parameters.core.postRefineAcceptOnlyIfBetter =
      coreInputs.postRefineAcceptOnlyIfBetter;
  parameters.core.debugImprovementDiagnostics =
      coreInputs.debugImprovementDiagnostics;
  parameters.core.debugIterationTsvPath = coreInputs.debugIterationTsvPath;
  parameters.core.destroyHeuristic = coreInputs.destroyHeuristic;
  parameters.core.acceptanceCriteria = coreInputs.acceptanceCriteria;
  parameters.core.optimizationObjective = coreInputs.optimizationObjective;
  parameters.core.acceptOnlyValidCandidates =
      coreInputs.acceptOnlyValidCandidates;
  parameters.core.repairHeuristic = coreInputs.repairHeuristic;
  parameters.core.enableNrrRepair = coreInputs.enableNrrRepair;
  parameters.core.nrrFallbackToStandard = coreInputs.nrrFallbackToStandard;
  parameters.core.nrrGlobalReassign = coreInputs.nrrGlobalReassign;
  parameters.core.forceNeighborhoodChangeOnReject =
      coreInputs.forceNeighborhoodChangeOnReject;
  parameters.core.nrrMiniSolver = coreInputs.nrrMiniSolver;
  parameters.core.repairMapfpcSolver = coreInputs.repairMapfpcSolver;
  parameters.core.repairMapfpcTimeoutSec = coreInputs.repairMapfpcTimeoutSec;
  parameters.core.regretType = coreInputs.regretType;
  parameters.core.alnsEnablePrecedenceAwareDestroy =
      coreInputs.alnsEnablePrecedenceAwareDestroy;
  parameters.core.softRecoveryDestroyMode =
      coreInputs.softRecoveryDestroyMode;
  parameters.core.softPersistentConflictGraph =
      coreInputs.softPersistentConflictGraph;
  parameters.core.seed = coreInputs.seed;
  parameters.lowLevel = lowLevelInputs;
  return parameters;
}
}  // namespace

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
  desc.add_options()(
      "lnsConflictWeight",
      po::value<double>()->default_value(0.75),
      "Weight of conflict signal in moving utility");
  desc.add_options()(
      "lnsCostWeight",
      po::value<double>()->default_value(0.25),
      "Weight of optimization objective signal in moving utility");
  desc.add_options()("agentNum,k", po::value<int>()->default_value(0),
                     "Number of agents to plan for");
  desc.add_options()("taskNum,l", po::value<int>()->default_value(0),
                     "Number of tasks to plan for");
  desc.add_options()(
      "neighborSize,n", po::value<int>()->default_value(0),
      "Size of the neighborhood (0 = adaptive formula based on agents/tasks/precedence)");
  desc.add_options()("maxIterations,i", po::value<int>()->default_value(0),
                     "Maximum number of iterations");
  desc.add_options()(
      "alnsEnablePrecedenceAwareDestroy",
      po::value<bool>()->default_value(true),
      "Allow ALNS to sample precedence_wait and low_slack destroy operators");
  desc.add_options()(
      "softRecoveryDestroyMode",
      po::value<bool>()->default_value(true),
      "If true, keep full ALNS destroy pool normally, but restrict to "
      "collision_soft/failure_soft while soft-recovery is active");
  desc.add_options()(
      "softPersistentConflictGraph",
      po::value<bool>()->default_value(false),
      "If true, soft destroy heuristics sample from a persistent "
      "accepted-solution conflict graph (pairs/agents)");
  desc.add_options()("severity,d", po::value<int>()->default_value(0),
                     "Debugging level");
  desc.add_options()("initialSolution,s",
                     po::value<string>()->default_value("portfolio"),
                     "Strategy for the initial solution (portfolio default for "
                     "best final SoC; use greedy/prioritized for faster time-to-best; "
                     "or use seeded_mapfpc_log with --initialSeedFromMapfpcLog)");
  desc.add_options()(
      "initialSeedFromMapfpcLog",
      po::value<string>()->default_value(""),
      "Path to MAPF-PC log (PBS/CBS) with TASK ASSIGNMENTS and TASK PATHS "
      "sections; if provided, seed initialization directly from this log");
  desc.add_options()(
      "postRefineWithMapfpc",
      po::value<bool>()->default_value(false),
      "After LNS, run MAPF-PC on a fixed assignment and optionally accept if "
      "better");
  desc.add_options()(
      "postRefineAssignmentSource",
      po::value<string>()->default_value("solution"),
      "Assignment source for post-refinement: 'solution' or 'log'");
  desc.add_options()(
      "postRefineAssignmentLog",
      po::value<string>()->default_value(""),
      "Path to MAPF-PC/PBS log used when postRefineAssignmentSource='log'");
  desc.add_options()(
      "postRefineSolver",
      po::value<string>()->default_value("pbs"),
      "MAPF-PC solver for post-refinement: 'pbs' or 'cbs'");
  desc.add_options()(
      "postRefineTimeoutSec",
      po::value<int>()->default_value(120),
      "MAPF-PC cutoff in seconds for post-refinement");
  desc.add_options()(
      "postRefineAcceptOnlyIfBetter",
      po::value<bool>()->default_value(true),
      "If true, adopt post-refined solution only when optimization objective strictly improves");
  desc.add_options()(
      "debugImprovementDiagnostics",
      po::value<bool>()->default_value(false),
      "Emit additional diagnostics on acceptance/improvement behavior and "
      "iteration time split");
  desc.add_options()(
      "debugIterationTsvPath",
      po::value<string>()->default_value(""),
      "Optional output TSV path for per-iteration debug diagnostics");
  desc.add_options()(
      "destroyHeuristic,H", po::value<string>()->default_value("alns"),
      "Destroy heuristic to use for creating the LNS neighborhood");
  desc.add_options()("acceptanceCriteria,c",
                     po::value<string>()->default_value("TA"),
                     "Acceptance criteria for new solutions");
  desc.add_options()(
      "optimizationObjective",
      po::value<string>()->default_value("soc"),
      "Optimization objective: 'soc' or 'makespan'");
  desc.add_options()(
      "acceptOnlyValidCandidates",
      po::value<bool>()->default_value(false),
      "If true, reject invalid candidates before applying acceptance criteria");
  desc.add_options()(
      "repairHeuristic",
      po::value<string>()->default_value("regret"),
      "Repair heuristic to use: 'regret'");
  desc.add_options()(
      "enableNrrRepair",
      po::value<bool>()->default_value(false),
      "If true, attempt Neighborhood Reoptimization Repair (NRR) before "
      "standard repair");
  desc.add_options()(
      "nrrFallbackToStandard",
      po::value<bool>()->default_value(true),
      "If true, failed NRR falls back to the configured standard repair; if "
      "false, failed NRR terminates repair for that iteration");
  desc.add_options()(
      "nrrGlobalReassign",
      po::value<bool>()->default_value(false),
      "If true, NRR iterative proposal evaluates insertions against all agents "
      "with dynamic frozen-occupancy demotion");
  desc.add_options()(
      "forceNeighborhoodChangeOnReject",
      po::value<bool>()->default_value(false),
      "If true, rejected iterations do not reuse the prior neighborhood seed");
  desc.add_options()(
      "nrrMiniSolver",
      po::value<string>()->default_value("cbs"),
      "NRR mini-solver mode: 'pbs', 'cbs', or 'auto' (default: cbs)");
  desc.add_options()(
      "repairMapfpcSolver",
      po::value<string>()->default_value("cbs"),
      "When repairHeuristic is MAPF-PC-based, solver to use: 'pbs' or 'cbs'");
  desc.add_options()(
      "repairMapfpcTimeoutSec",
      po::value<int>()->default_value(30),
      "When repairHeuristic is MAPF-PC-based, timeout in seconds per repair "
      "attempt");
  desc.add_options()("seed",
                     po::value<unsigned int>()->default_value(0),
                     "Random seed (0 = time-based)");
  desc.add_options()("regretType,r",
                     po::value<string>()->default_value("absolute"),
                     "Type of regret metric to use i.e relative or absolute");
  desc.add_options()(
      "lowLevelPlanner",
      po::value<string>()->default_value("mlastar"),
      "Low-level planner to use: 'mlastar' or 'sipps'");
  desc.add_options()(
      "sippsSuboptimality",
      po::value<double>()->default_value(1.0),
      "Embedded SIPPS low-level suboptimality bound (>=1.0)");
  desc.add_options()(
      "plannerParityCheck",
      po::bool_switch()->default_value(false),
      "Debug mode: in SIPPS runs, shadow each low-level segment with MLA* and "
      "log parity mismatches");
  desc.add_options()(
      "lowLevelSegmentTimeout",
      po::value<double>()->default_value(600.0),
      "Per-segment timeout (seconds) for low-level planner searches");
  desc.add_options()(
      "lowLevelStructuralPrePrune",
      po::value<bool>()->default_value(false),
      "If true, skip low-level search for candidates with structural "
      "infeasibility certificates (goal-permanent/start-trapped/static-disconnected)");

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
  const string initialSeedFromMapfpcLogCli =
      vm["initialSeedFromMapfpcLog"].as<string>();
  const string initialSeedFromMapfpcLog = initialSeedFromMapfpcLogCli;
  const bool postRefineWithMapfpc =
      vm["postRefineWithMapfpc"].as<bool>();
  const string postRefineAssignmentSource =
      vm["postRefineAssignmentSource"].as<string>();
  const string postRefineAssignmentLog =
      vm["postRefineAssignmentLog"].as<string>();
  const string postRefineSolver = vm["postRefineSolver"].as<string>();
  const int postRefineTimeoutSec = vm["postRefineTimeoutSec"].as<int>();
  const bool postRefineAcceptOnlyIfBetter =
      vm["postRefineAcceptOnlyIfBetter"].as<bool>();
  const bool debugImprovementDiagnostics =
      vm["debugImprovementDiagnostics"].as<bool>();
  const string debugIterationTsvPath =
      vm["debugIterationTsvPath"].as<string>();
  if (!initialSeedFromMapfpcLog.empty()) {
    initialSolutionStrategy = "seeded_mapfpc_log";
  }
  if (initialSolutionStrategy != "greedy" &&
      initialSolutionStrategy != "prioritized" &&
      initialSolutionStrategy != "portfolio" &&
      initialSolutionStrategy != "seeded_mapfpc_log" &&
      initialSolutionStrategy.find("sota") == string::npos) {
    PLOGE << "Incorrect initial solution strategy provided. Please choose from "
             "'greedy', 'prioritized', 'portfolio', "
             "'sota_cbs', 'sota_pbs', or provide --initialSeedFromMapfpcLog"
          << "\n";
    return 1;
  }
  if (postRefineAssignmentSource != "solution" &&
      postRefineAssignmentSource != "log") {
    PLOGE << "postRefineAssignmentSource must be 'solution' or 'log'\n";
    return 1;
  }
  if (postRefineWithMapfpc && postRefineAssignmentSource == "log" &&
      postRefineAssignmentLog.empty()) {
    PLOGE << "postRefineAssignmentLog is required when "
             "postRefineAssignmentSource='log'\n";
    return 1;
  }
  if (postRefineSolver != "pbs" && postRefineSolver != "cbs") {
    PLOGE << "postRefineSolver must be 'pbs' or 'cbs'\n";
    return 1;
  }
  if (postRefineTimeoutSec <= 0) {
    PLOGE << "postRefineTimeoutSec must be positive\n";
    return 1;
  }
  string destroyHeuristic = vm["destroyHeuristic"].as<string>();
  if (destroyHeuristic != "conflict" && destroyHeuristic != "worst" &&
      destroyHeuristic != "random" && destroyHeuristic != "shaw" &&
      destroyHeuristic != "precedence_wait" &&
      destroyHeuristic != "low_slack" &&
      destroyHeuristic != "collision_soft" &&
      destroyHeuristic != "failure_soft" &&
      destroyHeuristic != "alns") {
    PLOGE << "The destroy heuristic provided is not supported! Please choose "
             "from 'conflict', 'worst', 'random', 'shaw', 'precedence_wait', "
             "'low_slack', 'collision_soft', 'failure_soft', "
             "and 'alns' removal operators\n";
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
  string optimizationObjective = vm["optimizationObjective"].as<string>();
  if (optimizationObjective != "soc" &&
      optimizationObjective != "makespan") {
    PLOGE << "optimizationObjective must be 'soc' or 'makespan'\n";
    return 1;
  }
  const bool acceptOnlyValidCandidates =
      vm["acceptOnlyValidCandidates"].as<bool>();
  const string repairHeuristic = vm["repairHeuristic"].as<string>();
  if (repairHeuristic != "regret") {
    PLOGE << "The repair heuristic provided is not supported! Please choose "
             "from 'regret'\n";
    return 1;
  }
  const bool enableNrrRepair = vm["enableNrrRepair"].as<bool>();
  const bool nrrFallbackToStandard = vm["nrrFallbackToStandard"].as<bool>();
  const bool nrrGlobalReassign = vm["nrrGlobalReassign"].as<bool>();
  const bool forceNeighborhoodChangeOnReject =
      vm["forceNeighborhoodChangeOnReject"].as<bool>();
  const bool softPersistentConflictGraph =
      vm["softPersistentConflictGraph"].as<bool>();
  const string nrrMiniSolver = vm["nrrMiniSolver"].as<string>();
  if (nrrMiniSolver != "pbs" && nrrMiniSolver != "cbs" &&
      nrrMiniSolver != "auto") {
    PLOGE << "nrrMiniSolver must be 'pbs', 'cbs', or 'auto'\n";
    return 1;
  }
  const string repairMapfpcSolver = vm["repairMapfpcSolver"].as<string>();
  if (repairMapfpcSolver != "pbs" && repairMapfpcSolver != "cbs") {
    PLOGE << "repairMapfpcSolver must be 'pbs' or 'cbs'\n";
    return 1;
  }
  const int repairMapfpcTimeoutSec = vm["repairMapfpcTimeoutSec"].as<int>();
  if (repairMapfpcTimeoutSec <= 0) {
    PLOGE << "repairMapfpcTimeoutSec must be positive\n";
    return 1;
  }
  string regretType = vm["regretType"].as<string>();
  if (regretType != "absolute" && regretType != "relative") {
    PLOGE << "The regret type provided is not supported! Please choose from "
             "'absolute' and 'relative' regret\n";
    return 1;
  }

  const string lowLevelPlanner = vm["lowLevelPlanner"].as<string>();
  if (lowLevelPlanner != "mlastar" && lowLevelPlanner != "sipps") {
    PLOGE << "The low-level planner provided is not supported! Please choose "
             "from 'mlastar' and 'sipps'\n";
    return 1;
  }
  const bool plannerParityCheck = vm["plannerParityCheck"].as<bool>();
  const double sippsSuboptimality = vm["sippsSuboptimality"].as<double>();
  const double lowLevelSegmentTimeout =
      vm["lowLevelSegmentTimeout"].as<double>();
  const bool lowLevelStructuralPrePrune =
      vm["lowLevelStructuralPrePrune"].as<bool>();
  if (!std::isfinite(sippsSuboptimality) || sippsSuboptimality < 1.0) {
    PLOGE << "sippsSuboptimality must be finite and >= 1.0\n";
    return 1;
  }
  if (lowLevelSegmentTimeout <= 0.0) {
    PLOGE << "lowLevelSegmentTimeout must be positive\n";
    return 1;
  }
  if (lowLevelPlanner != "sipps" && std::abs(sippsSuboptimality - 1.0) > 1e-9) {
    PLOGW << "sippsSuboptimality is set to " << sippsSuboptimality
          << " but lowLevelPlanner is '" << lowLevelPlanner
          << "'; this option is ignored unless lowLevelPlanner='sipps'.\n";
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

  const int agentNum = vm["agentNum"].as<int>();
  const int taskNum = vm["taskNum"].as<int>();
  const int neighborSize = vm["neighborSize"].as<int>();
  const int maxIterations = vm["maxIterations"].as<int>();
  const bool alnsEnablePrecedenceAwareDestroy =
      vm["alnsEnablePrecedenceAwareDestroy"].as<bool>();
  const bool softRecoveryDestroyMode =
      vm["softRecoveryDestroyMode"].as<bool>();
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
  const double cutoffTime = vm["cutoffTime"].as<double>();
  if (!std::isfinite(cutoffTime) || cutoffTime < 0.0) {
    PLOGE << "cutoffTime must be finite and non-negative\n";
    return 1;
  }
  if (cutoffTime == 0.0) {
    PLOGW << "cutoffTime is 0: planner will stop immediately after start\n";
  }
  const double lnsConflictWeight = vm["lnsConflictWeight"].as<double>();
  const double lnsCostWeight = vm["lnsCostWeight"].as<double>();
  if (!std::isfinite(lnsConflictWeight) || lnsConflictWeight < 0.0) {
    PLOGE << "lnsConflictWeight must be finite and non-negative\n";
    return 1;
  }
  if (!std::isfinite(lnsCostWeight) || lnsCostWeight < 0.0) {
    PLOGE << "lnsCostWeight must be finite and non-negative\n";
    return 1;
  }
  if (lnsConflictWeight == 0.0 && lnsCostWeight == 0.0) {
    PLOGE << "At least one of lnsConflictWeight or lnsCostWeight must be > 0\n";
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

  int effectiveNeighborSize = neighborSize;
  if (neighborSize == 0) {
    const double agentScale =
        static_cast<double>(std::max(1, instance.getAgentNum()));
    const double taskScale =
        static_cast<double>(std::max(1, instance.getTasksNum()));
    const double precedenceScale = static_cast<double>(std::max(
        1, static_cast<int>(instance.getInputPrecedenceConstraintsRef().size())));
    const double numerator = 10.0 * std::pow(taskScale / 100.0, 0.20);
    const double denominator = std::pow(agentScale / 10.0, 0.35) *
                               std::pow(precedenceScale / 80.0, 0.20);
    const int adaptiveNeighborSize =
        static_cast<int>(std::lround(numerator / denominator));
    effectiveNeighborSize = std::max(4, std::min(14, adaptiveNeighborSize));
    PLOGI << "neighborSize auto mode selected n=" << effectiveNeighborSize
          << " (agents=" << instance.getAgentNum()
          << ", tasks=" << instance.getTasksNum()
          << ", precedence_edges="
          << instance.getInputPrecedenceConstraintsRef().size() << ")\n";
  }

  CoreParameterInputs coreInputs{};
  coreInputs.neighborhoodSize = effectiveNeighborSize;
  coreInputs.timeLimit = cutoffTime;
  coreInputs.lnsConflictWeight = lnsConflictWeight;
  coreInputs.lnsCostWeight = lnsCostWeight;
  coreInputs.initialSolutionStrategy = initialSolutionStrategy;
  coreInputs.initialSeedFromMapfpcLog = initialSeedFromMapfpcLog;
  coreInputs.postRefineWithMapfpc = postRefineWithMapfpc;
  coreInputs.postRefineAssignmentSource = postRefineAssignmentSource;
  coreInputs.postRefineAssignmentLog = postRefineAssignmentLog;
  coreInputs.postRefineSolver = postRefineSolver;
  coreInputs.postRefineTimeoutSec = postRefineTimeoutSec;
  coreInputs.postRefineAcceptOnlyIfBetter = postRefineAcceptOnlyIfBetter;
  coreInputs.debugImprovementDiagnostics = debugImprovementDiagnostics;
  coreInputs.debugIterationTsvPath = debugIterationTsvPath;
  coreInputs.destroyHeuristic = destroyHeuristic;
  coreInputs.acceptanceCriteria = acceptanceCriteria;
  coreInputs.optimizationObjective = optimizationObjective;
  coreInputs.acceptOnlyValidCandidates = acceptOnlyValidCandidates;
  coreInputs.repairHeuristic = repairHeuristic;
  coreInputs.enableNrrRepair = enableNrrRepair;
  coreInputs.nrrFallbackToStandard = nrrFallbackToStandard;
  coreInputs.nrrGlobalReassign = nrrGlobalReassign;
  coreInputs.forceNeighborhoodChangeOnReject =
      forceNeighborhoodChangeOnReject;
  coreInputs.nrrMiniSolver = nrrMiniSolver;
  coreInputs.repairMapfpcSolver = repairMapfpcSolver;
  coreInputs.repairMapfpcTimeoutSec = repairMapfpcTimeoutSec;
  coreInputs.regretType = regretType;
  coreInputs.alnsEnablePrecedenceAwareDestroy =
      alnsEnablePrecedenceAwareDestroy;
  coreInputs.softRecoveryDestroyMode = softRecoveryDestroyMode;
  coreInputs.softPersistentConflictGraph = softPersistentConflictGraph;
  coreInputs.seed = seed;

  LNSParams::LowLevel lowLevelInputs{};
  lowLevelInputs.parityCheck = plannerParityCheck;
  lowLevelInputs.planner = lowLevelPlanner;
  lowLevelInputs.sippsSuboptimality = sippsSuboptimality;
  lowLevelInputs.segmentTimeout = lowLevelSegmentTimeout;
  lowLevelInputs.structuralPrePrune = lowLevelStructuralPrePrune;

  LNSParams parameters = buildLnsParams(coreInputs, lowLevelInputs);
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
  printRunSummaryReport(*lnsInstance, anytimeSolution, success, feasibleStats);
  return 0;
}
