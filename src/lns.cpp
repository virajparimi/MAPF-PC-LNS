#include "lns.hpp"
#include <boost/process.hpp>
#if defined(__has_include)
#if __has_include(<boost/process/null.hpp>)
#include <boost/process/null.hpp>
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 1
#else
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 0
#endif
#else
#define MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL 0
#endif
#include <cmath>
#include <limits>
#include <filesystem>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include "common.hpp"
#include "utils.hpp"

namespace {
struct TaskAssignmentIndex {
  vector<int> owner;
  vector<int> pos;
};

int clampSocToInt(long long soc, const char* context) {
  if (soc > std::numeric_limits<int>::max()) {
    PLOGW << context << ": sum of costs overflowed int; clamping to INT_MAX\n";
    return std::numeric_limits<int>::max();
  }
  if (soc < std::numeric_limits<int>::min()) {
    PLOGW << context << ": sum of costs underflowed int; clamping to INT_MIN\n";
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(soc);
}

TaskAssignmentIndex buildTaskAssignmentIndex(
    const vector<vector<int>>& assignments, int numTasks) {
  TaskAssignmentIndex index;
  index.owner.assign(numTasks, UNASSIGNED);
  index.pos.assign(numTasks, -1);
  int duplicateTaskOwners = 0;
  for (int agent = 0; agent < (int)assignments.size(); agent++) {
    for (int p = 0; p < (int)assignments[agent].size(); p++) {
      const int task = assignments[agent][p];
      if (task < 0 || task >= numTasks) {
        continue;
      }
      if (index.owner[task] != UNASSIGNED) {
        duplicateTaskOwners++;
        if (duplicateTaskOwners <= 5) {
          PLOGE << "Duplicate task ownership detected while building assignment index"
                << " for task " << task << " (existing agent="
                << index.owner[task] << ", new agent=" << agent << ")\n";
        }
      }
      index.owner[task] = agent;
      index.pos[task] = p;
    }
  }
  if (duplicateTaskOwners > 5) {
    PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
          << " tasks while building assignment index.\n";
  }
  return index;
}

TaskAssignmentIndex buildCurrentTaskAssignmentIndex(const Solution& solution,
                                                    int numTasks) {
  TaskAssignmentIndex index;
  index.owner.assign(numTasks, UNASSIGNED);
  index.pos.assign(numTasks, -1);
  int duplicateTaskOwners = 0;
  for (int agent = 0; agent < solution.numOfAgents; agent++) {
    const auto& assignments = solution.agents[agent].taskAssignments;
    for (int p = 0; p < (int)assignments.size(); p++) {
      const int task = assignments[p];
      if (task < 0 || task >= numTasks) {
        continue;
      }
      if (index.owner[task] != UNASSIGNED) {
        duplicateTaskOwners++;
        if (duplicateTaskOwners <= 5) {
          PLOGE << "Duplicate task ownership detected in current solution for task "
                << task << " (existing agent=" << index.owner[task]
                << ", new agent=" << agent << ")\n";
        }
      }
      index.owner[task] = agent;
      index.pos[task] = p;
    }
  }
  if (duplicateTaskOwners > 5) {
    PLOGE << "Duplicate task ownership detected for " << duplicateTaskOwners
          << " tasks in current solution assignment index.\n";
  }
  return index;
}
}  // namespace

vector<pair<int, int>> LNS::buildFullPrecedenceConstraints(
    bool includeIntraConstraints) const {
  const auto& inputConstraints = instance_.getInputPrecedenceConstraintsRef();
  size_t totalConstraints = inputConstraints.size();
  if (includeIntraConstraints) {
    for (int agent = 0; agent < instance_.getAgentNum(); ++agent) {
      totalConstraints +=
          solution_.agents[agent].intraPrecedenceConstraints.size();
    }
  }

  vector<pair<int, int>> precedenceConstraints;
  precedenceConstraints.reserve(totalConstraints);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputConstraints.begin(),
                               inputConstraints.end());
  if (includeIntraConstraints) {
    for (int agent = 0; agent < instance_.getAgentNum(); ++agent) {
      const auto& intra = solution_.agents[agent].intraPrecedenceConstraints;
      precedenceConstraints.insert(precedenceConstraints.end(),
                                   intra.begin(),
                                   intra.end());
    }
  }
  return precedenceConstraints;
}

AgentTaskPath LNS::runLowLevelSearch(SingleAgentSolver& solver,
                                     ConstraintTable& constraintTable,
                                     int startTime, int stage,
                                     int lowerBound) {
  const uint64_t expandedBefore = solver.numExpanded;
  const uint64_t generatedBefore = solver.numGenerated;
  lowLevelCalls_++;
  AgentTaskPath path =
      solver.findPathSegment(constraintTable, startTime, stage, lowerBound);
  lowLevelExpanded_ += (solver.numExpanded - expandedBefore);
  lowLevelGenerated_ += (solver.numGenerated - generatedBefore);
  return path;
}

LNS::LNS(int numOfIterations, const Instance& instance,
         const LNSParams& parameters)
    : numOfIterations_(numOfIterations),
      instance_(instance),
      seed_(parameters.core.seed),
      rng_(parameters.core.seed),
      solution_(instance),
      previousSolution_(instance) {
  plannerStartTime_ = Time::now();
  neighborSize_ = parameters.core.neighborhoodSize;
  timeLimit_ = parameters.core.timeLimit;
  temperature_ = parameters.core.temperature;
  coolingCoefficient_ = parameters.core.coolingCoefficient;
  heatingCoefficient_ = parameters.core.heatingCoefficient;
  tolerance_ = parameters.core.tolerance;
  shawDistanceWeight_ = parameters.core.shawDistanceWeight;
  shawTemporalWeight_ = parameters.core.shawTemporalWeight;
  lnsConflictWeight_ = parameters.core.lnsConflictWeight;
  lnsCostWeight_ = parameters.core.lnsCostWeight;
  initialSolutionStrategy = parameters.core.initialSolutionStrategy;
  destroyHeuristic = parameters.core.destroyHeuristic;
  acceptanceCriteria = parameters.core.acceptanceCriteria;
  regretType = parameters.core.regretType;
  regretCandidateTopK_ = std::max(0, parameters.core.regretCandidateTopK);
  repairIncludeNonAncestorAgents_ =
      parameters.core.repairIncludeNonAncestorAgents;
  if (parameters.lowLevel.planner == "sipps") {
    lowLevelPlannerType_ = LowLevelPlannerType::sipps;
  } else {
    lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  }
  lowLevelSegmentTimeout_ = max(0.0, parameters.lowLevel.segmentTimeout);
  plannerParityCheck_ = parameters.lowLevel.parityCheck;
  plannerParityMaxLogs_ = parameters.lowLevel.parityMaxLogs;
  market_.heuristics = parameters.market.heuristics;
  market_.bucketDt = max(1, parameters.market.bucketDt);
  market_.vertexBucketCapacity = max(1, parameters.market.vertexBucketCapacity);
  market_.edgeBucketCapacity = max(1, parameters.market.edgeBucketCapacity);
  market_.updateOnAcceptedOnly = parameters.market.updateOnAcceptedOnly;
  market_.updatePeriodAccepted = max(1, parameters.market.updatePeriodAccepted);
  market_.eta = max(0.0, parameters.market.eta);
  market_.rho = parameters.market.rho;
  market_.priceCap = max(0.0, parameters.market.priceCap);
  market_.gamma = max(0.0, parameters.market.gamma);
  market_.acceptanceGuards = parameters.market.acceptanceGuards;
  market_.tauP = max(0.0, parameters.market.tauP);
  market_.tauW = max(0.0, parameters.market.tauW);
  market_.destroyWeightPrice = parameters.market.destroyWeightPrice;
  market_.destroyWeightWait = parameters.market.destroyWeightWait;
  market_.destroyWeightRoot = parameters.market.destroyWeightRoot;
  market_.seedTopFrac = min(1.0, max(0.0, parameters.market.seedTopFrac));
  market_.randomDestroyQuota =
      min(1.0, max(0.0, parameters.market.randomDestroyQuota));
  market_.cooldownIters = max(0, parameters.market.cooldownIters);
  market_.dUp = max(0, parameters.market.dUp);
  market_.dDown = max(0, parameters.market.dDown);
  market_.closureCap = parameters.market.closureCap;
  market_.repairTieBreak = parameters.market.repairTieBreak;
  market_.repairBlend = parameters.market.repairBlend;
  market_.tieBreakEpsSoc = max(0.0, parameters.market.tieBreakEpsSoc);
  market_.lambdaPrice = max(0.0, parameters.market.lambdaPrice);
  market_.lambdaWait = max(0.0, parameters.market.lambdaWait);
  if (market_.closureCap <= 0) {
    market_.closureCap = neighborSize_;
  }
  market_.taskCooldownUntilIter.assign(instance_.getTasksNum(), 0);

  // Ensure both working solutions use the selected low-level planner.
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].pathPlanner = createSharedPlanner(agent);
    previousSolution_.agents[agent].pathPlanner = createSharedPlanner(agent);
  }

  incrementalRegret_ = parameters.core.incrementalRegret;
  if (parameters.core.incrementalRegretMode == "descendants") {
    incrementalRegretMode_ = IncrementalRegretMode::descendants;
  } else {
    incrementalRegretMode_ = IncrementalRegretMode::descendants_and_agent;
  }
  regretStamp_.assign(instance_.getTasksNum(), 0);
  regretBestOption_.assign(instance_.getTasksNum(), {UNASSIGNED, -1});
  regretSecondBestOption_.assign(instance_.getTasksNum(), {UNASSIGNED, -1});
}

std::shared_ptr<SingleAgentSolver> LNS::createSharedPlanner(int agent) const {
  std::shared_ptr<SingleAgentSolver> planner;
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      planner = std::make_shared<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_shared<MultiLabelSpaceTimeAStar>(instance_, agent);
      break;
  }
  planner->setSegmentTimeout(lowLevelSegmentTimeout_);
  return planner;
}

std::unique_ptr<SingleAgentSolver> LNS::createLocalPlanner(int agent) const {
  std::unique_ptr<SingleAgentSolver> planner;
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      // Local repair planners always set explicit goals before search.
      planner = std::make_unique<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_, false);
      break;
    case LowLevelPlannerType::mlastar:
    default:
      planner = std::make_unique<MultiLabelSpaceTimeAStar>(instance_, agent, false);
      break;
  }
  planner->setSegmentTimeout(lowLevelSegmentTimeout_);
  return planner;
}

int LNS::marketTimeBucket(int timestep) const {
  if (market_.bucketDt <= 1) {
    return timestep;
  }
  return timestep / market_.bucketDt;
}

uint64_t LNS::makeMarketVertexKey(int location, int bucket) const {
  uint64_t x = 0;
  x ^= (uint64_t)(uint32_t)location;
  x ^= ((uint64_t)(uint32_t)bucket) << 32;
  x ^= 0x9E3779B97F4A7C15ULL;
  return LLNode::mix64(x);
}

uint64_t LNS::makeMarketEdgeKey(int from, int to, int bucket) const {
  uint64_t x = 0xD1B54A32D192ED03ULL;
  x = LLNode::mix64(x ^ ((uint64_t)(uint32_t)from + 0x9E3779B97F4A7C15ULL));
  x = LLNode::mix64(x ^ ((uint64_t)(uint32_t)to + 0xC2B2AE3D27D4EB4FULL));
  x = LLNode::mix64(x ^ ((uint64_t)(uint32_t)bucket + 0x165667B19E3779F9ULL));
  return x;
}

void LNS::computeTaskScheduleMetricsFromIndex(
    const vector<int>& taskPosByTask, vector<TaskScheduleMetrics>& perTask,
    vector<double>* blockedWaitSum) const {
  const int taskCount = instance_.getTasksNum();
  perTask.assign(taskCount, TaskScheduleMetrics());
  if (blockedWaitSum != nullptr) {
    blockedWaitSum->assign(taskCount, 0.0);
  }
  const auto& predecessors = instance_.getAncestorsRef();

  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    const int pos = (task >= 0 && task < (int)taskPosByTask.size())
                        ? taskPosByTask[task]
                        : -1;
    if (pos < 0) {
      continue;
    }
    if (pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
    if (taskPath.empty()) {
      continue;
    }

    TaskScheduleMetrics metric;
    metric.valid = true;
    metric.end = taskPath.endTime();
    const int taskLocation = instance_.getTaskLocations(task);
    int arrive = taskPath.endTime();
    for (int i = 0; i < (int)taskPath.size(); i++) {
      if (taskPath[i].location == taskLocation) {
        arrive = taskPath.beginTime + i;
        break;
      }
    }
    metric.arrive = arrive;
    metric.release = 0;
    metric.blocker = UNASSIGNED;

    if (task >= 0 && task < (int)predecessors.size()) {
      for (int pred : predecessors[task]) {
        const int predAgent =
            (pred >= 0 && pred < (int)solution_.taskAgentMap.size())
                ? solution_.taskAgentMap[pred]
                : UNASSIGNED;
        if (predAgent == UNASSIGNED) {
          continue;
        }
        const int predPos = (pred >= 0 && pred < (int)taskPosByTask.size())
                                ? taskPosByTask[pred]
                                : -1;
        if (predPos < 0) {
          continue;
        }
        if (predPos < 0 ||
            predPos >= (int)solution_.agents[predAgent].taskPaths.size()) {
          continue;
        }
        const AgentTaskPath& predPath = solution_.agents[predAgent].taskPaths[predPos];
        if (predPath.empty()) {
          continue;
        }
        const int predEnd = predPath.endTime();
        if (predEnd > metric.release) {
          metric.release = predEnd;
          metric.blocker = pred;
        }
      }
    }
    metric.start = max(metric.arrive, metric.release);
    metric.waitPrec = max(0, metric.start - metric.arrive);
    perTask[task] = metric;
    if (blockedWaitSum != nullptr && metric.blocker != UNASSIGNED) {
      (*blockedWaitSum)[metric.blocker] += metric.waitPrec;
    }
  }
}

void LNS::computeTaskScheduleMetrics(vector<TaskScheduleMetrics>& perTask,
                                     vector<double>* blockedWaitSum) const {
  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  computeTaskScheduleMetricsFromIndex(currentIndex.pos, perTask, blockedWaitSum);
}

double LNS::computeTaskMarketExposure(int task, bool normalized) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0.0;
  }
  const int agent = solution_.taskAgentMap[task];
  if (agent == UNASSIGNED) {
    return 0.0;
  }
  const int pos = solution_.getLocalTaskIndex(agent, task);
  if (pos == UNASSIGNED) {
    return 0.0;
  }
  if (pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
    return 0.0;
  }
  const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
  if (taskPath.empty()) {
    return 0.0;
  }
  return computeMarketExposureFromPath(taskPath, normalized);
}

double LNS::computeMarketExposureFromPath(const AgentTaskPath& taskPath,
                                          bool normalized) const {
  if (taskPath.empty()) {
    return 0.0;
  }

  double totalExposure = 0.0;
  int resourceCount = 0;
  for (int i = 0; i < (int)taskPath.size(); i++) {
    const int timestep = taskPath.beginTime + i;
    const int bucket = marketTimeBucket(timestep);
    const int location = taskPath[i].location;
    const uint64_t vKey = makeMarketVertexKey(location, bucket);
    const auto vIt = market_.vertexPrices.find(vKey);
    if (vIt != market_.vertexPrices.end()) {
      totalExposure += vIt->second;
    }
    resourceCount++;

    if (i > 0) {
      const int prevLocation = taskPath[i - 1].location;
      const uint64_t eKey = makeMarketEdgeKey(prevLocation, location, bucket);
      const auto eIt = market_.edgePrices.find(eKey);
      if (eIt != market_.edgePrices.end()) {
        totalExposure += eIt->second;
      }
      resourceCount++;
    }
  }
  if (!normalized) {
    return totalExposure;
  }
  return totalExposure / max(1, resourceCount);
}

int LNS::computeTaskPrecedenceWaitFromState(
    int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
    const vector<vector<AgentTaskPath>>& agentTaskPaths,
    const vector<pair<int, int>>& precedenceConstraints) const {
  (void)precedenceConstraints;
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }

  const TaskAssignmentIndex stateIndex =
      buildTaskAssignmentIndex(agentTaskAssignments, instance_.getTasksNum());
  const int taskAgent =
      (task >= 0 && task < (int)stateIndex.owner.size()) ? stateIndex.owner[task]
                                                          : UNASSIGNED;
  const int taskPos =
      (task >= 0 && task < (int)stateIndex.pos.size()) ? stateIndex.pos[task] : -1;
  if (taskAgent == UNASSIGNED || taskPos < 0 ||
      taskPos >= (int)agentTaskPaths[taskAgent].size()) {
    return 0;
  }

  const AgentTaskPath& taskPath = agentTaskPaths[taskAgent][taskPos];
  if (taskPath.empty()) {
    return 0;
  }

  int arrive = taskPath.endTime();
  for (int i = 0; i < (int)taskPath.size(); i++) {
    if (taskPath[i].location == taskLocation) {
      arrive = taskPath.beginTime + i;
      break;
    }
  }

  int release = 0;
  const int taskCount = instance_.getTasksNum();
  vector<char> seenPredecessor(taskCount, 0);
  auto consumePredecessor = [&](int pred) {
    if (pred < 0 || pred >= taskCount) {
      return;
    }
    if (seenPredecessor[pred]) {
      return;
    }
    seenPredecessor[pred] = 1;
    const int predAgent = (pred >= 0 && pred < (int)stateIndex.owner.size())
                              ? stateIndex.owner[pred]
                              : UNASSIGNED;
    const int predPos = (pred >= 0 && pred < (int)stateIndex.pos.size())
                            ? stateIndex.pos[pred]
                            : -1;
    if (predAgent != UNASSIGNED && predPos >= 0 &&
        predPos < (int)agentTaskPaths[predAgent].size() &&
        !agentTaskPaths[predAgent][predPos].empty()) {
      release = max(release, agentTaskPaths[predAgent][predPos].endTime());
    }
  };

  // Base input precedence predecessors can be queried in O(in-degree(task)).
  const auto& baseAncestors = instance_.getAncestorsRef();
  if (task >= 0 && task < (int)baseAncestors.size()) {
    for (int pred : baseAncestors[task]) {
      consumePredecessor(pred);
    }
  }

  // Dynamic intra-agent predecessor in the current assignment state.
  if (taskPos > 0 && taskAgent >= 0 &&
      taskAgent < (int)agentTaskAssignments.size() &&
      taskPos - 1 < (int)agentTaskAssignments[taskAgent].size()) {
    consumePredecessor(agentTaskAssignments[taskAgent][taskPos - 1]);
  }

  return max(0, release - arrive);
}

int LNS::computeTaskPrecedenceWaitInCurrentSolution(int task) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }
  const int taskAgent = solution_.taskAgentMap[task];
  if (taskAgent == UNASSIGNED) {
    return 0;
  }

  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  const int agent = taskAgent;
  const int taskPos =
      (task >= 0 && task < (int)currentIndex.pos.size()) ? currentIndex.pos[task]
                                                          : -1;
  if (taskPos < 0) {
    return 0;
  }
  if (taskPos < 0 || taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
    return 0;
  }
  const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[taskPos];
  if (taskPath.empty()) {
    return 0;
  }

  const int taskLocation = instance_.getTaskLocations(task);
  int arrive = taskPath.endTime();
  for (int i = 0; i < (int)taskPath.size(); i++) {
    if (taskPath[i].location == taskLocation) {
      arrive = taskPath.beginTime + i;
      break;
    }
  }

  int release = 0;
  const auto& ancestors = instance_.getAncestorsRef();
  if (task < 0 || task >= (int)ancestors.size()) {
    return 0;
  }
  for (int pred : ancestors[task]) {
    if (pred < 0 || pred >= (int)solution_.taskAgentMap.size()) {
      continue;
    }
    const int predAgent = solution_.taskAgentMap[pred];
    if (predAgent == UNASSIGNED) {
      continue;
    }
    const int predPos =
        (pred >= 0 && pred < (int)currentIndex.pos.size()) ? currentIndex.pos[pred]
                                                            : -1;
    if (predPos < 0) {
      continue;
    }
    if (predPos < 0 ||
        predPos >= (int)solution_.agents[predAgent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& predPath = solution_.agents[predAgent].taskPaths[predPos];
    if (predPath.empty()) {
      continue;
    }
    release = max(release, predPath.endTime());
  }

  return max(0, release - arrive);
}

double LNS::computeSolutionMarketPressure() const {
  const int taskCount = instance_.getTasksNum();
  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, taskCount);
  double pressure = 0.0;
  for (int task = 0; task < taskCount; task++) {
    const int agent =
        (task >= 0 && task < (int)solution_.taskAgentMap.size())
            ? solution_.taskAgentMap[task]
            : UNASSIGNED;
    if (agent == UNASSIGNED) {
      continue;
    }
    const int taskPos =
        (task >= 0 && task < (int)currentIndex.pos.size()) ? currentIndex.pos[task]
                                                            : -1;
    if (taskPos < 0 || taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[taskPos];
    if (taskPath.empty()) {
      continue;
    }
    pressure += computeMarketExposureFromPath(taskPath, true);
  }
  return pressure;
}

double LNS::computeSolutionPrecedenceWaitFromIndex(
    const vector<int>& taskPosByTask) const {
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetricsFromIndex(taskPosByTask, perTask, nullptr);
  double totalWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
  }
  return totalWait;
}

double LNS::computeSolutionPrecedenceWait() const {
  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  return computeSolutionPrecedenceWaitFromIndex(currentIndex.pos);
}

bool LNS::passMarketAcceptanceGuards(double candidatePressure,
                                     double candidateWait) const {
  if (!market_.acceptanceGuards) {
    return true;
  }
  if (incumbentSolution_.agentPaths.empty()) {
    return true;
  }
  if (!std::isfinite(market_.bestPressure) || !std::isfinite(market_.bestWait)) {
    return true;
  }
  const bool pressureOK = candidatePressure <= market_.bestPressure + market_.tauP;
  const bool waitOK = candidateWait <= market_.bestWait + market_.tauW;
  return pressureOK && waitOK;
}

void LNS::updateMarketStateFromCurrentSolution() {
  if (!market_.heuristics) {
    return;
  }

  unordered_map<uint64_t, int> vertexDemand;
  unordered_map<uint64_t, int> edgeDemand;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const AgentTaskPath& path = solution_.agents[agent].path;
    if (path.empty()) {
      continue;
    }
    for (int i = 0; i < (int)path.size(); i++) {
      const int timestep = path.beginTime + i;
      const int bucket = marketTimeBucket(timestep);
      const int location = path[i].location;
      vertexDemand[makeMarketVertexKey(location, bucket)]++;
      if (i > 0) {
        const int prevLocation = path[i - 1].location;
        edgeDemand[makeMarketEdgeKey(prevLocation, location, bucket)]++;
      }
    }
  }

  const double eta =
      market_.eta / std::sqrt(1.0 + (double)market_.stats.updates);

  int64_t contendedResources = 0;
  double contendedPriceSum = 0.0;
  double maxPrice = 0.0;

  auto updateCategory = [&](const unordered_map<uint64_t, int>& demand,
                            unordered_map<uint64_t, double>& prices,
                            unordered_map<uint64_t, double>& excessHat,
                            int bucketCapacity) {
    vector<uint64_t> keys;
    keys.reserve(demand.size() + prices.size() + excessHat.size());
    for (const auto& kv : demand) {
      keys.push_back(kv.first);
    }
    for (const auto& kv : prices) {
      keys.push_back(kv.first);
    }
    for (const auto& kv : excessHat) {
      keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    unordered_map<uint64_t, double> nextPrices;
    unordered_map<uint64_t, double> nextExcessHat;
    nextPrices.reserve(prices.size() + demand.size());
    nextExcessHat.reserve(excessHat.size() + demand.size());

    for (uint64_t key : keys) {
      const auto itDemand = demand.find(key);
      const int d = (itDemand == demand.end()) ? 0 : itDemand->second;
      const int excess = max(0, d - max(1, bucketCapacity));
      const auto itOldPrice = prices.find(key);
      const auto itOldHat = excessHat.find(key);
      const double oldPrice = (itOldPrice == prices.end()) ? 0.0 : itOldPrice->second;
      const double oldHat = (itOldHat == excessHat.end()) ? 0.0 : itOldHat->second;
      const double newHat = market_.rho * oldHat + (1.0 - market_.rho) * excess;

      double newPrice = oldPrice;
      if (excess > 0) {
        const double basePrice = max(oldPrice, 1.0);
        newPrice = min(market_.priceCap, basePrice * std::exp(eta * newHat));
      } else {
        newPrice = oldPrice * max(0.0, 1.0 - market_.gamma);
      }

      if (newPrice > 1e-9 || newHat > 1e-9) {
        nextPrices[key] = newPrice;
        nextExcessHat[key] = newHat;
      }
      if (excess > 0) {
        contendedResources++;
        contendedPriceSum += newPrice;
        maxPrice = max(maxPrice, newPrice);
      }
    }

    prices.swap(nextPrices);
    excessHat.swap(nextExcessHat);
  };

  updateCategory(vertexDemand, market_.vertexPrices, market_.vertexExcessHat,
                 market_.vertexBucketCapacity);
  updateCategory(edgeDemand, market_.edgePrices, market_.edgeExcessHat,
                 market_.edgeBucketCapacity);

  vector<double> allPrices;
  allPrices.reserve(market_.vertexPrices.size() + market_.edgePrices.size());
  double totalPriceMass = 0.0;
  for (const auto& kv : market_.vertexPrices) {
    if (kv.second > 0.0) {
      allPrices.push_back(kv.second);
      totalPriceMass += kv.second;
    }
  }
  for (const auto& kv : market_.edgePrices) {
    if (kv.second > 0.0) {
      allPrices.push_back(kv.second);
      totalPriceMass += kv.second;
    }
  }
  double topPriceMassFrac = 0.0;
  if (!allPrices.empty() && totalPriceMass > 1e-12) {
    std::sort(allPrices.begin(), allPrices.end(), std::greater<double>());
    const int topK = max(1, (int)std::ceil((double)allPrices.size() * 0.01));
    double topMass = 0.0;
    for (int i = 0; i < topK && i < (int)allPrices.size(); i++) {
      topMass += allPrices[i];
    }
    topPriceMassFrac = topMass / totalPriceMass;
  }

  const TaskAssignmentIndex currentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetricsFromIndex(currentIndex.pos, perTask, nullptr);
  double totalWait = 0.0;
  double maxWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
    maxWait = max(maxWait, (double)metric.waitPrec);
  }

  market_.stats.updates++;
  market_.stats.contendedResources = contendedResources;
  market_.stats.meanPriceContended =
      contendedResources > 0 ? contendedPriceSum / (double)contendedResources
                             : 0.0;
  market_.stats.maxPrice = maxPrice;
  market_.stats.topPriceMassFrac = topPriceMassFrac;
  market_.stats.totalPrecedenceWait = totalWait;
  market_.stats.maxPrecedenceWait = maxWait;
}

void LNS::maybeUpdateMarketState(bool accepted) {
  if (!market_.heuristics) {
    return;
  }
  if (market_.updateOnAcceptedOnly) {
    if (!accepted) {
      return;
    }
    market_.acceptedCounter++;
    if (market_.acceptedCounter % market_.updatePeriodAccepted != 0) {
      return;
    }
  } else {
    market_.updateCounter++;
    if (market_.updateCounter % market_.updatePeriodAccepted != 0) {
      return;
    }
  }
  updateMarketStateFromCurrentSolution();
}


bool LNS::buildGreedySolutionWithMAPFPC(const string& variant) {

  // Reset solution state in case this is called more than once.
  Solution freshSolution(instance_);
  solution_ = freshSolution;

  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  bool readingTaskAssignments = false, readingTaskPaths = false;

  auto parseInt = [](const std::string& s) -> std::optional<int> {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      return std::nullopt;
    }
    const size_t end = s.find_last_not_of(" \t\r\n");
    try {
      return std::stoi(s.substr(begin, end - begin + 1));
    } catch (...) {
      return std::nullopt;
    }
  };

  string solver;
  if (variant == "sota_cbs") {
    solver = "CBS";
  } else if (variant == "sota_pbs") {
    solver = "PBS";
  } else {
    PLOGE << "Initial solution solver using MAPF-PC variant not supported\n";
    return false;
  }

  // Run a child process to spawn the MAPC-PC codebase with the current map and agent informations
  namespace bp = boost::process;
  bp::ipstream inputStream;
  std::string taskAssignmentExe;
  if (std::filesystem::exists("./MAPF-PC/build_local/bin/task_assignment")) {
    taskAssignmentExe = "./MAPF-PC/build_local/bin/task_assignment";
  } else {
    taskAssignmentExe = "./MAPF-PC/build/bin/task_assignment";
  }
  const std::vector<std::string> args = {
      "-m", instance_.getMapName(),
      "-a", instance_.getAgentTaskFName(),
      "-k", std::to_string(instance_.getAgentNum()),
      "-t", "120",
      "-d", std::to_string(seed_),
      "--solver", solver,
  };
#if MAPF_PC_LNS_HAS_BOOST_PROCESS_NULL
  bp::child child(taskAssignmentExe, bp::args(args), bp::std_out > inputStream,
                  bp::std_err > bp::null);
#else
  // Some Boost.Process installations don't ship <boost/process/null.hpp>.
  // In that case, don't suppress stderr.
  bp::child child(taskAssignmentExe, bp::args(args), bp::std_out > inputStream);
#endif

  // The output sequence of the MAPF-PC codebase is as follows:
  // 1. Output TASK ASSIGNMENTS
  // Output Agent # and then followed by the task sequences in order
  // 2. Output some other internal stuff
  // 3. Output TASK PATHS
  // Output Agent # and then followed by the task locations (non-linearized) with @ after the first location with begin time right after and -> between each location

  int agent = -1;
  string line;
  auto parseAgentHeader = [](const std::string& s) -> bool {
    // Expected format: "Agent <id>"
    // Avoid matching unrelated stderr/log lines that merely contain "Agent".
    std::istringstream iss(s);
    std::string tag;
    int id = -1;
    if (!(iss >> tag >> id)) {
      return false;
    }
    return tag == "Agent";
  };
  while (std::getline(inputStream, line)) {
    if (line.empty()) {
      continue;
    }

    // If the agent variable exceeds the total number of agents we are working with then we have read all the task assignments or all the task paths
    if (agent >= instance_.getAgentNum()) {
      if (readingTaskAssignments) {
        readingTaskAssignments = false;
      } else if (readingTaskPaths) {
        readingTaskPaths = false;
      }
    }

    // Check if the following set of lines will be for task assignments or task paths
    if (line == "TASK ASSIGNMENTS") {
      readingTaskAssignments = true;
      readingTaskPaths = false;
      agent = -1;
      continue;
    } else if (line == "TASK PATHS") {
      readingTaskAssignments = false;
      readingTaskPaths = true;
      agent = -1;
      continue;
    }

    // Use the Agent # to increment the agent variable
    if (parseAgentHeader(line)) {
      agent++;
    }
    // Otherwise if we are supposed to read the task assignments then we split that line using ',' as the delimiter and extract the tokens one by one
    // Eg: Agent 1
    //     1, 2, 3, 4,
    else if (readingTaskAssignments && agent > -1) {
      string token;
      size_t pos = 0;
      while ((pos = line.find(',')) != string::npos) {
        token = line.substr(0, pos);
        const auto task = parseInt(token);
        if (!task.has_value()) {
          PLOGE << "Failed to parse task assignment token: '" << token << "'\n";
          child.terminate();
          child.wait();
          return false;
        }
        solution_.assignTaskToAgent(agent, *task);
        line.erase(0, pos + 1);
      }
      // Handle the last token if the line doesn't end with a comma.
      if (const auto task = parseInt(line); task.has_value()) {
        solution_.assignTaskToAgent(agent, *task);
      }
      solution_.agents[agent].taskPaths.resize(
          solution_.agents[agent].taskAssignments.size(), AgentTaskPath());
    }
    // If we are not reading the task assignments then we must be reading the task paths.
    // Eg: Agent 1
    //     6 @ 0 -> 22 -> 23 @ 2 -> 24 @ 3 -> 25 -> 26 ->
    else if (readingTaskPaths && agent > -1) {
      bool ok = true;
      auto isWhitespaceOnly = [](const std::string& s) -> bool {
        return s.find_first_not_of(" \t\r\n") == std::string::npos;
      };
      auto hasDigit = [](const std::string& s) -> bool {
        return s.find_first_of("0123456789") != std::string::npos;
      };
      auto consumeToken = [&](const std::string& rawToken, AgentTaskPath& taskPath,
                              int& taskIndex) {
        if (rawToken.empty() || isWhitespaceOnly(rawToken)) {
          return;
        }
        string token = rawToken;
        if (isWhitespaceOnly(token)) {
          return;
        }
        if (token.find('@') != string::npos) {
          if (!taskPath.empty()) {
            // Mark the last location of the previous task as goal.
            taskPath.path.back().isGoal = true;
            solution_.agents[agent].taskPaths[taskIndex] = taskPath;
            initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
                taskPath;
            taskPath = AgentTaskPath();
          }
          taskIndex++;

          const size_t atPos = token.find('@');
          const auto loc = parseInt(token.substr(0, atPos));
          if (!loc.has_value()) {
            if (!hasDigit(token.substr(0, atPos))) {
              return;
            }
            PLOGE << "Failed to parse path location token: '" << token << "'\n";
            ok = false;
            return;
          }
          PathEntry pEntry = {false, *loc};

          if (taskIndex > 0) {
            if (taskIndex - 1 >=
                    (int)solution_.agents[agent].taskPaths.size() ||
                solution_.agents[agent].taskPaths[taskIndex - 1].empty()) {
              PLOGE << "buildGreedySolutionWithMAPFPC: previous task path "
                       "missing/empty for agent "
                    << agent << ", taskIndex " << taskIndex << "\n";
              ok = false;
              return;
            }
            const int previousLocation =
                solution_.agents[agent].taskPaths[taskIndex - 1]
                    .back()
                    .location;
            taskPath.path.push_back(PathEntry{false, previousLocation});
          }
          taskPath.path.push_back(pEntry);

          token.erase(0, atPos + 1);
          if (taskIndex > 0) {
            const auto beginTime = parseInt(token);
            if (!beginTime.has_value()) {
              PLOGE << "Failed to parse task begin time token: '" << token
                    << "'\n";
              ok = false;
              return;
            }
            taskPath.beginTime = *beginTime - 1;
          } else {
            taskPath.beginTime = 0;
          }
        } else {
          const auto loc = parseInt(token);
          if (!loc.has_value()) {
            if (!hasDigit(token)) {
              return;
            }
            PLOGE << "Failed to parse path location token: '" << token << "'\n";
            ok = false;
            return;
          }
          taskPath.path.push_back(PathEntry{false, *loc});
        }
      };

      AgentTaskPath taskPath;
      size_t pos = 0;
      int taskIndex = -1;
      while ((pos = line.find("->")) != string::npos) {
        consumeToken(line.substr(0, pos), taskPath, taskIndex);
        line.erase(0, pos + 2);
      }

      // Process any trailing token after the last "->".
      consumeToken(line, taskPath, taskIndex);

      // Add the last task path.
      if (!taskPath.empty() && taskIndex >= 0) {
        taskPath.path.back().isGoal = true;
        solution_.agents[agent].taskPaths[taskIndex] = taskPath;
        initialPaths_[solution_.agents[agent].taskAssignments[taskIndex]] =
            taskPath;
      }
      if (!ok) {
        child.terminate();
        child.wait();
        return false;
      }
    }
  }

  child.wait();
  if (child.exit_code() != 0) {
    PLOGE << "MAPF-PC task_assignment exited with code " << child.exit_code()
          << "\n";
    return false;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(agent));
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolutionWithMAPFPC: failed to join agent paths\n";
    return false;
  }

  // Gather the information
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildGreedySolutionWithMAPFPC");
  return true;
}

bool LNS::buildGreedySolution() {

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
  }

  // Assign tasks
  if (!greedyTaskAssignment(&instance_, &solution_)) {
    PLOGE << "Failed to compute greedy task assignment\n";
    return false;
  }
  size_t expectedIntraSize = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    vector<int> taskLocations = instance_.getTaskLocations(agentTasks);
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(agentTasks.size(), AgentTaskPath());
    if (!agentTasks.empty()) {
      expectedIntraSize += agentTasks.size() - 1;
    }
  }

  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.reserve(inputPrecedenceConstraints.size() +
                                expectedIntraSize);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());

  // Compute the precedence constraints based on current task assignments
  // Intra-agent precedence constraints
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
      precedenceConstraints.emplace_back(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Find paths based on the task assignments
  // First we need to sort the tasks based on the precedence constraints
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return success;
  }

  // Following the topological order we find the paths for each task
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());
  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  vector<char> plannedTasks(instance_.getTasksNum(), 0);
  for (int id : planningOrder) {

    const int task = id;
    int startTime = 0;
    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildGreedySolution: task " << task
            << " is not assigned to any agent\n";
      return false;
    }
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildGreedySolution: invalid local task index " << taskPosition
            << " for task " << task << " on agent " << agent << "\n";
      return false;
    }
    if (taskPosition != 0) {
      int previousTask =
          solution_.agents[agent].taskAssignments[taskPosition - 1];
      if (initialPaths_[previousTask].empty()) {
        PLOGE << "buildGreedySolution: missing path for predecessor task "
              << previousTask << "\n";
        return false;
      }
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    PLOGI << "Planning for agent " << agent << " and task " << task << "\n";

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    buildConstraintTable(constraintTable, task);
    // Strengthen greedy initialization with collision constraints from tasks
    // that have already been planned in this pass. This keeps initialization
    // robust across low-level planners that may choose different but valid
    // shortest paths under precedence-only constraints.
    for (int plannedTask = 0; plannedTask < instance_.getTasksNum();
         plannedTask++) {
      if (!plannedTasks[plannedTask]) {
        continue;
      }
      const int plannedAgent = solution_.getAgentWithTask(plannedTask);
      if (plannedAgent == UNASSIGNED) {
        continue;
      }
      const int plannedTaskPos =
          (plannedTask >= 0 && plannedTask < (int)assignmentIndex.pos.size())
              ? assignmentIndex.pos[plannedTask]
              : UNASSIGNED;
      if (plannedTaskPos == UNASSIGNED ||
          plannedTaskPos >=
              (int)solution_.agents[plannedAgent].taskPaths.size()) {
        continue;
      }
      const auto& plannedPath =
          solution_.agents[plannedAgent].taskPaths[plannedTaskPos];
      if (plannedPath.empty()) {
        continue;
      }
      const bool waitAtGoal =
          !solution_.agents[plannedAgent].taskAssignments.empty() &&
          plannedTask ==
              solution_.agents[plannedAgent].taskAssignments.back();
      constraintTable.addPath(plannedPath, waitAtGoal);
    }
    initialPaths_[id] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[id].empty()) {
      PLOGE << "No path exists for agent " << agent << " and task " << task
            << "\n";
      return false;
    }
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[id];
    plannedTasks[task] = 1;
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolution: failed to join agent paths\n";
    return false;
  }

  // Gather the information
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts +=
        static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
  }
  solution_.sumOfCosts = clampSocToInt(initialSumOfCosts, "buildGreedySolution");
  return true;
}

bool LNS::buildGreedySolutionPrecedenceOnly() {

  // Reset any previous task->agent mapping.
  for (int& assignedAgent : solution_.taskAgentMap) {
    assignedAgent = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
  }

  // Assign tasks (greedy), but do not build collision constraints.
  if (!greedyTaskAssignment(&instance_, &solution_)) {
    PLOGE << "Failed to compute greedy task assignment\n";
    return false;
  }
  size_t expectedIntraSize = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& agentTasks = solution_.getAgentGlobalTasks(agent);
    vector<int> taskLocations = instance_.getTaskLocations(agentTasks);
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(agentTasks.size(), AgentTaskPath());
    if (!agentTasks.empty()) {
      expectedIntraSize += agentTasks.size() - 1;
    }
  }

  // Global precedence constraints = input + intra-agent ordering.
  vector<pair<int, int>> precedenceConstraints;
  const auto& inputPrecedenceConstraints =
      instance_.getInputPrecedenceConstraintsRef();
  precedenceConstraints.reserve(inputPrecedenceConstraints.size() +
                                expectedIntraSize);
  precedenceConstraints.insert(precedenceConstraints.end(),
                               inputPrecedenceConstraints.begin(),
                               inputPrecedenceConstraints.end());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task = 1; task < (int)solution_.getAgentGlobalTasks(agent).size();
         task++) {
      solution_.agents[agent].insertPrecedenceConstraint(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
      precedenceConstraints.emplace_back(
          solution_.agents[agent].taskAssignments[task - 1],
          solution_.agents[agent].taskAssignments[task]);
    }
  }

  // Compute a topological planning order.
  vector<int> planningOrder;
  bool success =
      topologicalSort(&instance_, precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return false;
  }
  const TaskAssignmentIndex assignmentIndex =
      buildCurrentTaskAssignmentIndex(solution_, instance_.getTasksNum());

  // Build predecessor adjacency to enforce precedence via earliest goal times.
  vector<vector<int>> predecessors(instance_.getTasksNum());
  for (const auto& e : precedenceConstraints) {
    predecessors[e.second].push_back(e.first);
  }

  // Plan each task segment with only precedence timing constraints (no paths
  // from other agents in the constraint table).
  initialPaths_.assign(instance_.getTasksNum(), AgentTaskPath());
  for (int task : planningOrder) {
    const int agent = solution_.getAgentWithTask(task);
    if (agent == UNASSIGNED) {
      PLOGE << "buildGreedySolutionPrecedenceOnly: task " << task
            << " is not assigned to any agent\n";
      return false;
    }
    const int taskPosition = (task >= 0 && task < (int)assignmentIndex.pos.size())
                                 ? assignmentIndex.pos[task]
                                 : UNASSIGNED;
    if (taskPosition < 0 ||
        taskPosition >= (int)solution_.agents[agent].taskAssignments.size() ||
        taskPosition >= (int)solution_.agents[agent].taskPaths.size()) {
      PLOGE << "buildGreedySolutionPrecedenceOnly: invalid local task index "
            << taskPosition << " for task " << task << " on agent " << agent
            << "\n";
      return false;
    }

    int startTime = 0;
    if (taskPosition != 0) {
      int previousTask = solution_.agents[agent].taskAssignments[taskPosition - 1];
      if (initialPaths_[previousTask].empty()) {
        PLOGE << "buildGreedySolutionPrecedenceOnly: missing path for predecessor task "
              << previousTask << "\n";
        return false;
      }
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    int earliestGoalTime = 0;
    for (int pred : predecessors[task]) {
      if (initialPaths_[pred].empty()) {
        PLOGE << "buildGreedySolutionPrecedenceOnly: missing path for precedence predecessor task "
              << pred << "\n";
        return false;
      }
      assert(!initialPaths_[pred].empty());
      earliestGoalTime =
          max(earliestGoalTime, initialPaths_[pred].endTimeChecked() + 1);
    }

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    constraintTable.goalLocation = instance_.getTaskLocations(task);
    constraintTable.lengthMin = max(constraintTable.lengthMin, earliestGoalTime);
    constraintTable.latestTimestep =
        max(constraintTable.latestTimestep, constraintTable.lengthMin);

    initialPaths_[task] = runLowLevelSearch(
        *solution_.agents[agent].pathPlanner, constraintTable, startTime,
        taskPosition, 0);
    if (initialPaths_[task].empty()) {
      PLOGE << "No path exists for agent " << agent << " and task " << task
            << "\n";
      return false;
    }
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[task];
  }

  // Join the individual task paths to form the agent's path.
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  if (!solution_.joinPaths(agentsToCompute)) {
    PLOGE << "buildGreedySolutionPrecedenceOnly: failed to join agent paths\n";
    return false;
  }

  // Gather the information.
  long long initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      initialSumOfCosts +=
          static_cast<long long>(solution_.agents[agent].path.endTimeOrZero());
    }
  }
  solution_.sumOfCosts =
      clampSocToInt(initialSumOfCosts, "buildGreedySolutionPrecedenceOnly");
  return true;
}

bool LNS::extractFeasibleSolution() {

  // Only update the feasible solution if the new solution has better cost!
  if (incumbentSolution_.agentPaths.empty() ||
      incumbentSolution_.sumOfCosts > solution_.sumOfCosts) {
    incumbentSolution_.numOfCols = instance_.numOfCols;
    incumbentSolution_.sumOfCosts = solution_.sumOfCosts;
    incumbentSolution_.agentPaths.resize(instance_.getAgentNum());
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      if (!solution_.agents[agent].taskAssignments.empty()) {
        incumbentSolution_.agentPaths[agent] = solution_.agents[agent].path;
      } else {
        incumbentSolution_.agentPaths[agent] = AgentTaskPath();
      }
    }
    return true;
  }
  return false;
}
