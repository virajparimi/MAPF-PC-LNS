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
#include <filesystem>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include "common.hpp"
#include "utils.hpp"

LNS::LNS(int numOfIterations, const Instance& instance,
         const LNSParams& parameters)
    : numOfIterations_(numOfIterations),
      instance_(instance),
      seed_(parameters.seed),
      rng_(parameters.seed),
      solution_(instance),
      previousSolution_(instance) {
  plannerStartTime_ = Time::now();
  neighborSize_ = parameters.neighborhoodSize;
  timeLimit_ = parameters.timeLimit;
  temperature_ = parameters.temperature;
  coolingCoefficient_ = parameters.coolingCoefficient;
  heatingCoefficient_ = parameters.heatingCoefficient;
  tolerance_ = parameters.tolerance;
  shawDistanceWeight_ = parameters.shawDistanceWeight;
  shawTemporalWeight_ = parameters.shawTemporalWeight;
  lnsConflictWeight_ = parameters.lnsConflictWeight;
  lnsCostWeight_ = parameters.lnsCostWeight;
  initialSolutionStrategy = parameters.initialSolutionStrategy;
  destroyHeuristic = parameters.destroyHeuristic;
  acceptanceCriteria = parameters.acceptanceCriteria;
  regretType = parameters.regretType;
  if (parameters.lowLevelPlanner == "sipps") {
    lowLevelPlannerType_ = LowLevelPlannerType::sipps;
  } else {
    lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  }
  plannerParityCheck_ = parameters.plannerParityCheck;
  plannerParityMaxLogs_ = parameters.plannerParityMaxLogs;
  marketHeuristics_ = parameters.marketHeuristics;
  marketBucketDt_ = max(1, parameters.marketBucketDt);
  marketVertexBucketCapacity_ = max(1, parameters.marketVertexBucketCapacity);
  marketEdgeBucketCapacity_ = max(1, parameters.marketEdgeBucketCapacity);
  marketUpdateOnAcceptedOnly_ = parameters.marketUpdateOnAcceptedOnly;
  marketUpdatePeriodAccepted_ = max(1, parameters.marketUpdatePeriodAccepted);
  marketEta_ = max(0.0, parameters.marketEta);
  marketRho_ = parameters.marketRho;
  marketPriceCap_ = max(0.0, parameters.marketPriceCap);
  marketGamma_ = max(0.0, parameters.marketGamma);
  marketAcceptanceGuards_ = parameters.marketAcceptanceGuards;
  marketTauP_ = max(0.0, parameters.marketTauP);
  marketTauW_ = max(0.0, parameters.marketTauW);
  marketDestroyWeightPrice_ = parameters.marketDestroyWeightPrice;
  marketDestroyWeightWait_ = parameters.marketDestroyWeightWait;
  marketDestroyWeightRoot_ = parameters.marketDestroyWeightRoot;
  marketSeedTopFrac_ = min(1.0, max(0.0, parameters.marketSeedTopFrac));
  marketRandomDestroyQuota_ =
      min(1.0, max(0.0, parameters.marketRandomDestroyQuota));
  marketCooldownIters_ = max(0, parameters.marketCooldownIters);
  marketDUp_ = max(0, parameters.marketDUp);
  marketDDown_ = max(0, parameters.marketDDown);
  marketClosureCap_ = parameters.marketClosureCap;
  marketRepairTieBreak_ = parameters.marketRepairTieBreak;
  marketRepairBlend_ = parameters.marketRepairBlend;
  marketTieBreakEpsSoc_ = max(0.0, parameters.marketTieBreakEpsSoc);
  marketLambdaPrice_ = max(0.0, parameters.marketLambdaPrice);
  marketLambdaWait_ = max(0.0, parameters.marketLambdaWait);
  if (marketClosureCap_ <= 0) {
    marketClosureCap_ = neighborSize_;
  }
  marketTaskCooldownUntilIter_.assign(instance_.getTasksNum(), 0);

  // Ensure both working solutions use the selected low-level planner.
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].pathPlanner = createSharedPlanner(agent);
    previousSolution_.agents[agent].pathPlanner = createSharedPlanner(agent);
  }

  incrementalRegret_ = parameters.incrementalRegret;
  if (parameters.incrementalRegretMode == "descendants") {
    incrementalRegretMode_ = IncrementalRegretMode::descendants;
  } else {
    incrementalRegretMode_ = IncrementalRegretMode::descendants_and_agent;
  }
  regretStamp_.assign(instance_.getTasksNum(), 0);
  regretBestOption_.assign(instance_.getTasksNum(), {UNASSIGNED, -1});
  regretSecondBestOption_.assign(instance_.getTasksNum(), {UNASSIGNED, -1});
}

std::shared_ptr<SingleAgentSolver> LNS::createSharedPlanner(int agent) const {
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      return std::make_shared<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_);
    case LowLevelPlannerType::mlastar:
    default:
      return std::make_shared<MultiLabelSpaceTimeAStar>(instance_, agent);
  }
}

std::unique_ptr<SingleAgentSolver> LNS::createLocalPlanner(int agent) const {
  switch (lowLevelPlannerType_) {
    case LowLevelPlannerType::sipps:
      return std::make_unique<MultiLabelSIPPS>(
          instance_, agent, plannerParityCheck_, plannerParityMaxLogs_);
    case LowLevelPlannerType::mlastar:
    default:
      return std::make_unique<MultiLabelSpaceTimeAStar>(instance_, agent);
  }
}

int LNS::marketTimeBucket(int timestep) const {
  if (marketBucketDt_ <= 1) {
    return timestep;
  }
  return timestep / marketBucketDt_;
}

uint64_t LNS::makeMarketVertexKey(int location, int bucket) const {
  uint64_t x = 0;
  x ^= (uint64_t)(uint32_t)location;
  x ^= ((uint64_t)(uint32_t)bucket) << 32;
  x ^= 0x9E3779B97F4A7C15ULL;
  return LLNode::mix64(x);
}

uint64_t LNS::makeMarketEdgeKey(int from, int to, int bucket) const {
  uint64_t x = 0;
  x ^= (uint64_t)(uint32_t)from;
  x ^= ((uint64_t)(uint32_t)to) << 21;
  x ^= ((uint64_t)(uint32_t)bucket) << 42;
  x ^= 0xD1B54A32D192ED03ULL;
  return LLNode::mix64(x);
}

void LNS::computeTaskScheduleMetrics(vector<TaskScheduleMetrics>& perTask,
                                     vector<double>* blockedWaitSum) const {
  const int taskCount = instance_.getTasksNum();
  perTask.assign(taskCount, TaskScheduleMetrics());
  if (blockedWaitSum != nullptr) {
    blockedWaitSum->assign(taskCount, 0.0);
  }
  const vector<vector<int>> predecessors = instance_.getAncestors();

  for (int task = 0; task < taskCount; task++) {
    const auto itAgent = solution_.taskAgentMap.find(task);
    if (itAgent == solution_.taskAgentMap.end()) {
      continue;
    }
    const int agent = itAgent->second;
    if (agent == UNASSIGNED) {
      continue;
    }
    const vector<int>& assignments = solution_.agents[agent].taskAssignments;
    const auto itPos = std::find(assignments.begin(), assignments.end(), task);
    if (itPos == assignments.end()) {
      continue;
    }
    const int pos = (int)(itPos - assignments.begin());
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
        const auto itPredAgent = solution_.taskAgentMap.find(pred);
        if (itPredAgent == solution_.taskAgentMap.end()) {
          continue;
        }
        const int predAgent = itPredAgent->second;
        if (predAgent == UNASSIGNED) {
          continue;
        }
        const vector<int>& predAssignments =
            solution_.agents[predAgent].taskAssignments;
        const auto itPredPos =
            std::find(predAssignments.begin(), predAssignments.end(), pred);
        if (itPredPos == predAssignments.end()) {
          continue;
        }
        const int predPos = (int)(itPredPos - predAssignments.begin());
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

double LNS::computeTaskMarketExposure(int task, bool normalized) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0.0;
  }
  const auto itAgent = solution_.taskAgentMap.find(task);
  if (itAgent == solution_.taskAgentMap.end()) {
    return 0.0;
  }
  const int agent = itAgent->second;
  if (agent == UNASSIGNED) {
    return 0.0;
  }
  const vector<int>& assignments = solution_.agents[agent].taskAssignments;
  const auto itPos = std::find(assignments.begin(), assignments.end(), task);
  if (itPos == assignments.end()) {
    return 0.0;
  }
  const int pos = (int)(itPos - assignments.begin());
  if (pos < 0 || pos >= (int)solution_.agents[agent].taskPaths.size()) {
    return 0.0;
  }
  const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[pos];
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
    const auto vIt = marketVertexPrices_.find(vKey);
    if (vIt != marketVertexPrices_.end()) {
      totalExposure += vIt->second;
    }
    resourceCount++;

    if (i > 0) {
      const int prevLocation = taskPath[i - 1].location;
      const uint64_t eKey = makeMarketEdgeKey(prevLocation, location, bucket);
      const auto eIt = marketEdgePrices_.find(eKey);
      if (eIt != marketEdgePrices_.end()) {
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
    const auto vIt = marketVertexPrices_.find(vKey);
    if (vIt != marketVertexPrices_.end()) {
      totalExposure += vIt->second;
    }
    resourceCount++;

    if (i > 0) {
      const int prevLocation = taskPath[i - 1].location;
      const uint64_t eKey = makeMarketEdgeKey(prevLocation, location, bucket);
      const auto eIt = marketEdgePrices_.find(eKey);
      if (eIt != marketEdgePrices_.end()) {
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
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }

  int taskAgent = UNASSIGNED;
  int taskPos = -1;
  for (int agent = 0; agent < (int)agentTaskAssignments.size(); agent++) {
    const auto it = std::find(agentTaskAssignments[agent].begin(),
                              agentTaskAssignments[agent].end(), task);
    if (it != agentTaskAssignments[agent].end()) {
      taskAgent = agent;
      taskPos = (int)(it - agentTaskAssignments[agent].begin());
      break;
    }
  }
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
  for (const auto& prec : precedenceConstraints) {
    if (prec.second != task) {
      continue;
    }
    const int pred = prec.first;
    bool predFound = false;
    int predEnd = 0;
    for (int agent = 0; agent < (int)agentTaskAssignments.size(); agent++) {
      const auto itPred = std::find(agentTaskAssignments[agent].begin(),
                                    agentTaskAssignments[agent].end(), pred);
      if (itPred == agentTaskAssignments[agent].end()) {
        continue;
      }
      const int predPos = (int)(itPred - agentTaskAssignments[agent].begin());
      if (predPos >= 0 && predPos < (int)agentTaskPaths[agent].size() &&
          !agentTaskPaths[agent][predPos].empty()) {
        predEnd = agentTaskPaths[agent][predPos].endTime();
        predFound = true;
      }
      break;
    }
    if (predFound) {
      release = max(release, predEnd);
    }
  }

  return max(0, release - arrive);
}

int LNS::computeTaskPrecedenceWaitInCurrentSolution(int task) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }
  const auto itTaskAgent = solution_.taskAgentMap.find(task);
  if (itTaskAgent == solution_.taskAgentMap.end() ||
      itTaskAgent->second == UNASSIGNED) {
    return 0;
  }

  const int agent = itTaskAgent->second;
  const vector<int>& assignments = solution_.agents[agent].taskAssignments;
  const auto itPos = std::find(assignments.begin(), assignments.end(), task);
  if (itPos == assignments.end()) {
    return 0;
  }
  const int taskPos = (int)(itPos - assignments.begin());
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
  for (const auto& prec : instance_.getInputPrecedenceConstraints()) {
    if (prec.second != task) {
      continue;
    }
    const int pred = prec.first;
    const auto itPredAgent = solution_.taskAgentMap.find(pred);
    if (itPredAgent == solution_.taskAgentMap.end() ||
        itPredAgent->second == UNASSIGNED) {
      continue;
    }
    const int predAgent = itPredAgent->second;
    const vector<int>& predAssignments =
        solution_.agents[predAgent].taskAssignments;
    const auto itPredPos =
        std::find(predAssignments.begin(), predAssignments.end(), pred);
    if (itPredPos == predAssignments.end()) {
      continue;
    }
    const int predPos = (int)(itPredPos - predAssignments.begin());
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
  double pressure = 0.0;
  for (int task = 0; task < instance_.getTasksNum(); task++) {
    const auto itAgent = solution_.taskAgentMap.find(task);
    if (itAgent == solution_.taskAgentMap.end() || itAgent->second == UNASSIGNED) {
      continue;
    }
    pressure += computeTaskMarketExposure(task, true);
  }
  return pressure;
}

double LNS::computeSolutionPrecedenceWait() const {
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetrics(perTask, nullptr);
  double totalWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
  }
  return totalWait;
}

bool LNS::passMarketAcceptanceGuards(double candidatePressure,
                                     double candidateWait) const {
  if (!marketAcceptanceGuards_) {
    return true;
  }
  if (incumbentSolution_.agentPaths.empty()) {
    return true;
  }
  if (!std::isfinite(marketBestPressure_) || !std::isfinite(marketBestWait_)) {
    return true;
  }
  const bool pressureOK = candidatePressure <= marketBestPressure_ + marketTauP_;
  const bool waitOK = candidateWait <= marketBestWait_ + marketTauW_;
  return pressureOK && waitOK;
}

void LNS::updateMarketStateFromCurrentSolution() {
  if (!marketHeuristics_) {
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
      marketEta_ / std::sqrt(1.0 + (double)marketStats_.updates);

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
      const double newHat = marketRho_ * oldHat + (1.0 - marketRho_) * excess;

      double newPrice = oldPrice;
      if (excess > 0) {
        const double basePrice = max(oldPrice, 1.0);
        newPrice = min(marketPriceCap_, basePrice * std::exp(eta * newHat));
      } else {
        newPrice = oldPrice * max(0.0, 1.0 - marketGamma_);
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

  updateCategory(vertexDemand, marketVertexPrices_, marketVertexExcessHat_,
                 marketVertexBucketCapacity_);
  updateCategory(edgeDemand, marketEdgePrices_, marketEdgeExcessHat_,
                 marketEdgeBucketCapacity_);

  vector<double> allPrices;
  allPrices.reserve(marketVertexPrices_.size() + marketEdgePrices_.size());
  double totalPriceMass = 0.0;
  for (const auto& kv : marketVertexPrices_) {
    if (kv.second > 0.0) {
      allPrices.push_back(kv.second);
      totalPriceMass += kv.second;
    }
  }
  for (const auto& kv : marketEdgePrices_) {
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

  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetrics(perTask, nullptr);
  double totalWait = 0.0;
  double maxWait = 0.0;
  for (const TaskScheduleMetrics& metric : perTask) {
    if (!metric.valid) {
      continue;
    }
    totalWait += metric.waitPrec;
    maxWait = max(maxWait, (double)metric.waitPrec);
  }

  marketStats_.updates++;
  marketStats_.contendedResources = contendedResources;
  marketStats_.meanPriceContended =
      contendedResources > 0 ? contendedPriceSum / (double)contendedResources
                             : 0.0;
  marketStats_.maxPrice = maxPrice;
  marketStats_.topPriceMassFrac = topPriceMassFrac;
  marketStats_.totalPrecedenceWait = totalWait;
  marketStats_.maxPrecedenceWait = maxWait;
}

void LNS::maybeUpdateMarketState(bool accepted) {
  if (!marketHeuristics_) {
    return;
  }
  if (marketUpdateOnAcceptedOnly_) {
    if (!accepted) {
      return;
    }
    marketAcceptedCounter_++;
    if (marketAcceptedCounter_ % marketUpdatePeriodAccepted_ != 0) {
      return;
    }
  } else {
    marketUpdateCounter_++;
    if (marketUpdateCounter_ % marketUpdatePeriodAccepted_ != 0) {
      return;
    }
  }
  updateMarketStateFromCurrentSolution();
}

void LNS::marketTatonnementRemoval(
    std::optional<set<Conflicts>> potentialNeighborhood) {
  PLOGD << "Using market tatonnement removal\n";

  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();
  lnsNeighborhood_.removedTasks.clear();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  vector<TaskScheduleMetrics> perTask;
  vector<double> blockedWaitSum;
  computeTaskScheduleMetrics(perTask, &blockedWaitSum);
  const vector<vector<int>> predecessors = instance_.getAncestors();
  const vector<vector<int>> successors = instance_.getSuccessors();

  vector<pair<int, double>> rankedTasks;
  rankedTasks.reserve(taskCount);
  const int currentIter = (int)iterationStats.size();
  for (int task = 0; task < taskCount; task++) {
    const auto itAgent = solution_.taskAgentMap.find(task);
    if (itAgent == solution_.taskAgentMap.end() || itAgent->second == UNASSIGNED) {
      continue;
    }
    const double exposure = computeTaskMarketExposure(task, true);
    const double wait = perTask[task].valid ? (double)perTask[task].waitPrec : 0.0;
    const int blocker = perTask[task].blocker;
    const double root = (blocker >= 0 && blocker < taskCount)
                            ? blockedWaitSum[blocker]
                            : 0.0;
    double burden = marketDestroyWeightPrice_ * exposure +
                    marketDestroyWeightWait_ * wait +
                    marketDestroyWeightRoot_ * root;
    if (task < (int)marketTaskCooldownUntilIter_.size() &&
        marketTaskCooldownUntilIter_[task] > currentIter) {
      burden *= 0.25;
    }
    rankedTasks.emplace_back(task, burden);
  }

  if (rankedTasks.empty()) {
    randomRemoval();
    return;
  }

  std::sort(rankedTasks.begin(), rankedTasks.end(),
            [](const pair<int, double>& lhs, const pair<int, double>& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  vector<char> selected(taskCount, 0);
  auto addTask = [&](int task) -> bool {
    if (task < 0 || task >= taskCount || selected[task]) {
      return false;
    }
    const auto itAgent = solution_.taskAgentMap.find(task);
    if (itAgent == solution_.taskAgentMap.end()) {
      return false;
    }
    const int agent = itAgent->second;
    if (agent == UNASSIGNED) {
      return false;
    }
    const vector<int>& assignments = solution_.agents[agent].taskAssignments;
    const auto itPos = std::find(assignments.begin(), assignments.end(), task);
    if (itPos == assignments.end()) {
      return false;
    }
    const int pos = (int)(itPos - assignments.begin());
    selected[task] = 1;
    lnsNeighborhood_.removedTasks.insert(Conflicts(task, agent, pos));
    return true;
  };

  if (potentialNeighborhood.has_value() && !potentialNeighborhood->empty() &&
      incumbentSolution_.agentPaths.empty()) {
    const int quota = max(1, cappedNeighborSize / 2);
    const set<Conflicts> conflictSeeds =
        extractNConflicts(quota, potentialNeighborhood.value());
    for (const Conflicts& conflict : conflictSeeds) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      addTask(conflict.task);
    }
  }

  const int randomSlots = min(
      cappedNeighborSize,
      max(0, (int)std::round((double)cappedNeighborSize * marketRandomDestroyQuota_)));
  const int targetNonRandom = max(0, cappedNeighborSize - randomSlots);

  int seedPoolSize = (int)std::ceil((double)rankedTasks.size() * marketSeedTopFrac_);
  seedPoolSize = max(1, min(seedPoolSize, (int)rankedTasks.size()));
  vector<int> seedPoolTasks;
  vector<double> seedPoolWeights;
  seedPoolTasks.reserve(seedPoolSize);
  seedPoolWeights.reserve(seedPoolSize);
  for (int i = 0; i < seedPoolSize; i++) {
    seedPoolTasks.push_back(rankedTasks[i].first);
    seedPoolWeights.push_back(max(1e-6, rankedTasks[i].second + 1e-6));
  }

  auto expandAncestors = [&](int rootTask, int maxDepth, int* closureCount) {
    if (rootTask < 0 || rootTask >= taskCount || maxDepth <= 0) {
      return;
    }
    std::queue<pair<int, int>> q;
    q.push({rootTask, 0});
    while (!q.empty() && (int)lnsNeighborhood_.removedTasks.size() < targetNonRandom) {
      const auto [task, depth] = q.front();
      q.pop();
      if (depth >= maxDepth || task < 0 || task >= taskCount) {
        continue;
      }
      for (int pred : predecessors[task]) {
        if ((int)lnsNeighborhood_.removedTasks.size() >= targetNonRandom ||
            *closureCount >= marketClosureCap_) {
          return;
        }
        if (addTask(pred)) {
          (*closureCount)++;
          q.push({pred, depth + 1});
        }
      }
    }
  };

  auto expandSuccessors = [&](int rootTask, int maxDepth, int* closureCount) {
    if (rootTask < 0 || rootTask >= taskCount || maxDepth <= 0) {
      return;
    }
    std::queue<pair<int, int>> q;
    q.push({rootTask, 0});
    while (!q.empty() && (int)lnsNeighborhood_.removedTasks.size() < targetNonRandom) {
      const auto [task, depth] = q.front();
      q.pop();
      if (depth >= maxDepth || task < 0 || task >= taskCount) {
        continue;
      }
      for (int succ : successors[task]) {
        if ((int)lnsNeighborhood_.removedTasks.size() >= targetNonRandom ||
            *closureCount >= marketClosureCap_) {
          return;
        }
        if (addTask(succ)) {
          (*closureCount)++;
          q.push({succ, depth + 1});
        }
      }
    }
  };

  while ((int)lnsNeighborhood_.removedTasks.size() < targetNonRandom &&
         !seedPoolTasks.empty()) {
    std::discrete_distribution<int> seedDist(seedPoolWeights.begin(),
                                             seedPoolWeights.end());
    const int sampledIdx = seedDist(rng_);
    const int seedTask = seedPoolTasks[sampledIdx];
    const int blocker =
        (seedTask >= 0 && seedTask < taskCount) ? perTask[seedTask].blocker
                                                : UNASSIGNED;
    const bool addedSeed = addTask(seedTask);
    if (addedSeed) {
      int closureCount = 0;
      if (blocker != UNASSIGNED && addTask(blocker)) {
        closureCount++;
      }
      expandAncestors(blocker, marketDUp_, &closureCount);
      expandSuccessors(blocker, marketDDown_, &closureCount);
      expandSuccessors(seedTask, marketDDown_, &closureCount);
    }
    seedPoolTasks.erase(seedPoolTasks.begin() + sampledIdx);
    seedPoolWeights.erase(seedPoolWeights.begin() + sampledIdx);
  }

  std::uniform_int_distribution<int> randomTaskDist(0, taskCount - 1);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    addTask(randomTaskDist(rng_));
  }

  if (marketCooldownIters_ > 0) {
    for (int task = 0; task < taskCount; task++) {
      if (selected[task]) {
        marketTaskCooldownUntilIter_[task] = currentIter + marketCooldownIters_;
      }
    }
  }
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
    if (line.find("Agent") != string::npos) {
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
            const int previousLocation = solution_.agents[agent]
                                             .taskPaths[taskIndex - 1]
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
    solution_.agents[agent].pathPlanner->computeHeuristics();
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
  solution_.joinPaths(agentsToCompute);

  // Gather the information
  int initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts += solution_.agents[agent].path.endTimeOrZero();
  }
  solution_.sumOfCosts = initialSumOfCosts;
  return true;
}

bool LNS::buildGreedySolution() {

  // Reset any previous task->agent mapping.
  for (auto& kv : solution_.taskAgentMap) {
    kv.second = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
  }

  // Assign tasks
  greedyTaskAssignment(&instance_, &solution_);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(agent));
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(
        solution_.getAgentGlobalTasks(agent).size(), AgentTaskPath());
    solution_.agents[agent].pathPlanner->computeHeuristics();
  }

  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();

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
      topologicalSort(&instance_, &precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return success;
  }

  // Following the topological order we find the paths for each task
  initialPaths_.resize(instance_.getTasksNum(), AgentTaskPath());
  for (int id : planningOrder) {

    int agent = solution_.getAgentWithTask(id), task = id,
        taskPosition = solution_.getLocalTaskIndex(agent, task), startTime = 0;
    if (taskPosition != 0) {
      int previousTask =
          solution_.agents[agent].taskAssignments[taskPosition - 1];
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTimeChecked();
    }

    PLOGI << "Planning for agent " << agent << " and task " << task << endl;

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    buildConstraintTable(constraintTable, task);
    initialPaths_[id] = solution_.agents[agent].pathPlanner->findPathSegment(
        constraintTable, startTime, taskPosition, 0);
    if (initialPaths_[id].empty()) {
      PLOGE << "No path exists for agent " << agent << " and task " << task
            << endl;
      return false;
    }
    solution_.agents[agent].taskPaths[taskPosition] = initialPaths_[id];
  }

  // Join the individual task paths to form the agent's path
  vector<int> agentsToCompute(instance_.getAgentNum());
  std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
  solution_.joinPaths(agentsToCompute);

  // Gather the information
  int initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    initialSumOfCosts += solution_.agents[agent].path.endTimeOrZero();
  }
  solution_.sumOfCosts = initialSumOfCosts;
  return true;
}

bool LNS::buildGreedySolutionPrecedenceOnly() {

  // Reset any previous task->agent mapping.
  for (auto& kv : solution_.taskAgentMap) {
    kv.second = UNASSIGNED;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    solution_.agents[agent].taskPaths.clear();
    solution_.agents[agent].path = AgentTaskPath();
    solution_.agents[agent].taskAssignments.clear();
    solution_.agents[agent].intraPrecedenceConstraints.clear();
  }

  // Assign tasks (greedy), but do not build collision constraints.
  greedyTaskAssignment(&instance_, &solution_);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(agent));
    solution_.agents[agent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[agent].taskPaths.resize(
        solution_.getAgentGlobalTasks(agent).size(), AgentTaskPath());
    solution_.agents[agent].pathPlanner->computeHeuristics();
  }

  // Global precedence constraints = input + intra-agent ordering.
  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();
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
      topologicalSort(&instance_, &precedenceConstraints, planningOrder);
  if (!success) {
    PLOGE << "Topological sorting failed\n";
    return false;
  }

  // Build predecessor adjacency to enforce precedence via earliest goal times.
  vector<vector<int>> predecessors(instance_.getTasksNum());
  for (const auto& e : precedenceConstraints) {
    predecessors[e.second].push_back(e.first);
  }

  // Plan each task segment with only precedence timing constraints (no paths
  // from other agents in the constraint table).
  initialPaths_.assign(instance_.getTasksNum(), AgentTaskPath());
  for (int task : planningOrder) {
    int agent = solution_.getAgentWithTask(task);
    int taskPosition = solution_.getLocalTaskIndex(agent, task);

    int startTime = 0;
    if (taskPosition != 0) {
      int previousTask = solution_.agents[agent].taskAssignments[taskPosition - 1];
      assert(!initialPaths_[previousTask].empty());
      startTime = initialPaths_[previousTask].endTime();
    }

    int earliestGoalTime = 0;
    for (int pred : predecessors[task]) {
      assert(!initialPaths_[pred].empty());
      earliestGoalTime =
          max(earliestGoalTime, initialPaths_[pred].endTimeChecked() + 1);
    }

    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
    constraintTable.goalLocation = instance_.getTaskLocations(task);
    constraintTable.lengthMin = max(constraintTable.lengthMin, earliestGoalTime);
    constraintTable.latestTimestep =
        max(constraintTable.latestTimestep, constraintTable.lengthMin);

    initialPaths_[task] = solution_.agents[agent].pathPlanner->findPathSegment(
        constraintTable, startTime, taskPosition, 0);
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
  solution_.joinPaths(agentsToCompute);

  // Gather the information.
  int initialSumOfCosts = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      initialSumOfCosts += solution_.agents[agent].path.endTime();
    }
  }
  solution_.sumOfCosts = initialSumOfCosts;
  return true;
}

bool LNS::extractFeasibleSolution() {

  // Only update the feasible solution if the new solution has better cost!
  if (incumbentSolution_.agentPaths.empty() ||
      incumbentSolution_.sumOfCosts > solution_.sumOfCosts) {
    incumbentSolution_.numOfCols = instance_.numOfCols;
    incumbentSolution_.sumOfCosts = solution_.sumOfCosts;
    if ((int)incumbentSolution_.agentPaths.size() == 0) {
      incumbentSolution_.agentPaths.resize(instance_.getAgentNum());
    }
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

void LNS::randomRemoval() {

  PLOGD << "Using random removal\n";

  // Clear old information about the LNS neighborhood. This should be the first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();
  lnsNeighborhood_.removedTasks.clear();

  // Sample tasks uniformly without replacement (partial Fisher-Yates),
  // avoiding repeated draws/duplicate checks.
  const int taskCount = instance_.getTasksNum();
  const int numToRemove = (neighborSize_ < taskCount) ? neighborSize_ : taskCount;
  vector<int> taskIds(taskCount);
  std::iota(taskIds.begin(), taskIds.end(), 0);
  for (int i = 0; i < numToRemove; i++) {
    std::uniform_int_distribution<int> distribution(i, taskCount - 1);
    const int j = distribution(rng_);
    std::swap(taskIds[i], taskIds[j]);
    const int randomTask = taskIds[i];
    const int randomTaskAgent = solution_.taskAgentMap[randomTask];
    assert(randomTaskAgent != UNASSIGNED);
    const int randomTaskPosition =
        solution_.getLocalTaskIndex(randomTaskAgent, randomTask);
    lnsNeighborhood_.removedTasks.insert(
        Conflicts(randomTask, randomTaskAgent, randomTaskPosition));
  }
}

void LNS::conflictRemoval(std::optional<set<Conflicts>> potentialNeighborhood) {

  PLOGD << "Using conflict-based removal\n";

  // Clear old information about the LNS neighborhood. This should be the first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();

  // Conflict removal operator should always be sent this argument!
  assert(potentialNeighborhood.has_value());

  // Extract N conflicts first. This can return N tasks where N <= neighborhood size
  lnsNeighborhood_.removedTasks =
      extractNConflicts(neighborSize_, potentialNeighborhood.value());

  if ((int)lnsNeighborhood_.removedTasks.size() < neighborSize_) {
    // In this case we have less conflicts than the neighborhood size of the LNS
    // so we need to augment this list with more tasks using random removal.
    std::uniform_int_distribution<int> distribution(0,
                                                    instance_.getTasksNum() - 1);
    while ((int)lnsNeighborhood_.removedTasks.size() < neighborSize_) {
      const int randomTask = distribution(rng_);
      if (lnsNeighborhood_.removedTasks.count(Conflicts(randomTask, 0, 0)) != 0) {
        continue;
      }
      const int randomTaskAgent = solution_.taskAgentMap[randomTask];
      assert(randomTaskAgent != UNASSIGNED);
      const int randomTaskPosition =
          solution_.getLocalTaskIndex(randomTaskAgent, randomTask);
      lnsNeighborhood_.removedTasks.insert(
          Conflicts(randomTask, randomTaskAgent, randomTaskPosition));
    }
  }
  // The else case should not happen since the 'extractNConflict' will never return more than neighborhood size set
}

void LNS::worstRemoval() {

  PLOGD << "Using worst removal\n";

  // Clear old information about the LNS neighborhood. This should be the first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();
  lnsNeighborhood_.removedTasks.clear();

  // Maintain a priority queue of (key, value) where key is the path length of a task and the value is the task. We need to do a reverse way to avoid making our own comparator
  ppq worstTasksOrder;
  for (int task = 0; task < instance_.getTasksNum(); task++) {
    int taskAgent = solution_.taskAgentMap[task];
    assert(taskAgent != UNASSIGNED);
    int taskPosition = solution_.getLocalTaskIndex(taskAgent, task);
    int taskPathSize =
        (int)solution_.agents[taskAgent].taskPaths[taskPosition].size();
    worstTasksOrder.emplace(taskPathSize, task);
  }
  while ((int)lnsNeighborhood_.removedTasks.size() < neighborSize_) {
    pair<int, int> worstTaskFromOrder = worstTasksOrder.top();
    int worstTask = worstTaskFromOrder.second;
    // No need to check whether this task was part of the removed tasks already or not since that cannot happen ever!
    int worstTaskAgent = solution_.taskAgentMap[worstTask];
    assert(worstTaskAgent != UNASSIGNED);
    int worstTaskPosition =
        solution_.getLocalTaskIndex(worstTaskAgent, worstTask);
    Conflicts conflict(worstTask, worstTaskAgent, worstTaskPosition);
    lnsNeighborhood_.removedTasks.insert(conflict);
    worstTasksOrder.pop();
  }
}

void LNS::shawRemoval(int prioritySize) {
  /*
  Shaw removal works by using the relatedness parameter ->
  r(task_i, task_j) = w1 * distance(task_i_goal, task_j_goal) 
                    + w2 * (abs(task_i_start_time - task_j_start_time) + abs(task_i_end_time - task_j_end_time))

  -> w1 and w2 are parameters that can be tuned
  -> distance(task_i_goal, task_j_goal) is the manhattan distance between the tasks
  -> task_i_start_time is the begin time of the task
  -> task_i_end_time is the end time of the task

  Need to remove N - 1 tasks after selecting the first task randomly. N can be user input parameter
  */

  // Clear old information about the LNS neighborhood. This should be the first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();
  lnsNeighborhood_.removedTasks.clear();

  // Randomly choose a task and remove it from the solution and add it to the neighborhood
  std::uniform_int_distribution<int> distribution(0,
                                                  instance_.getTasksNum() - 1);

  // Sample a random task and remove it!
  int randomTask = distribution(rng_);
  int randomTaskAgent = solution_.taskAgentMap[randomTask];
  assert(randomTaskAgent != UNASSIGNED);
  int randomTaskPosition =
      solution_.getLocalTaskIndex(randomTaskAgent, randomTask);
  Conflicts randomConflict(randomTask, randomTaskAgent, randomTaskPosition);
  lnsNeighborhood_.removedTasks.insert(randomConflict);
  PLOGD << "Shaw Removal Step -> Random Task " << randomTask << " is removed!"
        << endl;

  const int taskCount = instance_.getTasksNum();
  const int cappedPrioritySize = min(prioritySize, taskCount);
  const int cappedNeighborSize = min(neighborSize_, taskCount);

  // Get information about random task
  int randomTaskLocation = instance_.getTaskLocations(randomTask);
  int randomTaskST =
      solution_.agents[randomTaskAgent].taskPaths[randomTaskPosition].beginTime;
  int randomTaskET =
      solution_.agents[randomTaskAgent].taskPaths[randomTaskPosition].endTime();

  // Initialize a queue to hold the related tasks and rank by relatedness
  pqRelatedTasks relatedQ;  // TODO: can change to ascending or descending here
  set<RelatedTasks, RelatedTasks::RelatedTasksComparator> expandedTasks;
  vector<bool> alreadyExpanded(taskCount, false);

  // Adding the random task first
  RelatedTasks randomRelatedTask(randomTask, randomTaskAgent,
                                 randomTaskPosition, randomTaskST, randomTaskET,
                                 -1, -1);
  expandedTasks.insert(randomRelatedTask);
  alreadyExpanded[randomTask] = true;

  auto pushRelatedCandidate = [&](int relatedTask) {
    // Get information about related task
    const int relatedTaskAgent = solution_.taskAgentMap[relatedTask];
    assert(relatedTaskAgent != UNASSIGNED);
    const int relatedTaskPosition =
        solution_.getLocalTaskIndex(relatedTaskAgent, relatedTask);

    // Compute the manhattan distance
    const int relatedTaskLocation = instance_.getTaskLocations(relatedTask);
    const int relatedManhattanDistance =
        instance_.getManhattanDistance(randomTaskLocation, relatedTaskLocation);

    // Get the temporal values
    const int relatedTaskST = solution_.agents[relatedTaskAgent]
                                  .taskPaths[relatedTaskPosition]
                                  .beginTime;
    const int relatedTaskET = solution_.agents[relatedTaskAgent]
                                  .taskPaths[relatedTaskPosition]
                                  .endTime();

    // Compute the relatedness (smaller => more related).
    const int temporalDiff =
        abs(randomTaskST - relatedTaskST) + abs(randomTaskET - relatedTaskET);
    const int relatedness =
        (int)(shawDistanceWeight_ * relatedManhattanDistance +
              shawTemporalWeight_ * temporalDiff);

    // Store information
    RelatedTasks relatedToRandomTask(relatedTask, relatedTaskAgent,
                                     relatedTaskPosition, relatedTaskST,
                                     relatedTaskET, relatedManhattanDistance,
                                     relatedness);
    expandedTasks.insert(relatedToRandomTask);
    relatedQ.emplace(relatedness, relatedToRandomTask);
    alreadyExpanded[relatedTask] = true;
  };

  // Fill candidates uniformly at random.
  while ((int)expandedTasks.size() < cappedPrioritySize) {
    const int relatedTask = distribution(rng_);
    if (alreadyExpanded[relatedTask]) {
      continue;
    }
    pushRelatedCandidate(relatedTask);
  }

  // Now remove the most-related tasks.
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
         !relatedQ.empty()) {
    RelatedTasks relatedTask = relatedQ.top().second;
    PLOGD << "Shaw Removal Step -> Related Task " << relatedTask.task
          << " is removed!" << endl;
    relatedQ.pop();
    // Add the related task to the neighborhood
    Conflicts relatedConflict(relatedTask.task, relatedTask.agent,
                              relatedTask.taskPosition);
    lnsNeighborhood_.removedTasks.insert(relatedConflict);
  }

  // If the candidate queue was exhausted (e.g., very small instances), augment
  // with random tasks to reach the requested neighborhood size.
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    const int t = distribution(rng_);
    if (lnsNeighborhood_.removedTasks.count(Conflicts(t, 0, 0)) != 0) {
      continue;
    }
    const int agent = solution_.taskAgentMap[t];
    assert(agent != UNASSIGNED);
    const int pos = solution_.getLocalTaskIndex(agent, t);
    lnsNeighborhood_.removedTasks.insert(Conflicts(t, agent, pos));
  }
}

void LNS::precedenceWaitRemoval(
    std::optional<set<Conflicts>> potentialNeighborhood) {
  PLOGD << "Using precedence-wait removal\n";

  // Clear old information about the LNS neighborhood. This should be the
  // first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();
  lnsNeighborhood_.removedTasks.clear();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  const vector<vector<int>> predecessors = instance_.getAncestors();
  const vector<vector<int>> successors = instance_.getSuccessors();

  vector<int> criticalPred(taskCount, UNASSIGNED);
  vector<int> releaseTime(taskCount, 0);
  vector<int> precedenceWait(taskCount, 0);
  vector<char> taskFeasible(taskCount, 1);
  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, -1);

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& taskPaths = solution_.agents[agent].taskPaths;
    const int maxPos = min((int)assignments.size(), (int)taskPaths.size());
    for (int pos = 0; pos < maxPos; pos++) {
      const int task = assignments[pos];
      if (task >= 0 && task < taskCount) {
        taskToAgent[task] = agent;
        taskToPosition[task] = pos;
      }
    }
  }

  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0 ||
        taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      taskFeasible[task] = 0;
      continue;
    }

    int prevEnd = 0;
    int prevLocation = instance_.getStartLocations()[agent];
    if (taskPos > 0) {
      const AgentTaskPath& previousTaskPath =
          solution_.agents[agent].taskPaths[taskPos - 1];
      if (!previousTaskPath.empty()) {
        prevEnd = previousTaskPath.endTime();
        prevLocation = previousTaskPath.back().location;
      } else {
        // Defensive fallback for malformed intermediate schedules.
        const int previousTask =
            solution_.agents[agent].taskAssignments[taskPos - 1];
        prevLocation = instance_.getTaskLocations(previousTask);
        prevEnd = 0;
      }
    }

    const int taskLocation = instance_.getTaskLocations(task);
    const int travelToTask =
        instance_.getManhattanDistance(prevLocation, taskLocation);
    const int earliestArrivalNoPrec = prevEnd + travelToTask;

    int maxPredEnd = std::numeric_limits<int>::min();
    int blocker = UNASSIGNED;
    for (int pred : predecessors[task]) {
      if (pred < 0 || pred >= taskCount) {
        continue;
      }
      const int predAgent = taskToAgent[pred];
      const int predPos = taskToPosition[pred];
      if (predAgent == UNASSIGNED) {
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
      if (predEnd > maxPredEnd) {
        maxPredEnd = predEnd;
        blocker = pred;
      }
    }

    if (blocker != UNASSIGNED) {
      criticalPred[task] = blocker;
      releaseTime[task] = maxPredEnd;
      precedenceWait[task] = max(0, maxPredEnd - earliestArrivalNoPrec);
    }
  }

  vector<int> order(taskCount);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int lhs, int rhs) {
                     if (taskFeasible[lhs] != taskFeasible[rhs]) {
                       return taskFeasible[lhs] > taskFeasible[rhs];
                     }
                     if (precedenceWait[lhs] != precedenceWait[rhs]) {
                       return precedenceWait[lhs] > precedenceWait[rhs];
                     }
                     if (releaseTime[lhs] != releaseTime[rhs]) {
                       return releaseTime[lhs] > releaseTime[rhs];
                     }
                     return lhs < rhs;
                   });

  vector<char> selected(taskCount, 0);
  auto addTask = [&](int task) -> bool {
    if (task < 0 || task >= taskCount || selected[task]) {
      return false;
    }
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0) {
      return false;
    }
    selected[task] = 1;
    lnsNeighborhood_.removedTasks.insert(Conflicts(task, agent, taskPos));
    return true;
  };

  // If the current solution is infeasible, keep a conflict-focused anchor so
  // this operator can still drive towards feasibility instead of only
  // precedence reshaping.
  if (potentialNeighborhood.has_value() && !potentialNeighborhood->empty()) {
    // Bootstrap phase: until we find the first feasible incumbent, prioritize
    // conflict-driven neighborhoods to quickly recover feasibility.
    const bool noFeasibleIncumbent = incumbentSolution_.agentPaths.empty();
    const int conflictQuota =
        noFeasibleIncumbent ? cappedNeighborSize : max(1, cappedNeighborSize / 2);
    const set<Conflicts> conflictSeeds =
        extractNConflicts(conflictQuota, potentialNeighborhood.value());
    for (const Conflicts& conflict : conflictSeeds) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      addTask(conflict.task);
    }
  }

  auto addOneHopNeighborhood = [&](int task) {
    if (task < 0 || task >= taskCount) {
      return;
    }
    for (int pred : predecessors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(pred);
    }
    for (int succ : successors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(succ);
    }
  };

  // Pass 1: prioritize tasks with positive precedence wait. Include the critical
  // predecessor and one-hop precedence neighbors to address root causes.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    if (!taskFeasible[task] || precedenceWait[task] <= 0) {
      continue;
    }
    const bool insertedSeed = addTask(task);
    if (!insertedSeed) {
      continue;
    }
    const int blocker = criticalPred[task];
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        blocker != UNASSIGNED) {
      addTask(blocker);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      addOneHopNeighborhood(task);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        blocker != UNASSIGNED) {
      addOneHopNeighborhood(blocker);
    }
  }

  // Pass 2: if needed, fill from global precedence-wait order.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    addTask(task);
  }

  // Pass 3: fallback random fill (defensive; should almost never trigger).
  std::uniform_int_distribution<int> distribution(0, taskCount - 1);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    addTask(distribution(rng_));
  }
}

void LNS::lowSlackRemoval(std::optional<set<Conflicts>> potentialNeighborhood) {
  PLOGD << "Using low-slack removal\n";

  // Clear old information about the LNS neighborhood. This should be the
  // first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();
  lnsNeighborhood_.removedTasks.clear();

  const int taskCount = instance_.getTasksNum();
  if (taskCount <= 0) {
    return;
  }
  const int cappedNeighborSize = min(neighborSize_, taskCount);
  if (cappedNeighborSize <= 0) {
    return;
  }

  const vector<vector<int>> predecessors = instance_.getAncestors();
  const vector<vector<int>> successors = instance_.getSuccessors();
  const int INF = std::numeric_limits<int>::max() / 4;

  vector<int> taskToAgent(taskCount, UNASSIGNED);
  vector<int> taskToPosition(taskCount, -1);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& taskPaths = solution_.agents[agent].taskPaths;
    const int maxPos = min((int)assignments.size(), (int)taskPaths.size());
    for (int pos = 0; pos < maxPos; pos++) {
      const int task = assignments[pos];
      if (task >= 0 && task < taskCount) {
        taskToAgent[task] = agent;
        taskToPosition[task] = pos;
      }
    }
  }

  vector<char> taskFeasible(taskCount, 1);
  vector<char> hasSuccessor(taskCount, 0);
  vector<int> slack(taskCount, INF);
  vector<int> tightSuccessor(taskCount, UNASSIGNED);

  for (int task = 0; task < taskCount; task++) {
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0 ||
        taskPos >= (int)solution_.agents[agent].taskPaths.size()) {
      taskFeasible[task] = 0;
      continue;
    }
    const AgentTaskPath& taskPath = solution_.agents[agent].taskPaths[taskPos];
    if (taskPath.empty()) {
      taskFeasible[task] = 0;
      continue;
    }

    const int endTask = taskPath.endTime();
    int minSuccStart = INF;
    int minSucc = UNASSIGNED;
    for (int succ : successors[task]) {
      if (succ < 0 || succ >= taskCount) {
        continue;
      }
      const int succAgent = taskToAgent[succ];
      const int succPos = taskToPosition[succ];
      if (succAgent == UNASSIGNED || succPos < 0 ||
          succPos >= (int)solution_.agents[succAgent].taskPaths.size()) {
        continue;
      }
      const AgentTaskPath& succPath = solution_.agents[succAgent].taskPaths[succPos];
      if (succPath.empty()) {
        continue;
      }
      hasSuccessor[task] = 1;
      const int succStart = succPath.beginTime;
      if (succStart < minSuccStart) {
        minSuccStart = succStart;
        minSucc = succ;
      }
    }

    if (minSucc != UNASSIGNED) {
      tightSuccessor[task] = minSucc;
      slack[task] = minSuccStart - endTask;
    }
  }

  vector<int> order(taskCount);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    if (taskFeasible[lhs] != taskFeasible[rhs]) {
      return taskFeasible[lhs] > taskFeasible[rhs];
    }
    if (hasSuccessor[lhs] != hasSuccessor[rhs]) {
      return hasSuccessor[lhs] > hasSuccessor[rhs];
    }
    if (slack[lhs] != slack[rhs]) {
      return slack[lhs] < slack[rhs];  // lower slack = tighter
    }
    return lhs < rhs;
  });

  vector<char> selected(taskCount, 0);
  auto addTask = [&](int task) -> bool {
    if (task < 0 || task >= taskCount || selected[task]) {
      return false;
    }
    const int agent = taskToAgent[task];
    const int taskPos = taskToPosition[task];
    if (agent == UNASSIGNED || taskPos < 0) {
      return false;
    }
    selected[task] = 1;
    lnsNeighborhood_.removedTasks.insert(Conflicts(task, agent, taskPos));
    return true;
  };

  // If the current solution is infeasible, keep a conflict-focused anchor so
  // this operator can recover feasibility.
  if (potentialNeighborhood.has_value() && !potentialNeighborhood->empty()) {
    const int conflictQuota = max(1, cappedNeighborSize / 2);
    const set<Conflicts> conflictSeeds =
        extractNConflicts(conflictQuota, potentialNeighborhood.value());
    for (const Conflicts& conflict : conflictSeeds) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        break;
      }
      addTask(conflict.task);
    }
  }

  auto addOneHopNeighborhood = [&](int task) {
    if (task < 0 || task >= taskCount) {
      return;
    }
    for (int pred : predecessors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(pred);
    }
    for (int succ : successors[task]) {
      if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
        return;
      }
      addTask(succ);
    }
  };

  // Pass 1: remove low-slack tasks and their tight successors / local DAG
  // neighbors to unlock precedence bottlenecks.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    if (!taskFeasible[task] || !hasSuccessor[task]) {
      continue;
    }
    const bool insertedSeed = addTask(task);
    if (!insertedSeed) {
      continue;
    }
    const int succ = tightSuccessor[task];
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        succ != UNASSIGNED) {
      addTask(succ);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
      addOneHopNeighborhood(task);
    }
    if ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize &&
        succ != UNASSIGNED) {
      addOneHopNeighborhood(succ);
    }
  }

  // Pass 2: if needed, fill from sorted slack order.
  for (int task : order) {
    if ((int)lnsNeighborhood_.removedTasks.size() >= cappedNeighborSize) {
      break;
    }
    addTask(task);
  }

  // Pass 3: random fallback to guarantee neighborhood size.
  std::uniform_int_distribution<int> distribution(0, taskCount - 1);
  while ((int)lnsNeighborhood_.removedTasks.size() < cappedNeighborSize) {
    addTask(distribution(rng_));
  }
}

void LNS::alnsRemoval(std::optional<set<Conflicts>> potentialNeighborhood) {

  // Clear old information about the LNS neighborhood. This should be the first thing that any removal operator must do!
  lnsNeighborhood_.patchedTasks.clear();
  lnsNeighborhood_.regretMaxHeap.clear();
  lnsNeighborhood_.commitedTasks.clear();
  lnsNeighborhood_.removedTasksPathSize.clear();

  adaptiveLNS_.alnsCounter++;

  // Cannot update the successes in the first iteration!
  if (iterationStats.size() != 1) {
    // Incorporate the results of the heuristic performance in the last iteration
    switch (iterationStats.back().quality) {
      case bestSolutionYet:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta1;
        break;
      case improvedSolution:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta2;
        break;
      case dowgradedButAccepted:
        adaptiveLNS_.success[adaptiveLNS_.destroyHeuristicHistory.back()] +=
            adaptiveLNS_.delta3;
        break;
      default:
        break;
    }
  }

  if (adaptiveLNS_.alnsCounter >= adaptiveLNS_.alnsCounterThreshold) {
    // Need to update the weights here!
    for (int i = 0; i < adaptiveLNS_.numDestroyHeuristics; i++) {
      if (adaptiveLNS_.used[i] > 0) {
        adaptiveLNS_.weights[i] =
            (1.0 - adaptiveLNS_.reactionFactor) * adaptiveLNS_.weights[i] +
            adaptiveLNS_.reactionFactor *
                (adaptiveLNS_.success[i] / adaptiveLNS_.used[i]);
      } else {
        adaptiveLNS_.weights[i] =
            (1.0 - adaptiveLNS_.reactionFactor) * adaptiveLNS_.weights[i];
      }
      adaptiveLNS_.used[i] = 0;
      adaptiveLNS_.success[i] = 0;
    }
    adaptiveLNS_.alnsCounter = 0;
  }
  // Sample the destroy heuristic and extract the neighborhood
  std::discrete_distribution<> distribution(adaptiveLNS_.weights.begin(),
                                            adaptiveLNS_.weights.end());

  int sampledDestroyHeuristic = distribution(rng_);
  adaptiveLNS_.recentDestroyHeuristic = sampledDestroyHeuristic;
  switch (sampledDestroyHeuristic) {
    case DestroyHeuristic::randomRemoval:  // RANDOM
      randomRemoval();
      break;
    case DestroyHeuristic::worstRemoval:  // WORST
      worstRemoval();
      break;
    case DestroyHeuristic::conflictRemoval:  // CONFLICT
      conflictRemoval(std::move(potentialNeighborhood));
      break;
    case DestroyHeuristic::shawRemoval:  // SHAW
      shawRemoval(neighborSize_ * 3);
      break;
    case DestroyHeuristic::precedenceWaitRemoval:  // PRECEDENCE WAIT
      precedenceWaitRemoval(std::move(potentialNeighborhood));
      break;
    case DestroyHeuristic::lowSlackRemoval:  // LOW SLACK
      lowSlackRemoval(std::move(potentialNeighborhood));
      break;
    case DestroyHeuristic::marketTatonnementRemoval:  // MARKET TATONNEMENT
      marketTatonnementRemoval(std::move(potentialNeighborhood));
      break;
    default:
      PLOGD << "Sampled a non-existant destroy heuristic!\n";
      static_assert(true);
  }

  adaptiveLNS_.used[sampledDestroyHeuristic] += 1;
  adaptiveLNS_.destroyHeuristicHistory.push_back(sampledDestroyHeuristic);
}

bool LNS::simulatedAnnealing() {

  bool accepted = false;
  double acceptanceProb =
      exp((previousSolution_.utility - solution_.utility) / temperature_);
  std::uniform_real_distribution<double> unit01(0.0, 1.0);
  if (unit01(rng_) < acceptanceProb) {
    // Use simulated annealing to potentially accept worse solutions!
    accepted = true;
  } else {
    // Reject this solution
    solution_ = previousSolution_;
    PLOGD << "Rejecting this solution!\n";
  }
  temperature_ *= coolingCoefficient_;
  return accepted;
}

bool LNS::thresholdAcceptance() {

  bool accepted = false;
  // In this case we are worse than the previous solution but within some threshold so we can accept this one
  if (solution_.utility - previousSolution_.utility < temperature_) {
    accepted = true;
  } else {
    // Reject this solution
    solution_ = previousSolution_;
    PLOGD << "Rejecting this solution!\n";
  }
  temperature_ *= coolingCoefficient_;
  return accepted;
}

bool LNS::oldBachelorsAcceptance() {

  bool accepted = false;
  if (solution_.utility - previousSolution_.utility < temperature_) {
    // Accept this solution and reduce the temperature
    temperature_ *= coolingCoefficient_;
    accepted = true;
  } else {
    // Reject this solution and increase the temperature
    solution_ = previousSolution_;
    temperature_ *= heatingCoefficient_;
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}

bool LNS::greatDelugeAlgorithm() {

  bool accepted = false;
  if (solution_.utility - previousSolution_.utility < temperature_) {
    // This temperature acts as a water level and we want to accept solutions that fall within some water level and corresponding increase it further for future iterations
    // Since we are effectively doing a minimization problem we need to decrease the temperature ONLY if we accept
    temperature_ *= coolingCoefficient_;
    accepted = true;
  } else {
    // Reject this solution but dont change the temperature'
    solution_ = previousSolution_;
    PLOGD << "Rejecting this solution\n";
  }
  return accepted;
}

bool LNS::run() {

  bool success = false;
  if (initialSolutionStrategy == "greedy") {
    // Run the greedy task assignment and subsequent path finding algorithm
    success = buildGreedySolution();
  } else if (initialSolutionStrategy == "greedy_precedence_only") {
    // Precedence-feasible, collision-infeasible warm start.
    success = buildGreedySolutionPrecedenceOnly();
  } else if (initialSolutionStrategy.find("sota") != string::npos) {
    // Run the greedy task assignment and use CBS-PC for finding the paths of agents
    success = buildGreedySolutionWithMAPFPC(initialSolutionStrategy);
  }

  if (!success && initialSolutionStrategy != "greedy") {
    success = buildGreedySolution();
  }

  // If the initial solution strategy failed then we cannot do anything!
  if (!success) {
    return success;
  }

  initialSolutionRuntime_ = ((fsec)(Time::now() - plannerStartTime_)).count();
  runtime = initialSolutionRuntime_;

  PLOGD << "Initial solution cost = " << solution_.sumOfCosts
        << ", Runtime = " << initialSolutionRuntime_ << endl;

  set<Conflicts> potentialNeighborhood;  // Need for the conflict removal case
  bool valid = validateSolution(&potentialNeighborhood);

  bool feasibleSolutionUpdated = false;
  if (valid) {
    feasibleSolutionUpdated = true;
    extractFeasibleSolution();
  }

  if (marketHeuristics_) {
    updateMarketStateFromCurrentSolution();
    if (valid) {
      marketBestPressure_ = computeSolutionMarketPressure();
      marketBestWait_ = computeSolutionPrecedenceWait();
    }
  }

  iterationStats.emplace_back(initialSolutionRuntime_, initialSolutionStrategy,
                              instance_.getAgentNum(), instance_.getTasksNum(),
                              solution_.sumOfCosts, feasibleSolutionUpdated,
                              bestSolutionYet);

  set<Conflicts> oldNeighborhood;

  // Needed to maintain a running mean and standard deviation which can then be used to standardize the sum of costs and conflicts in accepting criteria functions
  MovingMetrics metrics(numOfIterations_, lnsConflictWeight_, lnsCostWeight_,
                        (int)potentialNeighborhood.size(),
                        solution_.sumOfCosts);
  solution_.utility = metrics.computeMovingMetrics(
      (int)potentialNeighborhood.size(), solution_.sumOfCosts);

  temperature_ = solution_.utility * (tolerance_ / 100);
  if (acceptanceCriteria == "SA") {
    temperature_ /= log(2);
  }

  previousSolution_ = solution_;

  // LNS loop
  while (runtime < timeLimit_ &&
         (int)iterationStats.size() < numOfIterations_ * 2) {
    const int previousSocForIter = previousSolution_.sumOfCosts;
    int alnsHeuristicForIter = -1;

    // These functions populate the LNS neighborhoods' removedTask parameter
    if (destroyHeuristic == "conflict") {
      conflictRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "worst") {
      worstRemoval();
    } else if (destroyHeuristic == "random") {
      randomRemoval();
    } else if (destroyHeuristic == "shaw") {
      shawRemoval(neighborSize_ * 3);
    } else if (destroyHeuristic == "precedence_wait") {
      precedenceWaitRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "low_slack") {
      lowSlackRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "market_tatonnement") {
      marketTatonnementRemoval(std::make_optional(potentialNeighborhood));
    } else if (destroyHeuristic == "alns") {
      alnsRemoval(std::make_optional(potentialNeighborhood));
      alnsHeuristicForIter = adaptiveLNS_.recentDestroyHeuristic;
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.selections[alnsHeuristicForIter]++;
      }
    } else {
      static_assert(true);
    }

    oldNeighborhood = lnsNeighborhood_.removedTasks;

    PLOGD << "Printing neighborhood conflict tasks\n";
    PLOGD << "Size: " << lnsNeighborhood_.removedTasks.size() << "\n";
    for (Conflicts conflictTask : lnsNeighborhood_.removedTasks) {
      PLOGD << "Conflicted Task : " << conflictTask.task << "\n";
    }

    prepareNextIteration();

    // This needs to happen after prepare iteration since we updated the conflictedTasks variable in the prepare next iteration function
    for (Conflicts conflictedTask : lnsNeighborhood_.removedTasks) {
      lnsNeighborhood_.commitedTasks.insert(
          make_pair(conflictedTask.task, false));
    }
    lnsNeighborhood_.immutableRemovedTasks = lnsNeighborhood_.removedTasks;

    // Repair: commit removed tasks back using regret.
    //
    // Default behavior recomputes regrets from scratch every commit.
    // Optional incremental mode recomputes regrets only for a dirty subset.
    lnsNeighborhood_.regretMaxHeap.clear();
    bool repairFailed = false;
    regretEvalStatsCurrent_.reset();
    {
      const int removedCount = (int)lnsNeighborhood_.removedTasks.size();
      regretEvalStatsCurrent_.neighborhoods++;
      regretEvalStatsTotal_.neighborhoods++;
      regretEvalStatsCurrent_.removedTasksSum += removedCount;
      regretEvalStatsTotal_.removedTasksSum += removedCount;
      regretEvalStatsCurrent_.removedTasksMax =
          max(regretEvalStatsCurrent_.removedTasksMax, (int64_t)removedCount);
      regretEvalStatsTotal_.removedTasksMax =
          max(regretEvalStatsTotal_.removedTasksMax, (int64_t)removedCount);
    }
    if (!incrementalRegret_) {
      // Compute regret for each of the tasks that are in the conflicting set
      // Pick the best one and repeat the whole process again
      while (!lnsNeighborhood_.removedTasks.empty()) {
        bool enoughSpace = computeRegret();
        if (!enoughSpace) {
          // We could not compute enough regrets for each task so we need to try
          // and reset the solution and try potentially with a different
          // neighborhood!
          repairFailed = true;
          break;
        }
        assert(!lnsNeighborhood_.regretMaxHeap.empty());
        Regret bestRegret = lnsNeighborhood_.regretMaxHeap.top();
        // Use the best regret task and insert it in its correct location
        commitBestRegretTask(bestRegret);
      }
    } else {
      // Initial regret computation for the neighborhood.
      incrementalRegretStatsCurrent_.reset();
      if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
        repairFailed = true;
      }

      int64_t stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
      int64_t stalePopsSinceRefresh = 0;
      int64_t commitsSinceRefresh = 0;

      while (!repairFailed && !lnsNeighborhood_.removedTasks.empty()) {
        // If we are spending too much effort discarding stale heap entries,
        // do a full refresh of all remaining regrets.
        //
        // This helps when dirty rules miss some dependencies (regret drift),
        // and also prevents the heap from filling up with stale entries.
        if (stalePopsSinceRefresh >= 100 &&
            stalePopsSinceRefresh > 2 * commitsSinceRefresh + 50) {
          lnsNeighborhood_.regretMaxHeap.clear();
          incrementalRegretStatsCurrent_.fullRefreshes++;
          incrementalRegretStatsTotal_.fullRefreshes++;
          if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
            repairFailed = true;
            break;
          }
          stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
          stalePopsSinceRefresh = 0;
          commitsSinceRefresh = 0;
        }

        const auto bestRegret = popNextValidRegret();
        const int64_t staleDelta =
            incrementalRegretStatsCurrent_.stalePops - stalePopsAtLastCheck;
        stalePopsAtLastCheck = incrementalRegretStatsCurrent_.stalePops;
        stalePopsSinceRefresh += staleDelta;

        if (!bestRegret.has_value()) {
          // Heap may have been exhausted by stale entries; rebuild for whatever
          // is left.
          incrementalRegretStatsCurrent_.heapRebuilds++;
          incrementalRegretStatsTotal_.heapRebuilds++;
          if (!recomputeRegretsForTasks(collectRemainingRemovedTasks())) {
            repairFailed = true;
          }
          continue;
        }

        const vector<int> endTimesBefore = computeCurrentTaskEndTimes();
        const vector<int> lastTaskBefore = computeCurrentLastTaskPerAgent();
        commitBestRegretTask(*bestRegret);
        incrementalRegretStatsCurrent_.commits++;
        incrementalRegretStatsTotal_.commits++;
        commitsSinceRefresh++;
        const vector<int> endTimesAfter = computeCurrentTaskEndTimes();
        const vector<int> lastTaskAfter = computeCurrentLastTaskPerAgent();

        const vector<int> dirtyTasks = computeDirtyTasksAfterCommit(
            endTimesBefore, endTimesAfter, lastTaskBefore, lastTaskAfter);
        if (!recomputeRegretsForTasks(dirtyTasks)) {
          repairFailed = true;
        }
      }

      // Stats are collected and printed in a dedicated summary section.
    }

    IterationQuality quality = IterationQuality::none;

    // If we could not successfully compute the regrets and commit to all the tasks in the neighborhood then we need to reset this neighborhood!
    if (repairFailed || !lnsNeighborhood_.removedTasks.empty()) {
      if (alnsHeuristicForIter >= 0 &&
          alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
        adaptiveLNS_.couldNotFind[alnsHeuristicForIter]++;
      }
      // Reject whatever we done till now
      solution_ = previousSolution_;
      feasibleSolutionUpdated = false;
      quality = IterationQuality::couldNotFind;
      runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
      iterationStats.emplace_back(runtime, "LNS", instance_.getAgentNum(),
                                  instance_.getTasksNum(), solution_.sumOfCosts,
                                  feasibleSolutionUpdated, quality);
      // Skip everything after this statement
      PLOGD << "Could not find paths for the neighborhood! Attempting a new "
               "neighborhood computation\n";
      maybeUpdateMarketState(false);
      continue;
    }

    // Join the individual paths that were found for each agent
    for (int i = 0; i < instance_.getAgentNum(); i++) {
      solution_.agents[i].path = AgentTaskPath();
    }
    vector<int> agentsToCompute(instance_.getAgentNum());
    std::iota(agentsToCompute.begin(), agentsToCompute.end(), 0);
    solution_.joinPaths(agentsToCompute);

    // Compute the updated sum of costs
    solution_.sumOfCosts = 0;
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      solution_.sumOfCosts += solution_.agents[agent].path.endTimeOrZero();
    }

    PLOGD << "Old sum of costs = " << previousSolution_.sumOfCosts << endl;
    PLOGD << "New sum of costs = " << solution_.sumOfCosts << endl;

    PLOGD << "Number of conflicts in old solution: "
          << (int)potentialNeighborhood.size() << endl;

    // Extract the set of conflicting tasks
    potentialNeighborhood.clear();
    valid = validateSolution(&potentialNeighborhood);

    PLOGD << "Number of conflicts in new solution: "
          << potentialNeighborhood.size() << endl;

    // Accept the solution only if the new one has higher utility compared to the old solution where utility is a weighted combination of the number of conflicts and sum of costs.
    // Compute the utility of this solution
    solution_.utility = metrics.computeMovingMetrics(
        (int)potentialNeighborhood.size(), solution_.sumOfCosts);

    if (!valid) {
      // Solution was not valid as we found some conflicts!
      feasibleSolutionUpdated = false;
      PLOGE << "The solution was not valid!\n";
    } else {
      if (extractFeasibleSolution()) {
        // This is the case when the feasible solution was updated!
        quality = IterationQuality::bestSolutionYet;
        feasibleSolutionUpdated = true;
      } else {
        feasibleSolutionUpdated = false;
      }
    }

    // Ensure that we are either accepting or rejecting a solution here!
    const int proposedSocForIter = solution_.sumOfCosts;
    const double candidatePressure =
        marketHeuristics_ ? computeSolutionMarketPressure() : 0.0;
    const double candidateWait =
        marketHeuristics_ ? computeSolutionPrecedenceWait() : 0.0;
    bool accepted = false;
    bool guardRejected = false;
    if (marketHeuristics_ && marketAcceptanceGuards_ &&
        !passMarketAcceptanceGuards(candidatePressure, candidateWait)) {
      solution_ = previousSolution_;
      accepted = false;
      guardRejected = true;
      PLOGD << "Rejecting this solution due to market acceptance guards\n";
    } else {
      if (acceptanceCriteria == "SA") {
        accepted = simulatedAnnealing();
      } else if (acceptanceCriteria == "TA") {
        accepted = thresholdAcceptance();
      } else if (acceptanceCriteria == "OBA") {
        accepted = oldBachelorsAcceptance();
      } else if (acceptanceCriteria == "GDA") {
        accepted = greatDelugeAlgorithm();
      } else {
        accepted = false;
        static_assert(true);
      }
    }

    if (alnsHeuristicForIter >= 0 &&
        alnsHeuristicForIter < adaptiveLNS_.numDestroyHeuristics) {
      const double deltaSoc = (double)(previousSocForIter - proposedSocForIter);
      adaptiveLNS_.deltaSocAll[alnsHeuristicForIter] += deltaSoc;
      if (valid) {
        adaptiveLNS_.feasible[alnsHeuristicForIter]++;
      }
      if (!accepted) {
        adaptiveLNS_.rejected[alnsHeuristicForIter]++;
      } else {
        adaptiveLNS_.accepted[alnsHeuristicForIter]++;
        adaptiveLNS_.deltaSocAccepted[alnsHeuristicForIter] += deltaSoc;
        if (previousSolution_.utility < solution_.utility) {
          adaptiveLNS_.downgradedAccepted[alnsHeuristicForIter]++;
        } else {
          adaptiveLNS_.improvedAccepted[alnsHeuristicForIter]++;
        }
        if (feasibleSolutionUpdated) {
          adaptiveLNS_.bestUpdates[alnsHeuristicForIter]++;
        }
      }
    }

    if (!accepted) {
      quality = IterationQuality::none;
      potentialNeighborhood = oldNeighborhood;
      if (guardRejected && marketHeuristics_) {
        // Candidate utility was discarded; keep previous temperature behavior
        // untouched and preserve previous solution.
      }
    } else {
      if (previousSolution_.utility < solution_.utility) {
        // We accepted a potentially worse solution to get out of local minima
        quality = IterationQuality::dowgradedButAccepted;
      } else {
        // We accepted a strictly better solution!
        quality = IterationQuality::improvedSolution;
      }
      previousSolution_ = solution_;
      if (marketHeuristics_) {
        marketBestPressure_ = min(marketBestPressure_, candidatePressure);
        marketBestWait_ = min(marketBestWait_, candidateWait);
      }
    }

    maybeUpdateMarketState(accepted);

    runtime = ((fsec)(Time::now() - plannerStartTime_)).count();
    double costToLog = (feasibleSolutionUpdated) ? incumbentSolution_.sumOfCosts
                                                 : solution_.sumOfCosts;
    iterationStats.emplace_back(runtime, "LNS", instance_.getAgentNum(),
                                instance_.getTasksNum(), costToLog,
                                feasibleSolutionUpdated, quality);
  }

  // printPaths();
  return !incumbentSolution_.agentPaths.empty();
}

void LNS::prepareNextIteration() {
  PLOGI << "Preparing the solution object for the next iteration\n";

  // Find the tasks that are following the earliest conflicting task as their paths need to be invalidated
  vector<vector<int>> successors = instance_.getSuccessors();
  // vector<pair<int, int>> precedenceConstraints =
  //     instance_.getInputPrecedenceConstraints();
  // vector<vector<int>> successors(instance_.getTasksNum());

  // for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
  //   precedenceConstraints.insert(
  //       precedenceConstraints.end(),
  //       solution_.agents[agent].intraPrecedenceConstraints.begin(),
  //       solution_.agents[agent].intraPrecedenceConstraints.end());
  // }

  // for (pair<int, int> precConstraint : precedenceConstraints) {
  //   successors[precConstraint.first].push_back(precConstraint.second);
  // }

  // We need to include all the successors of the original conflicted tasks to ensure that we dont try to find their paths later down the line because otherwise we will face errors since the ancestors of those successor tasks wont have paths.
  // Iterate over a snapshot: we only want successors of the original tasks,
  // and we don't want to repeatedly expand successors of successors as we
  // insert into `removedTasks`.
  const vector<Conflicts> originalRemovedTasks(lnsNeighborhood_.removedTasks.begin(),
                                              lnsNeighborhood_.removedTasks.end());
  for (const Conflicts& conflictTask : originalRemovedTasks) {
    set<int> successorsOfConflictTask =
        reachableSet(conflictTask.task, successors);
    for (int successorOfConflictTask : successorsOfConflictTask) {
      int successorAgent = solution_.taskAgentMap[successorOfConflictTask];
      assert(successorAgent != UNASSIGNED);
      int successorTaskPosition =
          solution_.getLocalTaskIndex(successorAgent, successorOfConflictTask);
      Conflicts successorConflict(successorOfConflictTask, successorAgent,
                                  successorTaskPosition);
      lnsNeighborhood_.removedTasks.insert(successorConflict);
    }
  }

  // If t_id is deleted then t_id + 1 task needs to be fixed
  set<int> tasksToFix, affectedAgents;
  for (Conflicts invalidTask : lnsNeighborhood_.removedTasks) {

    int agent = invalidTask.agent, taskPosition = invalidTask.taskPosition;
    PLOGD << "Invalidating task: " << invalidTask.task << ", Agent: " << agent
          << " at position: " << taskPosition << " with path length = "
          << solution_.agents[agent].taskPaths[taskPosition].size() << endl;

    // If the invalidated task was not the last local task of this agent then t_id + 1 exists
    // If the invalid task was not the last task
    if (invalidTask.task != solution_.getAgentGlobalTasks(agent).back()) {
      int nextTask = UNDEFINED, nextTaskPosition = taskPosition + 1;
      while (nextTask == UNDEFINED &&
             nextTaskPosition <
                 (int)solution_.getAgentGlobalTasks(agent).size()) {
        nextTask = solution_.getAgentGlobalTasks(agent)[nextTaskPosition];
        nextTaskPosition++;
      }
      PLOGD << "Found a potential next task!\n";
      // Next task can still be undefined in the case where that task was removed in the previous iteration of this loop
      if (lnsNeighborhood_.removedTasks.count(
              Conflicts(nextTask, agent, taskPosition + 1)) == 0 &&
          nextTask != UNDEFINED) {
        tasksToFix.insert(nextTask);
        PLOGD << "Next task: " << nextTask << endl;
      }
    }

    affectedAgents.insert(agent);

    // Marking past information about this conflicting task
    solution_.taskAgentMap[invalidTask.task] = UNASSIGNED;
    solution_.agents[agent].clearIntraAgentPrecedenceConstraint(
        invalidTask.task);
    // Needs to happen after clearing precedence constraints
    solution_.agents[agent].taskAssignments[taskPosition] = UNDEFINED;

    lnsNeighborhood_.removedTasksPathSize.insert(
        make_pair(invalidTask.task,
                  solution_.agents[agent].taskPaths[taskPosition].size()));
    solution_.agents[agent].taskPaths[taskPosition] = AgentTaskPath();
  }

  // Marking past information about conflicting tasks
  for (int affAgent : affectedAgents) {

    // For an affected agent there can be multiple conflicting tasks so need to do it this way
    solution_.agents[affAgent].path = AgentTaskPath();
    solution_.agents[affAgent].taskAssignments.erase(
        std::remove_if(solution_.agents[affAgent].taskAssignments.begin(),
                       solution_.agents[affAgent].taskAssignments.end(),
                       [](int task) { return task == UNDEFINED; }),
        solution_.agents[affAgent].taskAssignments.end());
    solution_.agents[affAgent].taskPaths.erase(
        std::remove_if(solution_.agents[affAgent].taskPaths.begin(),
                       solution_.agents[affAgent].taskPaths.end(),
                       [](const Path& p) { return p.empty(); }),
        solution_.agents[affAgent].taskPaths.end());

    vector<int> taskLocations =
        instance_.getTaskLocations(solution_.getAgentGlobalTasks(affAgent));
    solution_.agents[affAgent].pathPlanner->setGoalLocations(taskLocations);
    solution_.agents[affAgent].pathPlanner->computeHeuristics();
  }

  lnsNeighborhood_.patchedTasks = tasksToFix;

  // Find the paths for the tasks whose previous tasks were removed
  for (int task : instance_.getInputPlanningOrder()) {
    if (tasksToFix.count(task) > 0) {

      PLOGD << "Going to find path for next task: " << task << endl;

      int startTime = 0, agent = solution_.getAgentWithTask(task);
      int taskPosition = solution_.getLocalTaskIndex(agent, task);

      if (taskPosition != 0) {
        startTime = solution_.agents[agent].taskPaths[taskPosition - 1].endTime();
      }
      assert(taskPosition <=
             (int)solution_.getAgentGlobalTasks(agent).size() - 1);

      ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
      buildConstraintTable(constraintTable, task);
      AgentTaskPath path = solution_.agents[agent].pathPlanner->findPathSegment(
          constraintTable, startTime, taskPosition, 0);
      // We must be able to find the path for the next task. If not then we cannot move forward!
      assert(!path.empty());
      solution_.agents[agent].taskPaths[taskPosition] = path;

      // Once the path was found fix the begin times for subsequent tasks of the agent
      patchAgentTaskPaths(agent, taskPosition);
    }
  }
}

bool LNS::computeRegret() {
  regretEvalStatsCurrent_.recomputeCalls++;
  regretEvalStatsTotal_.recomputeCalls++;
  lnsNeighborhood_.regretMaxHeap.clear();
  for (Conflicts conflictTask : lnsNeighborhood_.removedTasks) {
    bool enoughSpace = computeRegretForTask(conflictTask.task);
    if (!enoughSpace) {
      return false;
    }
  }
  return true;
}

vector<int> LNS::collectRemainingRemovedTasks() const {
  vector<int> tasks;
  tasks.reserve(lnsNeighborhood_.removedTasks.size());
  for (const Conflicts& conflict : lnsNeighborhood_.removedTasks) {
    tasks.push_back(conflict.task);
  }
  return tasks;
}

vector<int> LNS::computeCurrentTaskEndTimes() const {
  vector<int> endTimes(instance_.getTasksNum(), -1);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int pos = 0; pos < (int)solution_.agents[agent].taskAssignments.size();
         pos++) {
      const int task = solution_.agents[agent].taskAssignments[pos];
      if (pos >= (int)solution_.agents[agent].taskPaths.size()) {
        continue;
      }
      if (solution_.agents[agent].taskPaths[pos].empty()) {
        continue;
      }
      endTimes[task] = solution_.agents[agent].taskPaths[pos].endTime();
    }
  }
  return endTimes;
}

vector<int> LNS::computeCurrentLastTaskPerAgent() const {
  vector<int> lastTasks(instance_.getAgentNum(), UNASSIGNED);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (!solution_.agents[agent].taskAssignments.empty()) {
      lastTasks[agent] = solution_.agents[agent].taskAssignments.back();
    }
  }
  return lastTasks;
}

bool LNS::recomputeRegretsForTasks(const vector<int>& tasks) {
  regretEvalStatsCurrent_.recomputeCalls++;
  regretEvalStatsTotal_.recomputeCalls++;
  incrementalRegretStatsCurrent_.recomputeCalls++;
  incrementalRegretStatsTotal_.recomputeCalls++;
  for (int task : tasks) {
    if (lnsNeighborhood_.commitedTasks.count(task) == 0 ||
        lnsNeighborhood_.commitedTasks[task]) {
      continue;
    }
    regretStamp_[task]++;
    incrementalRegretStatsCurrent_.recomputedTasks++;
    incrementalRegretStatsTotal_.recomputedTasks++;
    const bool enoughSpace = computeRegretForTask(task);
    if (!enoughSpace) {
      return false;
    }
  }
  return true;
}

std::optional<Regret> LNS::popNextValidRegret() {
  while (!lnsNeighborhood_.regretMaxHeap.empty()) {
    Regret r = lnsNeighborhood_.regretMaxHeap.top();
    lnsNeighborhood_.regretMaxHeap.pop();

    if (lnsNeighborhood_.commitedTasks.count(r.task) == 0 ||
        lnsNeighborhood_.commitedTasks[r.task]) {
      incrementalRegretStatsCurrent_.stalePops++;
      incrementalRegretStatsTotal_.stalePops++;
      continue;
    }
    if (!incrementalRegret_ || r.stamp == regretStamp_[r.task]) {
      return r;
    }
    incrementalRegretStatsCurrent_.stalePops++;
    incrementalRegretStatsTotal_.stalePops++;
  }
  return std::nullopt;
}

vector<int> LNS::computeDirtyTasksAfterCommit(const vector<int>& endTimesBefore,
                                             const vector<int>& endTimesAfter,
                                             const vector<int>& lastTaskBefore,
                                             const vector<int>& lastTaskAfter) {
  vector<int> changedTasks;
  changedTasks.reserve(instance_.getTasksNum());
  for (int task = 0; task < instance_.getTasksNum(); task++) {
    if (endTimesAfter[task] < 0) {
      continue;
    }
    if (endTimesAfter[task] != endTimesBefore[task]) {
      changedTasks.push_back(task);
    }
  }
  incrementalRegretStatsCurrent_.changedSum += (int64_t)changedTasks.size();
  incrementalRegretStatsTotal_.changedSum += (int64_t)changedTasks.size();
  incrementalRegretStatsCurrent_.changedMax =
      max(incrementalRegretStatsCurrent_.changedMax,
          (int64_t)changedTasks.size());
  incrementalRegretStatsTotal_.changedMax =
      max(incrementalRegretStatsTotal_.changedMax, (int64_t)changedTasks.size());

  vector<bool> affectedAgents(instance_.getAgentNum(), false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if (lastTaskBefore[agent] != lastTaskAfter[agent]) {
      affectedAgents[agent] = true;
    }
  }
  for (int task : changedTasks) {
    const int agent = solution_.taskAgentMap[task];
    if (agent != UNASSIGNED && agent >= 0 && agent < instance_.getAgentNum()) {
      affectedAgents[agent] = true;
    }
  }

  vector<bool> isDirty(instance_.getTasksNum(), false);

  const vector<vector<int>> successors = instance_.getSuccessors();
  std::vector<int> stack;
  stack.reserve(changedTasks.size());
  for (int t : changedTasks) {
    stack.push_back(t);
  }
  while (!stack.empty()) {
    const int current = stack.back();
    stack.pop_back();
    if (current < 0 || current >= instance_.getTasksNum()) {
      continue;
    }
    if (isDirty[current]) {
      continue;
    }
    isDirty[current] = true;
    for (int succ : successors[current]) {
      if (!isDirty[succ]) {
        stack.push_back(succ);
      }
    }
  }

  if (incrementalRegretMode_ == IncrementalRegretMode::descendants_and_agent) {
    for (const Conflicts& conflict : lnsNeighborhood_.removedTasks) {
      const int task = conflict.task;
      const int bestAgent = regretBestOption_[task].first;
      const int secondAgent = regretSecondBestOption_[task].first;
      if ((bestAgent != UNASSIGNED && affectedAgents[bestAgent]) ||
          (secondAgent != UNASSIGNED && affectedAgents[secondAgent])) {
        isDirty[task] = true;
      }
    }
  }

  vector<int> dirtyTasks;
  dirtyTasks.reserve(lnsNeighborhood_.removedTasks.size());
  for (const Conflicts& conflict : lnsNeighborhood_.removedTasks) {
    if (isDirty[conflict.task]) {
      dirtyTasks.push_back(conflict.task);
    }
  }
  incrementalRegretStatsCurrent_.dirtySum += (int64_t)dirtyTasks.size();
  incrementalRegretStatsTotal_.dirtySum += (int64_t)dirtyTasks.size();
  incrementalRegretStatsCurrent_.dirtyMax =
      max(incrementalRegretStatsCurrent_.dirtyMax, (int64_t)dirtyTasks.size());
  incrementalRegretStatsTotal_.dirtyMax =
      max(incrementalRegretStatsTotal_.dirtyMax, (int64_t)dirtyTasks.size());
  return dirtyTasks;
}

bool LNS::computeRegretForTask(int task) {
  regretEvalStatsCurrent_.tasksEvaluated++;
  regretEvalStatsTotal_.tasksEvaluated++;
  pairing_heap<Utility, compare<Utility::CompareUtilities>> serviceTimes;

  // The task has to start after the earliest time step but needs to finish before the latest time step. However we cannot give any guarantee on the latest timestep so we only work with the earliest timestep
  int earliestTimestep = 0;

  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    precedenceConstraints.insert(
        precedenceConstraints.end(),
        solution_.agents[agent].intraPrecedenceConstraints.begin(),
        solution_.agents[agent].intraPrecedenceConstraints.end());
  }

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }
  set<int> ancestorsOfTask = reachableSet(task, ancestors);
  ancestorsOfTask.erase(task);

  vector<vector<int>> agentTaskAssignments(instance_.getAgentNum());
  vector<vector<AgentTaskPath>> agentTaskPaths(instance_.getAgentNum());
  vector<vector<pair<int, int>>> agentPrecedenceConstraints(
      instance_.getAgentNum());

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    agentTaskAssignments[agent] = solution_.agents[agent].taskAssignments;
    agentTaskPaths[agent] = solution_.agents[agent].taskPaths;
  }

  for (int ancestorTask : ancestorsOfTask) {
    if (lnsNeighborhood_.commitedTasks.count(ancestorTask) > 0 &&
        !lnsNeighborhood_.commitedTasks[ancestorTask]) {
      // This task's path will not exist currently!
      int ancestorTaskAgent = previousSolution_.taskAgentMap[ancestorTask];
      assert(ancestorTask != UNASSIGNED);
      int ancestorTaskLocalIndex =
          previousSolution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
      int ancestorTaskLocalIndexRelativeToSolution = extractOldLocalTaskIndex(
          ancestorTask,
          previousSolution_.agents[ancestorTaskAgent].taskAssignments,
          agentTaskAssignments[ancestorTaskAgent]);
      agentTaskAssignments[ancestorTaskAgent].insert(
          agentTaskAssignments[ancestorTaskAgent].begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          ancestorTask);
      agentTaskPaths[ancestorTaskAgent].insert(
          agentTaskPaths[ancestorTaskAgent].begin() +
              ancestorTaskLocalIndexRelativeToSolution,
          previousSolution_.agents[ancestorTaskAgent]
              .taskPaths[ancestorTaskLocalIndex]);
    }
  }

  precedenceConstraints = instance_.getInputPrecedenceConstraints();
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0; localTask < (int)agentTaskAssignments[agent].size();
         localTask++) {
      if (localTask > 0) {
        agentPrecedenceConstraints[agent].emplace_back(
            agentTaskAssignments[agent][localTask - 1],
            agentTaskAssignments[agent][localTask]);
        agentTaskPaths[agent][localTask].beginTime =
            agentTaskPaths[agent][localTask - 1].endTime();
      } else {
        agentTaskPaths[agent][localTask].beginTime = 0;
      }

      if ((localTask == 0 &&
           agentTaskPaths[agent][localTask].front().location !=
               instance_.getStartLocations()[agent]) ||
          (localTask > 0 &&
           agentTaskPaths[agent][localTask - 1].path.back().location !=
               agentTaskPaths[agent][localTask].path.front().location)) {
        // if (lnsNeighborhood_.patchedTasks.count(
        //         agentTaskAssignments[agent][localTask]) == 0) {

        int startTime = 0;
        if (localTask > 0) {
          startTime = agentTaskPaths[agent][localTask - 1].endTime();
        }
        vector<int> goalLocations =
            instance_.getTaskLocations(agentTaskAssignments[agent]);
        auto localPlanner = createLocalPlanner(agent);
        localPlanner->setGoalLocations(goalLocations);
        localPlanner->computeHeuristics();

        ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
        TaskRegretPacket taskPacket = {agentTaskAssignments[agent][localTask],
                                       agent, localTask, -1};
        // TODO: Possible incomplete precedence constraints here!
        buildConstraintTable(constraintTable, taskPacket,
                             goalLocations[localTask], &agentTaskAssignments,
                             &agentTaskPaths, &precedenceConstraints);
        AgentTaskPath path = localPlanner->findPathSegment(
            constraintTable, startTime, localTask, 0);
        // We must be able to find the path for the next task. If not then we cannot move forward!
        assert(!path.empty());
        agentTaskPaths[agent][localTask] = path;

        // } else {
        //   int taskAgent =
        //       previousSolution_
        //           .taskAgentMap[agentTaskAssignments[agent][localTask]];
        //   assert(taskAgent != UNASSIGNED);
        //   int taskLocalIndex = previousSolution_.getLocalTaskIndex(
        //       taskAgent, agentTaskAssignments[agent][localTask]);
        //   agentTaskPaths[agent][localTask] =
        //       previousSolution_.agents[taskAgent].taskPaths[taskLocalIndex];
        // }
      }
    }
    precedenceConstraints.insert(precedenceConstraints.end(),
                                 agentPrecedenceConstraints[agent].begin(),
                                 agentPrecedenceConstraints[agent].end());
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0; localTask < (int)agentTaskAssignments[agent].size();
         localTask++) {
      if (localTask > 0) {
        assert(agentTaskPaths[agent][localTask - 1].path.back().location ==
               agentTaskPaths[agent][localTask].path.front().location);
      } else {
        assert(agentTaskPaths[agent][localTask].front().location ==
               instance_.getStartLocations()[agent]);
      }
    }
  }

  ancestors.clear();
  ancestors.resize(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }
  ancestorsOfTask = reachableSet(task, ancestors);
  ancestorsOfTask.erase(task);

  for (int ancestorTask : ancestorsOfTask) {
    int ancestorTaskAgent = UNDEFINED;
    if (lnsNeighborhood_.commitedTasks.count(ancestorTask) > 0 &&
        !lnsNeighborhood_.commitedTasks[ancestorTask]) {
      ancestorTaskAgent = previousSolution_.taskAgentMap[ancestorTask];
    } else {
      ancestorTaskAgent = solution_.taskAgentMap[ancestorTask];
    }
    if (ancestorTaskAgent < 0 || ancestorTaskAgent >= instance_.getAgentNum()) {
      PLOGE << "Invalid ancestor agent " << ancestorTaskAgent
            << " for task " << ancestorTask << "\n";
      return false;
    }
    assert(ancestorTaskAgent != UNASSIGNED);
    auto ancestorTaskIt =
        find(agentTaskAssignments[ancestorTaskAgent].begin(),
             agentTaskAssignments[ancestorTaskAgent].end(), ancestorTask);
    if (ancestorTaskIt == agentTaskAssignments[ancestorTaskAgent].end()) {
      PLOGE << "Ancestor task " << ancestorTask
            << " missing from temporary assignment for agent "
            << ancestorTaskAgent << "\n";
      return false;
    }
    int ancestorTaskPosition =
        (int)distance(agentTaskAssignments[ancestorTaskAgent].begin(),
                      ancestorTaskIt);
    if (ancestorTaskPosition >=
            (int)agentTaskPaths[ancestorTaskAgent].size() ||
        agentTaskPaths[ancestorTaskAgent][ancestorTaskPosition].empty()) {
      PLOGE << "Ancestor path missing for task " << ancestorTask
            << " at position " << ancestorTaskPosition << "\n";
      return false;
    }
    earliestTimestep =
        max(earliestTimestep, agentTaskPaths[ancestorTaskAgent][ancestorTaskPosition]
                                   .endTimeChecked() +
                                   1);
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {

    TaskRegretPacket regretPacket = {task, agent, -1, earliestTimestep};
    computeRegretForTaskWithAgent(regretPacket, &agentTaskAssignments,
                                  &agentTaskPaths, &precedenceConstraints,
                                  &serviceTimes);
  }

  if ((int)serviceTimes.size() < 2) {
    // This is the case when we run out of heap i.e there are not enough options left to compute the regret for this task!
    PLOGD << "Ran out of service time options for task " << task
          << " inside the regular compute regret function\n";
    return false;
  }
  Utility bestUtility = serviceTimes.top();
  serviceTimes.pop();
  Utility secondBestUtility = serviceTimes.top();

  regretBestOption_[task] = {bestUtility.agent, bestUtility.taskPosition};
  regretSecondBestOption_[task] = {secondBestUtility.agent,
                                   secondBestUtility.taskPosition};

  double value = 0;
  if (regretType == "absolute") {
    value = secondBestUtility.value - bestUtility.value;
  } else {
    value = (secondBestUtility.value + 1) / (bestUtility.value + 1);
  }
  Regret regret(task, bestUtility.agent, bestUtility.taskPosition,
                bestUtility.pathLength, bestUtility.agentTasksLen,
                (int)serviceTimes.size(), value, regretStamp_[task]);
  lnsNeighborhood_.regretMaxHeap.push(regret);
  return true;
}

void LNS::computeRegretForTaskWithAgent(
    TaskRegretPacket regretPacket, vector<vector<int>>* agentTaskAssignments,
    vector<vector<AgentTaskPath>>* agentTaskPaths,
    vector<pair<int, int>>* precedenceConstraints,
    pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes) {

  regretEvalStatsCurrent_.agentEvaluations++;
  regretEvalStatsTotal_.agentEvaluations++;

  // Compute the first position along the agent's task assignments where we can insert this task
  int firstValidPosition = 0;
  for (int j = (int)(*agentTaskAssignments)[regretPacket.agent].size() - 1;
       j >= 0; j--) {
    int beginTime = (*agentTaskPaths)[regretPacket.agent][j].beginTime,
        endTime = (*agentTaskPaths)[regretPacket.agent][j].endTime();
    if ((regretPacket.earliestTimestep > endTime) ||
        (regretPacket.earliestTimestep <= endTime &&
         regretPacket.earliestTimestep >= beginTime)) {
      firstValidPosition = j + 1;
      break;
    }
  }

  for (int j = firstValidPosition;
       j <= (int)(*agentTaskAssignments)[regretPacket.agent].size(); j++) {

    regretEvalStatsCurrent_.candidateInsertionsTried++;
    regretEvalStatsTotal_.candidateInsertionsTried++;

    if (find_if(begin(lnsNeighborhood_.removedTasks),
                end(lnsNeighborhood_.removedTasks),
                [regretPacket, j](Conflicts conflict) {
                  return regretPacket.task == conflict.task &&
                         regretPacket.agent == conflict.agent &&
                         j == conflict.taskPosition;
                }) != end(lnsNeighborhood_.removedTasks)) {
      // We dont want to compute regret for the same agent, task positions that led to the original conflict!
      continue;
    }
    regretPacket.taskPosition = j;
    // Create a copy of the task assignments, paths and corresponding precedence constraints so that they dont get modified
    vector<vector<AgentTaskPath>> temporaryAgentTaskPaths = *agentTaskPaths;
    vector<vector<int>> temporaryAgentTaskAssignments = *agentTaskAssignments;
    vector<pair<int, int>> temporaryPrecedenceConstraints =
        *precedenceConstraints;
    std::variant<bool, Utility> insertCulmination = insertTask(
        regretPacket, &temporaryAgentTaskPaths, &temporaryAgentTaskAssignments,
        &temporaryPrecedenceConstraints);
    if (std::holds_alternative<Utility>(insertCulmination)) {
      regretEvalStatsCurrent_.candidateInsertionsFeasible++;
      regretEvalStatsTotal_.candidateInsertionsFeasible++;
      serviceTimes->push(std::get<Utility>(insertCulmination));
    }
  }
}

// Need the task paths, assignments and precedence constraints as pointers so that we can reuse this code when commiting as we can make in-place changes to these data-structures
std::variant<bool, Utility> LNS::insertTask(
    TaskRegretPacket regretPacket,
    vector<vector<AgentTaskPath>>* agentTaskPaths,
    vector<vector<int>>* agentTaskAssignments,
    vector<pair<int, int>>* precedenceConstraints) {

  double pathSizeChange = 0;
  int startTime = 0, previousTask = UNDEFINED, nextTask = UNDEFINED;

  // The task paths are all the task paths when we dont commit but if we commit they will be agent specific task paths
  vector<vector<AgentTaskPath>>& agentTaskPathsRef = *agentTaskPaths;
  vector<vector<int>>& agentTaskAssignmentsRef = *agentTaskAssignments;
  vector<pair<int, int>>& precedenceConstraintsRef = *precedenceConstraints;

  int agentTasksSize = (int)agentTaskAssignmentsRef[regretPacket.agent].size();
  double value = INT_MAX;

  // In this case we are inserting a task not at the last position
  if (regretPacket.taskPosition < agentTasksSize) {

    nextTask =
        agentTaskAssignmentsRef[regretPacket.agent][regretPacket.taskPosition];
    pathSizeChange =
        (double)agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition]
            .size();

    agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition] =
        AgentTaskPath();

    agentTaskAssignmentsRef[regretPacket.agent].insert(
        agentTaskAssignmentsRef[regretPacket.agent].begin() +
            regretPacket.taskPosition,
        regretPacket.task);
    agentTaskPathsRef[regretPacket.agent].insert(
        agentTaskPathsRef[regretPacket.agent].begin() +
            regretPacket.taskPosition,
        AgentTaskPath());

    // Invalidate the path of the next task
    // Compute the path size of the next task before you remove it!
    precedenceConstraintsRef.emplace_back(regretPacket.task, nextTask);

    // If we are NOT inserting at the start position then we need to take care of the previous task as well
    if (regretPacket.taskPosition != 0) {
      previousTask = agentTaskAssignmentsRef[regretPacket.agent]
                                            [regretPacket.taskPosition - 1];
      // TODO: Technically the task can start being processed before the previous task ends. This is more conservative but need to check if there are better ways to tackle this.
      startTime =
          agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition - 1]
              .endTime();
      precedenceConstraintsRef.erase(
          std::remove_if(
              precedenceConstraintsRef.begin(), precedenceConstraintsRef.end(),
              [previousTask, nextTask](pair<int, int> x) {
                return x.first == previousTask && x.second == nextTask;
              }),
          precedenceConstraintsRef.end());
      precedenceConstraintsRef.emplace_back(previousTask, regretPacket.task);
    }
  }
  // In this case we are inserting at the very end
  else if (regretPacket.taskPosition == agentTasksSize && agentTasksSize != 0) {

    previousTask = agentTaskAssignmentsRef[regretPacket.agent]
                                          [regretPacket.taskPosition - 1];
    startTime =
        agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition - 1]
            .endTime();

    agentTaskAssignmentsRef[regretPacket.agent].push_back(regretPacket.task);
    precedenceConstraintsRef.emplace_back(previousTask, regretPacket.task);
    agentTaskPathsRef[regretPacket.agent].emplace_back();
  } else if (agentTasksSize == 0) {
    // This is the rare-case when the agent has no tasks assigned to it.
    assert(regretPacket.taskPosition == 0);

    startTime = 0;
    agentTaskAssignmentsRef[regretPacket.agent].push_back(regretPacket.task);
    agentTaskPathsRef[regretPacket.agent].emplace_back();
  }

  if (nextTask >= 0) {
    // The task paths reference does not have ancestor information about next task, so we need to add those in

    vector<vector<int>> ancestors(instance_.getTasksNum());
    for (pair<int, int> precConstraint : precedenceConstraintsRef) {
      ancestors[precConstraint.second].push_back(precConstraint.first);
    }
    set<int> ancestorsOfNextTask = reachableSet(nextTask, ancestors);
    ancestorsOfNextTask.erase(nextTask);

    for (int nextTaskAncestor : ancestorsOfNextTask) {
      if (lnsNeighborhood_.commitedTasks.count(nextTaskAncestor) > 0 &&
          !lnsNeighborhood_.commitedTasks[nextTaskAncestor] &&
          nextTaskAncestor != regretPacket.task) {
        // This task's path will not exist currently!
        int nextTaskAncestorAgent =
            previousSolution_.taskAgentMap[nextTaskAncestor];
        assert(nextTaskAncestor != UNASSIGNED);
        if (std::find(agentTaskAssignmentsRef[nextTaskAncestorAgent].begin(),
                      agentTaskAssignmentsRef[nextTaskAncestorAgent].end(),
                      nextTaskAncestor) ==
            agentTaskAssignmentsRef[nextTaskAncestorAgent].end()) {

          int ancestorTaskLocalIndex = previousSolution_.getLocalTaskIndex(
              nextTaskAncestorAgent, nextTaskAncestor);
          int ancestorTaskLocalIndexRelativeToSolution =
              extractOldLocalTaskIndex(
                  nextTaskAncestor,
                  previousSolution_.agents[nextTaskAncestorAgent]
                      .taskAssignments,
                  agentTaskAssignmentsRef[nextTaskAncestorAgent]);
          agentTaskAssignmentsRef[nextTaskAncestorAgent].insert(
              agentTaskAssignmentsRef[nextTaskAncestorAgent].begin() +
                  ancestorTaskLocalIndexRelativeToSolution,
              nextTaskAncestor);
          agentTaskPathsRef[nextTaskAncestorAgent].insert(
              agentTaskPathsRef[nextTaskAncestorAgent].begin() +
                  ancestorTaskLocalIndexRelativeToSolution,
              previousSolution_.agents[nextTaskAncestorAgent]
                  .taskPaths[ancestorTaskLocalIndex]);
        }
      }
    }

    precedenceConstraintsRef = instance_.getInputPrecedenceConstraints();
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      for (int localTask = 0;
           localTask < (int)agentTaskAssignmentsRef[agent].size();
           localTask++) {

        if (localTask > 0) {
          precedenceConstraintsRef.emplace_back(
              agentTaskAssignmentsRef[agent][localTask - 1],
              agentTaskAssignmentsRef[agent][localTask]);
          agentTaskPathsRef[agent][localTask].beginTime =
              agentTaskPathsRef[agent][localTask - 1].endTime();
        } else {
          agentTaskPathsRef[agent][localTask].beginTime = 0;
        }
        const bool touchesRegretTask =
            (regretPacket.task == agentTaskAssignmentsRef[agent][localTask]) ||
            (localTask > 0 &&
             regretPacket.task == agentTaskAssignmentsRef[agent][localTask - 1]);
        const bool touchesNextTask =
            (nextTask == agentTaskAssignmentsRef[agent][localTask]) ||
            (localTask > 0 &&
             nextTask == agentTaskAssignmentsRef[agent][localTask - 1]);
        if (touchesRegretTask || touchesNextTask) {
          continue;
        }

        if ((localTask == 0 &&
             agentTaskPathsRef[agent][localTask].front().location !=
                 instance_.getStartLocations()[agent]) ||
            (localTask > 0 &&
             agentTaskPathsRef[agent][localTask - 1].path.back().location !=
                 agentTaskPathsRef[agent][localTask].path.front().location)) {
          // if (lnsNeighborhood_.patchedTasks.count(
          //         agentTaskAssignmentsRef[agent][localTask]) == 0) {

          int startTime = 0;
          if (localTask > 0) {
            startTime = agentTaskPathsRef[agent][localTask - 1].endTime();
          }
          vector<int> goalLocations =
              instance_.getTaskLocations(agentTaskAssignmentsRef[agent]);
          auto localPlanner = createLocalPlanner(agent);
          localPlanner->setGoalLocations(goalLocations);
          localPlanner->computeHeuristics();

          ConstraintTable constraintTable(instance_.numOfCols,
                                          instance_.mapSize);
          TaskRegretPacket taskPacket = {
              agentTaskAssignmentsRef[agent][localTask], agent, localTask, -1};
          buildConstraintTable(constraintTable, taskPacket,
                               goalLocations[localTask],
                               &agentTaskAssignmentsRef, &agentTaskPathsRef,
                               &precedenceConstraintsRef);
          AgentTaskPath path = localPlanner->findPathSegment(
              constraintTable, startTime, localTask, 0);
          // We must be able to find the path for the next task. If not then we cannot move forward!
          assert(!path.empty());
          agentTaskPathsRef[agent][localTask] = path;

          // } else {

          //   int taskAgent =
          //       previousSolution_
          //           .taskAgentMap[agentTaskAssignmentsRef[agent][localTask]];
          //   assert(taskAgent != UNASSIGNED);
          //   int taskLocalIndex = previousSolution_.getLocalTaskIndex(
          //       taskAgent, agentTaskAssignmentsRef[agent][localTask]);
          //   agentTaskPathsRef[agent][localTask] =
          //       previousSolution_.agents[taskAgent].taskPaths[taskLocalIndex];
          //   agentTaskPathsRef[agent][localTask].beginTime =
          //       agentTaskPathsRef[agent][localTask - 1].endTime();
          // }
        }
      }
    }

    vector<int> planningOrder;
    bool result =
        topologicalSort(&instance_, &precedenceConstraintsRef, planningOrder);
    if (!result) {
      return false;
    }

    int taskPosition = find(agentTaskAssignmentsRef[regretPacket.agent].begin(),
                            agentTaskAssignmentsRef[regretPacket.agent].end(),
                            regretPacket.task) -
                       agentTaskAssignmentsRef[regretPacket.agent].begin();
    vector<int> goalLocations =
        instance_.getTaskLocations(agentTaskAssignmentsRef[regretPacket.agent]);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    auto localPlanner = createLocalPlanner(regretPacket.agent);
    localPlanner->setGoalLocations(goalLocations);
    localPlanner->computeHeuristics();

    buildConstraintTable(constraintTable, regretPacket,
                         goalLocations[taskPosition], &agentTaskAssignmentsRef,
                         &agentTaskPathsRef, &precedenceConstraintsRef);
    AgentTaskPath path = localPlanner->findPathSegment(
        constraintTable, startTime, taskPosition, 0);
    if (path.empty()) {
      return false;
    }
    agentTaskPathsRef[regretPacket.agent][taskPosition] = path;
    value = path.size();
    startTime = agentTaskPathsRef[regretPacket.agent][taskPosition].endTime();

    // Need to recompute the positions as we might add paths for parent tasks before reaching here!
    int nextTaskPosition =
        find(agentTaskAssignmentsRef[regretPacket.agent].begin(),
             agentTaskAssignmentsRef[regretPacket.agent].end(), nextTask) -
        agentTaskAssignmentsRef[regretPacket.agent].begin();
    TaskRegretPacket nextTaskPacket = {
        nextTask, regretPacket.agent, nextTaskPosition, {}};
    buildConstraintTable(constraintTable, nextTaskPacket,
                         goalLocations[nextTaskPosition],
                         &agentTaskAssignmentsRef, &agentTaskPathsRef,
                         &precedenceConstraintsRef, true);
    AgentTaskPath nextPath = localPlanner->findPathSegment(
        constraintTable, startTime, nextTaskPosition, 0);
    if (nextPath.empty()) {
      return false;
    }
    agentTaskPathsRef[regretPacket.agent][nextTaskPosition] = nextPath;
    value += nextPath.size();
  } else {

    vector<int> planningOrder;
    bool result =
        topologicalSort(&instance_, &precedenceConstraintsRef, planningOrder);
    if (!result) {
      return false;
    }

    vector<int> goalLocations =
        instance_.getTaskLocations(agentTaskAssignmentsRef[regretPacket.agent]);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    auto localPlanner = createLocalPlanner(regretPacket.agent);
    localPlanner->setGoalLocations(goalLocations);
    localPlanner->computeHeuristics();

    buildConstraintTable(constraintTable, regretPacket,
                         goalLocations[regretPacket.taskPosition],
                         &agentTaskAssignmentsRef, &agentTaskPathsRef,
                         &precedenceConstraintsRef);
    AgentTaskPath path = localPlanner->findPathSegment(
        constraintTable, startTime, regretPacket.taskPosition, 0);
    if (path.empty()) {
      return false;
    }
    agentTaskPathsRef[regretPacket.agent][regretPacket.taskPosition] = path;
    value = path.size();
  }

  const auto pathLength = value;
  const double baseDeltaSoc =
      value - (lnsNeighborhood_.removedTasksPathSize.at(regretPacket.task) +
               pathSizeChange);
  double adjustedValue = baseDeltaSoc;
  double deltaExposure = 0.0;
  double deltaWait = 0.0;

  if (marketRepairTieBreak_ || marketRepairBlend_) {
    const int task = regretPacket.task;
    const int taskLocation = instance_.getTaskLocations(task);
    const double oldExposure = computeTaskMarketExposure(task, true);
    const int oldWait = computeTaskPrecedenceWaitInCurrentSolution(task);

    const auto itTaskPos = find(agentTaskAssignmentsRef[regretPacket.agent].begin(),
                                agentTaskAssignmentsRef[regretPacket.agent].end(),
                                task);
    if (itTaskPos != agentTaskAssignmentsRef[regretPacket.agent].end()) {
      const int currentTaskPos =
          (int)(itTaskPos - agentTaskAssignmentsRef[regretPacket.agent].begin());
      if (currentTaskPos >= 0 &&
          currentTaskPos < (int)agentTaskPathsRef[regretPacket.agent].size()) {
        const AgentTaskPath& insertedTaskPath =
            agentTaskPathsRef[regretPacket.agent][currentTaskPos];
        const double newExposure =
            computeMarketExposureFromPath(insertedTaskPath, true);
        const int newWait = computeTaskPrecedenceWaitFromState(
            task, taskLocation, agentTaskAssignmentsRef, agentTaskPathsRef,
            precedenceConstraintsRef);
        deltaExposure = newExposure - oldExposure;
        deltaWait = (double)newWait - (double)oldWait;
      }
    }

    const bool applyBlend = marketRepairBlend_;
    const bool applyTieBreak =
        marketRepairTieBreak_ && std::abs(baseDeltaSoc) <= marketTieBreakEpsSoc_;
    if (applyBlend || applyTieBreak) {
      adjustedValue +=
          marketLambdaPrice_ * deltaExposure + marketLambdaWait_ * deltaWait;
    }
  }

  Utility utility(regretPacket.agent, regretPacket.taskPosition, (int)pathLength,
                  (int)agentTaskAssignmentsRef[regretPacket.agent].size(),
                  adjustedValue, baseDeltaSoc, deltaExposure, deltaWait);
  return utility;
}

void LNS::commitAncestorTaskOf(
    int globalTask, std::optional<pair<bool, int>> committingNextTask) {
  // We are going to commit some ancestor of this global task. We need to ensure that the paths of all the required ancestors of this task are in order before we can commit the global task and any next task that may exist
  // If the boolean flag commitingNextTask is set then it means that the global task was the next task of some other task and we need to ensure that the ancestors of this next task are in order. This additional check is required as the first if condition changes depending on it.
  // The corresponding integer entry would be the global task id of the main task that we wanted to commit.

  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    precedenceConstraints.insert(
        precedenceConstraints.end(),
        solution_.agents[agent].intraPrecedenceConstraints.begin(),
        solution_.agents[agent].intraPrecedenceConstraints.end());
  }

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }
  set<int> ancestorsOfTask = reachableSet(globalTask, ancestors);
  ancestorsOfTask.erase(globalTask);

  for (int ancestorTask : ancestorsOfTask) {
    if (lnsNeighborhood_.commitedTasks.count(ancestorTask) > 0 &&
        !lnsNeighborhood_.commitedTasks[ancestorTask]) {
      if (committingNextTask.has_value() &&
          committingNextTask.value().second == ancestorTask) {
        continue;
      }

      int ancestorTaskAgent = previousSolution_.taskAgentMap[ancestorTask];
      assert(ancestorTaskAgent != UNASSIGNED);

      PLOGD << "Commiting ancestor task " << ancestorTask << " to agent "
            << ancestorTaskAgent << " using previous solution" << endl;

      int ancestorTaskPositionRelativeToSolution = extractOldLocalTaskIndex(
          ancestorTask,
          previousSolution_.agents[ancestorTaskAgent].taskAssignments,
          solution_.agents[ancestorTaskAgent].taskAssignments);
      int ancestorTaskPosition =
          previousSolution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);

      AgentTaskPath ancestorPath = previousSolution_.agents[ancestorTaskAgent]
                                       .taskPaths[ancestorTaskPosition];

      solution_.agents[ancestorTaskAgent].pathPlanner->goalLocations.insert(
          solution_.agents[ancestorTaskAgent]
                  .pathPlanner->goalLocations.begin() +
              ancestorTaskPositionRelativeToSolution,
          instance_.getTaskLocations(ancestorTask));
      solution_.agents[ancestorTaskAgent].pathPlanner->computeHeuristics();

      solution_.agents[ancestorTaskAgent].taskAssignments.insert(
          solution_.agents[ancestorTaskAgent].taskAssignments.begin() +
              ancestorTaskPositionRelativeToSolution,
          ancestorTask);

      solution_.agents[ancestorTaskAgent].insertIntraAgentPrecedenceConstraint(
          ancestorTask, ancestorTaskPositionRelativeToSolution);

      solution_.agents[ancestorTaskAgent].taskPaths.insert(
          solution_.agents[ancestorTaskAgent].taskPaths.begin() +
              ancestorTaskPositionRelativeToSolution,
          ancestorPath);

      solution_.taskAgentMap[ancestorTask] = ancestorTaskAgent;

      // The ancestor of the task that was in the conflict set has now been committed using its old path, hence we need to mark it as resolved now
      markResolved(ancestorTask);
    }
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0;
         localTask < (int)solution_.agents[agent].taskAssignments.size();
         localTask++) {

      if (localTask > 0) {
        solution_.agents[agent].taskPaths[localTask].beginTime =
            solution_.agents[agent].taskPaths[localTask - 1].endTime();
      } else {
        solution_.agents[agent].taskPaths[localTask].beginTime = 0;
      }
      const bool touchesGlobalTask =
          (globalTask == solution_.agents[agent].taskAssignments[localTask]) ||
          (localTask > 0 &&
           globalTask ==
               solution_.agents[agent].taskAssignments[localTask - 1]);
      if (touchesGlobalTask) {
        // We have not found the path for this global task yet so its task path would be empty placeholder!
        continue;
      }

      const bool touchesCommittingNext =
          committingNextTask.has_value() &&
          ((committingNextTask.value().second ==
            solution_.agents[agent].taskAssignments[localTask]) ||
           (localTask > 0 &&
            committingNextTask.value().second ==
                solution_.agents[agent].taskAssignments[localTask - 1]));
      if (touchesCommittingNext) {
        // We wont have the path for the original commiting task here yet
        continue;
      }

      if ((localTask == 0 &&
           solution_.agents[agent].taskPaths[localTask].front().location !=
               solution_.agents[agent].pathPlanner->startLocation) ||
          (localTask > 0 && solution_.agents[agent]
                                    .taskPaths[localTask - 1]
                                    .path.back()
                                    .location != solution_.agents[agent]
                                                     .taskPaths[localTask]
                                                     .path.front()
                                                     .location)) {
        // if (lnsNeighborhood_.patchedTasks.count(
        //         solution_.agents[agent].taskAssignments[localTask]) == 0) {
        // static_assert(true);
        int startTime = 0;
        if (localTask > 0) {
          startTime =
              solution_.agents[agent].taskPaths[localTask - 1].endTime();
        }
        ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);
        buildConstraintTable(
            constraintTable,
            solution_.agents[agent].taskAssignments[localTask]);
        AgentTaskPath path =
            solution_.agents[agent].pathPlanner->findPathSegment(
                constraintTable, startTime, localTask, 0);
        // We must be able to find the path for the next task. If not then we cannot move forward!
        assert(!path.empty());
        solution_.agents[agent].taskPaths[localTask] = path;

        // } else {
        //   int taskAgent =
        //       previousSolution_.taskAgentMap[solution_.agents[agent]
        //                                          .taskAssignments[localTask]];
        //   assert(taskAgent != UNASSIGNED);
        //   int taskLocalIndex = previousSolution_.getLocalTaskIndex(
        //       taskAgent, solution_.agents[agent].taskAssignments[localTask]);
        //   solution_.agents[agent].taskPaths[localTask] =
        //       previousSolution_.agents[taskAgent].taskPaths[taskLocalIndex];
        //   solution_.agents[agent].taskPaths[localTask].beginTime =
        //       solution_.agents[agent].taskPaths[localTask - 1].endTime();
        // }
      }
      lnsNeighborhood_.patchedTasks.erase(
          solution_.agents[agent].taskAssignments[localTask]);
    }
  }

  // Run a validity check!
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int localTask = 0;
         localTask < (int)solution_.agents[agent].taskAssignments.size();
         localTask++) {
      const bool touchesGlobalTask =
          (globalTask == solution_.agents[agent].taskAssignments[localTask]) ||
          (localTask > 0 &&
           globalTask ==
               solution_.agents[agent].taskAssignments[localTask - 1]);
      if (touchesGlobalTask) {
        // We have not found the path for this global task yet so its task path would be empty placeholder!
        continue;
      }

      const bool touchesCommittingNext =
          committingNextTask.has_value() &&
          ((committingNextTask.value().second ==
            solution_.agents[agent].taskAssignments[localTask]) ||
           (localTask > 0 &&
            committingNextTask.value().second ==
                solution_.agents[agent].taskAssignments[localTask - 1]));
      if (touchesCommittingNext) {
        // We wont have the path for the original commiting task here yet
        continue;
      }

      if (localTask > 0) {
        assert(
            solution_.agents[agent]
                .taskPaths[localTask - 1]
                .path.back()
                .location ==
            solution_.agents[agent].taskPaths[localTask].path.front().location);
      } else {
        assert(solution_.agents[agent].taskPaths[localTask].front().location ==
               solution_.agents[agent].pathPlanner->startLocation);
      }
    }
  }
}

void LNS::commitBestRegretTask(Regret bestRegret) {

  PLOGD << "Commiting for task " << bestRegret.task << " to agent "
        << bestRegret.agent << " with regret = " << bestRegret.value << endl;

  TaskRegretPacket bestRegretPacket = {
      bestRegret.task, bestRegret.agent, bestRegret.taskPosition, {}};

  commitAncestorTaskOf(bestRegret.task, std::nullopt);

  // At this point any previously empty paths must be resolved and we can use the insert task function to commit to the actual best regret task
  insertBestRegretTask(bestRegretPacket);
  markResolved(bestRegret.task);
}

void LNS::insertBestRegretTask(TaskRegretPacket bestRegretPacket) {

  int startTime = 0, previousTask = UNDEFINED, nextTask = UNDEFINED;

  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    precedenceConstraints.insert(
        precedenceConstraints.end(),
        solution_.agents[agent].intraPrecedenceConstraints.begin(),
        solution_.agents[agent].intraPrecedenceConstraints.end());
  }

  int agentTasksSize =
      (int)solution_.agents[bestRegretPacket.agent].taskAssignments.size();

  // In this case we are inserting a task not at the last position
  if (bestRegretPacket.taskPosition < agentTasksSize) {

    nextTask = solution_.agents[bestRegretPacket.agent]
                   .taskAssignments[bestRegretPacket.taskPosition];
    solution_.agents[bestRegretPacket.agent]
        .taskPaths[bestRegretPacket.taskPosition] = AgentTaskPath();

    solution_.agents[bestRegretPacket.agent].taskAssignments.insert(
        solution_.agents[bestRegretPacket.agent].taskAssignments.begin() +
            bestRegretPacket.taskPosition,
        bestRegretPacket.task);
    precedenceConstraints.emplace_back(bestRegretPacket.task, nextTask);

    // If we are NOT inserting at the start position then we need to take care of the previous task as well
    if (bestRegretPacket.taskPosition != 0) {
      previousTask = solution_.agents[bestRegretPacket.agent]
                         .taskAssignments[bestRegretPacket.taskPosition - 1];
      // TODO: Technically the task can start being processed before the previous task ends. This is more conservative but need to check if there are better ways to tackle this.
      startTime = solution_.agents[bestRegretPacket.agent]
                      .taskPaths[bestRegretPacket.taskPosition - 1]
                      .endTime();
      precedenceConstraints.erase(
          std::remove_if(
              precedenceConstraints.begin(), precedenceConstraints.end(),
              [previousTask, nextTask](pair<int, int> x) {
                return x.first == previousTask && x.second == nextTask;
              }),
          precedenceConstraints.end());
      precedenceConstraints.emplace_back(previousTask, bestRegretPacket.task);
    }

    // Insert an empty path at that task position
    solution_.agents[bestRegretPacket.agent].taskPaths.insert(
        solution_.agents[bestRegretPacket.agent].taskPaths.begin() +
            bestRegretPacket.taskPosition,
        AgentTaskPath());
  }
  // In this case we are inserting at the very end
  else if (bestRegretPacket.taskPosition == agentTasksSize &&
           agentTasksSize != 0) {

    previousTask = solution_.agents[bestRegretPacket.agent]
                       .taskAssignments[bestRegretPacket.taskPosition - 1];
    startTime = solution_.agents[bestRegretPacket.agent]
                    .taskPaths[bestRegretPacket.taskPosition - 1]
                    .endTime();

    solution_.agents[bestRegretPacket.agent].taskAssignments.push_back(
        bestRegretPacket.task);
    precedenceConstraints.emplace_back(previousTask, bestRegretPacket.task);

    solution_.agents[bestRegretPacket.agent].taskPaths.emplace_back();
  } else if (agentTasksSize == 0) {
    // The rare-case when the agent has no tasks assigned to them
    assert(bestRegretPacket.taskPosition == 0);

    startTime = 0;
    solution_.agents[bestRegretPacket.agent].taskAssignments.push_back(
        bestRegretPacket.task);
    solution_.agents[bestRegretPacket.agent].taskPaths.emplace_back();
  }

  if (nextTask >= 0) {

    commitAncestorTaskOf(
        nextTask, std::make_optional(make_pair(true, bestRegretPacket.task)));

    vector<int> goalLocations = instance_.getTaskLocations(
        solution_.agents[bestRegretPacket.agent].taskAssignments);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    solution_.agents[bestRegretPacket.agent].pathPlanner->setGoalLocations(
        goalLocations);
    solution_.agents[bestRegretPacket.agent].pathPlanner->computeHeuristics();

    // Need to recompute this task position as we may have added tasks in the agent's task queue when trying to account for the next task parents
    int taskPosition =
        find(solution_.agents[bestRegretPacket.agent].taskAssignments.begin(),
             solution_.agents[bestRegretPacket.agent].taskAssignments.end(),
             bestRegretPacket.task) -
        solution_.agents[bestRegretPacket.agent].taskAssignments.begin();

    buildConstraintTable(constraintTable, bestRegretPacket.task);
    AgentTaskPath path =
        solution_.agents[bestRegretPacket.agent].pathPlanner->findPathSegment(
            constraintTable, startTime, taskPosition, 0);
    // We must be able to insert this path for this task as its the best regret path and we did try it before i.e it must have succeeded then
    assert(!path.empty());
    solution_.agents[bestRegretPacket.agent].taskPaths[taskPosition] = path;
    solution_.agents[bestRegretPacket.agent]
        .insertIntraAgentPrecedenceConstraint(bestRegretPacket.task,
                                              taskPosition);
    solution_.taskAgentMap[bestRegretPacket.task] = bestRegretPacket.agent;

    buildConstraintTable(constraintTable, nextTask);
    int nextTaskPosition =
        find(solution_.agents[bestRegretPacket.agent].taskAssignments.begin(),
             solution_.agents[bestRegretPacket.agent].taskAssignments.end(),
             nextTask) -
        solution_.agents[bestRegretPacket.agent].taskAssignments.begin();
    assert(nextTaskPosition - 1 >= 0);
    startTime = solution_.agents[bestRegretPacket.agent]
                    .taskPaths[nextTaskPosition - 1]
                    .endTime();
    AgentTaskPath nextPath =
        solution_.agents[bestRegretPacket.agent].pathPlanner->findPathSegment(
            constraintTable, startTime, nextTaskPosition, 0);
    // We must be able to insert this path for this task as its the best regret path and we did try it before i.e it must have succeeded then
    assert(!nextPath.empty());
    solution_.agents[bestRegretPacket.agent].taskPaths[nextTaskPosition] =
        nextPath;
  } else {

    vector<int> goalLocations = instance_.getTaskLocations(
        solution_.agents[bestRegretPacket.agent].taskAssignments);
    ConstraintTable constraintTable(instance_.numOfCols, instance_.mapSize);

    solution_.agents[bestRegretPacket.agent].pathPlanner->setGoalLocations(
        goalLocations);
    solution_.agents[bestRegretPacket.agent].pathPlanner->computeHeuristics();

    buildConstraintTable(constraintTable, bestRegretPacket.task);
    AgentTaskPath path =
        solution_.agents[bestRegretPacket.agent].pathPlanner->findPathSegment(
            constraintTable, startTime, bestRegretPacket.taskPosition, 0);
    // We must be able to insert this path for this task as its the best regret path and we did try it before i.e it must have succeeded then
    assert(!path.empty());
    solution_.agents[bestRegretPacket.agent]
        .taskPaths[bestRegretPacket.taskPosition] = path;
    solution_.agents[bestRegretPacket.agent]
        .insertIntraAgentPrecedenceConstraint(bestRegretPacket.task,
                                              bestRegretPacket.taskPosition);
    solution_.taskAgentMap[bestRegretPacket.task] = bestRegretPacket.agent;
  }

  patchAgentTaskPaths(bestRegretPacket.agent, 0);
}

void LNS::buildConstraintTable(ConstraintTable& constraintTable,
                               TaskRegretPacket taskPacket, int taskLocation,
                               vector<vector<int>>* agentTaskAssignments,
                               vector<vector<AgentTaskPath>>* agentTaskPaths,
                               vector<pair<int, int>>* precedenceConstraints,
                               bool findingNextTask) {

  vector<vector<AgentTaskPath>>& agentTaskPathsRef = *agentTaskPaths;
  vector<vector<int>>& agentTaskAssignmentsRef = *agentTaskAssignments;
  vector<pair<int, int>>& precedenceConstraintsRef = *precedenceConstraints;

  constraintTable.goalLocation = taskLocation;

  vector<vector<int>> ancestors = instance_.getAncestors();
  // TODO: We used input precedence constraints here but to me it seems like the input precedence constraints should be augmented by the precedence constraints of the agent we are considering here as well!
  for (pair<int, int> precedenceConstraint : precedenceConstraintsRef) {
    ancestors[precedenceConstraint.second].push_back(
        precedenceConstraint.first);
  }

  set<int> ancestorsOfTask = reachableSet(taskPacket.task, ancestors);
  ancestorsOfTask.erase(taskPacket.task);

  // Loop through the last task map to gather the actual final tasks of the agents
  vector<bool> finalTasks(instance_.getTasksNum(), false);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    if ((int)agentTaskAssignmentsRef[agent].size() > 0) {
      int lastTask = agentTaskAssignmentsRef[agent].back();
      finalTasks[lastTask] = true;
    }
  }

  // Add the paths of the prior tasks to the constraint table with information about whether they were their agent's final tasks or not
  for (int ancestorTask : ancestorsOfTask) {

    int ancestorTaskAgent = UNDEFINED;
    if (findingNextTask &&
        ancestorTask == agentTaskAssignmentsRef[taskPacket.agent]
                                               [taskPacket.taskPosition - 1]) {
      ancestorTaskAgent = taskPacket.agent;
    } else if (lnsNeighborhood_.commitedTasks.count(ancestorTask) > 0 &&
               !lnsNeighborhood_.commitedTasks[ancestorTask]) {
      ancestorTaskAgent = previousSolution_.taskAgentMap[ancestorTask];
      assert(ancestorTaskAgent != UNASSIGNED);
    } else {
      ancestorTaskAgent = solution_.taskAgentMap[ancestorTask];
      assert(ancestorTaskAgent != UNASSIGNED);
    }

    int ancestorTaskLocalIndex =
        std::find(begin(agentTaskAssignmentsRef[ancestorTaskAgent]),
                  end(agentTaskAssignmentsRef[ancestorTaskAgent]),
                  ancestorTask) -
        begin(agentTaskAssignmentsRef[ancestorTaskAgent]);
    assert(
        !agentTaskPathsRef[ancestorTaskAgent][ancestorTaskLocalIndex].empty());
    bool waitAtGoal = finalTasks[ancestorTask];
    constraintTable.addPath(
        agentTaskPathsRef[ancestorTaskAgent][ancestorTaskLocalIndex],
        waitAtGoal);

    constraintTable.lengthMin = max(
        constraintTable.lengthMin,
        agentTaskPathsRef[ancestorTaskAgent][ancestorTaskLocalIndex].endTime() +
            1);
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
}

void LNS::buildConstraintTable(ConstraintTable& constraintTable, int task) {

  constraintTable.goalLocation = instance_.getTaskLocations(task);

  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();

  if (iterationStats.size() > 1) {
    for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
      precedenceConstraints.insert(
          precedenceConstraints.end(),
          solution_.agents[agent].intraPrecedenceConstraints.begin(),
          solution_.agents[agent].intraPrecedenceConstraints.end());
    }
  }

  vector<vector<int>> ancestors(instance_.getTasksNum());
  for (pair<int, int> precConstraint : precedenceConstraints) {
    if (precConstraint.first < 0 || precConstraint.second < 0 ||
        precConstraint.first >= instance_.getTasksNum() ||
        precConstraint.second >= instance_.getTasksNum()) {
      continue;
    }
    ancestors[precConstraint.second].push_back(precConstraint.first);
  }

  set<int> ancestorsOfTask = reachableSet(task, ancestors);
  ancestorsOfTask.erase(task);

  for (int ancestorTask : ancestorsOfTask) {
    int ancestorTaskAgent = solution_.taskAgentMap[ancestorTask];
    assert(ancestorTaskAgent != UNASSIGNED);
    int ancestorTaskPosition =
        solution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
    bool waitAtGoal =
        ancestorTask ==
        (int)solution_.agents[ancestorTaskAgent].taskAssignments.back();
    constraintTable.addPath(
        solution_.agents[ancestorTaskAgent].taskPaths[ancestorTaskPosition],
        waitAtGoal);
  }

  for (int ancestorTask : ancestorsOfTask) {
    int ancestorTaskAgent = solution_.getAgentWithTask(ancestorTask);
    int ancestorTaskPosition =
        solution_.getLocalTaskIndex(ancestorTaskAgent, ancestorTask);
    assert(!solution_.agents[ancestorTaskAgent]
                .taskPaths[ancestorTaskPosition]
                .empty());
    constraintTable.lengthMin =
        max(constraintTable.lengthMin, solution_.agents[ancestorTaskAgent]
                                               .taskPaths[ancestorTaskPosition]
                                               .endTime() +
                                           1);
  }

  constraintTable.latestTimestep =
      max(constraintTable.latestTimestep, constraintTable.lengthMin);
}

int LNS::extractOldLocalTaskIndex(int task, vector<int> oldTaskQueue,
                                  vector<int> newTaskQueue) {
  int localTaskPositionOffset = 0;
  // We need to compute the offset as we can invalidate multiple tasks associated with an agent. This means that simply querying the previous solution agent's task index is not enough as the it would be more than the actual task position value for the current solution
  for (int localTask : oldTaskQueue) {
    // We dont need to bother for the tasks that come after the current one since we are considering them in planning order
    if (localTask == task) {
      break;
    }

    // This local task should not be in the new task queue otherwise we have accounted for it before! If not then the offset should only be incremented if it was in conflict set
    if (find_if(begin(lnsNeighborhood_.immutableRemovedTasks),
                end(lnsNeighborhood_.immutableRemovedTasks),
                [localTask](Conflicts conflict) {
                  return conflict.task == localTask;
                }) != end(lnsNeighborhood_.immutableRemovedTasks) &&
        find_if(begin(newTaskQueue), end(newTaskQueue), [localTask](int task) {
          return task == localTask;
        }) == end(newTaskQueue)) {
      localTaskPositionOffset++;
    }
  }
  int index = 0;
  for (; index < (int)oldTaskQueue.size(); index++) {
    if (oldTaskQueue[index] == task) {
      break;
    }
  }
  assert(index < (int)oldTaskQueue.size());
  return index - localTaskPositionOffset;
}

set<int> LNS::reachableSet(int source, vector<vector<int>> edgeList) {
  set<int> result;
  stack<int> q({source});
  while (!q.empty()) {
    int current = q.top();
    q.pop();
    if (result.count(current) > 0) {
      continue;
    }
    result.insert(current);
    for (int sink : edgeList[current]) {
      if (result.count(sink) == 0) {
        q.push(sink);
      }
    }
  }
  return result;
}

void LNS::markResolved(int globalTask) {
  auto it = std::find_if(begin(lnsNeighborhood_.removedTasks),
                         end(lnsNeighborhood_.removedTasks),
                         [globalTask](Conflicts conflicts) {
                           return conflicts.task == globalTask;
                         });
  while (it != end(lnsNeighborhood_.removedTasks)) {
    it = lnsNeighborhood_.removedTasks.erase(it);
    it = std::find_if(it, end(lnsNeighborhood_.removedTasks),
                      [globalTask](Conflicts conflicts) {
                        return conflicts.task == globalTask;
                      });
  }
  lnsNeighborhood_.removedTasksPathSize.erase(globalTask);
  lnsNeighborhood_.commitedTasks[globalTask] = true;
}

void LNS::patchAgentTaskPaths(int agent, int taskPosition) {
  if (taskPosition == 0) {
    // If we are the first task then ensure that we begin at 0
    solution_.agents[agent].taskPaths[taskPosition].beginTime = 0;
  }
  for (int k = taskPosition + 1;
       k < (int)solution_.agents[agent].taskAssignments.size(); k++) {
    solution_.agents[agent].taskPaths[k].beginTime =
        solution_.agents[agent].taskPaths[k - 1].endTime();
  }
}

bool LNS::validateSolution(set<Conflicts>* conflictedTasks) {

  bool result = true;

  vector<pair<int, int>> precedenceConstraints =
      instance_.getInputPrecedenceConstraints();
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    precedenceConstraints.insert(
        precedenceConstraints.end(),
        solution_.agents[agent].intraPrecedenceConstraints.begin(),
        solution_.agents[agent].intraPrecedenceConstraints.end());
  }

  for (int task = 0; task < instance_.getTasksNum(); task++) {
    int taskAgent = solution_.taskAgentMap[task];
    assert(taskAgent != UNASSIGNED);
    int taskPosition = solution_.getLocalTaskIndex(taskAgent, task);
    if (solution_.agents[taskAgent].taskPaths[taskPosition].empty()) {
      result = false;
      return result;
    }
  }

  // Check that the precedence constraints are not violated
  for (pair<int, int> precedenceConstraint : precedenceConstraints) {

    int agentA = solution_.getAgentWithTask(precedenceConstraint.first),
        agentB = solution_.getAgentWithTask(precedenceConstraint.second);
    int taskPositionA =
            solution_.getLocalTaskIndex(agentA, precedenceConstraint.first),
        taskPositionB =
            solution_.getLocalTaskIndex(agentB, precedenceConstraint.second);

    if (solution_.agents[agentA].path.timeStamps[taskPositionA] >=
        solution_.agents[agentB].path.timeStamps[taskPositionB]) {
      PLOGE << "Temporal conflict between agent " << agentA << " doing task "
            << precedenceConstraint.first << " and agent " << agentB
            << " doing task " << precedenceConstraint.second << endl;
      result = false;
      if (conflictedTasks == nullptr) {
        return false;
      }
      Conflicts conflictA(precedenceConstraint.first, agentA, taskPositionA);
      Conflicts conflictB(precedenceConstraint.second, agentB, taskPositionB);
      conflictedTasks->insert(conflictA);
      conflictedTasks->insert(conflictB);
    }
  }

  for (int agentI = 0; agentI < instance_.getAgentNum(); agentI++) {
    for (int agentJ = 0; agentJ < instance_.getAgentNum(); agentJ++) {
      if (agentI == agentJ) {
        continue;
      }
      if (solution_.agents[agentI].taskAssignments.empty() ||
          solution_.agents[agentJ].taskAssignments.empty()) {
        continue;
      }
      size_t minPathLength = solution_.agents[agentI].path.size() <
                                     solution_.agents[agentJ].path.size()
                                 ? solution_.agents[agentI].path.size()
                                 : solution_.agents[agentJ].path.size();
      for (int timestep = 0; timestep < (int)minPathLength; timestep++) {
        int locationAgentI =
            solution_.agents[agentI].path.at(timestep).location;
        int locationAgentJ =
            solution_.agents[agentJ].path.at(timestep).location;

        // Check that any two agents are not at the same location at the same timestep
        if (locationAgentI == locationAgentJ) {
          pair<int, int> coord = instance_.getCoordinate(locationAgentI);
          PLOGE << "Agents " << agentI << " and " << agentJ
                << " collide with each other at (" << coord.first << ", "
                << coord.second << ") at timestep " << timestep << endl;
          result = false;
          if (conflictedTasks == nullptr) {
            return false;
          }
          for (int taskIdx = 0;
               taskIdx < (int)solution_.getAgentGlobalTasks(agentI).size();
               taskIdx++) {
            if (solution_.agents[agentI].path.timeStamps[taskIdx] > timestep) {
              Conflicts conflict(solution_.getAgentGlobalTasks(agentI, taskIdx),
                                 agentI, taskIdx);
              conflictedTasks->insert(conflict);
              break;
            }
          }
          for (int taskIdx = 0;
               taskIdx < (int)solution_.getAgentGlobalTasks(agentJ).size();
               taskIdx++) {
            if (solution_.agents[agentJ].path.timeStamps[taskIdx] > timestep) {
              Conflicts conflict(solution_.getAgentGlobalTasks(agentJ, taskIdx),
                                 agentJ, taskIdx);
              conflictedTasks->insert(conflict);
              break;
            }
          }
        }
        // Check that any two agents are not following the same edge in the opposite direction at the same timestep
        else if (timestep < (int)minPathLength - 1 &&
                 locationAgentI ==
                     solution_.agents[agentJ].path.at(timestep + 1).location &&
                 locationAgentJ ==
                     solution_.agents[agentI].path.at(timestep + 1).location) {
          pair<int, int> coordI = instance_.getCoordinate(locationAgentI),
                         coordJ = instance_.getCoordinate(locationAgentJ);
          PLOGE << "Agents " << agentI << " and " << agentJ
                << " collide with each other at (" << coordI.first << ", "
                << coordI.second << ") --> (" << coordJ.first << ", "
                << coordJ.second << ") at timestep " << timestep << endl;
          result = false;
          if (conflictedTasks == nullptr) {
            return false;
          }
          for (int taskIdx = 0;
               taskIdx < (int)solution_.getAgentGlobalTasks(agentI).size();
               taskIdx++) {
            if (solution_.agents[agentI].path.timeStamps[taskIdx] > timestep) {
              Conflicts conflict(solution_.getAgentGlobalTasks(agentI, taskIdx),
                                 agentI, taskIdx);
              conflictedTasks->insert(conflict);
              break;
            }
          }
          for (int taskIdx = 0;
               taskIdx < (int)solution_.getAgentGlobalTasks(agentJ).size();
               taskIdx++) {
            if (solution_.agents[agentJ].path.timeStamps[taskIdx] > timestep) {
              Conflicts conflict(solution_.getAgentGlobalTasks(agentJ, taskIdx),
                                 agentJ, taskIdx);
              conflictedTasks->insert(conflict);
              break;
            }
          }
        }
      }

      // Check that any two agents are not at the same location at the same timestep where one agent might be waiting already
      if (solution_.agents[agentI].path.size() !=
          solution_.agents[agentJ].path.size()) {
        int smallerPathAgent = solution_.agents[agentI].path.size() <
                                       solution_.agents[agentJ].path.size()
                                   ? agentI
                                   : agentJ;
        int largerPathAgent = solution_.agents[agentI].path.size() <
                                      solution_.agents[agentJ].path.size()
                                  ? agentJ
                                  : agentI;
        int lastLocationOfSmallerPathAgent =
            solution_.agents[smallerPathAgent].path.back().location;
        for (int timestep = (int)minPathLength;
             timestep < (int)solution_.agents[largerPathAgent].path.size();
             timestep++) {
          int locationOfLargerPathAgent =
              solution_.agents[largerPathAgent].path.at(timestep).location;
          if (lastLocationOfSmallerPathAgent == locationOfLargerPathAgent) {
            pair<int, int> coord =
                instance_.getCoordinate(locationOfLargerPathAgent);
            PLOGE << "Agents " << agentI << " and " << agentJ
                  << " collide with each other at (" << coord.first << ", "
                  << coord.second << ") at timestep " << timestep << endl;
            result = false;
            if (conflictedTasks == nullptr) {
              return false;
            }
            if (solution_.agents[agentI].path.timeStamps
                    [(int)solution_.getAgentGlobalTasks(agentI).size() - 1] <
                timestep) {
              int taskIdx = solution_.getAgentGlobalTasks(agentI).size() - 1;
              Conflicts conflict(solution_.getAgentGlobalTasks(agentI, taskIdx),
                                 agentI, taskIdx);
              conflictedTasks->insert(conflict);
            } else {
              for (int taskIdx = 0;
                   taskIdx < (int)solution_.getAgentGlobalTasks(agentI).size();
                   taskIdx++) {
                if (solution_.agents[agentI].path.timeStamps[taskIdx] >
                    timestep) {
                  Conflicts conflict(
                      solution_.getAgentGlobalTasks(agentI, taskIdx), agentI,
                      taskIdx);
                  conflictedTasks->insert(conflict);
                  break;
                }
              }
            }
            if (solution_.agents[agentJ].path.timeStamps
                    [(int)solution_.getAgentGlobalTasks(agentJ).size() - 1] <
                timestep) {
              int taskIdx = solution_.getAgentGlobalTasks(agentJ).size() - 1;
              Conflicts conflict(solution_.getAgentGlobalTasks(agentJ, taskIdx),
                                 agentJ, taskIdx);
              conflictedTasks->insert(conflict);
            } else {
              for (int taskIdx = 0;
                   taskIdx < (int)solution_.getAgentGlobalTasks(agentJ).size();
                   taskIdx++) {
                if (solution_.agents[agentJ].path.timeStamps[taskIdx] >
                    timestep) {
                  Conflicts conflict(
                      solution_.getAgentGlobalTasks(agentJ, taskIdx), agentJ,
                      taskIdx);
                  conflictedTasks->insert(conflict);
                  break;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

void LNS::printPaths() const {
  for (int i = 0; i < instance_.getAgentNum(); i++) {
    const int agentPathSize = solution_.agents[i].path.endTimeOrZero();
    cout << "Agent " << i << " (cost = " << agentPathSize << "): ";
    cout << "\n\tPaths:\n\t";
    for (int t = 0; t < (int)solution_.agents[i].path.size(); t++) {
      pair<int, int> coord =
          instance_.getCoordinate(solution_.agents[i].path.at(t).location);
      cout << "(" << coord.first << ", " << coord.second << ")@" << t;
      if (solution_.agents[i].path.at(t).isGoal) {
        cout << "*";
      }
      if (i != (int)solution_.agents[i].path.size() - 1) {
        cout << " -> ";
      }
    }
    cout << endl;
    cout << "\tTimestamps:\n\t";
    for (int j = 0; j < (int)solution_.getAgentGlobalTasks(i).size(); j++) {
      pair<int, int> goalCoord = instance_.getCoordinate(
          solution_.agents[i].pathPlanner->goalLocations[j]);
      cout << "(" << goalCoord.first << ", " << goalCoord.second << ")@"
           << solution_.agents[i].path.timeStamps[j];
      if (j != (int)solution_.getAgentGlobalTasks(i).size() - 1) {
        cout << " -> ";
      }
    }
    cout << endl;
    cout << "\tTasks:\n\t";
    for (int j = 0; j < (int)solution_.getAgentGlobalTasks(i).size(); j++) {
      cout << solution_.getAgentGlobalTasks(i)[j];
      if (j != (int)solution_.getAgentGlobalTasks(i).size() - 1) {
        cout << " -> ";
      }
    }
    cout << endl;
  }
}

Solution& Solution::operator=(const Solution& other) {
  if (this == &other) {
    return *this;
  }
  this->numOfTasks = other.numOfTasks;
  this->numOfAgents = other.numOfAgents;
  this->sumOfCosts = other.sumOfCosts;

  this->agents = other.agents;
  this->taskAgentMap = other.taskAgentMap;

  return *this;
}
