#include "lns.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "common.hpp"
#include "lns_internal_helpers.hpp"

namespace {
inline int findTaskPositionInAssignments(const std::vector<int>& assignments,
                                         int task) {
  for (int i = 0; i < static_cast<int>(assignments.size()); i++) {
    if (assignments[i] == task) {
      return i;
    }
  }
  return -1;
}
}  // namespace

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
  const vector<int>& taskPosByTask = getCurrentTaskPositionIndexByTask();
  computeTaskScheduleMetricsFromIndex(taskPosByTask, perTask, blockedWaitSum);
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

  auto weightedResourcePrice =
      [](const unordered_map<uint64_t, double>& prices,
         const unordered_map<uint64_t, double>& excessHat,
         uint64_t key) -> double {
    const auto pIt = prices.find(key);
    if (pIt == prices.end()) {
      return 0.0;
    }
    const auto eIt = excessHat.find(key);
    if (eIt == excessHat.end()) {
      return 0.0;
    }
    const double excessWeight = max(0.0, eIt->second);
    if (excessWeight <= 0.0) {
      return 0.0;
    }
    return pIt->second * excessWeight;
  };

  double totalExposure = 0.0;
  int activeResourceCount = 0;
  for (int i = 0; i < (int)taskPath.size(); i++) {
    const int timestep = taskPath.beginTime + i;
    const int bucket = marketTimeBucket(timestep);
    const int location = taskPath[i].location;
    const uint64_t vKey = makeMarketVertexKey(location, bucket);
    const double vertexContribution =
        weightedResourcePrice(market_.vertexPrices, market_.vertexExcessHat,
                              vKey);
    if (vertexContribution > 0.0) {
      totalExposure += vertexContribution;
      activeResourceCount++;
    }

    if (i > 0) {
      const int prevLocation = taskPath[i - 1].location;
      const uint64_t eKey = makeMarketEdgeKey(prevLocation, location, bucket);
      const double edgeContribution =
          weightedResourcePrice(market_.edgePrices, market_.edgeExcessHat, eKey);
      if (edgeContribution > 0.0) {
        totalExposure += edgeContribution;
        activeResourceCount++;
      }
    }
  }
  if (!normalized) {
    return totalExposure;
  }
  // Normalize only over resources with positive congestion-weighted
  // contribution, so long uncongested paths do not dominate the score.
  return totalExposure / max(1, activeResourceCount);
}

double LNS::computeMarketMarginalReliefFromPath(
    const AgentTaskPath& taskPath,
    const unordered_map<uint64_t, int>& vertexDemand,
    const unordered_map<uint64_t, int>& edgeDemand, bool normalized) const {
  if (taskPath.empty()) {
    return 0.0;
  }

  unordered_map<uint64_t, int> vertexUsage;
  unordered_map<uint64_t, int> edgeUsage;
  vertexUsage.reserve(taskPath.size());
  edgeUsage.reserve(taskPath.size());

  for (int i = 0; i < (int)taskPath.size(); i++) {
    const int timestep = taskPath.beginTime + i;
    const int bucket = marketTimeBucket(timestep);
    const int location = taskPath[i].location;
    vertexUsage[makeMarketVertexKey(location, bucket)]++;

    if (i > 0) {
      const int prevLocation = taskPath[i - 1].location;
      edgeUsage[makeMarketEdgeKey(prevLocation, location, bucket)]++;
    }
  }

  auto computeCategoryRelief =
      [](const unordered_map<uint64_t, int>& usage,
         const unordered_map<uint64_t, int>& demand,
         const unordered_map<uint64_t, double>& prices,
         const unordered_map<uint64_t, double>& excessHat,
         int capacity, double* totalRelief,
         int* activeResources) {
        const int safeCapacity = max(1, capacity);
        for (const auto& kv : usage) {
          const uint64_t key = kv.first;
          const int usageCount = kv.second;
          if (usageCount <= 0) {
            continue;
          }

          const auto pIt = prices.find(key);
          if (pIt == prices.end()) {
            continue;
          }
          const auto eIt = excessHat.find(key);
          if (eIt == excessHat.end()) {
            continue;
          }
          const double excessWeight = max(0.0, eIt->second);
          if (excessWeight <= 0.0) {
            continue;
          }
          const double unitPressure = pIt->second * excessWeight;
          if (unitPressure <= 0.0) {
            continue;
          }

          const auto dIt = demand.find(key);
          const int currentDemand = (dIt == demand.end()) ? 0 : dIt->second;
          if (currentDemand <= safeCapacity) {
            continue;
          }

          const int oldExcess = max(0, currentDemand - safeCapacity);
          const int newExcess =
              max(0, currentDemand - usageCount - safeCapacity);
          const int relievedExcess = oldExcess - newExcess;
          if (relievedExcess <= 0) {
            continue;
          }

          *totalRelief += unitPressure * relievedExcess;
          (*activeResources)++;
        }
      };

  double totalRelief = 0.0;
  int activeResources = 0;
  computeCategoryRelief(vertexUsage, vertexDemand, market_.vertexPrices,
                        market_.vertexExcessHat, market_.vertexBucketCapacity,
                        &totalRelief, &activeResources);
  computeCategoryRelief(edgeUsage, edgeDemand, market_.edgePrices,
                        market_.edgeExcessHat, market_.edgeBucketCapacity,
                        &totalRelief, &activeResources);

  if (!normalized) {
    return totalRelief;
  }
  return totalRelief / max(1, activeResources);
}

int LNS::computeTaskPrecedenceWaitFromState(
    int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
    const vector<vector<AgentTaskPath>>& agentTaskPaths) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }

  const auto& baseAncestors = instance_.getAncestorsRef();
  vector<int> relevantTasks;
  relevantTasks.reserve(8);
  relevantTasks.push_back(task);
  if (task >= 0 && task < (int)baseAncestors.size()) {
    for (int pred : baseAncestors[task]) {
      if (pred >= 0 && pred < instance_.getTasksNum()) {
        relevantTasks.push_back(pred);
      }
    }
  }
  std::sort(relevantTasks.begin(), relevantTasks.end());
  relevantTasks.erase(std::unique(relevantTasks.begin(), relevantTasks.end()),
                      relevantTasks.end());

  struct TaskStatePos {
    int agent = UNASSIGNED;
    int pos = -1;
  };
  unordered_map<int, TaskStatePos> relevantLookup;
  relevantLookup.reserve(relevantTasks.size() * 2 + 1);
  for (int currentAgent = 0; currentAgent < (int)agentTaskAssignments.size();
       currentAgent++) {
    const auto& assignments = agentTaskAssignments[currentAgent];
    for (int currentPos = 0; currentPos < (int)assignments.size(); currentPos++) {
      const int assignedTask = assignments[currentPos];
      if (!std::binary_search(relevantTasks.begin(), relevantTasks.end(),
                              assignedTask)) {
        continue;
      }
      relevantLookup[assignedTask] = {currentAgent, currentPos};
    }
  }

  const auto taskIt = relevantLookup.find(task);
  if (taskIt == relevantLookup.end()) {
    return 0;
  }
  const int taskAgent = taskIt->second.agent;
  const int taskPos = taskIt->second.pos;
  if (taskAgent < 0 || taskAgent >= (int)agentTaskPaths.size() || taskPos < 0 ||
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
    if (pred < 0 || pred >= taskCount || seenPredecessor[pred]) {
      return;
    }
    seenPredecessor[pred] = 1;
    const auto predIt = relevantLookup.find(pred);
    if (predIt == relevantLookup.end()) {
      return;
    }
    const int predAgent = predIt->second.agent;
    const int predPos = predIt->second.pos;
    if (predAgent != UNASSIGNED && predAgent >= 0 &&
        predAgent < (int)agentTaskPaths.size() && predPos >= 0 &&
        predPos < (int)agentTaskPaths[predAgent].size() &&
        !agentTaskPaths[predAgent][predPos].empty()) {
      release = max(release, agentTaskPaths[predAgent][predPos].endTime());
    }
  };

  if (task >= 0 && task < (int)baseAncestors.size()) {
    for (int pred : baseAncestors[task]) {
      consumePredecessor(pred);
    }
  }

  if (taskPos > 0 && taskAgent >= 0 &&
      taskAgent < (int)agentTaskAssignments.size() &&
      taskPos - 1 < (int)agentTaskAssignments[taskAgent].size()) {
    consumePredecessor(agentTaskAssignments[taskAgent][taskPos - 1]);
  }

  return max(0, release - arrive);
}

void LNS::buildMarketDemandFromCurrentOccupancy(
    unordered_map<uint64_t, int>& vertexDemand,
    unordered_map<uint64_t, int>& edgeDemand) const {
  vertexDemand.clear();
  edgeDemand.clear();

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    const int horizon = getAgentOccupancyHorizon(agent, true);
    if (horizon <= 0) {
      continue;
    }

    int prevLocation = UNDEFINED;
    for (int timestep = 0; timestep < horizon; timestep++) {
      const int location = getAgentLocationAt(agent, timestep, true);
      if (location == UNDEFINED) {
        prevLocation = UNDEFINED;
        continue;
      }
      const int bucket = marketTimeBucket(timestep);
      vertexDemand[makeMarketVertexKey(location, bucket)]++;
      if (prevLocation != UNDEFINED) {
        edgeDemand[makeMarketEdgeKey(prevLocation, location, bucket)]++;
      }
      prevLocation = location;
    }
  }
}

int LNS::computeTaskPrecedenceWaitInCurrentSolution(int task) const {
  if (task < 0 || task >= instance_.getTasksNum()) {
    return 0;
  }
  const int taskAgent = solution_.taskAgentMap[task];
  if (taskAgent == UNASSIGNED) {
    return 0;
  }

  const int agent = taskAgent;
  if (agent < 0 || agent >= static_cast<int>(solution_.agents.size())) {
    return 0;
  }
  const int taskPos = findTaskPositionInAssignments(
      solution_.agents[agent].taskAssignments, task);
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
  const int taskCount = instance_.getTasksNum();
  vector<char> seenPredecessor(taskCount, 0);
  auto consumePredecessor = [&](int pred) {
    if (pred < 0 || pred >= taskCount || seenPredecessor[pred]) {
      return;
    }
    seenPredecessor[pred] = 1;
    const int predAgent = (pred >= 0 && pred < (int)solution_.taskAgentMap.size())
                              ? solution_.taskAgentMap[pred]
                              : UNASSIGNED;
    if (predAgent < 0 || predAgent >= static_cast<int>(solution_.agents.size())) {
      return;
    }
    const int predPos = findTaskPositionInAssignments(
        solution_.agents[predAgent].taskAssignments, pred);
    if (predAgent != UNASSIGNED && predPos >= 0 &&
        predPos < (int)solution_.agents[predAgent].taskPaths.size() &&
        !solution_.agents[predAgent].taskPaths[predPos].empty()) {
      release = max(release, solution_.agents[predAgent].taskPaths[predPos].endTime());
    }
  };

  const auto& ancestors = instance_.getAncestorsRef();
  if (task >= 0 && task < (int)ancestors.size()) {
    for (int pred : ancestors[task]) {
      consumePredecessor(pred);
    }
  }
  if (taskPos > 0 && agent >= 0 && agent < instance_.getAgentNum() &&
      taskPos - 1 < (int)solution_.agents[agent].taskAssignments.size()) {
    consumePredecessor(solution_.agents[agent].taskAssignments[taskPos - 1]);
  }

  return max(0, release - arrive);
}

double LNS::computeSolutionMarketPressure() const {
  auto weightedResourcePrice =
      [](const unordered_map<uint64_t, double>& prices,
         const unordered_map<uint64_t, double>& excessHat,
         uint64_t key) -> double {
    const auto pIt = prices.find(key);
    if (pIt == prices.end()) {
      return 0.0;
    }
    const auto eIt = excessHat.find(key);
    if (eIt == excessHat.end()) {
      return 0.0;
    }
    const double excessWeight = max(0.0, eIt->second);
    if (excessWeight <= 0.0) {
      return 0.0;
    }
    return pIt->second * excessWeight;
  };

  unordered_map<uint64_t, int> vertexDemand;
  unordered_map<uint64_t, int> edgeDemand;
  buildMarketDemandFromCurrentOccupancy(vertexDemand, edgeDemand);

  double pressure = 0.0;
  for (const auto& kv : vertexDemand) {
    const double unitContribution =
        weightedResourcePrice(market_.vertexPrices, market_.vertexExcessHat, kv.first);
    if (unitContribution <= 0.0) {
      continue;
    }
    pressure += unitContribution * kv.second;
  }
  for (const auto& kv : edgeDemand) {
    const double unitContribution =
        weightedResourcePrice(market_.edgePrices, market_.edgeExcessHat, kv.first);
    if (unitContribution <= 0.0) {
      continue;
    }
    pressure += unitContribution * kv.second;
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
  const vector<int>& taskPosByTask = getCurrentTaskPositionIndexByTask();
  return computeSolutionPrecedenceWaitFromIndex(taskPosByTask);
}

bool LNS::passMarketAcceptanceGuards(double previousPressure,
                                     double candidatePressure,
                                     double previousWait,
                                     double candidateWait,
                                     bool candidateIsWorse) const {
  if (!market_.acceptanceGuards) {
    return true;
  }
  // Keep exploration intact: apply guards only for non-improving candidates.
  if (!candidateIsWorse) {
    return true;
  }
  if (!std::isfinite(previousPressure) || !std::isfinite(previousWait)) {
    return true;
  }
  const bool pressureOK = candidatePressure <= previousPressure + market_.tauP;
  const bool waitOK = candidateWait <= previousWait + market_.tauW;
  return pressureOK && waitOK;
}

bool LNS::marketDestroyStabilityReady() const {
  if (!market_.heuristics || !market_.destroyRequireStable) {
    return true;
  }
  if (!market_.hasStabilityBaseline || market_.stats.updates < 2) {
    return false;
  }
  if (!std::isfinite(market_.stats.priceRelL1DeltaEma) ||
      !std::isfinite(market_.stats.topPriceMassDeltaEma) ||
      !std::isfinite(market_.stats.contendedJaccardEma)) {
    return false;
  }
  return market_.stats.priceRelL1DeltaEma <= market_.stabilityMaxRelPriceDelta &&
         market_.stats.topPriceMassDeltaEma <= market_.stabilityMaxTopMassDelta &&
         market_.stats.contendedJaccardEma >=
             market_.stabilityMinContendedJaccard;
}

void LNS::updateMarketStateFromCurrentSolution() {
  if (!market_.heuristics) {
    return;
  }

  unordered_map<uint64_t, int> vertexDemand;
  unordered_map<uint64_t, int> edgeDemand;
  buildMarketDemandFromCurrentOccupancy(vertexDemand, edgeDemand);

  const double eta =
      market_.eta / std::sqrt(1.0 + (double)market_.stats.updates);

  int64_t contendedResources = 0;
  double contendedPriceSum = 0.0;
  double maxPrice = 0.0;
  double totalAbsPriceDelta = 0.0;
  double totalOldPriceMass = 0.0;
  unordered_set<uint64_t> contendedVertices;
  unordered_set<uint64_t> contendedEdges;
  contendedVertices.reserve(vertexDemand.size());
  contendedEdges.reserve(edgeDemand.size());

  auto updateCategory = [&](const unordered_map<uint64_t, int>& demand,
                            unordered_map<uint64_t, double>& prices,
                            unordered_map<uint64_t, double>& excessHat,
                            int bucketCapacity,
                            unordered_set<uint64_t>* contendedKeys) {
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

      constexpr double kExcessHatEps = 1e-9;
      double newPrice = oldPrice;
      if (newHat > kExcessHatEps) {
        const double basePrice =
            (oldPrice > 0.0) ? oldPrice : market_.priceInit;
        newPrice = min(market_.priceCap, basePrice * std::exp(eta * newHat));
      } else {
        newPrice = oldPrice * max(0.0, 1.0 - market_.gamma);
      }

      totalAbsPriceDelta += std::abs(newPrice - oldPrice);
      totalOldPriceMass += oldPrice;

      if (newPrice > 1e-9 || newHat > 1e-9) {
        nextPrices[key] = newPrice;
        nextExcessHat[key] = newHat;
      }
      if (excess > 0) {
        contendedResources++;
        contendedPriceSum += newPrice;
        maxPrice = max(maxPrice, newPrice);
        if (contendedKeys != nullptr) {
          contendedKeys->insert(key);
        }
      }
    }

    prices.swap(nextPrices);
    excessHat.swap(nextExcessHat);
  };

  updateCategory(vertexDemand, market_.vertexPrices, market_.vertexExcessHat,
                 market_.vertexBucketCapacity, &contendedVertices);
  updateCategory(edgeDemand, market_.edgePrices, market_.edgeExcessHat,
                 market_.edgeBucketCapacity, &contendedEdges);

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

  const double relPriceL1Delta =
      (totalOldPriceMass > 1e-9)
          ? (totalAbsPriceDelta / totalOldPriceMass)
          : ((totalAbsPriceDelta > 1e-9) ? 1.0 : 0.0);
  const double previousTopPriceMassFrac = market_.stats.topPriceMassFrac;
  const double topPriceMassDelta =
      (market_.stats.updates > 0)
          ? std::abs(topPriceMassFrac - previousTopPriceMassFrac)
          : 0.0;
  double contendedJaccard = 1.0;
  if (market_.hasStabilityBaseline) {
    const auto computeIntersectionSize =
        [](const unordered_set<uint64_t>& lhs,
           const unordered_set<uint64_t>& rhs) -> size_t {
      if (lhs.empty() || rhs.empty()) {
        return 0;
      }
      const unordered_set<uint64_t>* smaller = &lhs;
      const unordered_set<uint64_t>* larger = &rhs;
      if (lhs.size() > rhs.size()) {
        smaller = &rhs;
        larger = &lhs;
      }
      size_t intersection = 0;
      for (uint64_t key : *smaller) {
        if (larger->find(key) != larger->end()) {
          intersection++;
        }
      }
      return intersection;
    };
    const size_t intersectionVertices = computeIntersectionSize(
        market_.prevContendedVertices, contendedVertices);
    const size_t unionVertices = market_.prevContendedVertices.size() +
                                 contendedVertices.size() -
                                 intersectionVertices;
    const size_t intersectionEdges =
        computeIntersectionSize(market_.prevContendedEdges, contendedEdges);
    const size_t unionEdges = market_.prevContendedEdges.size() +
                              contendedEdges.size() - intersectionEdges;
    const size_t intersectionTotal = intersectionVertices + intersectionEdges;
    const size_t unionTotal = unionVertices + unionEdges;
    contendedJaccard =
        (unionTotal == 0)
            ? 1.0
            : ((double)intersectionTotal / (double)unionTotal);
  }
  const auto emaUpdate = [&](double previous, double sample) -> double {
    if (!market_.hasStabilityBaseline) {
      return sample;
    }
    const double alpha = market_.stabilityEmaAlpha;
    return (1.0 - alpha) * previous + alpha * sample;
  };
  const double priceRelL1DeltaEma =
      emaUpdate(market_.stats.priceRelL1DeltaEma, relPriceL1Delta);
  const double topPriceMassDeltaEma =
      emaUpdate(market_.stats.topPriceMassDeltaEma, topPriceMassDelta);
  const double contendedJaccardEma =
      emaUpdate(market_.stats.contendedJaccardEma, contendedJaccard);

  const vector<int>& taskPosByTask = getCurrentTaskPositionIndexByTask();
  vector<TaskScheduleMetrics> perTask;
  computeTaskScheduleMetricsFromIndex(taskPosByTask, perTask, nullptr);
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
  market_.stats.priceRelL1Delta = relPriceL1Delta;
  market_.stats.priceRelL1DeltaEma = priceRelL1DeltaEma;
  market_.stats.topPriceMassDelta = topPriceMassDelta;
  market_.stats.topPriceMassDeltaEma = topPriceMassDeltaEma;
  market_.stats.contendedJaccard = contendedJaccard;
  market_.stats.contendedJaccardEma = contendedJaccardEma;
  market_.stats.totalPrecedenceWait = totalWait;
  market_.stats.maxPrecedenceWait = maxWait;
  market_.prevContendedVertices.swap(contendedVertices);
  market_.prevContendedEdges.swap(contendedEdges);
  market_.hasStabilityBaseline = true;
}

void LNS::maybeUpdateMarketState(bool accepted, bool candidateStateUpdate) {
  if (!market_.heuristics) {
    return;
  }
  if (candidateStateUpdate) {
    if (market_.updateOnAcceptedOnly || !market_.updateFromCandidate) {
      return;
    }
    market_.candidateUpdateConsumed = true;
    market_.updateCounter++;
    if (market_.updateCounter % market_.updatePeriodAccepted != 0) {
      return;
    }
    updateMarketStateFromCurrentSolution();
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
    if (market_.updateFromCandidate && market_.candidateUpdateConsumed) {
      return;
    }
    market_.updateCounter++;
    if (market_.updateCounter % market_.updatePeriodAccepted != 0) {
      return;
    }
  }
  updateMarketStateFromCurrentSolution();
}
