#pragma once

#include <plog/Log.h>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include "common.hpp"
#include "constrainttable.hpp"
#include "mlastar.hpp"
#include "sipps.hpp"

enum class LowLevelPlannerType {
  mlastar = 0,
  sipps = 1,
};

enum DestroyHeuristic {
  randomRemoval = 0,
  worstRemoval = 1,
  conflictRemoval = 2,
  shawRemoval = 3,
  precedenceWaitRemoval = 4,
  lowSlackRemoval = 5,
  marketTatonnementRemoval = 6,
  destroyHeuristicCount = 7
};

struct MarketStats {
  int64_t updates = 0;
  int64_t contendedResources = 0;
  double meanPriceContended = 0.0;
  double maxPrice = 0.0;
  double topPriceMassFrac = 0.0;
  double totalPrecedenceWait = 0.0;
  double maxPrecedenceWait = 0.0;

  void reset() { *this = MarketStats(); }
};

struct Agent {
  int id;
  AgentTaskPath path;
  vector<int> taskAssignments;
  vector<AgentTaskPath> taskPaths;
  vector<pair<int, int>> intraPrecedenceConstraints;
  std::shared_ptr<SingleAgentSolver> pathPlanner = nullptr;

  Agent(const Agent&) = default;
  Agent(Agent&&) = delete;
  Agent& operator=(const Agent& other) {
    if (this == &other) {
      return *this;
    }
    this->id = other.id;
    this->path = other.path;
    this->taskPaths = other.taskPaths;
    this->taskAssignments = other.taskAssignments;
    this->intraPrecedenceConstraints = other.intraPrecedenceConstraints;

    // Copy the path planner details
    this->pathPlanner->numExpanded = other.pathPlanner->numExpanded;
    this->pathPlanner->numGenerated = other.pathPlanner->numGenerated;

    this->pathPlanner->heuristic = other.pathPlanner->heuristic;
    this->pathPlanner->goalLocations = other.pathPlanner->goalLocations;
    this->pathPlanner->heuristicLandmarks =
        other.pathPlanner->heuristicLandmarks;
    return *this;
  }
  Agent& operator=(Agent&&) = delete;
  Agent(const Instance& instance, int id) : id(id) {
    pathPlanner = std::make_shared<MultiLabelSpaceTimeAStar>(instance, id);
  }
  ~Agent() = default;

  int getLocalTaskIndex(int globalTask) const {
    for (int i = 0; i < (int)taskAssignments.size(); i++) {
      if (taskAssignments[i] == globalTask) {
        return i;
      }
    }
    assert(false);
    return UNASSIGNED;
  }

  inline void insertPrecedenceConstraint(int taskA, int taskB) {
    assert(std::find(taskAssignments.begin(), taskAssignments.end(), taskA) !=
           taskAssignments.end());
    assert(std::find(taskAssignments.begin(), taskAssignments.end(), taskB) !=
           taskAssignments.end());
    intraPrecedenceConstraints.emplace_back(taskA, taskB);
  }

  // This function inserts a precedence constraint when adding a new task into the agent task queue. This would involve removing the existing precedence constraint between the task before it and after it and then adding two new precedence constraints
  void insertIntraAgentPrecedenceConstraint(int task, int taskPosition) {
    int previousTask = UNDEFINED, nextTask = UNDEFINED;
    if (taskPosition != 0) {
      previousTask = taskAssignments[taskPosition - 1];
    }
    if (taskPosition != (int)taskAssignments.size() - 1) {
      nextTask = taskAssignments[taskPosition + 1];
    }
    intraPrecedenceConstraints.erase(
        std::remove_if(intraPrecedenceConstraints.begin(),
                       intraPrecedenceConstraints.end(),
                       [previousTask, nextTask](pair<int, int> x) {
                         return (
                             (x.first == previousTask && x.second == nextTask));
                       }),
        intraPrecedenceConstraints.end());
    if (previousTask >= 0) {
      intraPrecedenceConstraints.insert(
          intraPrecedenceConstraints.begin() + taskPosition - 1,
          make_pair(previousTask, task));
    }
    if (nextTask >= 0) {
      intraPrecedenceConstraints.insert(
          intraPrecedenceConstraints.begin() + taskPosition,
          make_pair(task, nextTask));
    }
  }

  void clearIntraAgentPrecedenceConstraint(int task) {
    assert(std::find(taskAssignments.begin(), taskAssignments.end(), task) !=
           taskAssignments.end());
    int taskPosition = getLocalTaskIndex(task);
    int previousTask = UNDEFINED, nextTask = UNDEFINED,
        previousTaskPosition = taskPosition - 1,
        nextTaskPosition = taskPosition + 1;
    while (previousTaskPosition >= 0 && (previousTask == UNDEFINED || previousTask == UNASSIGNED)) {
      previousTask = taskAssignments[previousTaskPosition];
      previousTaskPosition--;
    }
    while (nextTaskPosition < (int)taskAssignments.size() && (nextTask == UNDEFINED || nextTask == UNASSIGNED)) {
      nextTask = taskAssignments[nextTaskPosition];
      nextTaskPosition++;
    }

    intraPrecedenceConstraints.erase(
        std::remove_if(intraPrecedenceConstraints.begin(),
                       intraPrecedenceConstraints.end(),
                       [task, previousTask, nextTask](pair<int, int> x) {
                         return (
                             (x.first == previousTask && x.second == task) ||
                             (x.first == task && x.second == nextTask));
                       }),
        intraPrecedenceConstraints.end());

    if (previousTask >= 0 && nextTask >= 0) {
      insertPrecedenceConstraint(previousTask, nextTask);
    }
  }
};

// We need min-heap for the utility since we want to quickly access the tasks which take the minimum time to complete
struct Utility {
  int agent, taskPosition;
  int pathLength, agentTasksLen;
  double value;
  double baseDeltaSoc;
  double deltaMarketExposure;
  double deltaPrecedenceWait;

  Utility() {
    agent = -1;
    taskPosition = -1;
    pathLength = -1;
    agentTasksLen = -1;
    value = std::numeric_limits<double>::max();
    baseDeltaSoc = 0.0;
    deltaMarketExposure = 0.0;
    deltaPrecedenceWait = 0.0;
  }

  Utility(int agent, int taskPosition, int pathLength, int agentTasksLen,
          double value, double baseDeltaSoc = 0.0,
          double deltaMarketExposure = 0.0,
          double deltaPrecedenceWait = 0.0)
      : agent(agent),
        taskPosition(taskPosition),
        pathLength(pathLength),
        agentTasksLen(agentTasksLen),
        value(value),
        baseDeltaSoc(baseDeltaSoc),
        deltaMarketExposure(deltaMarketExposure),
        deltaPrecedenceWait(deltaPrecedenceWait) {}

  struct CompareUtilities {
    bool operator()(const Utility& lhs, const Utility& rhs) const {
      if (lhs.value != rhs.value) {
        return lhs.value > rhs.value;
      }
      // Now that the regret values are same we move to compare the path lengths and prefer the regret with smaller path
      if (lhs.pathLength != rhs.pathLength) {
        return lhs.pathLength > rhs.pathLength;
      }
      // If even the path lengths are same then we will move to using the agent tasks queue lengths
      return lhs.agentTasksLen >= rhs.agentTasksLen;
    }
  };
};

// We need max-heap for the regret since we want to quickly access the task whose regret would be maximum
struct Regret {
  int task, agent, taskPosition;
  int pathLength, agentTasksLen, maxOptionsLeft;
  double value;
  uint32_t stamp = 0;

  Regret(int task, int agent, int taskPosition, int pathLength,
         int agentTasksLen, int maxOptionsLeft, double value,
         uint32_t stamp = 0)
      : task(task),
        agent(agent),
        taskPosition(taskPosition),
        pathLength(pathLength),
        agentTasksLen(agentTasksLen),
        maxOptionsLeft(maxOptionsLeft),
        value(value),
        stamp(stamp) {}

  struct CompareRegrets {
    bool operator()(const Regret& lhs, const Regret& rhs) const {
      if (lhs.value != rhs.value) {
        return lhs.value < rhs.value;
      }
      // Now that the regret values are same we move to compare the number of options left for them and prefer the regret with smaller number of options left to make it more likely that it will be picked first
      if (lhs.maxOptionsLeft != rhs.maxOptionsLeft) {
        return lhs.maxOptionsLeft > rhs.maxOptionsLeft;
      }
      // Now that the number of options left are same we move to compare the path lengths and prefer the regret with smaller path
      if (lhs.pathLength != rhs.pathLength) {
        return lhs.pathLength > rhs.pathLength;
      }
      // If even the path lengths are same then we will move to using the agent tasks queue lengths
      return lhs.agentTasksLen >= rhs.agentTasksLen;
    }
  };
};

struct TaskRegretPacket {
  int task, agent, taskPosition, earliestTimestep;
};

struct Conflicts {
  int task, agent, taskPosition;
  Conflicts(int task, int agent, int taskPosition) {
    this->task = task;
    this->agent = agent;
    this->taskPosition = taskPosition;
  }
  bool operator<(const Conflicts& right) const {
    return this->task < right.task;
  }
};

struct Neighbor {
  set<int> patchedTasks;
  map<int, bool> commitedTasks;
  map<int, int> removedTasksPathSize;
  set<Conflicts> removedTasks, immutableRemovedTasks;
  pairing_heap<Regret, compare<Regret::CompareRegrets>> regretMaxHeap;
};

struct FeasibleSolution {
 public:
  int sumOfCosts{}, numOfCols{};
  vector<AgentTaskPath> agentPaths;

  inline int getRowCoordinate(int id) const { return id / numOfCols; }
  inline int getColCoordinate(int id) const { return id % numOfCols; }
  inline pair<int, int> getCoordinate(int id) const {
    return make_pair(getRowCoordinate(id), getColCoordinate(id));
  }

  string toString() {
    string result =
        "Feasible Solution\n\tSum Of Costs = " + std::to_string(sumOfCosts) +
        "\n";
    for (int agent = 0; agent < (int)agentPaths.size(); agent++) {
      result += "Agent " + std::to_string(agent) +
                " (cost = " + std::to_string(agentPaths[agent].endTimeOrZero()) +
                "): \n\tPaths:\n\t";
      for (int t = 0; t < (int)agentPaths[agent].path.size(); t++) {
        pair<int, int> coord =
            getCoordinate(agentPaths[agent].path.at(t).location);
        result += "(" + std::to_string(coord.first) + ", " +
                  std::to_string(coord.second) + ")@" + std::to_string(t);
        if (agentPaths[agent].path.at(t).isGoal) {
          result += "*";
        }
        if (t != (int)agentPaths[agent].path.size() - 1) {
          result += " -> ";
        }
      }
      result += "\n";
    }
    return result;
  }
};

class Solution {
 public:
  int sumOfCosts{};
  double utility{};
  vector<Agent> agents;
  map<int, int> taskAgentMap;  // (key, value) - (global task, agent)
  int numOfAgents, numOfTasks;

  Solution(const Solution&) = default;
  Solution(Solution&&) = delete;
  Solution& operator=(Solution&&) = delete;
  ~Solution() = default;

  Solution(const Instance& instance) {
    numOfTasks = instance.getTasksNum();
    numOfAgents = instance.getAgentNum();
    agents.reserve(numOfAgents);
    for (int i = 0; i < numOfAgents; i++) {
      agents.emplace_back(instance, i);
    }
    for (int i = 0; i < numOfTasks; i++) {
      taskAgentMap.insert(make_pair(i, UNASSIGNED));
    }
  }

  Solution& operator=(const Solution& other);

  int getAgentWithTask(int globalTask) { return taskAgentMap[globalTask]; }

  int getLocalTaskIndex(int agent, int globalTask) const {
    return agents[agent].getLocalTaskIndex(globalTask);
  }

  inline vector<int> getAgentGlobalTasks(int agent) const {
    return agents[agent].taskAssignments;
  }
  inline int getAgentGlobalTasks(int agent, int taskIndex) const {
    return agents[agent].taskAssignments[taskIndex];
  }

  inline void assignTaskToAgent(int agent, int task) {
    taskAgentMap[task] = agent;
    agents[agent].taskAssignments.push_back(task);
  }

  inline void assignTaskToAgent(int agent, int task, int taskPosition) {
    assert(taskPosition >= 0 &&
           taskPosition <= (int)agents[agent].taskAssignments.size());
    agents[agent].taskAssignments.insert(
        agents[agent].taskAssignments.begin() + taskPosition, task);
  }

  // Do we need this function?
  void joinPaths(const vector<int>& agentsToCompute) {
    for (int agent : agentsToCompute) {

      assert(getAgentGlobalTasks(agent).size() ==
             agents[agent].pathPlanner->goalLocations.size());
      assert(getAgentGlobalTasks(agent).size() ==
             agents[agent].taskPaths.size());

      agents[agent].path = AgentTaskPath();
      for (int i = 0; i < (int)getAgentGlobalTasks(agent).size(); i++) {
        if (i == 0) {
          agents[agent].path.path.push_back(agents[agent].taskPaths[i].front());
        }
        assert((int)agents[agent].path.size() - 1 ==
               agents[agent].taskPaths[i].beginTime);
        for (int j = 1; j < (int)agents[agent].taskPaths[i].size(); j++) {
          agents[agent].path.path.push_back(agents[agent].taskPaths[i].at(j));
        }
        agents[agent].path.timeStamps.push_back(agents[agent].path.size() - 1);
      }
    }
  }
};

// Hold information to extract related tasks for shaw removal operator
struct RelatedTasks {
  int task, agent, taskPosition;
  int startTime, endTime, distance;
  // Lower value of relatedness means that the tasks are more related!
  double relatedness;

  RelatedTasks(int task, int agent, int taskPosition, int startTime,
               int endTime, int distance, int relation)
      : task(task),
        agent(agent),
        taskPosition(taskPosition),
        startTime(startTime),
        endTime(endTime),
        distance(distance),
        relatedness(relation) {}

  // Priority queue for relatedness
  struct RelationCompare {
    // Mininimum heap comparator
    bool operator()(const pair<int, RelatedTasks>& task1,
                    const pair<int, RelatedTasks>& task2) {
      return task1.first >= task2.first;
    }
  };

  // Comparator for custom Related Tasks struct
  struct RelatedTasksComparator {
    bool operator()(RelatedTasks task1, RelatedTasks task2) const {
      return task1.task < task2.task;
    }
  };
};

using pqRelatedTasks = std::priority_queue<pair<int, RelatedTasks>,
                                           vector<pair<int, RelatedTasks>>,
                                           RelatedTasks::RelationCompare>;

struct ALNS {

  // Parameter values copied from 'https://d-nb.info/1072464683/34'
  int alnsCounter = 0, alnsCounterThreshold = 100,
      numDestroyHeuristics = (int)DestroyHeuristic::destroyHeuristicCount,
      recentDestroyHeuristic = -1;
  vector<int> destroyHeuristicHistory;
  double r1 = 65, r2 = 45, r3 = 25;
  double delta1 = r1 + r2 + r3, delta2 = r2 + r3, delta3 = r3;
  double reactionFactor = 0.35;
  vector<double> weights, used, success;
  vector<int64_t> selections, accepted, rejected, feasible, bestUpdates,
      improvedAccepted, downgradedAccepted, couldNotFind;
  vector<double> deltaSocAll, deltaSocAccepted;

  ALNS() {

    // Initialize the vectors
    for (int i = 0; i < numDestroyHeuristics; i++) {
      weights.push_back(1);
      used.push_back(0);
      success.push_back(0);
      selections.push_back(0);
      accepted.push_back(0);
      rejected.push_back(0);
      feasible.push_back(0);
      bestUpdates.push_back(0);
      improvedAccepted.push_back(0);
      downgradedAccepted.push_back(0);
      couldNotFind.push_back(0);
      deltaSocAll.push_back(0.0);
      deltaSocAccepted.push_back(0.0);
    }
  }
};

struct LNSParams {
  int neighborhoodSize;
  double timeLimit, temperature, coolingCoefficient, heatingCoefficient,
      tolerance, shawDistanceWeight, shawTemporalWeight, lnsConflictWeight,
      lnsCostWeight;
  string initialSolutionStrategy, destroyHeuristic, acceptanceCriteria,
      regretType, lowLevelPlanner;
  bool incrementalRegret = false;
  bool plannerParityCheck = false;
  int plannerParityMaxLogs = 10;
  bool marketHeuristics = false;
  int marketBucketDt = 3;
  int marketVertexBucketCapacity = 2;
  int marketEdgeBucketCapacity = 2;
  bool marketUpdateOnAcceptedOnly = true;
  int marketUpdatePeriodAccepted = 1;
  double marketEta = 0.05;
  double marketRho = 0.9;
  double marketPriceCap = 50.0;
  double marketGamma = 0.01;
  bool marketAcceptanceGuards = false;
  double marketTauP = 0.0;
  double marketTauW = 0.0;
  double marketDestroyWeightPrice = 1.0;
  double marketDestroyWeightWait = 2.0;
  double marketDestroyWeightRoot = 1.5;
  double marketSeedTopFrac = 0.2;
  double marketRandomDestroyQuota = 0.15;
  int marketCooldownIters = 3;
  int marketDUp = 1;
  int marketDDown = 1;
  int marketClosureCap = 0;
  bool marketRepairTieBreak = false;
  bool marketRepairBlend = false;
  double marketTieBreakEpsSoc = 0.0;
  double marketLambdaPrice = 0.0;
  double marketLambdaWait = 0.0;
  double lowLevelSegmentTimeout = 600.0;
  // Supported: "descendants", "descendants+agent".
  string incrementalRegretMode = "descendants+agent";
  unsigned int seed = 0;

  LNSParams(int neighborhoodSize, double timeLimit, double temperature,
            double coolingCoefficient, double heatingCoefficient,
            double tolerance, double shawDistanceWeight,
            double shawTemporalWeight, double lnsConflictWeight,
            double lnsCostWeight, string initialSolutionStrategy,
            string destroyHeuristic, string acceptanceCriteria,
            string regretType, bool incrementalRegret,
            string incrementalRegretMode, unsigned int seed,
            bool plannerParityCheck = false,
            int plannerParityMaxLogs = 10,
            bool marketHeuristics = false,
            int marketBucketDt = 3,
            int marketVertexBucketCapacity = 2,
            int marketEdgeBucketCapacity = 2,
            bool marketUpdateOnAcceptedOnly = true,
            int marketUpdatePeriodAccepted = 1,
            double marketEta = 0.05,
            double marketRho = 0.9,
            double marketPriceCap = 50.0,
            double marketGamma = 0.01,
            bool marketAcceptanceGuards = false,
            double marketTauP = 0.0,
            double marketTauW = 0.0,
            double marketDestroyWeightPrice = 1.0,
            double marketDestroyWeightWait = 2.0,
            double marketDestroyWeightRoot = 1.5,
            double marketSeedTopFrac = 0.2,
            double marketRandomDestroyQuota = 0.15,
            int marketCooldownIters = 3,
            int marketDUp = 1,
            int marketDDown = 1,
            int marketClosureCap = 0,
            bool marketRepairTieBreak = false,
            bool marketRepairBlend = false,
            double marketTieBreakEpsSoc = 0.0,
            double marketLambdaPrice = 0.0,
            double marketLambdaWait = 0.0,
            string lowLevelPlanner = "mlastar",
            double lowLevelSegmentTimeout = 600.0)
      : neighborhoodSize(neighborhoodSize),
        timeLimit(timeLimit),
        temperature(temperature),
        coolingCoefficient(coolingCoefficient),
        heatingCoefficient(heatingCoefficient),
        tolerance(tolerance),
        shawDistanceWeight(shawDistanceWeight),
        shawTemporalWeight(shawTemporalWeight),
        lnsConflictWeight(lnsConflictWeight),
        lnsCostWeight(lnsCostWeight),
        initialSolutionStrategy(std::move(initialSolutionStrategy)),
        destroyHeuristic(std::move(destroyHeuristic)),
        acceptanceCriteria(std::move(acceptanceCriteria)),
        regretType(std::move(regretType)),
        lowLevelPlanner(std::move(lowLevelPlanner)),
        incrementalRegret(incrementalRegret),
        plannerParityCheck(plannerParityCheck),
        plannerParityMaxLogs(plannerParityMaxLogs),
        marketHeuristics(marketHeuristics),
        marketBucketDt(marketBucketDt),
        marketVertexBucketCapacity(marketVertexBucketCapacity),
        marketEdgeBucketCapacity(marketEdgeBucketCapacity),
        marketUpdateOnAcceptedOnly(marketUpdateOnAcceptedOnly),
        marketUpdatePeriodAccepted(marketUpdatePeriodAccepted),
        marketEta(marketEta),
        marketRho(marketRho),
        marketPriceCap(marketPriceCap),
        marketGamma(marketGamma),
        marketAcceptanceGuards(marketAcceptanceGuards),
        marketTauP(marketTauP),
        marketTauW(marketTauW),
        marketDestroyWeightPrice(marketDestroyWeightPrice),
        marketDestroyWeightWait(marketDestroyWeightWait),
        marketDestroyWeightRoot(marketDestroyWeightRoot),
        marketSeedTopFrac(marketSeedTopFrac),
        marketRandomDestroyQuota(marketRandomDestroyQuota),
        marketCooldownIters(marketCooldownIters),
        marketDUp(marketDUp),
        marketDDown(marketDDown),
        marketClosureCap(marketClosureCap),
        marketRepairTieBreak(marketRepairTieBreak),
        marketRepairBlend(marketRepairBlend),
        marketTieBreakEpsSoc(marketTieBreakEpsSoc),
        marketLambdaPrice(marketLambdaPrice),
        marketLambdaWait(marketLambdaWait),
        lowLevelSegmentTimeout(lowLevelSegmentTimeout),
        incrementalRegretMode(std::move(incrementalRegretMode)),
        seed(seed) {}
};

class LNS {
 public:
  struct RegretEvalStats {
    int64_t recomputeCalls = 0;
    int64_t tasksEvaluated = 0;
    int64_t agentEvaluations = 0;
    int64_t candidateInsertionsTried = 0;
    int64_t candidateInsertionsFeasible = 0;
    int64_t neighborhoods = 0;
    int64_t removedTasksSum = 0;
    int64_t removedTasksMax = 0;

    void reset() { *this = RegretEvalStats(); }
  };

  struct IncrementalRegretStats {
    int64_t commits = 0;
    int64_t heapRebuilds = 0;
    int64_t fullRefreshes = 0;
    int64_t stalePops = 0;
    int64_t recomputeCalls = 0;
    int64_t recomputedTasks = 0;
    int64_t dirtySum = 0;
    int64_t dirtyMax = 0;
    int64_t changedSum = 0;
    int64_t changedMax = 0;

    void reset() { *this = IncrementalRegretStats(); }
  };

 private:
 int numOfIterations_;
  bool incrementalRegret_ = false;
  LowLevelPlannerType lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  double lowLevelSegmentTimeout_ = 600.0;
  bool plannerParityCheck_ = false;
  int plannerParityMaxLogs_ = 10;
  bool marketHeuristics_ = false;
  int marketBucketDt_ = 3;
  int marketVertexBucketCapacity_ = 2;
  int marketEdgeBucketCapacity_ = 2;
  bool marketUpdateOnAcceptedOnly_ = true;
  int marketUpdatePeriodAccepted_ = 1;
  int marketAcceptedCounter_ = 0;
  int marketUpdateCounter_ = 0;
  double marketEta_ = 0.05;
  double marketRho_ = 0.9;
  double marketPriceCap_ = 50.0;
  double marketGamma_ = 0.01;
  bool marketAcceptanceGuards_ = false;
  double marketTauP_ = 0.0;
  double marketTauW_ = 0.0;
  double marketDestroyWeightPrice_ = 1.0;
  double marketDestroyWeightWait_ = 2.0;
  double marketDestroyWeightRoot_ = 1.5;
  double marketSeedTopFrac_ = 0.2;
  double marketRandomDestroyQuota_ = 0.15;
  int marketCooldownIters_ = 3;
  int marketDUp_ = 1;
  int marketDDown_ = 1;
  int marketClosureCap_ = 0;
  bool marketRepairTieBreak_ = false;
  bool marketRepairBlend_ = false;
  double marketTieBreakEpsSoc_ = 0.0;
  double marketLambdaPrice_ = 0.0;
  double marketLambdaWait_ = 0.0;
  double marketBestPressure_ = std::numeric_limits<double>::infinity();
  double marketBestWait_ = std::numeric_limits<double>::infinity();

  unordered_map<uint64_t, double> marketVertexPrices_;
  unordered_map<uint64_t, double> marketEdgePrices_;
  unordered_map<uint64_t, double> marketVertexExcessHat_;
  unordered_map<uint64_t, double> marketEdgeExcessHat_;
  vector<int> marketTaskCooldownUntilIter_;
  MarketStats marketStats_;

  struct TaskScheduleMetrics {
    bool valid = false;
    int arrive = 0;
    int release = 0;
    int start = 0;
    int end = 0;
    int waitPrec = 0;
    int blocker = UNASSIGNED;
  };

  enum class IncrementalRegretMode { descendants, descendants_and_agent };
  IncrementalRegretMode incrementalRegretMode_ =
      IncrementalRegretMode::descendants_and_agent;
  vector<uint32_t> regretStamp_;
  vector<pair<int, int>> regretBestOption_;
  vector<pair<int, int>> regretSecondBestOption_;

  RegretEvalStats regretEvalStatsCurrent_;
  RegretEvalStats regretEvalStatsTotal_;

  IncrementalRegretStats incrementalRegretStatsCurrent_;
  IncrementalRegretStats incrementalRegretStatsTotal_;

 protected:
  ALNS adaptiveLNS_;
  int neighborSize_;
  Neighbor lnsNeighborhood_;
  const Instance& instance_;
  unsigned int seed_ = 0;
  std::mt19937 rng_;
  vector<AgentTaskPath> initialPaths_;
  FeasibleSolution incumbentSolution_;
  Solution solution_, previousSolution_;
  double timeLimit_, initialSolutionRuntime_ = 0, temperature_ = 100,
                     coolingCoefficient_ = 0.99975,
                     heatingCoefficient_ = 1.00025, tolerance_ = 5,
                     shawDistanceWeight_ = 9, shawTemporalWeight_ = 3,
                     lnsConflictWeight_ = 0.75, lnsCostWeight_ = 0.25;
  high_resolution_clock::time_point plannerStartTime_;

 public:
  double runtime = 0;
  int numOfFailures = 0, sumOfCosts = 0;
  list<IterationStats> iterationStats;
  string initialSolutionStrategy, destroyHeuristic, acceptanceCriteria,
      regretType;

  LNS(int numOfIterations, const Instance& instance,
      const LNSParams& parameters);

  inline Instance getInstance() { return instance_; }

  bool run();

  bool buildGreedySolution();
  // Precedence-feasible initial solution that ignores inter-agent collisions.
  bool buildGreedySolutionPrecedenceOnly();
  bool buildGreedySolutionWithMAPFPC(const string& variant);

  void prepareNextIteration();
  void markResolved(int globalTask);
  // Patches the task paths of an agent such that the begin times and end times match up
  void patchAgentTaskPaths(int agent, int taskPosition);

  void printPaths() const;
  bool validateSolution(set<Conflicts>* conflictedTasks = nullptr);

  void buildConstraintTable(ConstraintTable& constraintTable, int task);

  void buildConstraintTable(ConstraintTable& constraintTable,
                            TaskRegretPacket taskPacket, int taskLocation,
                            vector<vector<int>>* agentTaskAssignments,
                            vector<vector<AgentTaskPath>>* agentTaskPaths,
                            vector<pair<int, int>>* precedenceConstraints,
                            bool findingNextTask = false);

  int extractOldLocalTaskIndex(int task, vector<int> oldTaskQueue,
                               vector<int> newTaskQueue = {});
  set<int> reachableSet(int source, vector<vector<int>> edgeList);

  bool computeRegret();
  bool computeRegretForTask(int task);
  void computeRegretForTaskWithAgent(
      TaskRegretPacket regretPacket, vector<vector<int>>* agentTaskAssignments,
      vector<vector<AgentTaskPath>>* agentTaskPaths,
      vector<pair<int, int>>* precedenceConstraints,
      pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes);

  bool recomputeRegretsForTasks(const vector<int>& tasks);
  std::optional<Regret> popNextValidRegret();
  vector<int> collectRemainingRemovedTasks() const;
  vector<int> computeCurrentTaskEndTimes() const;
  vector<int> computeCurrentLastTaskPerAgent() const;
  vector<int> computeDirtyTasksAfterCommit(const vector<int>& endTimesBefore,
                                          const vector<int>& endTimesAfter,
                                          const vector<int>& lastTaskBefore,
                                          const vector<int>& lastTaskAfter);
  std::shared_ptr<SingleAgentSolver> createSharedPlanner(int agent) const;
  std::unique_ptr<SingleAgentSolver> createLocalPlanner(int agent) const;

  void commitBestRegretTask(Regret bestRegret);
  void commitAncestorTaskOf(int globalTask,
                            std::optional<pair<bool, int>> committingNextTask);

  std::variant<bool, Utility> insertTask(
      TaskRegretPacket regretPacket,
      vector<vector<AgentTaskPath>>* agentTaskPaths,
      vector<vector<int>>* agentTaskAssignments,
      vector<pair<int, int>>* precedenceConstraints);
  void insertBestRegretTask(TaskRegretPacket bestRegretPacket);

  Solution getSolution() { return solution_; }
  ALNS getAdaptiveLNS() { return adaptiveLNS_; }
  std::optional<IncrementalRegretStats> getIncrementalRegretStats() const {
    if (!incrementalRegret_) {
      return std::nullopt;
    }
    return incrementalRegretStatsTotal_;
  }
  RegretEvalStats getRegretEvalStats() const { return regretEvalStatsTotal_; }
  string getIncrementalRegretMode() const {
    return incrementalRegretMode_ == IncrementalRegretMode::descendants
               ? "descendants"
               : "descendants+agent";
  }

  bool extractFeasibleSolution();
  FeasibleSolution getFeasibleSolution() { return incumbentSolution_; }
  MarketStats getMarketStats() const { return marketStats_; }

  void randomRemoval();
  void worstRemoval();
  void conflictRemoval(std::optional<set<Conflicts>> potentialNeighborhood);
  void shawRemoval(int prioritySize);
  void precedenceWaitRemoval(
      std::optional<set<Conflicts>> potentialNeighborhood = std::nullopt);
  void lowSlackRemoval(
      std::optional<set<Conflicts>> potentialNeighborhood = std::nullopt);
  void marketTatonnementRemoval(
      std::optional<set<Conflicts>> potentialNeighborhood = std::nullopt);
  void alnsRemoval(std::optional<set<Conflicts>> potentialNeighborhood);

  bool simulatedAnnealing();
  bool thresholdAcceptance();
  bool oldBachelorsAcceptance();
  bool greatDelugeAlgorithm();

  int marketTimeBucket(int timestep) const;
  uint64_t makeMarketVertexKey(int location, int bucket) const;
  uint64_t makeMarketEdgeKey(int from, int to, int bucket) const;
  void computeTaskScheduleMetrics(
      vector<TaskScheduleMetrics>& perTask,
      vector<double>* blockedWaitSum = nullptr) const;
  double computeTaskMarketExposure(int task, bool normalized) const;
  double computeSolutionMarketPressure() const;
  double computeSolutionPrecedenceWait() const;
  bool passMarketAcceptanceGuards(double candidatePressure,
                                  double candidateWait) const;
  void updateMarketStateFromCurrentSolution();
  void maybeUpdateMarketState(bool accepted);
  double computeMarketExposureFromPath(const AgentTaskPath& taskPath,
                                       bool normalized) const;
  int computeTaskPrecedenceWaitFromState(
      int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
      const vector<vector<AgentTaskPath>>& agentTaskPaths,
      const vector<pair<int, int>>& precedenceConstraints) const;
  int computeTaskPrecedenceWaitInCurrentSolution(int task) const;

  void computeMovingMetrics(int numberOfConflicts, int sumOfCosts);

  void printAgents() const {
    for (int i = 0; i < instance_.getAgentNum(); i++) {
      pair<int, int> startLoc =
          instance_.getCoordinate(instance_.getStartLocationsRef()[i]);
      PLOGI << "Agent " << i << " : S = (" << startLoc.first << ", "
            << startLoc.second << ") ;\nGoals : \n";
      for (int j = 0; j < (int)solution_.agents[i].taskAssignments.size();
           j++) {
        pair<int, int> goalLoc =
            instance_.getCoordinate(solution_.agents[i].taskAssignments[j]);
        PLOGI << "\t" << j << " : (" << goalLoc.first << " , " << goalLoc.second
              << ")\n";
      }
    }
  }
};
