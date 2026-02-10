#pragma once

#include <plog/Log.h>
#include <cmath>
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

  Agent(const Agent& other)
      : id(other.id),
        path(other.path),
        taskAssignments(other.taskAssignments),
        taskPaths(other.taskPaths),
        intraPrecedenceConstraints(other.intraPrecedenceConstraints),
        pathPlanner(clonePlanner(other.pathPlanner, other.id)) {}
  Agent(Agent&&) noexcept = default;
  Agent& operator=(const Agent& other) {
    if (this == &other) {
      return *this;
    }
    id = other.id;
    path = other.path;
    taskPaths = other.taskPaths;
    taskAssignments = other.taskAssignments;
    intraPrecedenceConstraints = other.intraPrecedenceConstraints;
    pathPlanner = clonePlanner(other.pathPlanner, other.id);
    return *this;
  }
  Agent& operator=(Agent&&) noexcept = default;
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
    PLOGE << "Agent::getLocalTaskIndex: task " << globalTask
          << " not found in agent " << id << " assignments\n";
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
    const int assignmentSize = (int)taskAssignments.size();
    if (taskPosition < 0 || taskPosition >= assignmentSize) {
      PLOGE << "insertIntraAgentPrecedenceConstraint: invalid taskPosition "
            << taskPosition << " for agent " << id << " with "
            << assignmentSize << " assigned tasks\n";
      assert(false);
      return;
    }
    int previousTask = UNDEFINED, nextTask = UNDEFINED;
    if (taskPosition > 0) {
      previousTask = taskAssignments[taskPosition - 1];
    }
    if (taskPosition + 1 < assignmentSize) {
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
    const auto addEdgeIfMissing = [this](int from, int to) {
      if (from < 0 || to < 0) {
        return;
      }
      const auto edge = std::make_pair(from, to);
      if (std::find(intraPrecedenceConstraints.begin(),
                    intraPrecedenceConstraints.end(),
                    edge) == intraPrecedenceConstraints.end()) {
        intraPrecedenceConstraints.emplace_back(edge);
      }
    };
    addEdgeIfMissing(previousTask, task);
    addEdgeIfMissing(task, nextTask);
  }

  void clearIntraAgentPrecedenceConstraint(int task) {
    assert(std::find(taskAssignments.begin(), taskAssignments.end(), task) !=
           taskAssignments.end());
    int taskPosition = getLocalTaskIndex(task);
    if (taskPosition == UNASSIGNED) {
      PLOGE << "clearIntraAgentPrecedenceConstraint: task " << task
            << " has no local index for agent " << id << "\n";
      return;
    }
    int previousTask = UNDEFINED, nextTask = UNDEFINED;

    // During destroy/repair, taskAssignments can temporarily contain tombstones
    // (UNDEFINED / UNASSIGNED) before compaction; skip them when reconnecting.
    const auto isConcreteTask = [](int value) {
      return value != UNDEFINED && value != UNASSIGNED;
    };
    for (int pos = taskPosition - 1; pos >= 0; --pos) {
      if (isConcreteTask(taskAssignments[pos])) {
        previousTask = taskAssignments[pos];
        break;
      }
    }
    for (int pos = taskPosition + 1; pos < (int)taskAssignments.size(); ++pos) {
      if (isConcreteTask(taskAssignments[pos])) {
        nextTask = taskAssignments[pos];
        break;
      }
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

 private:
  static std::shared_ptr<SingleAgentSolver> clonePlanner(
      const std::shared_ptr<SingleAgentSolver>& source, int agentId) {
    if (source == nullptr) {
      return nullptr;
    }
    return source->cloneForAgent(agentId);
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
    static int64_t valueKey(double value) {
      constexpr double kScale = 1e6;
      if (!std::isfinite(value)) {
        return value > 0 ? std::numeric_limits<int64_t>::max()
                         : std::numeric_limits<int64_t>::min();
      }
      constexpr double kMaxSafe =
          static_cast<double>(std::numeric_limits<int64_t>::max()) / kScale;
      constexpr double kMinSafe =
          static_cast<double>(std::numeric_limits<int64_t>::min()) / kScale;
      if (value >= kMaxSafe) {
        return std::numeric_limits<int64_t>::max();
      }
      if (value <= kMinSafe) {
        return std::numeric_limits<int64_t>::min();
      }
      return static_cast<int64_t>(std::llround(value * kScale));
    }

    bool operator()(const Utility& lhs, const Utility& rhs) const {
      const int64_t lhsValueKey = valueKey(lhs.value);
      const int64_t rhsValueKey = valueKey(rhs.value);
      if (lhsValueKey != rhsValueKey) {
        return lhsValueKey > rhsValueKey;
      }
      // Now that the regret values are same we move to compare the path lengths and prefer the regret with smaller path
      if (lhs.pathLength != rhs.pathLength) {
        return lhs.pathLength > rhs.pathLength;
      }
      // If even the path lengths are same then we will move to using the agent tasks queue lengths
      return lhs.agentTasksLen > rhs.agentTasksLen;
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
    static int64_t valueKey(double value) {
      constexpr double kScale = 1e6;
      if (!std::isfinite(value)) {
        return value > 0 ? std::numeric_limits<int64_t>::max()
                         : std::numeric_limits<int64_t>::min();
      }
      constexpr double kMaxSafe =
          static_cast<double>(std::numeric_limits<int64_t>::max()) / kScale;
      constexpr double kMinSafe =
          static_cast<double>(std::numeric_limits<int64_t>::min()) / kScale;
      if (value >= kMaxSafe) {
        return std::numeric_limits<int64_t>::max();
      }
      if (value <= kMinSafe) {
        return std::numeric_limits<int64_t>::min();
      }
      return static_cast<int64_t>(std::llround(value * kScale));
    }

    bool operator()(const Regret& lhs, const Regret& rhs) const {
      const int64_t lhsValueKey = valueKey(lhs.value);
      const int64_t rhsValueKey = valueKey(rhs.value);
      if (lhsValueKey != rhsValueKey) {
        return lhsValueKey < rhsValueKey;
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
      return lhs.agentTasksLen > rhs.agentTasksLen;
    }
  };
};

struct TaskRegretPacket {
  int task, agent, taskPosition, earliestTimestep;
};

struct Conflicts {
  int task, agent, taskPosition;
  // Identity is task-based: each global task can appear at most once per
  // neighborhood.
  Conflicts(int task, int agent, int taskPosition) {
    this->task = task;
    this->agent = agent;
    this->taskPosition = taskPosition;
  }
};
using ConflictMap = map<int, Conflicts>;

struct Neighbor {
  set<int> patchedTasks;
  // Task-indexed state vectors.
  // committedTasks: -1 = unknown/absent, 0 = pending, 1 = committed.
  vector<int8_t> committedTasks;
  // removedTasksPathSize: -1 = absent, otherwise prior path size.
  vector<int> removedTasksPathSize;
  ConflictMap removedTasks, immutableRemovedTasks;
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

  string toString() const {
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
  vector<int> taskAgentMap;  // index=global task, value=agent
  int numOfAgents, numOfTasks;

  Solution(const Solution&) = default;
  Solution(Solution&&) noexcept = default;
  Solution& operator=(Solution&&) noexcept = default;
  ~Solution() = default;

  explicit Solution(const Instance& instance) {
    numOfTasks = instance.getTasksNum();
    numOfAgents = instance.getAgentNum();
    agents.reserve(numOfAgents);
    for (int i = 0; i < numOfAgents; i++) {
      agents.emplace_back(instance, i);
    }
    taskAgentMap.assign(numOfTasks, UNASSIGNED);
  }

  Solution& operator=(const Solution& other);

  int getAgentWithTask(int globalTask) const {
    if (globalTask < 0 || globalTask >= (int)taskAgentMap.size()) {
      assert(false);
      return UNASSIGNED;
    }
    return taskAgentMap[globalTask];
  }

  int getLocalTaskIndex(int agent, int globalTask) const {
    if (agent < 0 || agent >= (int)agents.size()) {
      PLOGE << "Solution::getLocalTaskIndex: invalid agent index " << agent
            << " for task " << globalTask << "\n";
      assert(false);
      return UNASSIGNED;
    }
    return agents[agent].getLocalTaskIndex(globalTask);
  }

  inline const vector<int>& getAgentGlobalTasks(int agent) const {
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
    taskAgentMap[task] = agent;
    agents[agent].taskAssignments.insert(
        agents[agent].taskAssignments.begin() + taskPosition, task);
  }

  // Rebuild joined agent paths from per-task segments.
  // Transactional semantics: if validation fails for any requested agent, no
  // agent path is modified and the function returns false.
  bool joinPaths(const vector<int>& agentsToCompute) {
    vector<pair<int, AgentTaskPath>> staged;
    staged.reserve(agentsToCompute.size());

    for (int agent : agentsToCompute) {
      if (agent < 0 || agent >= (int)agents.size()) {
        PLOGE << "joinPaths: invalid agent index " << agent << "\n";
        return false;
      }
      if (agents[agent].pathPlanner == nullptr) {
        PLOGE << "joinPaths: missing path planner for agent " << agent << "\n";
        return false;
      }

      const auto& assignments = getAgentGlobalTasks(agent);
      const auto& taskPaths = agents[agent].taskPaths;
      const auto& plannerGoals = agents[agent].pathPlanner->goalLocations;
      if (assignments.size() != plannerGoals.size()) {
        PLOGE << "joinPaths: goal count mismatch for agent " << agent
              << " (assignments=" << assignments.size()
              << ", goals=" << plannerGoals.size() << ")\n";
        return false;
      }
      if (assignments.size() != taskPaths.size()) {
        PLOGE << "joinPaths: task path count mismatch for agent " << agent
              << " (assignments=" << assignments.size()
              << ", taskPaths=" << taskPaths.size() << ")\n";
        return false;
      }

      AgentTaskPath joined;
      for (int i = 0; i < (int)assignments.size(); i++) {
        const auto& segment = taskPaths[i];
        if (segment.empty()) {
          PLOGE << "joinPaths: empty segment at agent " << agent
                << ", local task " << i << "\n";
          return false;
        }

        if (i == 0) {
          joined.path.push_back(segment.front());
        } else {
          if ((int)joined.size() - 1 != segment.beginTime) {
            PLOGE << "joinPaths: beginTime mismatch for agent " << agent
                  << ", local task " << i << " (expected "
                  << (int)joined.size() - 1 << ", got " << segment.beginTime
                  << ")\n";
            return false;
          }
          if (joined.path.back().location != segment.front().location) {
            PLOGE << "joinPaths: discontinuity for agent " << agent
                  << ", local task " << i << "\n";
            return false;
          }
        }

        for (int j = 1; j < (int)segment.size(); j++) {
          joined.path.push_back(segment.at(j));
        }
        joined.timeStamps.push_back((int)joined.path.size() - 1);
      }
      staged.emplace_back(agent, std::move(joined));
    }

    for (auto& entry : staged) {
      agents[entry.first].path = std::move(entry.second);
    }
    return true;
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
      return task1.first > task2.first;
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
    weights.assign(numDestroyHeuristics, 1.0);
    used.assign(numDestroyHeuristics, 0.0);
    success.assign(numDestroyHeuristics, 0.0);
    selections.assign(numDestroyHeuristics, 0);
    accepted.assign(numDestroyHeuristics, 0);
    rejected.assign(numDestroyHeuristics, 0);
    feasible.assign(numDestroyHeuristics, 0);
    bestUpdates.assign(numDestroyHeuristics, 0);
    improvedAccepted.assign(numDestroyHeuristics, 0);
    downgradedAccepted.assign(numDestroyHeuristics, 0);
    couldNotFind.assign(numDestroyHeuristics, 0);
    deltaSocAll.assign(numDestroyHeuristics, 0.0);
    deltaSocAccepted.assign(numDestroyHeuristics, 0.0);
  }
};

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
    string destroyHeuristic;
    string acceptanceCriteria;
    string regretType;
    bool incrementalRegret = false;
    // Supported: "descendants", "descendants+agent".
    string incrementalRegretMode = "descendants+agent";
    unsigned int seed = 0;
  } core;

  struct LowLevel {
    string planner = "mlastar";
    double segmentTimeout = 600.0;
    bool parityCheck = false;
    int parityMaxLogs = 10;
  } lowLevel;

  struct Market {
    bool heuristics = false;
    int bucketDt = 3;
    int vertexBucketCapacity = 2;
    int edgeBucketCapacity = 2;
    bool updateOnAcceptedOnly = true;
    int updatePeriodAccepted = 1;
    double eta = 0.05;
    double rho = 0.9;
    double priceCap = 50.0;
    double gamma = 0.01;
    bool acceptanceGuards = false;
    double tauP = 0.0;
    double tauW = 0.0;
    double destroyWeightPrice = 1.0;
    double destroyWeightWait = 2.0;
    double destroyWeightRoot = 1.5;
    double seedTopFrac = 0.2;
    double randomDestroyQuota = 0.15;
    int cooldownIters = 3;
    int dUp = 1;
    int dDown = 1;
    int closureCap = 0;
    bool repairTieBreak = false;
    bool repairBlend = false;
    double tieBreakEpsSoc = 0.0;
    double lambdaPrice = 0.0;
    double lambdaWait = 0.0;
  } market;
};

class LNS {
 public:
  struct LowLevelSearchStats {
    uint64_t calls = 0;
    uint64_t expanded = 0;
    uint64_t generated = 0;
  };

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
  struct MarketState : LNSParams::Market {
    // Runtime-only market state. Configuration fields are inherited from
    // LNSParams::Market to avoid duplicated declarations.
    int acceptedCounter = 0;
    int updateCounter = 0;
    double bestPressure = std::numeric_limits<double>::infinity();
    double bestWait = std::numeric_limits<double>::infinity();

    unordered_map<uint64_t, double> vertexPrices;
    unordered_map<uint64_t, double> edgePrices;
    unordered_map<uint64_t, double> vertexExcessHat;
    unordered_map<uint64_t, double> edgeExcessHat;
    vector<int> taskCooldownUntilIter;
    MarketStats stats;
  };
  MarketState market_;

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

  vector<pair<int, int>> buildFullPrecedenceConstraints(
      bool includeIntraConstraints = true) const;
  AgentTaskPath runLowLevelSearch(SingleAgentSolver& solver,
                                  ConstraintTable& constraintTable,
                                  int startTime, int stage, int lowerBound);
  void clearNeighborhood();

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
  uint64_t lowLevelCalls_ = 0;
  uint64_t lowLevelExpanded_ = 0;
  uint64_t lowLevelGenerated_ = 0;
  double timeLimit_, initialSolutionRuntime_ = 0, temperature_ = 100,
                     coolingCoefficient_ = 0.99975,
                     heatingCoefficient_ = 1.00025, tolerance_ = 5,
                     shawDistanceWeight_ = 9, shawTemporalWeight_ = 3,
                     lnsConflictWeight_ = 0.75, lnsCostWeight_ = 0.25;
  Time::time_point plannerStartTime_;

 public:
  double runtime = 0;
  int numOfFailures = 0, sumOfCosts = 0;
  vector<IterationStats> iterationStats;
  string initialSolutionStrategy, destroyHeuristic, acceptanceCriteria,
      regretType;

  LNS(int numOfIterations, const Instance& instance,
      const LNSParams& parameters);

  inline const Instance& getInstance() const { return instance_; }

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
  bool validateSolution(ConflictMap* conflictedTasks = nullptr);
  void addConflictingTask(int agent, int timestep, ConflictMap* out) const;

  void buildConstraintTable(ConstraintTable& constraintTable, int task);

  void buildConstraintTable(ConstraintTable& constraintTable,
                            TaskRegretPacket taskPacket, int taskLocation,
                            vector<vector<int>>* agentTaskAssignments,
                            vector<vector<AgentTaskPath>>* agentTaskPaths,
                            vector<pair<int, int>>* precedenceConstraints,
                            bool findingNextTask = false);

  int extractOldLocalTaskIndex(
      int task, const vector<int>& oldTaskQueue,
      const vector<int>& newTaskQueue = vector<int>());
  vector<char> reachableSet(int source, const vector<vector<int>>& edgeList);

  bool computeRegret();
  bool computeRegretForTask(int task);
  bool computeRegretForTask(
      int task,
      const vector<pair<int, int>>& fullPrecedenceConstraints);
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

  bool commitBestRegretTask(Regret bestRegret);
  bool commitAncestorTaskOf(
      int globalTask, std::optional<pair<bool, int>> committingNextTask);

  std::variant<bool, Utility> insertTask(
      TaskRegretPacket regretPacket,
      vector<vector<AgentTaskPath>>* agentTaskPaths,
      vector<vector<int>>* agentTaskAssignments,
      vector<pair<int, int>>* precedenceConstraints);
  bool insertBestRegretTask(TaskRegretPacket bestRegretPacket);

  const Solution& getSolution() const { return solution_; }
  const ALNS& getAdaptiveLNSRef() const { return adaptiveLNS_; }
  ALNS getAdaptiveLNS() const { return adaptiveLNS_; }
  LowLevelSearchStats getLowLevelSearchStats() const {
    return {lowLevelCalls_, lowLevelExpanded_, lowLevelGenerated_};
  }
  std::optional<IncrementalRegretStats> getIncrementalRegretStats() const {
    if (!incrementalRegret_) {
      return std::nullopt;
    }
    return incrementalRegretStatsTotal_;
  }
  const RegretEvalStats& getRegretEvalStatsRef() const {
    return regretEvalStatsTotal_;
  }
  RegretEvalStats getRegretEvalStats() const { return regretEvalStatsTotal_; }
  string getIncrementalRegretMode() const {
    return incrementalRegretMode_ == IncrementalRegretMode::descendants
               ? "descendants"
               : "descendants+agent";
  }

  bool extractFeasibleSolution();
  const FeasibleSolution& getFeasibleSolution() const {
    return incumbentSolution_;
  }
  MarketStats getMarketStats() const { return market_.stats; }

  void randomRemoval();
  void worstRemoval();
  void conflictRemoval(std::optional<ConflictMap> potentialNeighborhood);
  void shawRemoval(int prioritySize);
  void precedenceWaitRemoval(
      std::optional<ConflictMap> potentialNeighborhood = std::nullopt);
  void lowSlackRemoval(
      std::optional<ConflictMap> potentialNeighborhood = std::nullopt);
  void marketTatonnementRemoval(
      std::optional<ConflictMap> potentialNeighborhood = std::nullopt);
  void alnsRemoval(std::optional<ConflictMap> potentialNeighborhood);

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
        const int globalTask = solution_.agents[i].taskAssignments[j];
        if (globalTask < 0 || globalTask >= instance_.getTasksNum()) {
          PLOGE << "\t" << j << " : invalid task id " << globalTask << "\n";
          continue;
        }
        const int goalLocation = instance_.getTaskLocations(globalTask);
        pair<int, int> goalLoc =
            instance_.getCoordinate(goalLocation);
        PLOGI << "\t" << j << " : (" << goalLoc.first << " , " << goalLoc.second
              << ")\n";
      }
    }
  }
};
