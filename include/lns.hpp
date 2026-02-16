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
  int64_t destroyWarmupSkipped = 0;
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
  // Optional post-completion trajectory (move-out/wait/return) used by
  // true-reposition modes. In Phase A this is plumbing only.
  AgentTaskPath terminalPath;
  bool terminalPathActive = false;
  vector<int> taskAssignments;
  vector<AgentTaskPath> taskPaths;
  vector<pair<int, int>> intraPrecedenceConstraints;
  // Best-effort inverse lookup cache: global task -> local index in
  // taskAssignments. Entries are validated before use, so stale hints are safe.
  mutable vector<int> localTaskIndexCache;
  // True when assignment-order mutations happened without eagerly updating
  // intraPrecedenceConstraints. This keeps hot-path updates O(1) and treats
  // intraPrecedenceConstraints as diagnostic-only cache.
  bool intraPrecedenceDirty = false;
  std::shared_ptr<SingleAgentSolver> pathPlanner = nullptr;

  Agent(const Agent& other)
      : id(other.id),
        path(other.path),
        terminalPath(other.terminalPath),
        terminalPathActive(other.terminalPathActive),
        taskAssignments(other.taskAssignments),
        taskPaths(other.taskPaths),
        intraPrecedenceConstraints(other.intraPrecedenceConstraints),
        localTaskIndexCache(other.localTaskIndexCache),
        intraPrecedenceDirty(other.intraPrecedenceDirty),
        pathPlanner(clonePlanner(other.pathPlanner, other.id)) {}
  Agent(Agent&&) noexcept = default;
  Agent& operator=(const Agent& other) {
    if (this == &other) {
      return *this;
    }
    const int oldId = id;
    id = other.id;
    path = other.path;
    terminalPath = other.terminalPath;
    terminalPathActive = other.terminalPathActive;
    taskPaths = other.taskPaths;
    taskAssignments = other.taskAssignments;
    intraPrecedenceConstraints = other.intraPrecedenceConstraints;
    localTaskIndexCache = other.localTaskIndexCache;
    intraPrecedenceDirty = other.intraPrecedenceDirty;
    if (other.pathPlanner == nullptr) {
      pathPlanner.reset();
    } else if (pathPlanner != nullptr && oldId == other.id &&
               pathPlanner->getName() == other.pathPlanner->getName()) {
      // Reuse existing planner allocation when type/agent identity match.
      pathPlanner->copyStateFrom(*other.pathPlanner);
    } else {
      pathPlanner = clonePlanner(other.pathPlanner, other.id);
    }
    return *this;
  }
  Agent& operator=(Agent&&) noexcept = default;
  Agent(const Instance& instance, int id) : id(id) {
    pathPlanner = std::make_shared<MultiLabelSpaceTimeAStar>(instance, id);
    localTaskIndexCache.assign(instance.getTasksNum(), UNASSIGNED);
  }
  ~Agent() = default;

  int getLocalTaskIndex(int globalTask) const {
    if (globalTask < 0) {
      PLOGE << "Agent::getLocalTaskIndex: invalid negative task "
            << globalTask << " for agent " << id << "\n";
      assert(false);
      return UNASSIGNED;
    }
    if (globalTask >= 0 && globalTask < (int)localTaskIndexCache.size()) {
      const int cached = localTaskIndexCache[globalTask];
      if (cached >= 0 && cached < (int)taskAssignments.size() &&
          taskAssignments[cached] == globalTask) {
        return cached;
      }
    }
    for (int i = 0; i < (int)taskAssignments.size(); i++) {
      const int task = taskAssignments[i];
      if (task >= 0 && task < (int)localTaskIndexCache.size()) {
        localTaskIndexCache[task] = i;
      }
      if (task == globalTask) {
        return i;
      }
    }
    if (globalTask >= 0 && globalTask < (int)localTaskIndexCache.size()) {
      localTaskIndexCache[globalTask] = UNASSIGNED;
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
    intraPrecedenceDirty = false;
  }

  // Inserts intra-agent precedence edges around a newly inserted task.
  // Precondition: `taskAssignments[taskPosition] == task` and taskPosition is
  // the position in the current (already-updated) assignment order.
  // Effect: if neighbors exist, replaces previousTask->nextTask with
  // previousTask->task and task->nextTask.
  void insertIntraAgentPrecedenceConstraint(int task, int taskPosition) {
    const int assignmentSize = (int)taskAssignments.size();
    if (taskPosition < 0 || taskPosition >= assignmentSize) {
      PLOGE << "insertIntraAgentPrecedenceConstraint: invalid taskPosition "
            << taskPosition << " for agent " << id << " with "
            << assignmentSize << " assigned tasks\n";
      assert(false);
      return;
    }
    if (taskAssignments[taskPosition] != task) {
      PLOGE << "insertIntraAgentPrecedenceConstraint: task mismatch at "
            << "position " << taskPosition << " for agent " << id
            << " (expected " << task << ", found "
            << taskAssignments[taskPosition] << ")\n";
      assert(false);
      return;
    }
    intraPrecedenceDirty = true;
  }

  void clearIntraAgentPrecedenceConstraint(int task) {
#ifndef NDEBUG
    const auto taskIt = std::find(taskAssignments.begin(), taskAssignments.end(),
                                  task);
    if (taskIt == taskAssignments.end()) {
      PLOGE << "clearIntraAgentPrecedenceConstraint: task " << task
            << " not found for agent " << id << "\n";
      return;
    }
#endif
    intraPrecedenceDirty = true;
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

struct TaskBaselineMetrics {
  double oldExposure = 0.0;
  int oldWait = 0;
  bool valid = false;
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
    return mapf_pc_lns::toCoordinate(id, numOfCols);
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
               int endTime, int distance, double relation)
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
    bool operator()(const pair<double, RelatedTasks>& task1,
                    const pair<double, RelatedTasks>& task2) {
      return task1.first > task2.first;
    }
  };

  // Comparator for custom Related Tasks struct
  struct RelatedTasksComparator {
    bool operator()(const RelatedTasks& task1,
                    const RelatedTasks& task2) const {
      return task1.task < task2.task;
    }
  };
};

using pqRelatedTasks = std::priority_queue<pair<double, RelatedTasks>,
                                           vector<pair<double, RelatedTasks>>,
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
      improvedAccepted, downgradedAccepted, couldNotFind, cascadeAborted;
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
    cascadeAborted.assign(numDestroyHeuristics, 0);
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
    // Fallback strategy when initialSolutionStrategy fails.
    // Supported: "greedy", "none".
    string initialSolutionFallback = "greedy";
    // Final-goal occupancy policy used when reserving completed task paths in
    // constraint tables:
    // - "stay": reserve final goal indefinitely (legacy behavior)
    // - "tail": reserve final goal for goalTailSteps and then release
    // - "reposition": MVP alias of tail-release (explicit move-out/return is
    //   not yet modeled in task paths)
    // - "reposition_true": explicit terminal move-out/return path (Phase B+)
    string goalOccupationMode = "stay";
    int goalTailSteps = 0;
    // Phase-D knobs for true reposition performance.
    int repositionMaxCandidates = 12;
    // 0 = scan full path horizon when detecting demand at final goals.
    int repositionDemandLookahead = 0;
    // Additional timesteps beyond active service horizon to reserve
    // terminalPath occupancy in constraint tables.
    int repositionReservationSlack = 64;
    // Emit per-(agent,task) greedy low-level segment diagnostics.
    bool greedySegmentDiagnostics = false;
    int greedySegmentDiagnosticsTopK = 10;
    string destroyHeuristic;
    string acceptanceCriteria;
    string regretType;
    bool incrementalRegret = false;
    // Add occupancy constraints from non-ancestor agents during repair
    // planning. This reduces collision-heavy candidates at the cost of extra
    // low-level constraint processing.
    bool repairIncludeNonAncestorAgents = true;
    // Candidate insertion budget per (task, agent) regret evaluation.
    // 0 means evaluate all candidate positions.
    int regretCandidateTopK = 0;
    // Hard cap on precedence successor-closure growth in prepareNextIteration.
    // If maxCascadeTasks == 0, use:
    //   max(maxCascadeFactor * neighborSize, neighborSize + 10)
    // If maxCascadeFactor <= 0 and maxCascadeTasks == 0, cap is disabled.
    double maxCascadeFactor = 3.0;
    int maxCascadeTasks = 0;
    // Enable touched-agent-scoped rollback restore.
    // Full-copy restore remains the default/fallback behavior.
    bool partialSolutionRestore = false;
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
    // ALNS-only warmup gate: MarketTatonnement is ineligible until at least
    // this many market price updates have been performed.
    // 0 disables warmup gating.
    int destroyWarmupUpdates = 3;
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
  struct RegretWorkspace;
  struct LowLevelSearchStats {
    uint64_t calls = 0;
    uint64_t expanded = 0;
    uint64_t generated = 0;
    uint64_t found = 0;
    uint64_t timeout = 0;
    uint64_t searchExhausted = 0;
    uint64_t invalidInput = 0;
    uint64_t budgetExhausted = 0;
    uint64_t unknown = 0;
  };

  struct RegretEvalStats {
    int64_t recomputeCalls = 0;
    int64_t tasksEvaluated = 0;
    int64_t agentEvaluations = 0;
    int64_t candidateInsertionsTried = 0;
    int64_t candidateInsertionsFeasible = 0;
    int64_t workspaceAgentsCloned = 0;
    int64_t workspaceMaxClonedPerTask = 0;
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

  struct CascadeStats {
    int64_t prepareCalls = 0;
    int64_t budgetAborts = 0;
    int64_t seedTasksSum = 0;
    int64_t closureTasksSum = 0;
    int64_t closureAddedSum = 0;
    int64_t closureTasksMax = 0;
    int64_t closureAddedMax = 0;

    void reset() { *this = CascadeStats(); }
  };

  struct TerminalRepositionStats {
    int64_t replansRequested = 0;
    int64_t agentsEvaluated = 0;
    int64_t agentsPlanned = 0;
    int64_t skippedNoDemand = 0;
    int64_t planningFailures = 0;
    int64_t candidateCacheHits = 0;
    int64_t candidateCacheMisses = 0;

    void reset() { *this = TerminalRepositionStats(); }
  };

  struct SolutionRestoreStats {
    int64_t restoreCalls = 0;
    int64_t fullRestores = 0;
    int64_t partialRestores = 0;
    int64_t partialRestoreFallbacks = 0;
    int64_t partialAgentsRestored = 0;

    void reset() { *this = SolutionRestoreStats(); }
  };

 private:
  int numOfIterations_;
  bool incrementalRegret_ = false;
  LowLevelPlannerType lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  double lowLevelSegmentTimeout_ = 600.0;
  bool plannerParityCheck_ = false;
  int plannerParityMaxLogs_ = 10;
  bool partialSolutionRestore_ = false;
  struct MarketState : LNSParams::Market {
    // Runtime-only market state. Configuration fields are inherited from
    // LNSParams::Market to avoid duplicated declarations.
    int64_t acceptedCounter = 0;
    int64_t updateCounter = 0;
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

  void buildFullPrecedenceConstraints(
      vector<pair<int, int>>& out,
      bool includeIntraConstraints = true) const;
  vector<pair<int, int>> buildFullPrecedenceConstraints(
      bool includeIntraConstraints = true) const;
  void computeTaskScheduleMetricsFromIndex(
      const vector<int>& taskPosByTask, vector<TaskScheduleMetrics>& perTask,
      vector<double>* blockedWaitSum = nullptr) const;
  double computeSolutionPrecedenceWaitFromIndex(
      const vector<int>& taskPosByTask) const;
  AgentTaskPath runLowLevelSearch(SingleAgentSolver& solver,
                                  ConstraintTable& constraintTable,
                                  int startTime, int stage, int lowerBound);
  double elapsedRuntimeSec() const;
  double remainingRuntimeBudgetSec() const;
  bool runtimeBudgetExhausted() const;
  int cascadeTaskBudget() const;
  void clearNeighborhood();
  vector<int> buildRollbackAgentHints(const vector<int>& baseAgents) const;
  void restoreSolutionFromPrevious(const vector<int>* agentHints = nullptr);

 protected:
  ALNS adaptiveLNS_;
  int neighborSize_;
  int regretCandidateTopK_ = 0;
  double maxCascadeFactor_ = 3.0;
  int maxCascadeTasks_ = 0;
  bool repairIncludeNonAncestorAgents_ = true;
  bool useTerminalPathsInValidation_ = false;
  bool lastPrepareAbortedByCascade_ = false;
  int lastPrepareSeedTasks_ = 0;
  int lastPrepareClosureTasks_ = 0;
  int lastPrepareClosureAdded_ = 0;
  vector<int> lastPrepareAffectedAgents_;
  vector<int> iterationRollbackHintAgents_;
  CascadeStats cascadeStats_;
  SolutionRestoreStats solutionRestoreStats_;
  vector<pair<int, int>> fullPrecedenceConstraintsScratch_;
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
  uint64_t lowLevelFound_ = 0;
  uint64_t lowLevelTimeout_ = 0;
  uint64_t lowLevelSearchExhausted_ = 0;
  uint64_t lowLevelInvalidInput_ = 0;
  uint64_t lowLevelBudgetExhausted_ = 0;
  uint64_t lowLevelUnknown_ = 0;
  SingleAgentSolver::SearchOutcome lastLowLevelOutcome_ =
      SingleAgentSolver::SearchOutcome::unknown;
  double lastLowLevelRemainingBudgetSec_ = 0.0;
  double lastLowLevelEffectiveTimeoutSec_ = 0.0;
  double initialTemperature_ = 0.0;
  double maxTemperature_ = std::numeric_limits<double>::infinity();
  double greatDelugeDecay_ = 0.0;
  double timeLimit_, initialSolutionRuntime_ = 0, temperature_ = 100,
                     coolingCoefficient_ = 0.99975,
                     heatingCoefficient_ = 1.00025, tolerance_ = 5,
                     shawDistanceWeight_ = 9, shawTemporalWeight_ = 3,
                     lnsConflictWeight_ = 0.75, lnsCostWeight_ = 0.25;
  Time::time_point plannerStartTime_;
  string goalOccupationMode_ = "stay";
  int goalTailSteps_ = 0;
  int repositionMaxCandidates_ = 12;
  int repositionDemandLookahead_ = 0;
  int repositionReservationSlack_ = 64;
  bool greedySegmentDiagnostics_ = false;
  int greedySegmentDiagnosticsTopK_ = 10;
  mutable unordered_map<int, vector<int>> parkingCandidatesCache_;
  TerminalRepositionStats terminalRepositionStats_;
  string initialSolutionRequested_;
  string initialSolutionEffective_;
  bool initialSolutionFallbackUsed_ = false;
  string initialSolutionFallbackReason_;

 public:
  double runtime = 0;
  int numOfFailures = 0, sumOfCosts = 0;
  vector<IterationStats> iterationStats;
  string initialSolutionStrategy, initialSolutionFallback, destroyHeuristic,
      acceptanceCriteria,
      regretType;

 private:
  void reservePathWithGoalPolicy(ConstraintTable& constraintTable,
                                 const AgentTaskPath& path,
                                 bool isFinalTask) const;
  int getServiceOccupancyEndExclusive(int agent) const;
  void reserveTerminalPathIfActive(ConstraintTable& constraintTable,
                                   int agent) const;
  int computeActiveServiceHorizon() const;
  bool didAgentServicePathChange(int agent) const;
  vector<int> selectTerminalReplanAgents(
      const vector<int>& candidateAgents) const;
  const vector<int>& getParkingCandidatesForGoal(int finalGoal);

 public:

  LNS(int numOfIterations, const Instance& instance,
      const LNSParams& parameters);

  inline const Instance& getInstance() const { return instance_; }

  bool run();

  bool buildGreedySolution();
  bool buildPrioritizedInitialSolution();
  // Precedence-feasible initial solution that ignores inter-agent collisions.
  bool buildGreedySolutionPrecedenceOnly();
  bool buildGreedySolutionWithMAPFPC(const string& variant);
  bool planTerminalReposition(const vector<int>& agentsToPlan,
                              bool fullRebuild);

  bool prepareNextIteration();
  void markResolved(int globalTask);
  // Patches the task paths of an agent such that the begin times and end times match up
  void patchAgentTaskPaths(int agent, int taskPosition);

  void printPaths() const;
  enum class OccupancySource { undefined, service, terminal };
  OccupancySource getAgentOccupancySourceAt(
      int agent, int timestep, bool includeTerminal = true) const;
  // Returns an agent's occupied location at timestep.
  // If includeTerminal is false, occupancy follows service + goal policy
  // (stay/tail/reposition), excluding explicit terminalPath.
  int getAgentLocationAt(int agent, int timestep,
                         bool includeTerminal = true) const;
  // Returns the occupancy horizon (exclusive upper bound) for collision checks.
  // Horizon is policy-aware for service occupancy and optionally includes
  // explicit terminalPath when includeTerminal is true.
  int getAgentOccupancyHorizon(int agent,
                               bool includeTerminal = true) const;
  bool validateSolution(ConflictMap* conflictedTasks = nullptr);
  void addConflictingTask(int agent, int timestep, ConflictMap* out) const;

  bool buildConstraintTable(ConstraintTable& constraintTable, int task);
  bool buildConstraintTable(
      ConstraintTable& constraintTable, int task,
      const vector<pair<int, int>>& precedenceConstraints);

  bool buildConstraintTable(ConstraintTable& constraintTable,
                            TaskRegretPacket taskPacket, int taskLocation,
                            RegretWorkspace& workspace,
                            vector<pair<int, int>>* precedenceConstraints,
                            bool findingNextTask = false);

  int extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue);
  int extractOldLocalTaskIndex(int task, const vector<int>& oldTaskQueue,
                               const vector<int>& newTaskQueue);
  vector<char> reachableSet(int source, const vector<vector<int>>& edgeList);

  bool computeRegret();
  bool computeRegretForTask(int task);
  bool computeRegretForTask(
      int task,
      const vector<pair<int, int>>& fullPrecedenceConstraints);
  void computeRegretForTaskWithAgent(
      TaskRegretPacket regretPacket, RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics& baselineMetrics,
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
      RegretWorkspace& workspace,
      vector<pair<int, int>>* precedenceConstraints,
      const TaskBaselineMetrics* baselineMetrics = nullptr,
      SingleAgentSolver* reusablePlanner = nullptr,
      bool rollbackAfter = false);
  bool insertBestRegretTask(TaskRegretPacket bestRegretPacket);

  const Solution& getSolution() const { return solution_; }
  const string& getInitialSolutionRequested() const {
    return initialSolutionRequested_;
  }
  const string& getInitialSolutionEffective() const {
    return initialSolutionEffective_;
  }
  bool wasInitialSolutionFallbackUsed() const {
    return initialSolutionFallbackUsed_;
  }
  const string& getInitialSolutionFallbackReason() const {
    return initialSolutionFallbackReason_;
  }
  const ALNS& getAdaptiveLNSRef() const { return adaptiveLNS_; }
  ALNS getAdaptiveLNS() const { return adaptiveLNS_; }
  bool lastPrepareAbortedByCascade() const {
    return lastPrepareAbortedByCascade_;
  }
  int lastPrepareClosureAdded() const { return lastPrepareClosureAdded_; }
  const CascadeStats& getCascadeStatsRef() const { return cascadeStats_; }
  int getCascadeTaskBudget() const { return cascadeTaskBudget(); }
  const SolutionRestoreStats& getSolutionRestoreStats() const {
    return solutionRestoreStats_;
  }
  bool isPartialSolutionRestoreEnabled() const { return partialSolutionRestore_; }
  const TerminalRepositionStats& getTerminalRepositionStats() const {
    return terminalRepositionStats_;
  }
  LowLevelSearchStats getLowLevelSearchStats() const {
    return {lowLevelCalls_,         lowLevelExpanded_,      lowLevelGenerated_,
            lowLevelFound_,         lowLevelTimeout_,       lowLevelSearchExhausted_,
            lowLevelInvalidInput_,  lowLevelBudgetExhausted_,
            lowLevelUnknown_};
  }
  const char* getLastLowLevelOutcomeName() const {
    return SingleAgentSolver::searchOutcomeName(lastLowLevelOutcome_);
  }
  double getLastLowLevelRemainingBudgetSec() const {
    return lastLowLevelRemainingBudgetSec_;
  }
  double getLastLowLevelEffectiveTimeoutSec() const {
    return lastLowLevelEffectiveTimeoutSec_;
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
  void conflictRemoval(const ConflictMap* potentialNeighborhood);
  void shawRemoval(int prioritySize);
  void precedenceWaitRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void lowSlackRemoval(const ConflictMap* potentialNeighborhood = nullptr);
  void marketTatonnementRemoval(
      const ConflictMap* potentialNeighborhood = nullptr);
  void alnsRemoval(const ConflictMap* potentialNeighborhood);

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
  bool passMarketAcceptanceGuards(double previousPressure,
                                  double candidatePressure,
                                  double previousWait,
                                  double candidateWait,
                                  bool candidateIsWorse) const;
  void updateMarketStateFromCurrentSolution();
  void maybeUpdateMarketState(bool accepted);
  double computeMarketExposureFromPath(const AgentTaskPath& taskPath,
                                       bool normalized) const;
  void buildMarketDemandFromCurrentOccupancy(
      unordered_map<uint64_t, int>& vertexDemand,
      unordered_map<uint64_t, int>& edgeDemand) const;
  int computeTaskPrecedenceWaitFromState(
      int task, int taskLocation, const vector<vector<int>>& agentTaskAssignments,
      const vector<vector<AgentTaskPath>>& agentTaskPaths) const;
  int computeTaskPrecedenceWaitFromWorkspace(int task, int taskLocation,
                                             const RegretWorkspace& workspace) const;
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
