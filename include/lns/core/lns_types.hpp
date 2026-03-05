#pragma once

#include <plog/Log.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_set>
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
  collisionSoftRemoval = 7,
  failureSoftRemoval = 8,
  destroyHeuristicCount = 9
};

struct MarketStats {
  int64_t updates = 0;
  int64_t destroyWarmupSkipped = 0;
  int64_t destroyUnstableSkipped = 0;
  int64_t contendedResources = 0;
  double meanPriceContended = 0.0;
  double maxPrice = 0.0;
  double topPriceMassFrac = 0.0;
  double priceRelL1Delta = 0.0;
  double priceRelL1DeltaEma = 0.0;
  double topPriceMassDelta = 0.0;
  double topPriceMassDeltaEma = 0.0;
  double contendedJaccard = 1.0;
  double contendedJaccardEma = 1.0;
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
    // This eager path explicitly appends an intra-edge into the diagnostic
    // cache, so this operation itself does not leave pending lazy updates.
    // Note: intra-edge correctness for planning/validation is derived from
    // taskAssignments order (not this cached vector) in buildFullPrecedenceConstraints.
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
  // Task-level data for reporting and diagnostics on the incumbent solution.
  vector<vector<int>> agentTaskAssignments;
  vector<vector<AgentTaskPath>> agentTaskPaths;
  vector<int> taskAgentMap;

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

    if (!agentTaskAssignments.empty()) {
      result += "TASK ASSIGNMENTS\n";
      for (int agent = 0; agent < (int)agentTaskAssignments.size(); agent++) {
        result += "Agent " + std::to_string(agent) + "\n";
        const auto& assignments = agentTaskAssignments[agent];
        for (int i = 0; i < (int)assignments.size(); i++) {
          if (i > 0) {
            result += ", ";
          }
          result += std::to_string(assignments[i]);
        }
        result += "\n";
      }
    }

    if (!agentTaskPaths.empty()) {
      result += "TASK PATHS\n";
      for (int agent = 0; agent < (int)agentTaskPaths.size(); agent++) {
        result += "Agent " + std::to_string(agent) + "\n";
        const auto& taskPaths = agentTaskPaths[agent];
        const bool hasAssignments =
            agent < (int)agentTaskAssignments.size();
        for (int localTask = 0; localTask < (int)taskPaths.size();
             localTask++) {
          int taskId = -1;
          if (hasAssignments &&
              localTask < (int)agentTaskAssignments[agent].size()) {
            taskId = agentTaskAssignments[agent][localTask];
          }
          if (taskId >= 0) {
            result += "Task " + std::to_string(taskId) + ": ";
          } else {
            result += "Task #" + std::to_string(localTask) + ": ";
          }

          const AgentTaskPath& taskPath = taskPaths[localTask];
          if (taskPath.empty()) {
            result += "(empty)\n";
            continue;
          }
          for (int step = 0; step < (int)taskPath.path.size(); step++) {
            pair<int, int> coord = getCoordinate(taskPath.path[step].location);
            result += "(" + std::to_string(coord.first) + ", " +
                      std::to_string(coord.second) + ")@" +
                      std::to_string(taskPath.beginTime + step);
            if (taskPath.path[step].isGoal) {
              result += "*";
            }
            if (step != (int)taskPath.path.size() - 1) {
              result += " -> ";
            }
          }
          result += "\n";
        }
      }
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
      if (assignments.empty()) {
        // Keep a canonical idle path for zero-task agents so downstream code
        // that expects a non-empty joined path (e.g., NRR seed capture) does
        // not fail on otherwise valid assignments.
        joined.path.push_back(
            PathEntry{false, agents[agent].pathPlanner->startLocation});
        staged.emplace_back(agent, std::move(joined));
        continue;
      }
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
      improvedAccepted, downgradedAccepted, couldNotFind, cascadeAborted,
      proposedBetter, proposedEqual, proposedWorse, acceptedWorse;
  vector<double> deltaSocAll, deltaSocAccepted, bestUpdateDeltaSocSum;

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
    proposedBetter.assign(numDestroyHeuristics, 0);
    proposedEqual.assign(numDestroyHeuristics, 0);
    proposedWorse.assign(numDestroyHeuristics, 0);
    acceptedWorse.assign(numDestroyHeuristics, 0);
    deltaSocAll.assign(numDestroyHeuristics, 0.0);
    deltaSocAccepted.assign(numDestroyHeuristics, 0.0);
    bestUpdateDeltaSocSum.assign(numDestroyHeuristics, 0.0);
  }
};
