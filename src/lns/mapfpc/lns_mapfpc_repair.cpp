#include "lns.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

#include "lns_internal_helpers.hpp"
#include "lns_mapfpc_internal.hpp"
#include "lns_temp_file_guard.hpp"

using lns_mapfpc_internal::hashAssignments;

bool LNS::runFixedAssignmentMapfpcRepair() {
  vector<vector<int>> assignments(instance_.getAgentNum());
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    assignments[agent] = solution_.agents[agent].taskAssignments;
  }

  vector<vector<pair<int, int>>> pendingInsertions(instance_.getAgentNum());
  vector<char> alreadyAssigned(instance_.getTasksNum(), 0);
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task : assignments[agent]) {
      if (task >= 0 && task < instance_.getTasksNum()) {
        alreadyAssigned[task] = 1;
      }
    }
  }

  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    const int task = conflict.task;
    if (task < 0 || task >= instance_.getTasksNum()) {
      PLOGE << "mapfpc_fixed_repair: invalid removed task id " << task << "\n";
      return false;
    }
    if (alreadyAssigned[task]) {
      continue;
    }

    int owner = UNASSIGNED;
    if (task < (int)previousSolution_.taskAgentMap.size()) {
      owner = previousSolution_.taskAgentMap[task];
    }
    if (owner == UNASSIGNED && task < (int)solution_.taskAgentMap.size()) {
      owner = solution_.taskAgentMap[task];
    }
    if (owner == UNASSIGNED) {
      owner = conflict.agent;
    }
    if (owner < 0 || owner >= instance_.getAgentNum()) {
      PLOGE << "mapfpc_fixed_repair: could not resolve owner for task "
            << task << "\n";
      return false;
    }

    int previousPos = previousSolution_.getLocalTaskIndex(owner, task);
    if (previousPos < 0) {
      previousPos = (int)assignments[owner].size();
    }
    pendingInsertions[owner].emplace_back(previousPos, task);
    alreadyAssigned[task] = 1;
  }

  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    auto& inserts = pendingInsertions[agent];
    if (inserts.empty()) {
      continue;
    }
    std::sort(inserts.begin(), inserts.end(),
              [](const pair<int, int>& lhs, const pair<int, int>& rhs) {
                if (lhs.first != rhs.first) {
                  return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
              });
    for (const auto& [targetPos, task] : inserts) {
      const int boundedPos = std::max(0, std::min(targetPos, (int)assignments[agent].size()));
      assignments[agent].insert(assignments[agent].begin() + boundedPos, task);
    }
  }

  vector<char> seen(instance_.getTasksNum(), 0);
  int assignedCount = 0;
  for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
    for (int task : assignments[agent]) {
      if (task < 0 || task >= instance_.getTasksNum()) {
        PLOGE << "mapfpc_fixed_repair: assignment task id out of range: "
              << task << "\n";
        return false;
      }
      if (seen[task]) {
        PLOGE << "mapfpc_fixed_repair: duplicate assignment for task " << task
              << "\n";
        return false;
      }
      seen[task] = 1;
      assignedCount++;
    }
  }
  if (assignedCount != instance_.getTasksNum()) {
    PLOGE << "mapfpc_fixed_repair: incomplete assignment (" << assignedCount
          << "/" << instance_.getTasksNum() << ")\n";
    return false;
  }

  const string sourceLabel =
      "repair_mapfpc_fixed(" + repairMapfpcSolver_ + ",removed=" +
      std::to_string(lnsNeighborhood_.immutableRemovedTasks.size()) + ")";
  return runMAPFPCOnAssignments(assignments, repairMapfpcSolver_,
                                repairMapfpcTimeoutSec_, sourceLabel);
}

bool LNS::runNeighborhoodFixedMapfpcRepair() {
  namespace fs = std::filesystem;
  const int numAgents = instance_.getAgentNum();
  const int numTasks = instance_.getTasksNum();
  if ((int)previousSolution_.agents.size() != numAgents) {
    PLOGW << "mapfpc_neighborhood_fixed: previous solution missing/invalid; "
             "falling back to global fixed MAPF-PC repair\n";
    return runFixedAssignmentMapfpcRepair();
  }

  vector<vector<int>> assignments(numAgents);
  vector<char> seenTasks(numTasks, 0);
  int assignedCount = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    assignments[agent] = previousSolution_.agents[agent].taskAssignments;
    for (int task : assignments[agent]) {
      if (task < 0 || task >= numTasks) {
        PLOGE << "mapfpc_neighborhood_fixed: assignment task out of range: "
              << task << "\n";
        return false;
      }
      if (seenTasks[task]) {
        PLOGE << "mapfpc_neighborhood_fixed: duplicate task in assignment: "
              << task << "\n";
        return false;
      }
      seenTasks[task] = 1;
      assignedCount++;
    }
  }
  if (assignedCount != numTasks) {
    PLOGW << "mapfpc_neighborhood_fixed: assignment incomplete (" << assignedCount
          << "/" << numTasks << "); falling back to global fixed repair\n";
    return runFixedAssignmentMapfpcRepair();
  }

  vector<char> mutableMask(numAgents, 0);
  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    const int task = conflict.task;
    int owner = UNASSIGNED;
    if (task >= 0 && task < (int)previousSolution_.taskAgentMap.size()) {
      owner = previousSolution_.taskAgentMap[task];
    }
    if (owner == UNASSIGNED) {
      owner = conflict.agent;
    }
    if (owner >= 0 && owner < numAgents) {
      mutableMask[owner] = 1;
    }
  }
  vector<int> mutableAgents;
  for (int agent = 0; agent < numAgents; agent++) {
    if (mutableMask[agent]) {
      mutableAgents.push_back(agent);
    }
  }
  if (mutableAgents.empty()) {
    PLOGW << "mapfpc_neighborhood_fixed: no mutable agents resolved from "
             "neighborhood; falling back to global fixed repair\n";
    return runFixedAssignmentMapfpcRepair();
  }

  vector<vector<int>> pathLocations(numAgents);
  vector<vector<int>> pathTimestamps(numAgents);
  for (int agent = 0; agent < numAgents; agent++) {
    const auto& joined = previousSolution_.agents[agent].path;
    if (joined.empty()) {
      PLOGW << "mapfpc_neighborhood_fixed: missing joined path for agent "
            << agent << "; falling back to global fixed repair\n";
      return runFixedAssignmentMapfpcRepair();
    }
    pathLocations[agent].reserve(joined.path.size());
    for (const auto& entry : joined.path) {
      pathLocations[agent].push_back(entry.location);
    }

    const auto& taskPaths = previousSolution_.agents[agent].taskPaths;
    if ((int)taskPaths.size() == (int)assignments[agent].size()) {
      pathTimestamps[agent].reserve(taskPaths.size());
      for (const auto& segment : taskPaths) {
        if (segment.empty()) {
          PLOGW << "mapfpc_neighborhood_fixed: empty task segment for agent "
                << agent << "; falling back to global fixed repair\n";
          return runFixedAssignmentMapfpcRepair();
        }
        pathTimestamps[agent].push_back(segment.endTime());
      }
    } else if ((int)joined.timeStamps.size() == (int)assignments[agent].size()) {
      pathTimestamps[agent] = joined.timeStamps;
    } else {
      PLOGW << "mapfpc_neighborhood_fixed: timestamp count mismatch for agent "
            << agent << "; falling back to global fixed repair\n";
      return runFixedAssignmentMapfpcRepair();
    }

    if (assignments[agent].empty() && pathTimestamps[agent].empty()) {
      pathTimestamps[agent].push_back(0);
    }
    int prevTs = -1;
    for (int ts : pathTimestamps[agent]) {
      if (ts < 0 || ts >= (int)pathLocations[agent].size() || ts < prevTs) {
        PLOGW << "mapfpc_neighborhood_fixed: invalid timestamp for agent "
              << agent << "; falling back to global fixed repair\n";
        return runFixedAssignmentMapfpcRepair();
      }
      prevTs = ts;
    }
  }

  std::error_code tempDirEc;
  fs::path tempDir = fs::temp_directory_path(tempDirEc);
  if (tempDirEc || tempDir.empty()) {
    tempDir = fs::current_path(tempDirEc);
  }
  if (tempDirEc || tempDir.empty()) {
    PLOGE << "mapfpc_neighborhood_fixed: failed to resolve temp directory\n";
    return false;
  }
  const auto token =
      std::to_string(seed_) + "_" +
      std::to_string((long long)Time::now().time_since_epoch().count());
  const fs::path assignmentPath = tempDir / ("mapfpc_neigh_assign_" + token + ".txt");
  const fs::path mutableAgentsPath =
      tempDir / ("mapfpc_neigh_mutable_" + token + ".txt");
  const fs::path initialPathsPath =
      tempDir / ("mapfpc_neigh_paths_" + token + ".txt");
  lns_temp_file::ScopedPathCleanup tempCleanup;
  tempCleanup.add(assignmentPath);
  tempCleanup.add(mutableAgentsPath);
  tempCleanup.add(initialPathsPath);

  string assignmentError;
  if (!writeMAPFPCAssignmentFile(assignments, assignmentPath.string(),
                                 assignmentError)) {
    PLOGE << "mapfpc_neighborhood_fixed: failed to write assignment file: "
          << assignmentError << "\n";
    return false;
  }

  {
    std::ofstream outMutable(mutableAgentsPath);
    if (!outMutable.is_open()) {
      PLOGE << "mapfpc_neighborhood_fixed: failed to open mutable agents file\n";
      return false;
    }
    outMutable << "MUTABLE_AGENTS\n";
    for (int i = 0; i < (int)mutableAgents.size(); i++) {
      if (i > 0) {
        outMutable << ' ';
      }
      outMutable << mutableAgents[i];
    }
    outMutable << "\n";
    outMutable.flush();
    if (!outMutable.good()) {
      PLOGE << "mapfpc_neighborhood_fixed: failed writing mutable agents file\n";
      return false;
    }
  }

  {
    std::ofstream outPaths(initialPathsPath);
    if (!outPaths.is_open()) {
      PLOGE << "mapfpc_neighborhood_fixed: failed to open initial paths file\n";
      return false;
    }
    outPaths << "AGENT_PATHS\n";
    for (int agent = 0; agent < numAgents; agent++) {
      outPaths << "Agent " << agent << "\n";
      outPaths << "locations:";
      for (int loc : pathLocations[agent]) {
        outPaths << " " << loc;
      }
      outPaths << "\n";
      outPaths << "timestamps:";
      for (int ts : pathTimestamps[agent]) {
        outPaths << " " << ts;
      }
      outPaths << "\n";
    }
    outPaths.flush();
    if (!outPaths.good()) {
      PLOGE << "mapfpc_neighborhood_fixed: failed writing initial paths file\n";
      return false;
    }
  }

  const string sourceLabel =
      "repair_mapfpc_neighborhood_fixed(" + repairMapfpcSolver_ +
      ",mutable_agents=" + std::to_string(mutableAgents.size()) +
      ",removed=" + std::to_string(lnsNeighborhood_.immutableRemovedTasks.size()) +
      ")";
  const bool success = runMAPFPCForAgentFile(
      instance_.getAgentTaskFName(), repairMapfpcSolver_, repairMapfpcTimeoutSec_,
      sourceLabel, assignmentPath.string(), mutableAgentsPath.string(),
      initialPathsPath.string());
  return success;
}

bool LNS::runNeighborhoodReassignGreedyMapfpcRepair() {
  const int numAgents = instance_.getAgentNum();
  const int numTasks = instance_.getTasksNum();
  if ((int)previousSolution_.agents.size() != numAgents) {
    PLOGW << "mapfpc_neighborhood_reassign_greedy: previous solution missing/"
             "invalid; falling back to global fixed MAPF-PC repair\n";
    return runFixedAssignmentMapfpcRepair();
  }

  vector<vector<int>> assignments(numAgents);
  vector<char> seenTasks(numTasks, 0);
  int assignedCount = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    assignments[agent] = previousSolution_.agents[agent].taskAssignments;
    for (int task : assignments[agent]) {
      if (task < 0 || task >= numTasks) {
        PLOGE << "mapfpc_neighborhood_reassign_greedy: assignment task out of "
                 "range: "
              << task << "\n";
        return false;
      }
      if (seenTasks[task]) {
        PLOGE << "mapfpc_neighborhood_reassign_greedy: duplicate task in "
                 "assignment: "
              << task << "\n";
        return false;
      }
      seenTasks[task] = 1;
      assignedCount++;
    }
  }
  if (assignedCount != numTasks) {
    PLOGW << "mapfpc_neighborhood_reassign_greedy: assignment incomplete ("
          << assignedCount << "/" << numTasks
          << "); falling back to global fixed repair\n";
    return runFixedAssignmentMapfpcRepair();
  }
  const vector<vector<int>> baseAssignments = assignments;

  vector<char> removedMask(numTasks, 0);
  vector<int> removedTasks;
  removedTasks.reserve(lnsNeighborhood_.immutableRemovedTasks.size());
  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    const int task = conflict.task;
    if (task < 0 || task >= numTasks) {
      PLOGE << "mapfpc_neighborhood_reassign_greedy: invalid removed task id "
            << task << "\n";
      return false;
    }
    if (!removedMask[task]) {
      removedMask[task] = 1;
      removedTasks.push_back(task);
    }
  }
  if (removedTasks.empty()) {
    PLOGI << "mapfpc_neighborhood_reassign_greedy: no removed tasks in "
             "neighborhood; reusing neighborhood-fixed MAPF-PC repair\n";
    return runNeighborhoodFixedMapfpcRepair();
  }

  auto assignmentIndex = buildTaskAssignmentIndex(assignments, numTasks);
  vector<char> mutableMask(numAgents, 0);
  for (const auto& [_, conflict] : lnsNeighborhood_.immutableRemovedTasks) {
    const int task = conflict.task;
    int owner = UNASSIGNED;
    if (task >= 0 && task < (int)assignmentIndex.owner.size()) {
      owner = assignmentIndex.owner[task];
    }
    if (owner == UNASSIGNED &&
        task >= 0 && task < (int)previousSolution_.taskAgentMap.size()) {
      owner = previousSolution_.taskAgentMap[task];
    }
    if (owner == UNASSIGNED) {
      owner = conflict.agent;
    }
    if (owner >= 0 && owner < numAgents) {
      mutableMask[owner] = 1;
    }
  }
  vector<int> mutableAgents;
  for (int agent = 0; agent < numAgents; agent++) {
    if (mutableMask[agent]) {
      mutableAgents.push_back(agent);
    }
  }
  if (mutableAgents.empty()) {
    PLOGW << "mapfpc_neighborhood_reassign_greedy: no mutable agents resolved "
             "from neighborhood; falling back to global fixed repair\n";
    return runFixedAssignmentMapfpcRepair();
  }

  for (int agent = 0; agent < numAgents; agent++) {
    auto& queue = assignments[agent];
    queue.erase(std::remove_if(queue.begin(), queue.end(),
                               [&](int task) {
                                 return task >= 0 && task < numTasks &&
                                        removedMask[task];
                               }),
                queue.end());
  }
  assignmentIndex = buildTaskAssignmentIndex(assignments, numTasks);

  vector<int> removedPlanningOrder;
  removedPlanningOrder.reserve(removedTasks.size());
  vector<char> removedSeen(numTasks, 0);
  for (int task : instance_.getInputPlanningOrderRef()) {
    if (task >= 0 && task < numTasks && removedMask[task] && !removedSeen[task]) {
      removedPlanningOrder.push_back(task);
      removedSeen[task] = 1;
    }
  }
  for (int task : removedTasks) {
    if (!removedSeen[task]) {
      removedPlanningOrder.push_back(task);
      removedSeen[task] = 1;
    }
  }

  const long long kInf = std::numeric_limits<long long>::max() / 4;
  const auto distanceOrInf = [&](int from, int to) {
    const int distance = instance_.getDistanceToGoal(to, from);
    if (distance >= MAX_TIMESTEP / 2) {
      return kInf;
    }
    return static_cast<long long>(distance);
  };

  for (int task : removedPlanningOrder) {
    const int taskLocation = instance_.getTaskLocations(task);
    long long bestScore = kInf;
    int bestAgent = UNASSIGNED;
    int bestPos = -1;

    for (int agent : mutableAgents) {
      const auto& queue = assignments[agent];
      int minPos = 0;
      int maxPos = (int)queue.size();
      for (int predecessor : instance_.getAncestorsRef(task)) {
        if (predecessor < 0 || predecessor >= numTasks) {
          continue;
        }
        if (assignmentIndex.owner[predecessor] == agent) {
          minPos = max(minPos, assignmentIndex.pos[predecessor] + 1);
        }
      }
      for (int successor : instance_.getSuccessorsRef(task)) {
        if (successor < 0 || successor >= numTasks) {
          continue;
        }
        if (assignmentIndex.owner[successor] == agent) {
          maxPos = min(maxPos, assignmentIndex.pos[successor]);
        }
      }
      if (minPos > maxPos) {
        continue;
      }

      for (int pos = minPos; pos <= maxPos; pos++) {
        const int prevLocation =
            (pos == 0) ? instance_.getStartLocationsRef()[agent]
                       : instance_.getTaskLocations(queue[pos - 1]);
        const long long prevToTask = distanceOrInf(prevLocation, taskLocation);
        if (prevToTask >= kInf) {
          continue;
        }
        long long score = prevToTask;
        if (pos < (int)queue.size()) {
          const int nextLocation = instance_.getTaskLocations(queue[pos]);
          const long long taskToNext = distanceOrInf(taskLocation, nextLocation);
          const long long prevToNext = distanceOrInf(prevLocation, nextLocation);
          if (taskToNext >= kInf || prevToNext >= kInf) {
            continue;
          }
          score += taskToNext - prevToNext;
        }
        if (score < bestScore ||
            (score == bestScore &&
             (bestAgent == UNASSIGNED || agent < bestAgent ||
              (agent == bestAgent && pos < bestPos)))) {
          bestScore = score;
          bestAgent = agent;
          bestPos = pos;
        }
      }
    }

    if (bestAgent == UNASSIGNED || bestPos < 0) {
      PLOGW << "mapfpc_neighborhood_reassign_greedy: no feasible greedy slot "
               "for removed task "
            << task << "; falling back to neighborhood-fixed MAPF-PC repair\n";
      return runNeighborhoodFixedMapfpcRepair();
    }
    assignments[bestAgent].insert(assignments[bestAgent].begin() + bestPos, task);
    auto& updatedQueue = assignments[bestAgent];
    for (int pos = bestPos; pos < (int)updatedQueue.size(); pos++) {
      const int queuedTask = updatedQueue[pos];
      if (queuedTask < 0 || queuedTask >= numTasks) {
        continue;
      }
      assignmentIndex.owner[queuedTask] = bestAgent;
      assignmentIndex.pos[queuedTask] = pos;
    }
  }

  vector<char> finalSeen(numTasks, 0);
  int finalCount = 0;
  for (int agent = 0; agent < numAgents; agent++) {
    for (int task : assignments[agent]) {
      if (task < 0 || task >= numTasks) {
        PLOGE << "mapfpc_neighborhood_reassign_greedy: final assignment task "
                 "out of range: "
              << task << "\n";
        return false;
      }
      if (finalSeen[task]) {
        PLOGE << "mapfpc_neighborhood_reassign_greedy: duplicate task after "
                 "greedy reassignment: "
              << task << "\n";
        return false;
      }
      finalSeen[task] = 1;
      finalCount++;
    }
  }
  if (finalCount != numTasks) {
    PLOGE << "mapfpc_neighborhood_reassign_greedy: final assignment incomplete ("
          << finalCount << "/" << numTasks << ")\n";
    return false;
  }

  const auto baseIndex = buildTaskAssignmentIndex(baseAssignments, numTasks);
  const auto finalIndex = buildTaskAssignmentIndex(assignments, numTasks);
  int allOwnerChanged = 0;
  int allPosChangedSameOwner = 0;
  int removedOwnerChanged = 0;
  int removedPosChangedSameOwner = 0;
  for (int task = 0; task < numTasks; task++) {
    const int beforeOwner = baseIndex.owner[task];
    const int afterOwner = finalIndex.owner[task];
    const int beforePos = baseIndex.pos[task];
    const int afterPos = finalIndex.pos[task];
    if (beforeOwner != afterOwner) {
      allOwnerChanged++;
      if (removedMask[task]) {
        removedOwnerChanged++;
      }
      continue;
    }
    if (beforePos != afterPos) {
      allPosChangedSameOwner++;
      if (removedMask[task]) {
        removedPosChangedSameOwner++;
      }
    }
  }
  const int removedMovedTotal = removedOwnerChanged + removedPosChangedSameOwner;
  std::ostringstream hashStream;
  hashStream << "0x" << std::hex << hashAssignments(assignments);
  PLOGI << "mapfpc_neighborhood_reassign_greedy: assignment delta "
        << "(removed=" << removedPlanningOrder.size()
        << ", moved_removed=" << removedMovedTotal
        << ", owner_changed_removed=" << removedOwnerChanged
        << ", pos_changed_removed=" << removedPosChangedSameOwner
        << ", owner_changed_all=" << allOwnerChanged
        << ", pos_changed_all=" << allPosChangedSameOwner
        << ", assignment_hash=" << hashStream.str() << ")";

  const string sourceLabel =
      "repair_mapfpc_neighborhood_reassign_greedy(" + repairMapfpcSolver_ +
      ",mutable_agents=" + std::to_string(mutableAgents.size()) +
      ",removed=" + std::to_string(removedPlanningOrder.size()) + ")";
  return runMAPFPCOnAssignments(assignments, repairMapfpcSolver_,
                                repairMapfpcTimeoutSec_, sourceLabel);
}

bool LNS::buildSeededSolutionFromMAPFPCLog(const string& logFilePath) {
  if (logFilePath.empty()) {
    PLOGE << "buildSeededSolutionFromMAPFPCLog: empty log path\n";
    return false;
  }
  initialSeedRuntimeFromLogSec_ = -1.0;
  {
    std::ifstream runtimeStream(logFilePath);
    if (runtimeStream.is_open()) {
      string line;
      double parsedRuntime = -1.0;
      auto trim = [](const string& input) -> string {
        const size_t first = input.find_first_not_of(" \t\r\n");
        if (first == string::npos) {
          return "";
        }
        const size_t last = input.find_last_not_of(" \t\r\n");
        return input.substr(first, last - first + 1);
      };
      auto parseNonNegativeFiniteDouble = [](const string& token,
                                             double& out) -> bool {
        std::istringstream stream(token);
        double value = -1.0;
        if (!(stream >> value)) {
          return false;
        }
        if (!std::isfinite(value) || value < 0.0) {
          return false;
        }
        out = value;
        return true;
      };
      while (std::getline(runtimeStream, line)) {
        const string trimmed = trim(line);
        const size_t runtimePos = line.find("Runtime");
        if (runtimePos == string::npos) {
          // MAPF-PC/PBS-CBS logs often encode runtime as
          //   Solved,<soc>,<runtime>,...
          //   Optimal,<soc>,<runtime>,...
          if (trimmed.rfind("Solved,", 0) == 0 ||
              trimmed.rfind("Optimal,", 0) == 0 ||
              trimmed.rfind("Timeout,", 0) == 0 ||
              trimmed.rfind("No solution,", 0) == 0) {
            std::vector<string> tokens;
            size_t start = 0;
            while (start <= trimmed.size()) {
              const size_t comma = trimmed.find(',', start);
              if (comma == string::npos) {
                tokens.push_back(trimmed.substr(start));
                break;
              }
              tokens.push_back(trimmed.substr(start, comma - start));
              start = comma + 1;
            }
            if (tokens.size() >= 3) {
              double runtimeValue = -1.0;
              if (parseNonNegativeFiniteDouble(trim(tokens[2]), runtimeValue)) {
                parsedRuntime = runtimeValue;
              }
            }
          }
          continue;
        }
        const size_t equalsPos = line.find('=', runtimePos);
        if (equalsPos != string::npos) {
          double runtimeValue = -1.0;
          if (parseNonNegativeFiniteDouble(
                  trim(line.substr(equalsPos + 1)), runtimeValue)) {
            parsedRuntime = runtimeValue;
          }
        }
      }
      if (parsedRuntime >= 0.0) {
        initialSeedRuntimeFromLogSec_ = parsedRuntime;
      }
    }
  }
  std::ifstream inputStream(logFilePath);
  if (!inputStream.is_open()) {
    PLOGE << "buildSeededSolutionFromMAPFPCLog: failed to open '" << logFilePath
          << "'\n";
    return false;
  }
  return parseMAPFPCStreamIntoSolution(inputStream, logFilePath);
}

bool LNS::runPostMAPFPCRefinement() {
  vector<vector<int>> assignments(instance_.getAgentNum());
  string assignmentSource = postRefineAssignmentSource_;

  if (assignmentSource == "log") {
    if (postRefineAssignmentLog_.empty()) {
      PLOGE << "post_refine_mapfpc: assignment source is 'log' but no log path "
               "was provided\n";
      return false;
    }
    std::ifstream inputStream(postRefineAssignmentLog_);
    if (!inputStream.is_open()) {
      PLOGE << "post_refine_mapfpc: failed to open assignment log '"
            << postRefineAssignmentLog_ << "'\n";
      return false;
    }
    if (!parseMAPFPCAssignmentsFromStream(inputStream, postRefineAssignmentLog_,
                                          assignments)) {
      PLOGE << "post_refine_mapfpc: failed to parse assignments from '"
            << postRefineAssignmentLog_ << "'\n";
      return false;
    }
  } else {
    bool usedIncumbent = false;
    if ((int)incumbentSolution_.agentTaskAssignments.size() ==
        instance_.getAgentNum()) {
      assignments = incumbentSolution_.agentTaskAssignments;
      usedIncumbent = true;
    } else {
      for (int agent = 0; agent < instance_.getAgentNum(); agent++) {
        assignments[agent] = solution_.agents[agent].taskAssignments;
      }
    }
    assignmentSource = usedIncumbent ? "incumbent_solution" : "current_solution";
  }

  const Solution previousSolution = solution_;
  const FeasibleSolution previousIncumbent = incumbentSolution_;
  const int baselineObjective =
      previousIncumbent.agentPaths.empty()
          ? computeObjectiveValue(previousSolution)
          : computeObjectiveValue(previousIncumbent);

  const string solverVariant = postRefineSolver_;
  const string sourceLabel =
      "post_refine_mapfpc(" + solverVariant + "," + assignmentSource + ")";
  if (!runMAPFPCOnAssignments(assignments, solverVariant, postRefineTimeoutSec_,
                              sourceLabel)) {
    solution_ = previousSolution;
    invalidateCurrentTaskAssignmentIndexCache();
    incumbentSolution_ = previousIncumbent;
    return false;
  }

  if (isGoalOccupationRepositionTrue()) {
    vector<int> allAgents(instance_.getAgentNum());
    std::iota(allAgents.begin(), allAgents.end(), 0);
    if (!planTerminalReposition(allAgents, true)) {
      PLOGE << "post_refine_mapfpc: terminal reposition planning failed\n";
      solution_ = previousSolution;
      invalidateCurrentTaskAssignmentIndexCache();
      incumbentSolution_ = previousIncumbent;
      return false;
    }
  }

  ConflictMap potentialNeighborhood;
  ValidationStats validationStats;
  const bool previousTerminalValidationFlag = useTerminalPathsInValidation_;
  useTerminalPathsInValidation_ = isGoalOccupationRepositionTrue();
  const bool valid = validateSolution(&potentialNeighborhood, &validationStats);
  useTerminalPathsInValidation_ = previousTerminalValidationFlag;
  if (!valid) {
    PLOGE << "post_refine_mapfpc: refined solution failed validation\n";
    solution_ = previousSolution;
    invalidateCurrentTaskAssignmentIndexCache();
    incumbentSolution_ = previousIncumbent;
    return false;
  }

  const int refinedObjective = currentObjectiveValue();
  const bool strictlyBetter = previousIncumbent.agentPaths.empty() ||
                              refinedObjective <
                                  computeObjectiveValue(previousIncumbent);
  if (postRefineAcceptOnlyIfBetter_ && !strictlyBetter) {
    PLOGI << "post_refine_mapfpc: rejected non-improving refinement (baseline="
          << baselineObjective << ", refined=" << refinedObjective
          << ", objective=" << optimizationObjective_ << ")\n";
    solution_ = previousSolution;
    invalidateCurrentTaskAssignmentIndexCache();
    incumbentSolution_ = previousIncumbent;
    return false;
  }

  if (strictlyBetter) {
    extractFeasibleSolution();
  } else {
    overwriteIncumbentFromCurrentSolution();
  }
  long long refinedMakespanLl = 0;
  for (const auto& agent : solution_.agents) {
    refinedMakespanLl = std::max(
        refinedMakespanLl, static_cast<long long>(agent.path.endTimeOrZero()));
  }
  const int refinedMakespan =
      refinedMakespanLl > std::numeric_limits<int>::max()
          ? std::numeric_limits<int>::max()
          : static_cast<int>(refinedMakespanLl);
  const double refinedPrecedenceWait = computeSolutionPrecedenceWait();
  runtime = elapsedRuntimeSec();
  appendIterationStatBounded(IterationStats(
      runtime, sourceLabel, instance_.getAgentNum(), instance_.getTasksNum(),
      refinedObjective, true,
      strictlyBetter ? IterationQuality::bestSolutionYet
                     : IterationQuality::improvedSolution,
      0, 0, refinedMakespan, refinedPrecedenceWait));
  PLOGI << "post_refine_mapfpc: accepted refinement (baseline="
        << baselineObjective << ", refined=" << refinedObjective
        << ", objective=" << optimizationObjective_
        << ", accept_only_if_better="
        << (postRefineAcceptOnlyIfBetter_ ? "true" : "false") << ")\n";
  return true;
}
