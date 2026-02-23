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
#include <deque>
#include <limits>
#include <filesystem>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <utility>

#include "common.hpp"
#include "lns_internal_helpers.hpp"
#include "utils.hpp"

bool LNS::planTerminalReposition(const vector<int>& agentsToPlan,
                                 bool fullRebuild) {
  if (goalOccupationMode_ != "reposition_true") {
    return true;
  }
  terminalRepositionStats_.replansRequested++;

  const int agentCount = instance_.getAgentNum();
  vector<char> shouldPlan(agentCount, 0);
  for (int agent : agentsToPlan) {
    if (agent >= 0 && agent < agentCount) {
      shouldPlan[agent] = 1;
    }
  }
  if (fullRebuild) {
    for (int agent = 0; agent < agentCount; agent++) {
      shouldPlan[agent] = 1;
      solution_.agents[agent].terminalPath = AgentTaskPath();
      solution_.agents[agent].terminalPathActive = false;
    }
  } else {
    for (int agent = 0; agent < agentCount; agent++) {
      if (shouldPlan[agent]) {
        solution_.agents[agent].terminalPath = AgentTaskPath();
        solution_.agents[agent].terminalPathActive = false;
      }
    }
  }

  vector<char> plannedTerminal(agentCount, 0);
  if (!fullRebuild) {
    for (int agent = 0; agent < agentCount; agent++) {
      if (!shouldPlan[agent] && solution_.agents[agent].terminalPathActive &&
          !solution_.agents[agent].terminalPath.empty()) {
        plannedTerminal[agent] = 1;
      }
    }
  }

  vector<pair<int, int>> planningOrder;
  planningOrder.reserve(agentCount);
  for (int agent = 0; agent < agentCount; agent++) {
    if (!shouldPlan[agent]) {
      continue;
    }
    const auto& assignments = solution_.agents[agent].taskAssignments;
    const auto& joinedPath = solution_.agents[agent].path;
    if (assignments.empty() || joinedPath.empty()) {
      continue;
    }
    const int completionTime = joinedPath.endTimeOrZero();
    planningOrder.emplace_back(completionTime, agent);
  }
  std::sort(planningOrder.begin(), planningOrder.end(),
            [](const pair<int, int>& lhs, const pair<int, int>& rhs) {
              if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });

  const auto reserveOtherAgents = [&](ConstraintTable& constraintTable,
                                      int planningAgent) {
    for (int otherAgent = 0; otherAgent < agentCount; otherAgent++) {
      if (otherAgent == planningAgent) {
        continue;
      }
      const auto& otherPath = solution_.agents[otherAgent].path;
      if (!otherPath.empty()) {
        constraintTable.addPath(otherPath, false);
      }
      if (plannedTerminal[otherAgent] &&
          solution_.agents[otherAgent].terminalPathActive &&
          !solution_.agents[otherAgent].terminalPath.empty()) {
        constraintTable.addPath(solution_.agents[otherAgent].terminalPath,
                                false);
      }
    }
  };

  for (const auto& [completionTime, agent] : planningOrder) {
    if (runtimeBudgetExhausted()) {
      return false;
    }
    terminalRepositionStats_.agentsEvaluated++;
    const auto& assignments = solution_.agents[agent].taskAssignments;
    if (assignments.empty()) {
      continue;
    }
    const int finalTask = assignments.back();
    if (finalTask < 0 || finalTask >= instance_.getTasksNum()) {
      PLOGE << "planTerminalReposition: invalid final task index " << finalTask
            << " for agent " << agent << "\n";
      return false;
    }
    const int finalGoal = instance_.getTaskLocations(finalTask);

    int lastDemandTime = -1;
    for (int otherAgent = 0; otherAgent < agentCount; otherAgent++) {
      if (otherAgent == agent) {
        continue;
      }
      const auto& otherPath = solution_.agents[otherAgent].path;
      if (otherPath.empty()) {
        continue;
      }
      const int fromTime = max(0, completionTime + 1);
      int demandScanEnd = (int)otherPath.size();
      for (int timestep = fromTime; timestep < demandScanEnd;
           timestep++) {
        if (otherPath.at(timestep).location == finalGoal) {
          lastDemandTime = max(lastDemandTime, timestep);
        }
      }
    }

    if (lastDemandTime < completionTime + 1) {
      terminalRepositionStats_.skippedNoDemand++;
      // Even without demand, keep explicit terminal occupancy at final goal.
      AgentTaskPath holdPath;
      holdPath.beginTime = completionTime + 1;
      holdPath.path.push_back(PathEntry{false, finalGoal});
      solution_.agents[agent].terminalPath = std::move(holdPath);
      solution_.agents[agent].terminalPathActive = true;
      plannedTerminal[agent] = 1;
      continue;
    }

    const vector<int>& parkingCandidates = getParkingCandidatesForGoal(finalGoal);
    if (parkingCandidates.empty()) {
      terminalRepositionStats_.planningFailures++;
      PLOGE << "planTerminalReposition: no parking candidate found for agent "
            << agent << " final goal " << finalGoal << "\n";
      return false;
    }

    bool planned = false;
    for (int parkingLocation : parkingCandidates) {
      if (runtimeBudgetExhausted()) {
        return false;
      }
      auto planner = createLocalPlanner(agent);
      planner->setGoalLocations(vector<int>{finalGoal, parkingLocation});
      ConstraintTable evacConstraints(instance_.numOfCols, instance_.mapSize);
      reserveOtherAgents(evacConstraints, agent);
      AgentTaskPath evacuationPath = runLowLevelSearch(
          *planner, evacConstraints, completionTime, 1, 0);
      if (evacuationPath.empty()) {
        continue;
      }

      const int evacuationEnd = evacuationPath.endTimeChecked();
      const int returnStart = max(evacuationEnd, lastDemandTime + 1);
      planner->setGoalLocations(vector<int>{parkingLocation, finalGoal});
      ConstraintTable returnConstraints(instance_.numOfCols, instance_.mapSize);
      reserveOtherAgents(returnConstraints, agent);
      AgentTaskPath returnPath = runLowLevelSearch(*planner, returnConstraints,
                                                   returnStart, 1, 0);
      if (returnPath.empty()) {
        continue;
      }

      AgentTaskPath terminalPath = evacuationPath;
      while (!terminalPath.empty() &&
             terminalPath.endTimeChecked() < returnStart) {
        terminalPath.path.push_back(PathEntry{false, parkingLocation});
      }
      for (int step = 1; step < (int)returnPath.size(); step++) {
        terminalPath.path.push_back(returnPath.at(step));
      }

      if (terminalPath.empty()) {
        continue;
      }
      solution_.agents[agent].terminalPath = std::move(terminalPath);
      solution_.agents[agent].terminalPathActive = true;
      plannedTerminal[agent] = 1;
      terminalRepositionStats_.agentsPlanned++;
      planned = true;
      break;
    }

    if (!planned) {
      terminalRepositionStats_.planningFailures++;
      PLOGE << "planTerminalReposition: failed to construct terminal path for "
            << "agent " << agent << " (goal " << finalGoal << ")\n";
      return false;
    }
  }

  return true;
}
