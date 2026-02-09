#include "report_exporter.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include "lns.hpp"

namespace {
std::tm localtimeSafe(std::time_t timeValue) {
  std::tm tmValue{};
#if defined(_WIN32)
  localtime_s(&tmValue, &timeValue);
#else
  localtime_r(&timeValue, &tmValue);
#endif
  return tmValue;
}
}  // namespace

void CBSReportExporter::printStart() {
  std::cout
      << "Loading task assignment, locations and precedence constraint data"
      << std::endl;
}

void CBSReportExporter::printSaveStatus() {
  if (outputFile.empty()) {
    std::cout << "No report was generated." << std::endl;
    return;
  }
  std::cout << "File is saved!" << std::endl;
  std::cout << "Name of file: " << outputFile << std::endl;
}

void CBSReportExporter::writeReport(const Instance* inst,
                                    const Solution* sol) {
  assert(inst != nullptr);
  assert(sol != nullptr);

  // Open file
  const auto now = std::chrono::system_clock::now();
  const std::time_t inTimeT = std::chrono::system_clock::to_time_t(now);
  const std::tm tmValue = localtimeSafe(inTimeT);

  std::stringstream datetime;
  // Avoid ':' in filenames (makes shell escaping unnecessary).
  datetime << std::put_time(&tmValue, "%Y-%m-%d_%H-%M-%S");

  const std::string fileName = "report_" + datetime.str() + ".txt";
  std::ofstream myFile(fileName);
  outputFile.clear();

  if (myFile.is_open()) {
    outputFile = fileName;
    // Get number of agents
    const int agentNum = inst->getAgentNum();
    myFile << agentNum << " # number of agents" << std::endl;
    myFile << "# Format:  num_of_goals sx sy g1x g1y g2x g2y ..."
           << std::endl;

    // Get global task ids for each agent
    for (int a = 0; a < agentNum; a++) {
      const vector<int>& globalTasks = sol->agents[a].taskAssignments;
      const int numTasks = (int)globalTasks.size();
      const pair<int, int> startLocAgent =
          inst->getCoordinate(inst->getStartLocationsRef()[a]);
      myFile << numTasks << "\t" << startLocAgent.second << "\t"
             << startLocAgent.first << "\t";
      for (int i = 0; i < numTasks; i++) {
        const int globalTask = globalTasks[i];
        const int taskLocation = inst->getTaskLocations(globalTask);
        const pair<int, int> taskLoc = inst->getCoordinate(taskLocation);
        myFile << taskLoc.second << "\t" << taskLoc.first << "\t";
      }
      myFile << std::endl;
    }

    // Write the precedence constraints
    myFile << "temporal cons:" << std::endl;

    const vector<pair<int, int>>& globalPc =
        inst->getInputPrecedenceConstraintsRef();

    for (const auto& pc : globalPc) {
      const int predecessor = pc.first;
      const int successor = pc.second;

      const auto itPredAgent = sol->taskAgentMap.find(predecessor);
      const auto itSuccAgent = sol->taskAgentMap.find(successor);
      if (itPredAgent == sol->taskAgentMap.end() ||
          itSuccAgent == sol->taskAgentMap.end()) {
        std::cerr << "Skipping precedence constraint with missing task "
                     "assignment: "
                  << predecessor << " -> " << successor << std::endl;
        continue;
      }
      const int predAgent = itPredAgent->second;
      const int succAgent = itSuccAgent->second;
      if (predAgent == UNASSIGNED || succAgent == UNASSIGNED) {
        std::cerr << "Skipping precedence constraint with unassigned task: "
                  << predecessor << " -> " << successor << std::endl;
        continue;
      }

      const int predLocalIndex = sol->getLocalTaskIndex(predAgent, predecessor);
      const int succLocalIndex = sol->getLocalTaskIndex(succAgent, successor);
      if (predLocalIndex == UNASSIGNED || succLocalIndex == UNASSIGNED) {
        std::cerr << "Skipping precedence constraint with missing local index: "
                  << predecessor << " -> " << successor << std::endl;
        continue;
      }

      myFile << predAgent << "\t" << predLocalIndex << "\t" << succAgent << "\t"
             << succLocalIndex << std::endl;
    }
  } else {
    std::cerr << "Failed to create report file: " << fileName << std::endl;
  }
}
