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
      << '\n';
}

void CBSReportExporter::printSaveStatus() {
  if (outputFile_.empty()) {
    std::cout << "No report was generated.\n";
    return;
  }
  std::cout << "File is saved!\n";
  std::cout << "Name of file: " << outputFile_ << '\n';
}

void CBSReportExporter::writeReport(const Instance* inst,
                                    const Solution* sol) {
  if (inst == nullptr || sol == nullptr) {
    std::cerr << "writeReport called with null instance or solution.\n";
    outputFile_.clear();
    return;
  }
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
  outputFile_.clear();

  bool writeOk = false;
  if (myFile.is_open()) {
    // Get number of agents
    const int agentNum = inst->getAgentNum();
    myFile << agentNum << " # number of agents\n";
    myFile << "# Format:  num_of_goals sx sy g1x g1y g2x g2y ...\n";

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
      myFile << '\n';
    }

    // Write the precedence constraints
    myFile << "temporal cons:\n";

    const vector<pair<int, int>>& globalPc =
        inst->getInputPrecedenceConstraintsRef();

    for (const auto& pc : globalPc) {
      const int predecessor = pc.first;
      const int successor = pc.second;

      if (predecessor < 0 || predecessor >= (int)sol->taskAgentMap.size() ||
          successor < 0 || successor >= (int)sol->taskAgentMap.size()) {
        std::cerr << "Skipping precedence constraint with missing task "
                     "assignment: "
                  << predecessor << " -> " << successor << '\n';
        continue;
      }
      const int predAgent = sol->taskAgentMap[predecessor];
      const int succAgent = sol->taskAgentMap[successor];
      if (predAgent == UNASSIGNED || succAgent == UNASSIGNED) {
        std::cerr << "Skipping precedence constraint with unassigned task: "
                  << predecessor << " -> " << successor << '\n';
        continue;
      }

      const int predLocalIndex = sol->getLocalTaskIndex(predAgent, predecessor);
      const int succLocalIndex = sol->getLocalTaskIndex(succAgent, successor);
      if (predLocalIndex == UNASSIGNED || succLocalIndex == UNASSIGNED) {
        std::cerr << "Skipping precedence constraint with missing local index: "
                  << predecessor << " -> " << successor << '\n';
        continue;
      }

      myFile << predAgent << "\t" << predLocalIndex << "\t" << succAgent << "\t"
             << succLocalIndex << '\n';
    }
    myFile.flush();
    writeOk = myFile.good();
    if (!writeOk) {
      std::cerr << "Report write failed for file: " << fileName << '\n';
    }
  } else {
    std::cerr << "Failed to create report file: " << fileName << '\n';
  }
  if (writeOk) {
    outputFile_ = fileName;
  }
}
