#include <plog/Log.h>
#include <plog/Severity.h>
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"

#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "constrainttable.hpp"
#include "instance.hpp"
#include "mlastar.hpp"
#include "sipps.hpp"

namespace {

struct ScenarioEntry {
  int sx = 0, sy = 0, gx = 0, gy = 0;
};

struct PlannerResult {
  bool success = false;
  bool valid = false;
  int soc = 0;
  int makespan = 0;
  uint64_t expanded = 0;
  uint64_t generated = 0;
  double runtimeSec = 0.0;
  vector<AgentTaskPath> paths;
};

bool loadScenarioEntries(const string& scenFile, bool skipTrivial,
                         vector<ScenarioEntry>& out) {
  out.clear();
  std::ifstream in(scenFile);
  if (!in.is_open()) {
    PLOGE << "Failed to open scenario file: " << scenFile << "\n";
    return false;
  }

  string header;
  if (!std::getline(in, header)) {
    PLOGE << "Scenario file is empty: " << scenFile << "\n";
    return false;
  }
  if (header.rfind("version", 0) != 0) {
    PLOGW << "Scenario header is unexpected: " << header << "\n";
  }

  string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    int bucket = 0, width = 0, height = 0;
    double dist = 0.0;
    string mapName;
    ScenarioEntry e;
    if (!(iss >> bucket >> mapName >> width >> height >> e.sx >> e.sy >> e.gx >>
          e.gy >> dist)) {
      PLOGE << "Malformed scenario row: " << line << "\n";
      return false;
    }
    if (skipTrivial && e.sx == e.gx && e.sy == e.gy) {
      continue;
    }
    out.push_back(e);
  }

  if (out.empty()) {
    PLOGE << "Scenario has no usable entries: " << scenFile << "\n";
    return false;
  }
  return true;
}

unsigned int normalizeSeed(unsigned int seed) {
  if (seed != 0u) {
    return seed;
  }
  std::uint64_t ticks = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  // Mix entropy before narrowing to 32-bit.
  ticks ^= ticks >> 30;
  ticks *= 0xbf58476d1ce4e5b9ULL;
  ticks ^= ticks >> 27;
  ticks *= 0x94d049bb133111ebULL;
  ticks ^= ticks >> 31;
  return static_cast<unsigned int>(ticks ^ (ticks >> 32));
}

ScenarioEntry sampleScenarioEntry(const vector<ScenarioEntry>& entries,
                                  unsigned int seed, size_t& sampledIndex) {
  assert(!entries.empty());
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> distribution(0, entries.size() - 1);
  sampledIndex = distribution(rng);
  return entries[sampledIndex];
}

string writeAgentsGoalsFile(const vector<ScenarioEntry>& entries) {
  const auto stamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() /
      ("planner_isolation_agents_" + std::to_string(stamp) + ".txt");
  std::ofstream out(tmp);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to create temporary agents file: " +
                             tmp.string());
  }

  const int n = (int)entries.size();
  out << n << "\n";
  for (const auto& e : entries) {
    out << e.sx << ", " << e.sy << "\n";
  }
  out << "\n";
  out << "tasks\n";
  out << n << "\n";
  for (const auto& e : entries) {
    out << e.gx << ", " << e.gy << "\n";
  }
  out << "\n";
  out << "temporal\n";
  out << "0\n";
  return tmp.string();
}

int getLocationAt(const AgentTaskPath& path, int timestep) {
  if (path.empty()) {
    return UNASSIGNED;
  }
  const int index = timestep - path.beginTime;
  if (index < 0) {
    return path.front().location;
  }
  if (index >= (int)path.size()) {
    return path.back().location;
  }
  return path[index].location;
}

bool validateNoCollisions(const vector<AgentTaskPath>& paths) {
  int maxT = 0;
  for (const auto& p : paths) {
    maxT = max(maxT, p.endTimeOrZero());
  }

  for (int t = 0; t <= maxT; t++) {
    for (int i = 0; i < (int)paths.size(); i++) {
      for (int j = i + 1; j < (int)paths.size(); j++) {
        const int li = getLocationAt(paths[i], t);
        const int lj = getLocationAt(paths[j], t);
        if (li == UNASSIGNED || lj == UNASSIGNED) {
          continue;
        }
        if (li == lj) {
          return false;
        }
        if (t > 0) {
          const int lip = getLocationAt(paths[i], t - 1);
          const int ljp = getLocationAt(paths[j], t - 1);
          if (lip == lj && ljp == li) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

std::unique_ptr<SingleAgentSolver> createPlanner(const Instance& instance,
                                                 int agent,
                                                 const string& plannerName) {
  if (plannerName == "mlastar") {
    return std::make_unique<MultiLabelSpaceTimeAStar>(instance, agent);
  }
  if (plannerName == "sipps") {
    return std::make_unique<MultiLabelSIPPS>(instance, agent);
  }
  throw std::runtime_error("Unsupported planner: " + plannerName);
}

PlannerResult runSingleAgentLowLevel(const Instance& instance,
                                     const string& plannerName,
                                     double segmentTimeoutSec) {
  PlannerResult result;
  result.paths.resize(1);

  auto t0 = std::chrono::high_resolution_clock::now();
  auto planner = createPlanner(instance, 0, plannerName);
  planner->setSegmentTimeout(segmentTimeoutSec);

  // Fixed assignment in this utility: one task for one agent.
  planner->setGoalLocations(vector<int>{instance.getTaskLocations(0)});

  ConstraintTable ct((size_t)instance.getCols(), (size_t)instance.mapSize);
  AgentTaskPath path = planner->findPathSegment(ct, 0, 0, 0);
  result.expanded += planner->numExpanded;
  result.generated += planner->numGenerated;
  if (path.empty()) {
    auto t1 = std::chrono::high_resolution_clock::now();
    result.runtimeSec = std::chrono::duration<double>(t1 - t0).count();
    result.success = false;
    return result;
  }
  result.paths[0] = std::move(path);
  auto t1 = std::chrono::high_resolution_clock::now();
  result.runtimeSec = std::chrono::duration<double>(t1 - t0).count();

  result.soc = result.paths[0].endTimeOrZero();
  result.makespan = result.paths[0].endTimeOrZero();
  result.valid = validateNoCollisions(result.paths);
  result.success = result.valid;
  return result;
}

PlannerResult runBfsSingleAgent(const Instance& instance) {
  PlannerResult result;
  result.paths.resize(1);

  const int start = instance.getStartLocationsRef()[0];
  const int goal = instance.getTaskLocations(0);
  auto t0 = std::chrono::high_resolution_clock::now();

  vector<int> parent(instance.mapSize, UNASSIGNED);
  vector<bool> visited(instance.mapSize, false);
  std::queue<int> q;
  q.push(start);
  visited[start] = true;
  parent[start] = start;
  result.generated = 1;

  bool found = false;
  while (!q.empty()) {
    const int current = q.front();
    q.pop();
    result.expanded++;
    if (current == goal) {
      found = true;
      break;
    }

    for (int next : instance.getNeighbors(current)) {
      if (visited[next]) {
        continue;
      }
      visited[next] = true;
      parent[next] = current;
      q.push(next);
      result.generated++;
    }
  }

  if (!found) {
    auto t1 = std::chrono::high_resolution_clock::now();
    result.runtimeSec = std::chrono::duration<double>(t1 - t0).count();
    result.success = false;
    return result;
  }

  vector<int> locations;
  for (int cur = goal; cur != start; cur = parent[cur]) {
    locations.push_back(cur);
  }
  locations.push_back(start);
  std::reverse(locations.begin(), locations.end());

  AgentTaskPath path;
  path.path.resize(locations.size());
  for (size_t i = 0; i < locations.size(); i++) {
    path[(int)i].location = locations[i];
    path[(int)i].isGoal = (i + 1 == locations.size());
  }
  result.paths[0] = std::move(path);

  auto t1 = std::chrono::high_resolution_clock::now();
  result.runtimeSec = std::chrono::duration<double>(t1 - t0).count();
  result.soc = result.paths[0].endTimeOrZero();
  result.makespan = result.paths[0].endTimeOrZero();
  result.valid = validateNoCollisions(result.paths);
  result.success = result.valid;
  return result;
}

void printPaths(const Instance& instance, const PlannerResult& result) {
  for (int agent = 0; agent < (int)result.paths.size(); agent++) {
    std::cout << "Agent " << agent << " (cost=" << result.paths[agent].endTimeOrZero()
              << "): ";
    for (int t = 0; t < (int)result.paths[agent].size(); t++) {
      const auto coord = instance.getCoordinate(result.paths[agent][t].location);
      std::cout << "(" << coord.first << "," << coord.second << ")@" << t;
      if (result.paths[agent][t].isGoal) {
        std::cout << "*";
      }
      if (t + 1 < (int)result.paths[agent].size()) {
        std::cout << " -> ";
      }
    }
    std::cout << "\n";
  }
}

int runPlanner(const Instance& instance, const string& plannerName,
               double segmentTimeoutSec, bool printPathFlag) {
  PlannerResult r;
  if (plannerName == "bfs") {
    r = runBfsSingleAgent(instance);
  } else {
    r = runSingleAgentLowLevel(instance, plannerName, segmentTimeoutSec);
  }
  std::cout << "=== Planner: " << plannerName << " ===\n";
  std::cout << "success=" << (r.success ? "true" : "false") << "\n";
  std::cout << "valid_no_conflicts=" << (r.valid ? "true" : "false") << "\n";
  std::cout << "soc=" << r.soc << "\n";
  std::cout << "makespan=" << r.makespan << "\n";
  std::cout << "runtime_sec=" << std::fixed << std::setprecision(6)
            << r.runtimeSec << "\n";
  std::cout << "expanded=" << r.expanded << "\n";
  std::cout << "generated=" << r.generated << "\n";
  if (printPathFlag) {
    printPaths(instance, r);
  }
  return r.success ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::error, &consoleAppender);

  namespace po = boost::program_options;
  po::options_description desc("planner_isolation options");
  desc.add_options()("help", "Produce help message");
  desc.add_options()("map,m", po::value<string>()->required(), "Input .map file");
  desc.add_options()("scen",
                     po::value<string>()->required(),
                     "Input .scen file");
  desc.add_options()("agents,k",
                     po::value<int>()->default_value(1),
                     "Number of agents/scenario rows to use (must be 1)");
  desc.add_options()("planner,p",
                     po::value<string>()->default_value("both"),
                     "Planner to run: mlastar | sipps | bfs | both | all");
  desc.add_options()("skipTrivial",
                     po::bool_switch()->default_value(true),
                     "Skip scenario rows with start==goal");
  desc.add_options()("seed",
                     po::value<unsigned int>()->default_value(1),
                     "Scenario sampling seed (0 = time-based)");
  desc.add_options()("printPaths",
                     po::bool_switch()->default_value(false),
                     "Print full agent paths");
  desc.add_options()("lowLevelSegmentTimeout",
                     po::value<double>()->default_value(600.0),
                     "Per-agent planning timeout in seconds");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n" << desc << "\n";
    return 1;
  }

  const string mapFile = vm["map"].as<string>();
  const string scenFile = vm["scen"].as<string>();
  const int agents = vm["agents"].as<int>();
  const string planner = vm["planner"].as<string>();
  const bool skipTrivial = vm["skipTrivial"].as<bool>();
  unsigned int seed = vm["seed"].as<unsigned int>();
  const bool printPathFlag = vm["printPaths"].as<bool>();
  const double segmentTimeoutSec = vm["lowLevelSegmentTimeout"].as<double>();

  if (agents != 1) {
    std::cerr << "--agents must be exactly 1 for planner_isolation\n";
    return 1;
  }
  if (segmentTimeoutSec <= 0.0) {
    std::cerr << "--lowLevelSegmentTimeout must be positive\n";
    return 1;
  }
  if (planner != "mlastar" && planner != "sipps" && planner != "bfs" &&
      planner != "both" && planner != "all") {
    std::cerr << "--planner must be one of: mlastar, sipps, bfs, both, all\n";
    return 1;
  }

  vector<ScenarioEntry> allEntries;
  if (!loadScenarioEntries(scenFile, skipTrivial, allEntries)) {
    return 1;
  }
  seed = normalizeSeed(seed);
  size_t sampledIndex = 0;
  ScenarioEntry sampled = sampleScenarioEntry(allEntries, seed, sampledIndex);
  std::cout << "seed = " << seed << "\n";
  std::cout << "sampled_scen_row = " << (sampledIndex + 2) << "\n";

  string tmpAgentsFile;
  try {
    tmpAgentsFile = writeAgentsGoalsFile(vector<ScenarioEntry>{sampled});
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  int rc = 0;
  try {
    Instance instance(mapFile, tmpAgentsFile, 1, 1);
    if (planner == "mlastar" || planner == "both") {
      rc = max(rc, runPlanner(instance, "mlastar", segmentTimeoutSec, printPathFlag));
    }
    if (planner == "sipps" || planner == "both") {
      rc = max(rc, runPlanner(instance, "sipps", segmentTimeoutSec, printPathFlag));
    }
    if (planner == "bfs" || planner == "all") {
      rc = max(rc, runPlanner(instance, "bfs", segmentTimeoutSec, printPathFlag));
    }
    if (planner == "all") {
      rc = max(rc, runPlanner(instance, "mlastar", segmentTimeoutSec, printPathFlag));
      rc = max(rc, runPlanner(instance, "sipps", segmentTimeoutSec, printPathFlag));
    }
  } catch (const std::exception& e) {
    std::cerr << "planner_isolation failed: " << e.what() << "\n";
    rc = 1;
  }

  if (!tmpAgentsFile.empty()) {
    std::error_code ec;
    std::filesystem::remove(tmpAgentsFile, ec);
  }
  return rc;
}
