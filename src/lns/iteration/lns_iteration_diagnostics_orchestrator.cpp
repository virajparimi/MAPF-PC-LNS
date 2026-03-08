#include "lns_iteration_diagnostics_orchestrator.hpp"

#include "lns.hpp"

#include <algorithm>

NeighborhoodDiagnosticsResult IterationDiagnosticsOrchestrator::analyzeNeighborhood(
    LNS& lns) {
  return lns.analyzeNeighborhoodDiagnostics();
}

NeighborhoodDiagnosticsResult LNS::analyzeNeighborhoodDiagnostics() {
  NeighborhoodDiagnosticsResult result;
  result.removedTasks = (int)lnsNeighborhood_.immutableRemovedTasks.size();

  result.neighborhoodFingerprint =
      computeNeighborhoodFingerprint(lnsNeighborhood_.immutableRemovedTasks);
  const auto [_, neighborhoodInserted] =
      neighborhoodDiagnosticsState_.seenFingerprints.insert(
          result.neighborhoodFingerprint);
  result.neighborhoodFingerprintSeenBefore = !neighborhoodInserted;
  if (result.neighborhoodFingerprintSeenBefore) {
    improvementDiagnosticsStats_.neighborhoodFingerprintRepeat++;
    neighborhoodDiagnosticsState_.repeatStreak++;
    result.neighborhoodRepeatStreak = neighborhoodDiagnosticsState_.repeatStreak;
    improvementDiagnosticsStats_.neighborhoodRepeatStreakMax = std::max(
        improvementDiagnosticsStats_.neighborhoodRepeatStreakMax,
        (int64_t)neighborhoodDiagnosticsState_.repeatStreak);
  } else {
    improvementDiagnosticsStats_.neighborhoodFingerprintUnique++;
    neighborhoodDiagnosticsState_.repeatStreak = 0;
    result.neighborhoodRepeatStreak = 0;
  }

  std::vector<int> currentNeighborhoodTasksSorted;
  currentNeighborhoodTasksSorted.reserve(
      lnsNeighborhood_.immutableRemovedTasks.size());
  for (const auto& [task, _] : lnsNeighborhood_.immutableRemovedTasks) {
    currentNeighborhoodTasksSorted.push_back(task);
  }
  if (!currentNeighborhoodTasksSorted.empty()) {
    std::string taskIdsCsv;
    taskIdsCsv.reserve(currentNeighborhoodTasksSorted.size() * 6);
    for (size_t idx = 0; idx < currentNeighborhoodTasksSorted.size(); idx++) {
      if (idx > 0) {
        taskIdsCsv.push_back(',');
      }
      taskIdsCsv += std::to_string(currentNeighborhoodTasksSorted[idx]);
    }
    result.removedTaskIdsCsv = std::move(taskIdsCsv);
  } else {
    result.removedTaskIdsCsv.clear();
  }

  if (!neighborhoodDiagnosticsState_.previousTasksSorted.empty()) {
    size_t i = 0;
    size_t j = 0;
    size_t intersection = 0;
    while (i < currentNeighborhoodTasksSorted.size() &&
           j < neighborhoodDiagnosticsState_.previousTasksSorted.size()) {
      if (currentNeighborhoodTasksSorted[i] ==
          neighborhoodDiagnosticsState_.previousTasksSorted[j]) {
        intersection++;
        i++;
        j++;
      } else if (currentNeighborhoodTasksSorted[i] <
                 neighborhoodDiagnosticsState_.previousTasksSorted[j]) {
        i++;
      } else {
        j++;
      }
    }
    const size_t unionCount = currentNeighborhoodTasksSorted.size() +
                              neighborhoodDiagnosticsState_.previousTasksSorted.size() -
                              intersection;
    const double jaccardPrev =
        unionCount > 0 ? (double)intersection / (double)unionCount : 1.0;
    result.neighborhoodJaccardPrev = jaccardPrev;
    improvementDiagnosticsStats_.neighborhoodJaccardPrevSum += jaccardPrev;
    improvementDiagnosticsStats_.neighborhoodJaccardPrevSamples++;
    if (result.neighborhoodFingerprintSeenBefore) {
      improvementDiagnosticsStats_.neighborhoodJaccardPrevRepeatSum +=
          jaccardPrev;
      improvementDiagnosticsStats_.neighborhoodJaccardPrevRepeatSamples++;
    }
  }
  neighborhoodDiagnosticsState_.previousTasksSorted.swap(currentNeighborhoodTasksSorted);

  improvementDiagnosticsStats_.neighborhoodsCount++;
  improvementDiagnosticsStats_.removedTasksTotal += result.removedTasks;
  improvementDiagnosticsStats_.removedTasksMax = std::max(
      improvementDiagnosticsStats_.removedTasksMax,
      (int64_t)result.removedTasks);

  return result;
}
