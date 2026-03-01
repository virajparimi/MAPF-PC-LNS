#include "lns_iteration_diagnostics_orchestrator.hpp"

#include "lns.hpp"

#include <algorithm>

NeighborhoodDiagnosticsResult IterationDiagnosticsOrchestrator::analyzeNeighborhood(
    LNS& lns) {
  NeighborhoodDiagnosticsResult result;
  result.removedTasks = (int)lns.lnsNeighborhood_.immutableRemovedTasks.size();

  result.neighborhoodFingerprint =
      lns.computeNeighborhoodFingerprint(lns.lnsNeighborhood_.immutableRemovedTasks);
  const auto [_, neighborhoodInserted] =
      lns.seenNeighborhoodFingerprints_.insert(result.neighborhoodFingerprint);
  result.neighborhoodFingerprintSeenBefore = !neighborhoodInserted;
  if (result.neighborhoodFingerprintSeenBefore) {
    lns.improvementDiagnosticsStats_.neighborhoodFingerprintRepeat++;
    lns.currentNeighborhoodRepeatStreak_++;
    result.neighborhoodRepeatStreak = lns.currentNeighborhoodRepeatStreak_;
    lns.improvementDiagnosticsStats_.neighborhoodRepeatStreakMax = std::max(
        lns.improvementDiagnosticsStats_.neighborhoodRepeatStreakMax,
        (int64_t)lns.currentNeighborhoodRepeatStreak_);
  } else {
    lns.improvementDiagnosticsStats_.neighborhoodFingerprintUnique++;
    lns.currentNeighborhoodRepeatStreak_ = 0;
    result.neighborhoodRepeatStreak = 0;
  }

  std::vector<int> currentNeighborhoodTasksSorted;
  currentNeighborhoodTasksSorted.reserve(
      lns.lnsNeighborhood_.immutableRemovedTasks.size());
  for (const auto& [task, _] : lns.lnsNeighborhood_.immutableRemovedTasks) {
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

  if (!lns.previousNeighborhoodTasksSorted_.empty()) {
    size_t i = 0;
    size_t j = 0;
    size_t intersection = 0;
    while (i < currentNeighborhoodTasksSorted.size() &&
           j < lns.previousNeighborhoodTasksSorted_.size()) {
      if (currentNeighborhoodTasksSorted[i] ==
          lns.previousNeighborhoodTasksSorted_[j]) {
        intersection++;
        i++;
        j++;
      } else if (currentNeighborhoodTasksSorted[i] <
                 lns.previousNeighborhoodTasksSorted_[j]) {
        i++;
      } else {
        j++;
      }
    }
    const size_t unionCount = currentNeighborhoodTasksSorted.size() +
                              lns.previousNeighborhoodTasksSorted_.size() -
                              intersection;
    const double jaccardPrev =
        unionCount > 0 ? (double)intersection / (double)unionCount : 1.0;
    result.neighborhoodJaccardPrev = jaccardPrev;
    lns.improvementDiagnosticsStats_.neighborhoodJaccardPrevSum += jaccardPrev;
    lns.improvementDiagnosticsStats_.neighborhoodJaccardPrevSamples++;
    if (result.neighborhoodFingerprintSeenBefore) {
      lns.improvementDiagnosticsStats_.neighborhoodJaccardPrevRepeatSum +=
          jaccardPrev;
      lns.improvementDiagnosticsStats_.neighborhoodJaccardPrevRepeatSamples++;
    }
  }
  lns.previousNeighborhoodTasksSorted_.swap(currentNeighborhoodTasksSorted);

  lns.improvementDiagnosticsStats_.neighborhoodsCount++;
  lns.improvementDiagnosticsStats_.removedTasksTotal += result.removedTasks;
  lns.improvementDiagnosticsStats_.removedTasksMax = std::max(
      lns.improvementDiagnosticsStats_.removedTasksMax,
      (int64_t)result.removedTasks);

  return result;
}
