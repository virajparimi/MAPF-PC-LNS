#pragma once

#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace mapf_pc_lns::nrr {

struct BoundaryWindows {
  std::vector<int> releaseByTask;
  std::vector<int> deadlineByTask;
  bool feasible = true;
  std::string failureReason;
};

// Converts precedence edges that cross destroyed/fixed partitions into
// per-task release/deadline windows using incumbent completion times.
inline BoundaryWindows computeBoundaryWindows(
    int numTasks, const std::vector<std::pair<int, int>>& precedenceEdges,
    const std::vector<char>& destroyedTaskMask,
    const std::vector<int>& incumbentCompletionTimes) {
  BoundaryWindows out;
  out.releaseByTask.assign(numTasks, 0);
  out.deadlineByTask.assign(numTasks, MAX_TIMESTEP);
  out.feasible = true;

  if (numTasks < 0 || (int)destroyedTaskMask.size() != numTasks ||
      (int)incumbentCompletionTimes.size() != numTasks) {
    out.feasible = false;
    out.failureReason = "invalid_input_sizes";
    return out;
  }

  for (const auto& edge : precedenceEdges) {
    const int u = edge.first;
    const int v = edge.second;
    if (u < 0 || u >= numTasks || v < 0 || v >= numTasks) {
      out.feasible = false;
      out.failureReason = "precedence_edge_out_of_range";
      return out;
    }

    const bool uDestroyed = destroyedTaskMask[u] != 0;
    const bool vDestroyed = destroyedTaskMask[v] != 0;

    if (!uDestroyed && vDestroyed) {
      const int tauU = incumbentCompletionTimes[u];
      if (tauU < 0) {
        out.feasible = false;
        out.failureReason = "missing_incumbent_time_for_fixed_predecessor";
        return out;
      }
      out.releaseByTask[v] = std::max(out.releaseByTask[v], tauU + 1);
    } else if (uDestroyed && !vDestroyed) {
      const int tauV = incumbentCompletionTimes[v];
      if (tauV < 0) {
        out.feasible = false;
        out.failureReason = "missing_incumbent_time_for_fixed_successor";
        return out;
      }
      out.deadlineByTask[u] = std::min(out.deadlineByTask[u], tauV - 1);
    }
  }

  for (int task = 0; task < numTasks; task++) {
    if (destroyedTaskMask[task] == 0) {
      continue;
    }
    if (out.releaseByTask[task] > out.deadlineByTask[task]) {
      out.feasible = false;
      out.failureReason = "boundary_window_infeasible";
      return out;
    }
  }

  return out;
}

}  // namespace mapf_pc_lns::nrr

