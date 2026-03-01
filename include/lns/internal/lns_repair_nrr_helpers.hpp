#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "constrainttable.hpp"
#include "instance.hpp"
#include "nrr_boundary_windows.hpp"

namespace lns_nrr_helpers {

constexpr long long kNrrInfCost = (1LL << 60);

struct NrrSlot {
  int agent = UNASSIGNED;
  int gap = 0;
  int copy = 0;
  int prevLocation = -1;
  int nextLocation = -1;
  int prevCompletion = 0;
  int approxArrival = 0;
};

struct DifferenceEdge {
  int from = 0;
  int to = 0;
  long long weight = 0;  // x_to <= x_from + weight
};

long long distOrInf(const Instance& instance, int fromLocation, int toLocation);

bool topologicalSortSubsetWithPriorityStrict(
    const Instance& instance, const vector<int>& tasks,
    const std::unordered_map<int, long long>& tiePriority,
    vector<int>& outOrdered);

bool differenceConstraintsFeasible(int numVars,
                                   const vector<DifferenceEdge>& edges);

std::string normalizeNrrFailureReason(const std::string& reason);

bool buildIterativeProposal(
    const Instance& instance, int numAgents, int numTasks,
    const vector<int>& destroyedTasks, const vector<char>& destroyedMask,
    const vector<int>& neighborhoodAgents,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    const ConstraintTable& frozenCt, vector<vector<int>>& proposedAssignments,
    std::string& failureReason);

bool runTemporalPrecheck(
    const Instance& instance, int numTasks,
    const vector<int>& neighborhoodAgents, const vector<int>& destroyedTasks,
    const vector<vector<int>>& proposedAssignments,
    const vector<int>& incumbentCompletion,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    std::string& failureReason);

}  // namespace lns_nrr_helpers
