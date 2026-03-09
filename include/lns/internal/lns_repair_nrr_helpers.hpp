#pragma once

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "constrainttable.hpp"
#include "instance.hpp"
#include "lns_types.hpp"
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

struct DestroyedTaskOwnerChange {
  int task = -1;
  int oldOwner = UNASSIGNED;
  int newOwner = UNASSIGNED;
};

struct ProposalMutationTrace {
  vector<int> touchedAgentsFinal;
  vector<DestroyedTaskOwnerChange> destroyedTaskOwnerChanges;
  vector<int> patchedImmediateSuccessorTasks;
  int64_t globalDemotions = 0;
  int64_t globalSlotEvaluations = 0;
  int64_t globalNoFeasibleSlotCount = 0;
};

class NrrFrozenOccupancyIndex {
 public:
  void clear();
  void buildFromPreviousSolution(const Instance& instance,
                                 const Solution& previousSolution);
  bool isVertexConstrained(int location, int timestep) const;
  bool isEdgeConstrained(int fromLocation, int toLocation,
                         int nextTimestep) const;
  bool demoteAgent(int agent);

 private:
  struct VertexTimeKey {
    int location = -1;
    int timestep = -1;
    bool operator==(const VertexTimeKey& other) const {
      return location == other.location && timestep == other.timestep;
    }
  };
  struct EdgeTimeKey {
    int from = -1;
    int to = -1;
    int timestep = -1;
    bool operator==(const EdgeTimeKey& other) const {
      return from == other.from && to == other.to &&
             timestep == other.timestep;
    }
  };
  struct VertexTimeKeyHash {
    std::size_t operator()(const VertexTimeKey& key) const {
      uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(key.location));
      h ^= (static_cast<uint64_t>(static_cast<uint32_t>(key.timestep)) << 32);
      h ^= (h >> 33);
      h *= 0xff51afd7ed558ccdULL;
      h ^= (h >> 33);
      return static_cast<std::size_t>(h);
    }
  };
  struct EdgeTimeKeyHash {
    std::size_t operator()(const EdgeTimeKey& key) const {
      uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(key.from));
      h ^= (static_cast<uint64_t>(static_cast<uint32_t>(key.to)) << 21);
      h ^= (static_cast<uint64_t>(static_cast<uint32_t>(key.timestep)) << 42);
      h ^= (h >> 29);
      h *= 0x9e3779b97f4a7c15ULL;
      h ^= (h >> 32);
      return static_cast<std::size_t>(h);
    }
  };
  struct AgentContribution {
    bool active = false;
    vector<VertexTimeKey> vertexKeys;
    vector<EdgeTimeKey> edgeKeys;
    vector<pair<int, int>> tailStartsByLoc;
  };

  void addPathContribution(int agent, const AgentTaskPath& path,
                           bool reserveTail);
  void decrementVertexKey(const VertexTimeKey& key);
  void decrementEdgeKey(const EdgeTimeKey& key);
  void decrementTailKey(int location, int startTimestep);

  vector<AgentContribution> perAgentContrib_;
  std::unordered_map<VertexTimeKey, int, VertexTimeKeyHash> vertexCounts_;
  std::unordered_map<EdgeTimeKey, int, EdgeTimeKeyHash> edgeCounts_;
  std::unordered_map<int, std::map<int, int>> tailStartsByLocation_;
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

bool buildIterativeProposalGlobal(
    const Instance& instance, int numAgents, int numTasks,
    const vector<int>& destroyedTasks, const vector<char>& destroyedMask,
    const vector<int>& candidateAgents,
    const vector<int>& initialTouchedAgents,
    const vector<int>& incumbentTaskOwnerByTask,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    NrrFrozenOccupancyIndex& frozenOccupancy,
    vector<vector<int>>& proposedAssignments, ProposalMutationTrace& trace,
    std::string& failureReason);

bool runTemporalPrecheck(
    const Instance& instance, int numTasks,
    const vector<int>& mutableAgents, const vector<int>& destroyedTasks,
    const vector<vector<int>>& proposedAssignments,
    const vector<int>& incumbentCompletion,
    const mapf_pc_lns::nrr::BoundaryWindows& boundaryWindows,
    std::string& failureReason);

}  // namespace lns_nrr_helpers
