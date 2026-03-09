#pragma once

#include <deque>
#include <limits>
#include <map>
#include <plog/Log.h>
#include <memory>
#include "astar.hpp"
#include "common.hpp"
#include "constrainttable.hpp"

namespace mlastar_detail {
inline size_t hashNodeIdentity(int location, int timestep, unsigned int stage,
                               bool waitAtGoal) {
  const uint64_t hLocation =
      LLNode::mix64((uint64_t)(uint32_t)location ^ 0x9E3779B97F4A7C15ULL);
  const uint64_t hTimestep =
      LLNode::mix64((uint64_t)(uint32_t)timestep ^ 0xC2B2AE3D27D4EB4FULL);
  const uint64_t hStage =
      LLNode::mix64((uint64_t)(uint32_t)stage ^ 0x165667B19E3779F9ULL);
  const uint64_t hWait =
      waitAtGoal ? 0xD1B54A32D192ED03ULL : 0x94D049BB133111EBULL;
  uint64_t combined = hLocation;
  combined ^=
      hTimestep + 0x9E3779B97F4A7C15ULL + (combined << 6) + (combined >> 2);
  combined ^= hStage + 0x9E3779B97F4A7C15ULL + (combined << 6) +
              (combined >> 2);
  combined ^= hWait + 0x9E3779B97F4A7C15ULL + (combined << 6) +
              (combined >> 2);
  return (size_t)LLNode::mix64(combined);
}
}  // namespace mlastar_detail

// Forward-declare so it can be used as the heap element type before the full
// class definition.
class MultiLabelAStarNode;

// d_ary_heap (array-backed) is used instead of pairing_heap to avoid the
// O(n) recursive merge_first_pair in pairing_heap::pop(), which overflows the
// stack for large search frontiers.
using MlAStarOpenHeap =
    boost::heap::d_ary_heap<MultiLabelAStarNode*,
                            boost::heap::arity<4>,
                            boost::heap::compare<LLNode::OpenCompareNode>,
                            boost::heap::mutable_<true>>;
using MlAStarFocalHeap =
    boost::heap::d_ary_heap<MultiLabelAStarNode*,
                            boost::heap::arity<4>,
                            boost::heap::compare<LLNode::FocalCompareNode>,
                            boost::heap::mutable_<true>>;

class MultiLabelAStarNode : public LLNode {
 public:
  MlAStarOpenHeap::handle_type openHandle;
  MlAStarFocalHeap::handle_type focalHandle;
  bool inFocal = false;
  int indexedFVal = std::numeric_limits<int>::min();
  int indexedFSlot = -1;

  MultiLabelAStarNode() = default;

  MultiLabelAStarNode(LLNode* parent, int location, int gVal, int hVal,
                      int timestep, int numOfConflicts, unsigned int stage)
      : LLNode(parent, location, gVal, hVal, timestep, numOfConflicts, stage) {}

  ~MultiLabelAStarNode() = default;

  struct NodeHasher {
    size_t operator()(const MultiLabelAStarNode* node) const {
      return mlastar_detail::hashNodeIdentity(
          node->location, node->timestep, node->stage, node->waitAtGoal);
    }
  };

  struct CompareNode {
    bool operator()(const MultiLabelAStarNode* lhs,
                    const MultiLabelAStarNode* rhs) const {
      return (lhs == rhs) ||
             ((lhs != nullptr) && (rhs != nullptr) &&
              lhs->location == rhs->location &&
              lhs->timestep == rhs->timestep && lhs->stage == rhs->stage &&
              lhs->waitAtGoal == rhs->waitAtGoal);
    }
  };
};

class MultiLabelSpaceTimeAStar : public SingleAgentSolver {
 private:
  struct NodeKey {
    int location = -1;
    int timestep = 0;
    unsigned int stage = 0;
    bool waitAtGoal = false;

    bool operator==(const NodeKey& other) const {
      return location == other.location && timestep == other.timestep &&
             stage == other.stage && waitAtGoal == other.waitAtGoal;
    }
  };

  struct NodeKeyHasher {
    size_t operator()(const NodeKey& key) const {
      return mlastar_detail::hashNodeIdentity(
          key.location, key.timestep, key.stage, key.waitAtGoal);
    }
  };

  static inline NodeKey makeNodeKey(int location, int timestep,
                                    unsigned int stage, bool waitAtGoal) {
    NodeKey key;
    key.location = location;
    key.timestep = timestep;
    key.stage = stage;
    key.waitAtGoal = waitAtGoal;
    return key;
  }

  int agent_ = UNASSIGNED;
  bool incrementalFocalRefresh_ = false;
  MlAStarOpenHeap openList_;
  MlAStarFocalHeap focalList_;
  // Optional f-value buckets for incremental focal refresh. Stale entries are
  // tolerated and filtered lazily on promotion.
  std::map<int, std::vector<MultiLabelAStarNode*>> openByF_;

  int minFVal_{}, lowerBound_{};

  boost::unordered_map<NodeKey, MultiLabelAStarNode*, NodeKeyHasher>
      allNodesTable_;
  std::deque<MultiLabelAStarNode> allNodesStorage_;

  void releaseNodes();
  void updateFocalList();
  void registerOpenNodeByF(MultiLabelAStarNode* node);
  void unregisterOpenNodeByF(MultiLabelAStarNode* node);
  inline MultiLabelAStarNode* popNode();
  inline void pushNode(MultiLabelAStarNode* node);
  void updatePath(const LLNode* goal, Path& path);

  void printSearchTree();

 public:
  MultiLabelSpaceTimeAStar(const Instance& instance, int agent,
                           bool initializeHeuristics = true,
                           bool incrementalFocalRefresh = true)
      : SingleAgentSolver(instance, agent, initializeHeuristics),
        agent_(agent),
        incrementalFocalRefresh_(incrementalFocalRefresh) {}
  std::shared_ptr<SingleAgentSolver> cloneForAgent(int agent) const override {
    auto cloned =
        std::make_shared<MultiLabelSpaceTimeAStar>(
            instance, agent, false, incrementalFocalRefresh_);
    copyPlannerStateTo(*cloned);
    if (agent != agent_) {
      // Cross-agent clone should keep the target agent's own goal model.
      cloned->setGoalLocations(instance.getTaskLocationsRef());
    }
    return cloned;
  }
  string getName() const override { return "MLAStar"; }
  AgentTaskPath findPathSegment(ConstraintTable& constraintTable, int startTime,
                                int stage, int lb) override;
};
