#pragma once

#include <deque>
#include <plog/Log.h>
#include <memory>
#include "astar.hpp"
#include "common.hpp"
#include "constrainttable.hpp"

class MultiLabelAStarNode : public LLNode {
 public:
  pairing_heap<MultiLabelAStarNode*,
               compare<LLNode::OpenCompareNode>>::handle_type openHandle;
  pairing_heap<MultiLabelAStarNode*,
               compare<LLNode::FocalCompareNode>>::handle_type focalHandle;

  MultiLabelAStarNode() = default;

  MultiLabelAStarNode(LLNode* parent, int location, int gVal, int hVal,
                      int timestep, int numOfConflicts, unsigned int stage)
      : LLNode(parent, location, gVal, hVal, timestep, numOfConflicts, stage) {}

  ~MultiLabelAStarNode() = default;

  struct NodeHasher {
    size_t operator()(const MultiLabelAStarNode* node) const {
      const uint64_t hLocation =
          LLNode::mix64((uint64_t)(uint32_t)node->location ^
                        0x9E3779B97F4A7C15ULL);
      const uint64_t hTimestep =
          LLNode::mix64((uint64_t)(uint32_t)node->timestep ^
                        0xC2B2AE3D27D4EB4FULL);
      const uint64_t hStage =
          LLNode::mix64((uint64_t)(uint32_t)node->stage ^
                        0x165667B19E3779F9ULL);
      const uint64_t hWait = node->waitAtGoal ? 0xD1B54A32D192ED03ULL
                                              : 0x94D049BB133111EBULL;
      uint64_t combined = hLocation;
      combined ^= hTimestep + 0x9E3779B97F4A7C15ULL + (combined << 6) +
                  (combined >> 2);
      combined ^= hStage + 0x9E3779B97F4A7C15ULL + (combined << 6) +
                  (combined >> 2);
      combined ^= hWait + 0x9E3779B97F4A7C15ULL + (combined << 6) +
                  (combined >> 2);
      return (size_t)LLNode::mix64(combined);
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
  int agent_ = UNASSIGNED;
  pairing_heap<MultiLabelAStarNode*,
               compare<MultiLabelAStarNode::OpenCompareNode>>
      openList_;
  pairing_heap<MultiLabelAStarNode*,
               compare<MultiLabelAStarNode::FocalCompareNode>>
      focalList_;

  int minFVal_{}, lowerBound_{};

  unordered_set<MultiLabelAStarNode*, MultiLabelAStarNode::NodeHasher,
                MultiLabelAStarNode::CompareNode>
      allNodesTable_;
  std::deque<MultiLabelAStarNode> allNodesStorage_;

  void releaseNodes();
  void updateFocalList();
  inline MultiLabelAStarNode* popNode();
  inline void pushNode(MultiLabelAStarNode* node);
  void updatePath(const LLNode* goal, Path& path);

  void printSearchTree();

 public:
  MultiLabelSpaceTimeAStar(const Instance& instance, int agent,
                           bool initializeHeuristics = true)
      : SingleAgentSolver(instance, agent, initializeHeuristics), agent_(agent) {}
  std::shared_ptr<SingleAgentSolver> cloneForAgent(int agent) const override {
    auto cloned =
        std::make_shared<MultiLabelSpaceTimeAStar>(instance, agent, false);
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
