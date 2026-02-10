#pragma once

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

  MultiLabelAStarNode(const MultiLabelAStarNode& old) : LLNode(old) {}

  MultiLabelAStarNode(LLNode* parent, int location, int gVal, int hVal,
                      int timestep, int numOfConflicts, unsigned int stage)
      : LLNode(parent, location, gVal, hVal, timestep, numOfConflicts, stage) {}

  ~MultiLabelAStarNode() = default;

  struct NodeHasher {
    size_t operator()(const MultiLabelAStarNode* node) const {
      uint64_t x = 0;
      x ^= (uint64_t)(uint32_t)node->location;
      x ^= ((uint64_t)(uint32_t)node->timestep) << 21;
      x ^= ((uint64_t)(uint32_t)node->stage) << 42;
      x ^= node->waitAtGoal ? 0xD1B54A32D192ED03ULL : 0ULL;
      return (size_t)LLNode::mix64(x);
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
  vector<std::unique_ptr<MultiLabelAStarNode>> allNodesStorage_;

  void releaseNodes();
  void updateFocalList();
  inline MultiLabelAStarNode* popNode();
  inline void pushNode(MultiLabelAStarNode* node);
  void updatePath(const LLNode* goal, Path& path);

  void printSearchTree();

 public:
  MultiLabelSpaceTimeAStar(const Instance& instance, int agent,
                           bool initializeHeuristics = true)
      : SingleAgentSolver(instance, agent, initializeHeuristics) {}
  std::shared_ptr<SingleAgentSolver> cloneForAgent(int agent) const override {
    auto cloned =
        std::make_shared<MultiLabelSpaceTimeAStar>(instance, agent, false);
    copyPlannerStateTo(*cloned);
    return cloned;
  }
  string getName() const override { return "MLAStar"; }
  AgentTaskPath findPathSegment(ConstraintTable& constraintTable, int startTime,
                                int stage, int lb) override;
};
