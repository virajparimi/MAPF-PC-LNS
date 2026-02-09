#include "mlastar.hpp"
#include <chrono>
#include "astar.hpp"
#include "common.hpp"

void MultiLabelSpaceTimeAStar::releaseNodes() {
  openList_.clear();
  focalList_.clear();
  for (MultiLabelAStarNode* node : allNodesTable_) {
    delete node;
  }
  allNodesTable_.clear();
}

inline void MultiLabelSpaceTimeAStar::pushNode(MultiLabelAStarNode* node) {
  numGenerated++;
  node->inOpenlist = true;
  node->openHandle = openList_.push(node);
  if (node->getFVal() <= lowerBound_) {
    node->focalHandle = focalList_.push(node);
  }
}

inline MultiLabelAStarNode* MultiLabelSpaceTimeAStar::popNode() {
  numExpanded++;
  MultiLabelAStarNode* node = focalList_.top();
  focalList_.pop();
  node->inOpenlist = false;
  openList_.erase(node->openHandle);
  return node;
}

void MultiLabelSpaceTimeAStar::updateFocalList() {
  MultiLabelAStarNode* openHead = openList_.top();
  // Focal list is always supposed to be the set of nodes in open list whose f-value does not exceed the minimum f-value of a node in open list
  if (openHead->getFVal() > minFVal_) {
    int newMinFVal = (int)openHead->getFVal();
    int newLowerBound = max(lowerBound_, newMinFVal);
    for (MultiLabelAStarNode* node : openList_) {
      if (node->getFVal() > lowerBound_ && node->getFVal() <= newLowerBound) {
        node->focalHandle = focalList_.push(node);
      }
    }
    minFVal_ = newMinFVal;
    lowerBound_ = newLowerBound;
  }
}

void MultiLabelSpaceTimeAStar::updatePath(const LLNode* goal, Path& path) {
  path.path.resize(goal->gVal + 1);

  const LLNode* current = goal;
  while (current != nullptr) {
    path[current->gVal].location = current->location;
    path[current->gVal].isGoal =
        current->parent != nullptr && current->stage != current->parent->stage;
    current = current->parent;
  }
}

AgentTaskPath MultiLabelSpaceTimeAStar::findPathSegment(
    ConstraintTable& constraintTable, int startTime, int stage, int lb) {
  high_resolution_clock::time_point timeStart = Time::now();
  int location = startLocation;
  if (stage != 0) {
    location = goalLocations[stage - 1];
  }

  AgentTaskPath path;
  path.beginTime = startTime;

  int holdingTime = constraintTable.lengthMin;
  if (stage == (int)goalLocations.size() - 1) {
    holdingTime = constraintTable.getHoldingTime();
  }

  // Use the precomputed grid heuristic (exact shortest-path distances) rather
  // than Manhattan distance to improve pruning while remaining admissible.
  auto* start = new MultiLabelAStarNode(
      nullptr, location, 0,
      max((*heuristic[stage])[location], holdingTime - startTime),
      startTime, 0, stage);

  // Ensure that the constraint table is built before we call this
  numGenerated++;
  start->inOpenlist = true;
  allNodesTable_.insert(start);
  minFVal_ = (int)start->getFVal();

  start->openHandle = openList_.push(start);
  start->focalHandle = focalList_.push(start);
  start->secondaryKey = -start->gVal;

  lowerBound_ = max(holdingTime - startTime, max(minFVal_, lb));

  while (!openList_.empty()) {
    updateFocalList();
    MultiLabelAStarNode* current = popNode();

    if (current->location == goalLocations[stage] &&
        current->timestep >= holdingTime) {
      updatePath(current, path);
      break;
    }

    if (current->timestep >= constraintTable.lengthMax) {
      continue;  // Why is this needed? Probably because this node is not "good" and needs to be discarded
    }

    const vector<int>& neighbors = instance.getNeighbors(current->location);
    for (int successor : neighbors) {
      int nextTimestep = current->timestep + 1;

      if (max(constraintTable.size, constraintTable.latestTimestep) + 1 <
          current->timestep) {
        nextTimestep--;  // What does this do?
      }

      if (constraintTable.constrained(successor, nextTimestep) ||
          constraintTable.constrained(current->location, successor,
                                      nextTimestep)) {
        continue;
      }

      // Setting the stage
      const unsigned int currentStage = current->stage;

      int successorGVal = current->gVal + 1;
      // getHeuristic(stage, successor)
      int successorHVal = max((*heuristic[currentStage])[successor],
                              holdingTime - nextTimestep);
      int successorInternalConflicts = current->numOfConflicts;
      auto* next = new MultiLabelAStarNode(current, successor, successorGVal,
                                           successorHVal, nextTimestep,
                                           successorInternalConflicts,
                                           currentStage);
      next->secondaryKey = -successorGVal;
      next->distanceToNext = (*heuristic[currentStage])[successor];

      if (next->stage == goalLocations.size() - 1 &&
          successor == goalLocations.back() &&
          current->location == goalLocations.back()) {
        next->waitAtGoal = true;
        next->refreshTieBreaker();
      }

      // Try to retrieve it from the hash table
      auto it = allNodesTable_.find(next);
      if (it == allNodesTable_.end()) {
        pushNode(next);
        allNodesTable_.insert(next);
        continue;
      }

      // If we found existing entry then we need to update it but only if its in the open list
      if ((*it)->getFVal() > next->getFVal() ||
          ((*it)->getFVal() == next->getFVal() &&
           LLNode::FocalCompareNode()((*it), next))) {
        if (!(*it)->inOpenlist) {
          (*it)->copy(*next);
          pushNode(*it);
        } else {
          bool addToFocal = false, updateInFocal = false, updateOpen = false;
          // New node can be in focal list
          if ((successorGVal + successorHVal) <= lowerBound_) {
            if ((*it)->getFVal() > lowerBound_) {
              addToFocal = true;  // Old node could not be in focal list
            } else {
              updateInFocal =
                  true;  // Old node could be in focal list so need to update
            }
          }
          if ((*it)->getFVal() > successorGVal + successorHVal) {
            updateOpen =
                true;  // This node would not have been popped yet from the open list
          }
          (*it)->copy(*next);
          if (updateOpen) {
            openList_.increase((*it)->openHandle);
          }
          if (addToFocal) {
            (*it)->focalHandle = focalList_.push(*it);
          }
          if (updateInFocal) {
            focalList_.update((*it)->focalHandle);
          }
        }
      }
      delete next;
    }

    {
      // We can stay at the same location for the next timestep.
      const int successor = current->location;
      int nextTimestep = current->timestep + 1;

      if (max(constraintTable.size, constraintTable.latestTimestep) + 1 <
          current->timestep) {
        continue;
      }

      if (constraintTable.constrained(successor, nextTimestep) ||
          constraintTable.constrained(current->location, successor,
                                      nextTimestep)) {
        continue;
      }

      // Setting the stage
      const unsigned int currentStage = current->stage;

      int successorGVal = current->gVal + 1;
      // getHeuristic(stage, successor)
      int successorHVal = max((*heuristic[currentStage])[successor],
                              holdingTime - nextTimestep);
      int successorInternalConflicts = current->numOfConflicts;
      auto* next = new MultiLabelAStarNode(current, successor, successorGVal,
                                           successorHVal, nextTimestep,
                                           successorInternalConflicts,
                                           currentStage);
      next->secondaryKey = -successorGVal;
      next->distanceToNext = (*heuristic[currentStage])[successor];

      if (next->stage == goalLocations.size() - 1 &&
          successor == goalLocations.back() &&
          current->location == goalLocations.back()) {
        next->waitAtGoal = true;
        next->refreshTieBreaker();
      }

      // Try to retrieve it from the hash table
      auto it = allNodesTable_.find(next);
      if (it == allNodesTable_.end()) {
        pushNode(next);
        allNodesTable_.insert(next);
        continue;
      }

      // If we found existing entry then we need to update it but only if its in the open list
      if ((*it)->getFVal() > next->getFVal() ||
          ((*it)->getFVal() == next->getFVal() &&
           LLNode::FocalCompareNode()((*it), next))) {
        if (!(*it)->inOpenlist) {
          (*it)->copy(*next);
          pushNode(*it);
        } else {
          bool addToFocal = false, updateInFocal = false, updateOpen = false;
          // New node can be in focal list
          if ((successorGVal + successorHVal) <= lowerBound_) {
            if ((*it)->getFVal() > lowerBound_) {
              addToFocal = true;  // Old node could not be in focal list
            } else {
              updateInFocal =
                  true;  // Old node could be in focal list so need to update
            }
          }
          if ((*it)->getFVal() > successorGVal + successorHVal) {
            updateOpen =
                true;  // This node would not have been popped yet from the open list
          }
          (*it)->copy(*next);
          if (updateOpen) {
            openList_.increase((*it)->openHandle);
          }
          if (addToFocal) {
            (*it)->focalHandle = focalList_.push(*it);
          }
          if (updateInFocal) {
            focalList_.update((*it)->focalHandle);
          }
        }
      }
      delete next;
    }
    auto timeEnd = ((fsec)(Time::now() - timeStart)).count();
    if (timeEnd > 600) {
        releaseNodes();
        return path;
    }
  }
  releaseNodes();
  return path;
}

void MultiLabelSpaceTimeAStar::printSearchTree() {
  std::cout << "Size of allNodesTable_: " << allNodesTable_.size() << "\n";
  vector<MultiLabelAStarNode> allNodesSorted;
  for (auto node : allNodesTable_) {
    allNodesSorted.push_back(*node);
  }
  std::sort(allNodesSorted.begin(), allNodesSorted.end(),
            [](const MultiLabelAStarNode& lhs, const MultiLabelAStarNode& rhs) {
              return lhs.timestep < rhs.timestep;
            });
  for (const auto& node : allNodesSorted) {
    std::cout << "Location: " << node.location << ", Position: ("
              << std::to_string(instance.getRowCoordinate(node.location))
              << ", "
              << std::to_string(instance.getColCoordinate(node.location))
              << "), G-Val: " << node.gVal << ", H-Val: " << node.hVal
              << ", Timestep: " << node.timestep << "\n";
  }
  std::cout << "\n\n";
}
