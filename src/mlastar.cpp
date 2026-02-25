#include "mlastar.hpp"
#include <chrono>
#include "astar.hpp"
#include "common.hpp"

void MultiLabelSpaceTimeAStar::releaseNodes() {
  // Boost pairing_heap::clear() recursively disposes heap-node trees.
  // Large/deep search frontiers can overflow the stack during teardown.
  // Drain both heaps iteratively instead.
  while (!openList_.empty()) {
    openList_.pop();
  }
  while (!focalList_.empty()) {
    focalList_.pop();
  }
  openByF_.clear();
  allNodesTable_.clear();
  // Release retained bucket capacity from boost::unordered_set to avoid
  // carrying high-water memory across successive searches.
  allNodesTable_.rehash(1);
  allNodesStorage_.clear();
}

void MultiLabelSpaceTimeAStar::registerOpenNodeByF(MultiLabelAStarNode* node) {
  if (!incrementalFocalRefresh_) {
    return;
  }
  const int fVal = node->getFVal();
  node->indexedFVal = fVal;
  openByF_[fVal].push_back(node);
}

void MultiLabelSpaceTimeAStar::pushNode(MultiLabelAStarNode* node) {
  numGenerated++;
  node->inOpenlist = true;
  node->openHandle = openList_.push(node);
  registerOpenNodeByF(node);
  if (node->getFVal() <= lowerBound_) {
    if (!node->inFocal) {
      node->focalHandle = focalList_.push(node);
      node->inFocal = true;
    }
  }
}

MultiLabelAStarNode* MultiLabelSpaceTimeAStar::popNode() {
  while (!focalList_.empty()) {
    MultiLabelAStarNode* node = focalList_.top();
    focalList_.pop();
    node->inFocal = false;
    if (!node->inOpenlist) {
      continue;
    }
    numExpanded++;
    node->inOpenlist = false;
    openList_.erase(node->openHandle);
    return node;
  }
  return nullptr;
}

void MultiLabelSpaceTimeAStar::updateFocalList() {
  if (openList_.empty()) {
    return;
  }
  MultiLabelAStarNode* openHead = openList_.top();
  // Focal list is always supposed to be the set of nodes in open list whose f-value does not exceed the minimum f-value of a node in open list
  if (focalList_.empty() || openHead->getFVal() > minFVal_) {
    int newMinFVal = (int)openHead->getFVal();
    int newLowerBound = max(lowerBound_, newMinFVal);
    const bool repopulatingFromEmpty = focalList_.empty();
    if (!repopulatingFromEmpty && newLowerBound == lowerBound_) {
      // No f-range expansion; nothing can become newly focal.
      minFVal_ = newMinFVal;
      return;
    }

    if (incrementalFocalRefresh_) {
      auto begin = repopulatingFromEmpty ? openByF_.begin()
                                         : openByF_.upper_bound(lowerBound_);
      const auto end = openByF_.upper_bound(newLowerBound);
      for (auto it = begin; it != end; ++it) {
        const int bucketFVal = it->first;
        for (MultiLabelAStarNode* node : it->second) {
          if (node == nullptr || node->inFocal || !node->inOpenlist) {
            continue;
          }
          const int currentFVal = node->getFVal();
          if (node->indexedFVal != bucketFVal || currentFVal != bucketFVal) {
            // Stale bucket entry from a prior in-place decrease-key update.
            continue;
          }
          if (!repopulatingFromEmpty && currentFVal <= lowerBound_) {
            continue;
          }
          if (currentFVal > newLowerBound) {
            continue;
          }
          node->focalHandle = focalList_.push(node);
          node->inFocal = true;
        }
      }
    } else {
      for (MultiLabelAStarNode* node : openList_) {
        const int fVal = node->getFVal();
        const bool newlyEligible =
            (fVal > lowerBound_ && fVal <= newLowerBound);
        const bool eligibleWhenEmpty =
            (repopulatingFromEmpty && fVal <= newLowerBound);
        if ((newlyEligible || eligibleWhenEmpty) && !node->inFocal) {
          node->focalHandle = focalList_.push(node);
          node->inFocal = true;
        }
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
    const int stageGoal = goalLocations[current->stage];
    const bool atStageGoal = (current->location == stageGoal);
    const bool parentAtSameStageGoal =
        current->parent != nullptr && current->parent->stage == current->stage &&
        current->parent->location == stageGoal;
    // Mark the first timestep that reaches the current stage goal.
    path[current->gVal].isGoal = atStageGoal && !parentAtSameStageGoal;
    current = current->parent;
  }
}

AgentTaskPath MultiLabelSpaceTimeAStar::findPathSegment(
    ConstraintTable& constraintTable, int startTime, int stage, int lb) {
  Time::time_point timeStart = Time::now();
  AgentTaskPath path;
  path.beginTime = startTime;
  setLastSearchOutcome(SearchOutcome::unknown);
  if (stage < 0 || stage >= (int)goalLocations.size()) {
    PLOGE << "MLA*: invalid stage " << stage
          << " for goal count " << goalLocations.size() << "\n";
    setLastSearchOutcome(SearchOutcome::invalid_input);
    return path;
  }
  int location = startLocation;
  if (stage != 0) {
    location = goalLocations[stage - 1];
  }

  int holdingTime = constraintTable.lengthMin;
  if (stage == (int)goalLocations.size() - 1) {
    holdingTime = constraintTable.getHoldingTime();
  }

  // Use the precomputed grid heuristic (exact shortest-path distances) rather
  // than Manhattan distance to improve pruning while remaining admissible.
  allNodesStorage_.emplace_back(
      nullptr, location, 0,
      max(getStageGoalDistance(stage, location), holdingTime - startTime),
      startTime, 0, stage);
  auto* start = &allNodesStorage_.back();

  // Ensure that the constraint table is built before we call this
  numGenerated++;
  start->inOpenlist = true;
  allNodesTable_.insert(start);
  minFVal_ = (int)start->getFVal();
  start->secondaryKey = -start->gVal;

  start->openHandle = openList_.push(start);
  registerOpenNodeByF(start);
  start->focalHandle = focalList_.push(start);
  start->inFocal = true;

  lowerBound_ = max(holdingTime - startTime, max(minFVal_, lb));
  const auto timedOut = [&]() -> bool {
    return ((fsec)(Time::now() - timeStart)).count() > segmentTimeoutSec;
  };
  constexpr uint32_t kTimeoutCheckStride = 32;
  uint32_t expansionsSinceTimeoutProbe = 0;

  while (!openList_.empty()) {
    if (timedOut()) {
      releaseNodes();
      setLastSearchOutcome(SearchOutcome::timeout);
      return path;
    }
    updateFocalList();
    MultiLabelAStarNode* current = popNode();
    if (current == nullptr) {
      // Defensive guard: do not dereference an empty focal heap.
      PLOGE << "MLA*: focal list empty while open list non-empty\n";
      releaseNodes();
      setLastSearchOutcome(SearchOutcome::search_exhausted);
      return path;
    }

    if (current->location == goalLocations[stage] &&
        current->timestep >= holdingTime) {
      updatePath(current, path);
      setLastSearchOutcome(SearchOutcome::found);
      break;
    }

    if (current->timestep >= constraintTable.lengthMax) {
      continue;
    }

    // Correctness-first policy: keep explicit timestep progression even beyond
    // the finite constraint horizon. This avoids relying on compressed-time
    // state equivalence between gVal and timestep.

    auto tryExpandSuccessor = [&](int successor, int nextTimestep) {
      if (constraintTable.constrained(successor, nextTimestep) ||
          constraintTable.constrained(current->location, successor,
                                      nextTimestep)) {
        return;
      }

      const unsigned int currentStage = current->stage;
      const int successorGVal = current->gVal + 1;
      const int successorHVal =
          max(getStageGoalDistance(currentStage, successor),
              holdingTime - nextTimestep);
      const int successorInternalConflicts = current->numOfConflicts;
      MultiLabelAStarNode probe(current, successor, successorGVal,
                                successorHVal, nextTimestep,
                                successorInternalConflicts, currentStage);
      probe.secondaryKey = -successorGVal;
      probe.distanceToNext = getStageGoalDistance(currentStage, successor);

      if (probe.stage == goalLocations.size() - 1 &&
          successor == goalLocations.back() &&
          current->location == goalLocations.back()) {
        probe.waitAtGoal = true;
        probe.refreshTieBreaker();
      }

      auto it = allNodesTable_.find(&probe);
      if (it == allNodesTable_.end()) {
        allNodesStorage_.emplace_back(
            current, successor, successorGVal, successorHVal, nextTimestep,
            successorInternalConflicts, currentStage);
        auto* next = &allNodesStorage_.back();
        next->waitAtGoal = probe.waitAtGoal;
        next->refreshTieBreaker();
        pushNode(next);
        allNodesTable_.insert(next);
        return;
      }

      if ((*it)->getFVal() > probe.getFVal() ||
          ((*it)->getFVal() == probe.getFVal() &&
           LLNode::FocalCompareNode()((*it), &probe))) {
        // allNodesTable_ hashes by (location,timestep,stage,waitAtGoal).
        // Keep this key invariant unchanged for in-place updates.
        assert((*it)->location == probe.location);
        assert((*it)->timestep == probe.timestep);
        assert((*it)->stage == probe.stage);
        assert((*it)->waitAtGoal == probe.waitAtGoal);
        probe.inOpenlist = (*it)->inOpenlist;
        if (!(*it)->inOpenlist) {
          // Do not mutate already-expanded nodes in-place; descendants may
          // still reference the old parent chain. Replace the table entry with
          // a fresh node for this state and re-open that node.
          allNodesStorage_.emplace_back(
              current, successor, successorGVal, successorHVal, nextTimestep,
              successorInternalConflicts, currentStage);
          auto* reopened = &allNodesStorage_.back();
          reopened->waitAtGoal = probe.waitAtGoal;
          reopened->refreshTieBreaker();
          allNodesTable_.erase(it);
          allNodesTable_.insert(reopened);
          pushNode(reopened);
        } else {
          bool addToFocal = false, updateInFocal = false, updateOpen = false;
          const int oldFVal = (*it)->getFVal();
          const int newFVal = successorGVal + successorHVal;
          if (newFVal <= lowerBound_) {
            if (oldFVal > lowerBound_) {
              addToFocal = true;
            } else {
              updateInFocal = true;
            }
          }
          if (oldFVal > newFVal) {
            updateOpen = true;
          }
          (*it)->overwriteSearchStateFrom(probe);
          if (oldFVal != newFVal) {
            registerOpenNodeByF(*it);
          }
          if (updateOpen) {
            openList_.increase((*it)->openHandle);
          }
          if (addToFocal) {
            if (!(*it)->inFocal) {
              (*it)->focalHandle = focalList_.push(*it);
              (*it)->inFocal = true;
            } else {
              updateInFocal = true;
            }
          }
          if (updateInFocal && (*it)->inFocal) {
            focalList_.update((*it)->focalHandle);
          }
        }
      }
    };

    const vector<int>& neighbors = instance.getNeighbors(current->location);
    for (int successor : neighbors) {
      expansionsSinceTimeoutProbe++;
      if (expansionsSinceTimeoutProbe >= kTimeoutCheckStride) {
        expansionsSinceTimeoutProbe = 0;
        if (timedOut()) {
          releaseNodes();
          setLastSearchOutcome(SearchOutcome::timeout);
          return path;
        }
      }
      const int nextTimestep = current->timestep + 1;
      tryExpandSuccessor(successor, nextTimestep);
    }

    // We can stay at the same location for the next timestep.
    expansionsSinceTimeoutProbe++;
    if (expansionsSinceTimeoutProbe >= kTimeoutCheckStride) {
      expansionsSinceTimeoutProbe = 0;
      if (timedOut()) {
        releaseNodes();
        setLastSearchOutcome(SearchOutcome::timeout);
        return path;
      }
    }
    const int successor = current->location;
    const int nextTimestep = current->timestep + 1;
    tryExpandSuccessor(successor, nextTimestep);
  }
  if (path.empty() && getLastSearchOutcome() == SearchOutcome::unknown) {
    setLastSearchOutcome(SearchOutcome::search_exhausted);
  }
  releaseNodes();
  return path;
}

void MultiLabelSpaceTimeAStar::printSearchTree() {
  std::cout << "Size of allNodesTable_: " << allNodesTable_.size() << "\n";
  vector<const MultiLabelAStarNode*> allNodesSorted;
  allNodesSorted.reserve(allNodesTable_.size());
  for (const auto* node : allNodesTable_) {
    allNodesSorted.push_back(node);
  }
  std::sort(allNodesSorted.begin(), allNodesSorted.end(),
            [](const MultiLabelAStarNode* lhs, const MultiLabelAStarNode* rhs) {
              return lhs->timestep < rhs->timestep;
            });
  for (const auto* node : allNodesSorted) {
    std::cout << "Location: " << node->location << ", Position: ("
              << std::to_string(instance.getRowCoordinate(node->location))
              << ", "
              << std::to_string(instance.getColCoordinate(node->location))
              << "), G-Val: " << node->gVal << ", H-Val: " << node->hVal
              << ", Timestep: " << node->timestep << "\n";
  }
  std::cout << "\n\n";
}
