#include "lns.hpp"
#include "lns_repair_internal.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

void LNS::computeRegretForTaskWithAgent(
    TaskRegretPacket regretPacket, RegretWorkspace& workspace,
    vector<pair<int, int>>* precedenceConstraints,
    const TaskBaselineMetrics& baselineMetrics,
    pairing_heap<Utility, compare<Utility::CompareUtilities>>* serviceTimes,
    const vector<int>* assignmentOwnerLookup,
    const vector<int>* assignmentPosLookup) {
  if (runtimeBudgetExhausted()) {
    return;
  }

  regretEvalStatsCurrent_.agentEvaluations++;
  regretEvalStatsTotal_.agentEvaluations++;

  const auto& assignmentsForAgent = workspace.assignments(regretPacket.agent);
  const auto& pathsForAgent = workspace.taskPaths(regretPacket.agent);
  if (assignmentsForAgent.size() != pathsForAgent.size()) {
    PLOGE << "computeRegretForTaskWithAgent: assignment/path size mismatch for "
          << "agent " << regretPacket.agent << "\n";
    return;
  }

  // Compute the first position along the agent's task assignments where we can insert this task
  int firstValidPosition = 0;
  for (int j = (int)assignmentsForAgent.size() - 1; j >= 0; j--) {
    int beginTime = pathsForAgent[j].beginTime;
    int endTime = pathsForAgent[j].endTime();
    if ((regretPacket.earliestTimestep > endTime) ||
        (regretPacket.earliestTimestep <= endTime &&
         regretPacket.earliestTimestep >= beginTime)) {
      firstValidPosition = j + 1;
      break;
    }
  }

  auto candidatePlanner = createLocalPlanner(regretPacket.agent);

  const auto originalConflictIt =
      lnsNeighborhood_.removedTasks.find(regretPacket.task);
  const bool hasOriginalConflict =
      (originalConflictIt != end(lnsNeighborhood_.removedTasks));

  vector<int> allCandidatePositions;
  allCandidatePositions.reserve((int)assignmentsForAgent.size() -
                                firstValidPosition + 1);
  for (int pos = firstValidPosition; pos <= (int)assignmentsForAgent.size();
       pos++) {
    allCandidatePositions.push_back(pos);
  }

  const bool useMarketShortlist =
      (repairHeuristic == "market_shortlist_regret");
  const bool useNormalizedWaitProxyInRegretShortlist =
      (repairHeuristic == "regret");
  const bool useNormalizedSuccessorPressureInRegretShortlist =
      (repairHeuristic == "regret");
  const bool collectShortlistDiagnostics =
      (repairHeuristic == "regret") && regretShortlistDiagnostics_;
  const int candidateCount = (int)allCandidatePositions.size();
  const int shortlistTopK = regretCandidateTopK_;
  const bool shortlistActive =
      shortlistTopK > 0 && shortlistTopK < candidateCount;

  vector<int> candidatePositions = allCandidatePositions;
  if (shortlistActive) {
    regretEvalStatsCurrent_.shortlistAgentEvaluations++;
    regretEvalStatsTotal_.shortlistAgentEvaluations++;

    struct CandidateScore {
      double score = std::numeric_limits<double>::infinity();
      int pos = -1;
    };
    auto buildSortedCandidates = [&](const vector<double>& scores) {
      vector<CandidateScore> ranked;
      ranked.reserve(allCandidatePositions.size());
      for (size_t idx = 0; idx < allCandidatePositions.size(); idx++) {
        ranked.push_back({scores[idx], allCandidatePositions[idx]});
      }
      std::sort(ranked.begin(), ranked.end(), [](const CandidateScore& lhs,
                                                 const CandidateScore& rhs) {
        if (lhs.score == rhs.score) {
          return lhs.pos < rhs.pos;
        }
        return lhs.score < rhs.score;
      });
      return ranked;
    };

    const int task = regretPacket.task;
    const int agent = regretPacket.agent;
    const int taskLocation = instance_.getTaskLocations(task);
    const auto& taskHeuristics = instance_.getHeuristicsRef(task);
    const auto& taskLocations = instance_.getTaskLocationsRef();
    const auto& startLocations = instance_.getStartLocationsRef();
    const auto& staticAncestors = instance_.getAncestorsRef();
    const auto& staticSuccessors = instance_.getSuccessorsRef();
    const auto& agentAssignments = assignmentsForAgent;
    const bool needNormalizedWaitProxyScores =
        useNormalizedWaitProxyInRegretShortlist || collectShortlistDiagnostics;
    const bool needNormalizedSuccessorPressureScores =
        useNormalizedSuccessorPressureInRegretShortlist ||
        collectShortlistDiagnostics;
    vector<double> rawScores(allCandidatePositions.size(),
                             std::numeric_limits<double>::infinity());
    vector<double> waitProxyScores(allCandidatePositions.size(), 0.0);
    vector<double> successorPressureScores(allCandidatePositions.size(), 0.0);
    vector<double> successorDepth1ComponentScores(allCandidatePositions.size(),
                                                  0.0);
    vector<double> successorDepthGt1ComponentScores(allCandidatePositions.size(),
                                                    0.0);
    vector<int> finiteScoreIndices;
    finiteScoreIndices.reserve(allCandidatePositions.size());
    struct ResolvedSuccessorSignal {
      int begin = -1;
      int depth = 1;
    };
    vector<ResolvedSuccessorSignal> successorSignals;
    if (task >= 0 && task < (int)staticSuccessors.size() &&
        task < (int)successorPressureStaticSignalsByTask_.size()) {
      const int taskCount = instance_.getTasksNum();
      AssignmentLookup builtWorkspaceIndex;
      const vector<int>* ownerLookup = assignmentOwnerLookup;
      const vector<int>* posLookup = assignmentPosLookup;
      if (ownerLookup == nullptr || posLookup == nullptr ||
          (int)ownerLookup->size() != taskCount ||
          (int)posLookup->size() != taskCount) {
        builtWorkspaceIndex = buildAssignmentLookup(workspace, taskCount);
        ownerLookup = &builtWorkspaceIndex.owner;
        posLookup = &builtWorkspaceIndex.pos;
      }
      successorSignals.reserve(successorPressureStaticSignalsByTask_[task].size());
      int64_t successorBeginFromPreviousCount = 0;
      int64_t successorPrecedenceClampCount = 0;
      double successorPrecedenceClampDeltaSum = 0.0;
      int64_t successorDepth1Signals = 0;
      int64_t successorDepthGt1Signals = 0;
      for (const auto& staticSignal : successorPressureStaticSignalsByTask_[task]) {
        const int successorTask = staticSignal.successorTask;
        if (successorTask < 0 || successorTask >= instance_.getTasksNum()) {
          continue;
        }
        int successorBegin = -1;
        // Prefer current workspace timing when successor is still assigned.
        const int workspaceSuccessorAgent =
            (successorTask >= 0 && successorTask < (int)ownerLookup->size())
                ? (*ownerLookup)[successorTask]
                : UNASSIGNED;
        const int workspaceSuccessorPos =
            (successorTask >= 0 && successorTask < (int)posLookup->size())
                ? (*posLookup)[successorTask]
                : -1;
        if (workspaceSuccessorAgent != UNASSIGNED && workspaceSuccessorPos >= 0 &&
            workspaceSuccessorAgent >= 0 &&
            workspaceSuccessorAgent < workspace.numAgents() &&
            workspaceSuccessorPos <
                (int)workspace.taskPaths(workspaceSuccessorAgent).size()) {
          const auto& successorPath =
              workspace.taskPaths(workspaceSuccessorAgent)[workspaceSuccessorPos];
          if (!successorPath.empty()) {
            successorBegin = successorPath.beginTime;
          }
        }

        if (successorBegin >= 0) {
          int successorRelease = 0;
          bool hasSuccessorRelease = false;
          auto consumePredecessorRelease = [&](int predecessorTask) {
            bool usedPreviousFallback = false;
            const int predecessorEnd = resolveTaskEndTimeFromMixedState(
                predecessorTask, workspace, *ownerLookup, *posLookup,
                previousSolution_, lnsNeighborhood_, &usedPreviousFallback);
            if (predecessorEnd >= 0) {
              if (usedPreviousFallback) {
                successorBeginFromPreviousCount++;
              }
              successorRelease = max(successorRelease, predecessorEnd + 1);
              hasSuccessorRelease = true;
            }
          };

          if (successorTask >= 0 && successorTask < (int)staticAncestors.size()) {
            for (int predecessorTask : staticAncestors[successorTask]) {
              consumePredecessorRelease(predecessorTask);
            }
          }

          if (workspaceSuccessorAgent != UNASSIGNED && workspaceSuccessorPos > 0 &&
              workspaceSuccessorAgent >= 0 &&
              workspaceSuccessorAgent < workspace.numAgents() &&
              workspaceSuccessorPos - 1 <
                  (int)workspace.assignments(workspaceSuccessorAgent).size()) {
            const int predecessorTask =
                workspace.assignments(workspaceSuccessorAgent)
                                    [workspaceSuccessorPos - 1];
            consumePredecessorRelease(predecessorTask);
          }

          if (hasSuccessorRelease && successorBegin < successorRelease) {
            successorPrecedenceClampCount++;
            successorPrecedenceClampDeltaSum +=
                (double)(successorRelease - successorBegin);
            successorBegin = successorRelease;
          }
        }

        if (successorBegin >= 0) {
          successorSignals.push_back(
              {successorBegin, staticSignal.depth});
          if (staticSignal.depth <= 1) {
            successorDepth1Signals++;
          } else {
            successorDepthGt1Signals++;
          }
        }
      }
      if (needNormalizedSuccessorPressureScores) {
        regretEvalStatsCurrent_.successorPressureDiagPrevFallbackCount +=
            successorBeginFromPreviousCount;
        regretEvalStatsTotal_.successorPressureDiagPrevFallbackCount +=
            successorBeginFromPreviousCount;
        regretEvalStatsCurrent_.successorPressureDiagPrecedenceClampCount +=
            successorPrecedenceClampCount;
        regretEvalStatsTotal_.successorPressureDiagPrecedenceClampCount +=
            successorPrecedenceClampCount;
        regretEvalStatsCurrent_.successorPressureDiagPrecedenceClampDeltaSum +=
            successorPrecedenceClampDeltaSum;
        regretEvalStatsTotal_.successorPressureDiagPrecedenceClampDeltaSum +=
            successorPrecedenceClampDeltaSum;
        regretEvalStatsCurrent_.successorPressureDiagDepth1Signals +=
            successorDepth1Signals;
        regretEvalStatsTotal_.successorPressureDiagDepth1Signals +=
            successorDepth1Signals;
        regretEvalStatsCurrent_.successorPressureDiagDepthGt1Signals +=
            successorDepthGt1Signals;
        regretEvalStatsTotal_.successorPressureDiagDepthGt1Signals +=
            successorDepthGt1Signals;
      }
    }
    bool successorSignalsInformative = true;
    if (needNormalizedSuccessorPressureScores) {
      if (successorSignals.size() < 2) {
        successorSignalsInformative = false;
      } else {
        int minBegin = std::numeric_limits<int>::max();
        int maxBegin = std::numeric_limits<int>::min();
        for (const auto& signal : successorSignals) {
          minBegin = std::min(minBegin, signal.begin);
          maxBegin = std::max(maxBegin, signal.begin);
        }
        if (maxBegin <= minBegin) {
          successorSignalsInformative = false;
        }
      }
    }
    const bool useSuccessorPressureInThisEval =
        needNormalizedSuccessorPressureScores && successorSignalsInformative;
    double repairPriceScale = 0.0;
    if (useMarketShortlist) {
      if (market_.repairNormalizeByObservedPrice) {
        repairPriceScale = market_.stats.maxPrice;
        if (repairPriceScale <= 1e-12) {
          for (const auto& kv : market_.vertexPrices) {
            repairPriceScale = std::max(repairPriceScale, kv.second);
          }
        }
      }
      if (repairPriceScale <= 1e-12) {
        repairPriceScale = (market_.priceCap > 0.0) ? market_.priceCap : 1.0;
      }
    }

    for (size_t idx = 0; idx < allCandidatePositions.size(); idx++) {
      const int pos = allCandidatePositions[idx];
      const int prevLocation =
          (pos == 0) ? startLocations[agent]
                     : taskLocations[agentAssignments[pos - 1]];
      const int dPrevTask = taskHeuristics[prevLocation];
      double score = std::numeric_limits<double>::infinity();
      int estimatedArrival = 0;
      bool hasEstimatedArrival = false;
      if (dPrevTask < MAX_TIMESTEP) {
        const int prevEndTime = (pos == 0) ? 0 : pathsForAgent[pos - 1].endTime();
        estimatedArrival = prevEndTime + dPrevTask;
        hasEstimatedArrival = true;
        if (pos < (int)agentAssignments.size()) {
          const int nextTask = agentAssignments[pos];
          const auto& nextHeuristics = instance_.getHeuristicsRef(nextTask);
          const int dPrevNext = nextHeuristics[prevLocation];
          const int dTaskNext = nextHeuristics[taskLocation];
          if (dPrevNext < MAX_TIMESTEP && dTaskNext < MAX_TIMESTEP) {
            score = (double)dPrevTask + (double)dTaskNext - (double)dPrevNext;
          }
        } else {
          score = (double)dPrevTask;
        }
      }

      if (useMarketShortlist && std::isfinite(score) && hasEstimatedArrival) {
        const int bucket = marketTimeBucket(estimatedArrival);
        const uint64_t vertexKey = makeMarketVertexKey(taskLocation, bucket);
        double vertexPrice = 0.0;
        const auto priceIt = market_.vertexPrices.find(vertexKey);
        if (priceIt != market_.vertexPrices.end()) {
          vertexPrice = priceIt->second;
        }
        const double normalizedPrice =
            (repairPriceScale > 0.0) ? (vertexPrice / repairPriceScale)
                                     : vertexPrice;
        // Under feasibility filtering, candidates are already at/after earliest
        // release. Use lateness beyond release to retain a meaningful
        // precedence signal for shortlist ranking.
        const int waitProxy =
            max(0, estimatedArrival - regretPacket.earliestTimestep);
        const double waitProxyBuckets =
            (market_.bucketDt > 0) ? (double)waitProxy / (double)market_.bucketDt
                                   : (double)waitProxy;
        score += market_.destroyWeightPrice * normalizedPrice +
                 market_.destroyWeightWait * waitProxyBuckets;
      }

      rawScores[idx] = score;
      if (std::isfinite(score)) {
        finiteScoreIndices.push_back((int)idx);
        if (needNormalizedWaitProxyScores && hasEstimatedArrival) {
          waitProxyScores[idx] =
              (double)max(0, estimatedArrival - regretPacket.earliestTimestep);
        }
        if (useSuccessorPressureInThisEval && hasEstimatedArrival) {
          double pressure = 0.0;
          double depth1Pressure = 0.0;
          double depthGt1Pressure = 0.0;
          for (const auto& successorSignal : successorSignals) {
            const int successorDelay =
                max(0, (estimatedArrival + 1) - successorSignal.begin);
            if (successorDelay <= 0) {
              continue;
            }
            const double delayContribution = (double)successorDelay;
            pressure += delayContribution;
            if (successorSignal.depth <= 1) {
              depth1Pressure += delayContribution;
            } else {
              depthGt1Pressure += delayContribution;
            }
          }
          successorPressureScores[idx] = pressure;
          successorDepth1ComponentScores[idx] = depth1Pressure;
          successorDepthGt1ComponentScores[idx] = depthGt1Pressure;
        }
      }
    }

    vector<double> combinedScores = rawScores;
    vector<double> successorOnlyScores = rawScores;
    vector<double> normalizedDistanceScores;
    vector<double> normalizedWaitProxyScores;
    vector<double> normalizedSuccessorPressureScores;
    if (needNormalizedWaitProxyScores) {
      normalizeSeriesByRobustScale(rawScores, finiteScoreIndices,
                                   normalizedDistanceScores);
      normalizeSeriesByRobustScale(waitProxyScores, finiteScoreIndices,
                                   normalizedWaitProxyScores);
    }
    if (useSuccessorPressureInThisEval) {
      if (normalizedDistanceScores.empty()) {
        normalizeSeriesByRobustScale(rawScores, finiteScoreIndices,
                                     normalizedDistanceScores);
      }
      normalizeSeriesByRobustScale(successorPressureScores, finiteScoreIndices,
                                   normalizedSuccessorPressureScores);
    } else if (needNormalizedSuccessorPressureScores) {
      if (normalizedDistanceScores.empty()) {
        normalizeSeriesByRobustScale(rawScores, finiteScoreIndices,
                                     normalizedDistanceScores);
      }
      normalizedSuccessorPressureScores.assign(rawScores.size(), 0.0);
    }
    for (int idx : finiteScoreIndices) {
      if (!normalizedDistanceScores.empty()) {
        combinedScores[idx] = normalizedDistanceScores[idx];
        successorOnlyScores[idx] = normalizedDistanceScores[idx];
      }
      if (needNormalizedWaitProxyScores) {
        combinedScores[idx] += normalizedWaitProxyScores[idx];
      }
      if (useSuccessorPressureInThisEval) {
        combinedScores[idx] += normalizedSuccessorPressureScores[idx];
        successorOnlyScores[idx] += normalizedSuccessorPressureScores[idx];
      }
    }

    vector<CandidateScore> rankedRaw = buildSortedCandidates(rawScores);
    vector<CandidateScore> rankedCombined;
    vector<CandidateScore> rankedSuccessorOnly;
    if (needNormalizedWaitProxyScores || needNormalizedSuccessorPressureScores) {
      rankedCombined = buildSortedCandidates(combinedScores);
    }
    if (needNormalizedSuccessorPressureScores) {
      rankedSuccessorOnly = useSuccessorPressureInThisEval
                                ? buildSortedCandidates(successorOnlyScores)
                                : rankedRaw;
    }

    if (collectShortlistDiagnostics) {
      regretEvalStatsCurrent_.waitProxyDiagEvaluations++;
      regretEvalStatsTotal_.waitProxyDiagEvaluations++;

      const int finiteCount = (int)finiteScoreIndices.size();
      regretEvalStatsCurrent_.waitProxyDiagFiniteCandidates += finiteCount;
      regretEvalStatsTotal_.waitProxyDiagFiniteCandidates += finiteCount;

      int nonZeroWaitCandidates = 0;
      bool hasPositiveWait = false;
      bool hasWaitVariation = false;
      bool hasActiveNormalizedWait = false;
      double minWait = std::numeric_limits<double>::infinity();
      double maxWait = -std::numeric_limits<double>::infinity();
      double meanAbsWaitZ = 0.0;
      double meanAbsDistanceZ = 0.0;

      for (int idx : finiteScoreIndices) {
        const double waitProxy = waitProxyScores[idx];
        if (waitProxy > 1e-9) {
          nonZeroWaitCandidates++;
          hasPositiveWait = true;
        }
        minWait = std::min(minWait, waitProxy);
        maxWait = std::max(maxWait, waitProxy);
        if (needNormalizedWaitProxyScores) {
          const double absWaitZ = std::abs(normalizedWaitProxyScores[idx]);
          const double absDistZ = std::abs(normalizedDistanceScores[idx]);
          meanAbsWaitZ += absWaitZ;
          meanAbsDistanceZ += absDistZ;
          if (absWaitZ > 1e-9) {
            hasActiveNormalizedWait = true;
          }
        }
      }

      if (finiteCount > 0) {
        meanAbsWaitZ /= (double)finiteCount;
        meanAbsDistanceZ /= (double)finiteCount;
      }
      if (finiteCount > 1 && maxWait - minWait > 1e-9) {
        hasWaitVariation = true;
      }

      regretEvalStatsCurrent_.waitProxyDiagNonZeroCandidates +=
          nonZeroWaitCandidates;
      regretEvalStatsTotal_.waitProxyDiagNonZeroCandidates +=
          nonZeroWaitCandidates;
      regretEvalStatsCurrent_.waitProxyDiagMeanAbsWaitZSum += meanAbsWaitZ;
      regretEvalStatsTotal_.waitProxyDiagMeanAbsWaitZSum += meanAbsWaitZ;
      regretEvalStatsCurrent_.waitProxyDiagMeanAbsDistanceZSum +=
          meanAbsDistanceZ;
      regretEvalStatsTotal_.waitProxyDiagMeanAbsDistanceZSum +=
          meanAbsDistanceZ;
      if (hasPositiveWait) {
        regretEvalStatsCurrent_.waitProxyDiagPositiveEvals++;
        regretEvalStatsTotal_.waitProxyDiagPositiveEvals++;
      }
      if (hasWaitVariation) {
        regretEvalStatsCurrent_.waitProxyDiagVaryingEvals++;
        regretEvalStatsTotal_.waitProxyDiagVaryingEvals++;
      }
      if (hasActiveNormalizedWait) {
        regretEvalStatsCurrent_.waitProxyDiagNormalizedActiveEvals++;
        regretEvalStatsTotal_.waitProxyDiagNormalizedActiveEvals++;
      }

      if (!rankedRaw.empty() && !rankedCombined.empty() &&
          rankedRaw.front().pos != rankedCombined.front().pos) {
        regretEvalStatsCurrent_.waitProxyDiagTop1Changed++;
        regretEvalStatsTotal_.waitProxyDiagTop1Changed++;
      }

      const int topK = std::min(shortlistTopK, (int)rankedRaw.size());
      if (topK > 0 && (int)rankedCombined.size() >= topK) {
        vector<char> inRawTopK(assignmentsForAgent.size() + 1, 0);
        for (int i = 0; i < topK; i++) {
          const int pos = rankedRaw[i].pos;
          if (pos >= 0 && pos < (int)inRawTopK.size()) {
            inRawTopK[pos] = 1;
          }
        }
        int overlap = 0;
        for (int i = 0; i < topK; i++) {
          const int pos = rankedCombined[i].pos;
          if (pos >= 0 && pos < (int)inRawTopK.size() && inRawTopK[pos]) {
            overlap++;
          }
        }
        if (overlap < topK) {
          regretEvalStatsCurrent_.waitProxyDiagTopKChanged++;
          regretEvalStatsTotal_.waitProxyDiagTopKChanged++;
        }
        const double overlapFrac = (double)overlap / (double)topK;
        regretEvalStatsCurrent_.waitProxyDiagTopKOverlapFracSum += overlapFrac;
        regretEvalStatsTotal_.waitProxyDiagTopKOverlapFracSum += overlapFrac;
      }

      if (needNormalizedSuccessorPressureScores) {
        regretEvalStatsCurrent_.successorPressureDiagEvaluations++;
        regretEvalStatsTotal_.successorPressureDiagEvaluations++;
        regretEvalStatsCurrent_.successorPressureDiagFiniteCandidates += finiteCount;
        regretEvalStatsTotal_.successorPressureDiagFiniteCandidates += finiteCount;

        int nonZeroSuccessorCandidates = 0;
        bool hasPositiveSuccessor = false;
        bool hasSuccessorVariation = false;
        bool hasActiveNormalizedSuccessor = false;
        bool hasActiveDescendantContribution = false;
        double minSuccessor = std::numeric_limits<double>::infinity();
        double maxSuccessor = -std::numeric_limits<double>::infinity();
        double meanAbsSuccessorZ = 0.0;
        double meanAbsDistanceZForSuccessor = 0.0;
        double meanDepth1Contribution = 0.0;
        double meanDepthGt1Contribution = 0.0;

        for (int idx : finiteScoreIndices) {
          const double successorPressure = successorPressureScores[idx];
          meanDepth1Contribution += successorDepth1ComponentScores[idx];
          meanDepthGt1Contribution += successorDepthGt1ComponentScores[idx];
          if (successorDepthGt1ComponentScores[idx] > 1e-9) {
            hasActiveDescendantContribution = true;
          }
          if (successorPressure > 1e-9) {
            nonZeroSuccessorCandidates++;
            hasPositiveSuccessor = true;
          }
          minSuccessor = std::min(minSuccessor, successorPressure);
          maxSuccessor = std::max(maxSuccessor, successorPressure);

          const double absSuccessorZ =
              std::abs(normalizedSuccessorPressureScores[idx]);
          const double absDistZ = std::abs(normalizedDistanceScores[idx]);
          meanAbsSuccessorZ += absSuccessorZ;
          meanAbsDistanceZForSuccessor += absDistZ;
          if (absSuccessorZ > 1e-9) {
            hasActiveNormalizedSuccessor = true;
          }
        }

        if (finiteCount > 0) {
          meanAbsSuccessorZ /= (double)finiteCount;
          meanAbsDistanceZForSuccessor /= (double)finiteCount;
          meanDepth1Contribution /= (double)finiteCount;
          meanDepthGt1Contribution /= (double)finiteCount;
        }
        if (finiteCount > 1 && maxSuccessor - minSuccessor > 1e-9) {
          hasSuccessorVariation = true;
        }

        regretEvalStatsCurrent_.successorPressureDiagNonZeroCandidates +=
            nonZeroSuccessorCandidates;
        regretEvalStatsTotal_.successorPressureDiagNonZeroCandidates +=
            nonZeroSuccessorCandidates;
        regretEvalStatsCurrent_.successorPressureDiagMeanAbsPressureZSum +=
            meanAbsSuccessorZ;
        regretEvalStatsTotal_.successorPressureDiagMeanAbsPressureZSum +=
            meanAbsSuccessorZ;
        regretEvalStatsCurrent_.successorPressureDiagMeanAbsDistanceZSum +=
            meanAbsDistanceZForSuccessor;
        regretEvalStatsTotal_.successorPressureDiagMeanAbsDistanceZSum +=
            meanAbsDistanceZForSuccessor;
        regretEvalStatsCurrent_.successorPressureDiagMeanDepth1ContributionSum +=
            meanDepth1Contribution;
        regretEvalStatsTotal_.successorPressureDiagMeanDepth1ContributionSum +=
            meanDepth1Contribution;
        regretEvalStatsCurrent_.successorPressureDiagMeanDepthGt1ContributionSum +=
            meanDepthGt1Contribution;
        regretEvalStatsTotal_.successorPressureDiagMeanDepthGt1ContributionSum +=
            meanDepthGt1Contribution;
        if (hasPositiveSuccessor) {
          regretEvalStatsCurrent_.successorPressureDiagPositiveEvals++;
          regretEvalStatsTotal_.successorPressureDiagPositiveEvals++;
        }
        if (hasSuccessorVariation) {
          regretEvalStatsCurrent_.successorPressureDiagVaryingEvals++;
          regretEvalStatsTotal_.successorPressureDiagVaryingEvals++;
        }
        if (hasActiveNormalizedSuccessor) {
          regretEvalStatsCurrent_.successorPressureDiagNormalizedActiveEvals++;
          regretEvalStatsTotal_.successorPressureDiagNormalizedActiveEvals++;
        }
        if (hasActiveDescendantContribution) {
          regretEvalStatsCurrent_.successorPressureDiagDescendantActiveEvals++;
          regretEvalStatsTotal_.successorPressureDiagDescendantActiveEvals++;
        }

        if (!rankedRaw.empty() && !rankedSuccessorOnly.empty() &&
            rankedRaw.front().pos != rankedSuccessorOnly.front().pos) {
          regretEvalStatsCurrent_.successorPressureDiagTop1Changed++;
          regretEvalStatsTotal_.successorPressureDiagTop1Changed++;
        }

        if (topK > 0 && (int)rankedSuccessorOnly.size() >= topK) {
          vector<char> inRawTopKForSuccessor(assignmentsForAgent.size() + 1, 0);
          for (int i = 0; i < topK; i++) {
            const int pos = rankedRaw[i].pos;
            if (pos >= 0 && pos < (int)inRawTopKForSuccessor.size()) {
              inRawTopKForSuccessor[pos] = 1;
            }
          }
          int overlapForSuccessor = 0;
          for (int i = 0; i < topK; i++) {
            const int pos = rankedSuccessorOnly[i].pos;
            if (pos >= 0 && pos < (int)inRawTopKForSuccessor.size() &&
                inRawTopKForSuccessor[pos]) {
              overlapForSuccessor++;
            }
          }
          if (overlapForSuccessor < topK) {
            regretEvalStatsCurrent_.successorPressureDiagTopKChanged++;
            regretEvalStatsTotal_.successorPressureDiagTopKChanged++;
          }
          const double overlapFracForSuccessor =
              (double)overlapForSuccessor / (double)topK;
          regretEvalStatsCurrent_.successorPressureDiagTopKOverlapFracSum +=
              overlapFracForSuccessor;
          regretEvalStatsTotal_.successorPressureDiagTopKOverlapFracSum +=
              overlapFracForSuccessor;
        }
      }
    }

    const bool shortlistUsesCombinedScores =
        useNormalizedWaitProxyInRegretShortlist ||
        useSuccessorPressureInThisEval;
    vector<CandidateScore> scored = shortlistUsesCombinedScores
                                        ? std::move(rankedCombined)
                                        : std::move(rankedRaw);

    candidatePositions.clear();
    candidatePositions.reserve(shortlistTopK);
    for (int i = 0; i < shortlistTopK; i++) {
      candidatePositions.push_back(scored[i].pos);
    }
  }

  int feasibleCandidatesForAgent = 0;
  auto evaluateCandidatePositions = [&](const vector<int>& positions) {
    for (int pos : positions) {
      if (runtimeBudgetExhausted()) {
        return;
      }

      regretEvalStatsCurrent_.candidateInsertionsTried++;
      regretEvalStatsTotal_.candidateInsertionsTried++;

      if (hasOriginalConflict &&
          originalConflictIt->second.agent == regretPacket.agent &&
          originalConflictIt->second.taskPosition == pos) {
        // We dont want to compute regret for the same agent, task positions
        // that led to the original conflict!
        continue;
      }
      regretPacket.taskPosition = pos;

      std::variant<bool, Utility> insertCulmination =
          insertTask(regretPacket, workspace, precedenceConstraints,
                     &baselineMetrics, candidatePlanner.get(), true);
      if (std::holds_alternative<Utility>(insertCulmination)) {
        regretEvalStatsCurrent_.candidateInsertionsFeasible++;
        regretEvalStatsTotal_.candidateInsertionsFeasible++;
        feasibleCandidatesForAgent++;
        serviceTimes->push(std::get<Utility>(insertCulmination));
      }
    }
  };

  evaluateCandidatePositions(candidatePositions);

  // Correctness-first guard: if market shortlist mode misses all feasible
  // candidates for this agent, fall back to the remaining insertion positions.
  if (shortlistActive && useMarketShortlist && feasibleCandidatesForAgent == 0) {
    vector<char> inShortlist(assignmentsForAgent.size() + 1, 0);
    for (int pos : candidatePositions) {
      if (pos >= 0 && pos < (int)inShortlist.size()) {
        inShortlist[pos] = 1;
      }
    }
    vector<int> fallbackPositions;
    fallbackPositions.reserve(allCandidatePositions.size() -
                              candidatePositions.size());
    for (int pos : allCandidatePositions) {
      if (pos >= 0 && pos < (int)inShortlist.size() && !inShortlist[pos]) {
        fallbackPositions.push_back(pos);
      }
    }

    if (!fallbackPositions.empty()) {
      regretEvalStatsCurrent_.shortlistFallbackEvaluations++;
      regretEvalStatsTotal_.shortlistFallbackEvaluations++;
      evaluateCandidatePositions(fallbackPositions);
      if (feasibleCandidatesForAgent > 0) {
        regretEvalStatsCurrent_.shortlistFallbackRecovered++;
        regretEvalStatsTotal_.shortlistFallbackRecovered++;
      }
    }
  }
}
