  int numOfIterations_;
  bool debugImprovementDiagnostics_ = false;
  string debugIterationTsvPath_;
  struct NeighborhoodDiagnosticsState {
    unordered_set<uint64_t> seenFingerprints;
    vector<int> previousTasksSorted;
    int repeatStreak = 0;
  };
  NeighborhoodDiagnosticsState neighborhoodDiagnosticsState_;
  vector<IterationDebugRecord> iterationDebugRecords_;
  bool incrementalRegret_ = false;
  LowLevelPlannerType lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  double lowLevelSippsSuboptimality_ = 1.0;
  double lowLevelSegmentTimeout_ = 600.0;
  bool plannerParityCheck_ = false;
  LNSMarketState market_;

  enum class IncrementalRegretMode { descendants, descendants_and_agent };
  IncrementalRegretMode incrementalRegretMode_ =
      IncrementalRegretMode::descendants_and_agent;
  vector<uint32_t> regretStamp_;
  vector<pair<int, int>> regretBestOption_;
  vector<pair<int, int>> regretSecondBestOption_;
  vector<vector<int>> regretCandidateAgents_;

  RegretEvalStats regretEvalStatsCurrent_;
  RegretEvalStats regretEvalStatsTotal_;

  IncrementalRegretStats incrementalRegretStatsCurrent_;
  IncrementalRegretStats incrementalRegretStatsTotal_;
