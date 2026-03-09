  int numOfIterations_;
  bool debugImprovementDiagnostics_ = false;
  string debugIterationTsvPath_;
  struct NeighborhoodDiagnosticsState {
    boost::unordered_set<uint64_t> seenFingerprints;
    vector<int> previousTasksSorted;
    int repeatStreak = 0;
  };
  NeighborhoodDiagnosticsState neighborhoodDiagnosticsState_;
  vector<IterationDebugRecord> iterationDebugRecords_;
  LowLevelPlannerType lowLevelPlannerType_ = LowLevelPlannerType::mlastar;
  double lowLevelSippsSuboptimality_ = 1.0;
  double lowLevelSegmentTimeout_ = 600.0;
  bool plannerParityCheck_ = false;

  RegretEvalStats regretEvalStatsCurrent_;
  RegretEvalStats regretEvalStatsTotal_;
