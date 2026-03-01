  int numOfIterations_;
  bool debugImprovementDiagnostics_ = false;
  string debugIterationTsvPath_;
  unordered_set<uint64_t> acceptedSolutionFingerprints_;
  unordered_set<uint64_t> seenNeighborhoodFingerprints_;
  vector<int> previousNeighborhoodTasksSorted_;
  int currentNeighborhoodRepeatStreak_ = 0;
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
