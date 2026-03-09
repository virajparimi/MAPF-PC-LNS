  double runtime = 0;
  int numOfFailures = 0, sumOfCosts = 0;
  int64_t invalidCandidateRejections = 0;
  vector<IterationStats> iterationStats;
  string initialSolutionStrategy, destroyHeuristic, acceptanceCriteria,
      repairHeuristic,
      regretType;
