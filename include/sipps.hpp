#pragma once

#include "astar.hpp"

class MultiLabelSIPPS : public SingleAgentSolver {
 public:
  MultiLabelSIPPS(const Instance& instance, int agent,
                  bool plannerParityCheck = false,
                  int plannerParityMaxLogs = 10)
      : SingleAgentSolver(instance, agent),
        agent_(agent),
        plannerParityCheck_(plannerParityCheck),
        plannerParityMaxLogs_(plannerParityMaxLogs) {}
  string getName() const override { return "SIPPS"; }
  AgentTaskPath findPathSegment(ConstraintTable& constraintTable, int startTime,
                                int stage, int lb) override;

 private:
  int agent_;
  bool plannerParityCheck_ = false;
  int plannerParityMaxLogs_ = 10;
};
