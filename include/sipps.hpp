#pragma once

#include "astar.hpp"

class MultiLabelSIPPS : public SingleAgentSolver {
 public:
  MultiLabelSIPPS(const Instance& instance, int agent,
                  bool plannerParityCheck = false,
                  int plannerParityMaxLogs = 10,
                  bool initializeHeuristics = true)
      : SingleAgentSolver(instance, agent, initializeHeuristics),
        agent_(agent),
        plannerParityCheck_(plannerParityCheck),
        plannerParityMaxLogs_(plannerParityMaxLogs) {}
  std::shared_ptr<SingleAgentSolver> cloneForAgent(int agent) const override {
    auto cloned = std::make_shared<MultiLabelSIPPS>(
        instance, agent, plannerParityCheck_, plannerParityMaxLogs_, false);
    copyPlannerStateTo(*cloned);
    if (agent != agent_) {
      // Cross-agent clone should keep the target agent's own goal model.
      cloned->setGoalLocations(instance.getTaskLocationsRef());
    }
    cloned->setPlannerParityLogsEmitted(plannerParityLogsEmitted_);
    return cloned;
  }
  void copyStateFrom(const SingleAgentSolver& other) override {
    SingleAgentSolver::copyStateFrom(other);
    const auto* sippsOther = dynamic_cast<const MultiLabelSIPPS*>(&other);
    if (sippsOther != nullptr) {
      plannerParityLogsEmitted_ = sippsOther->plannerParityLogsEmitted_;
    }
  }
  string getName() const override { return "SIPPS"; }
  bool isPlannerParityCheckEnabled() const { return plannerParityCheck_; }
  int getPlannerParityMaxLogs() const { return plannerParityMaxLogs_; }
  int getPlannerParityLogsEmitted() const { return plannerParityLogsEmitted_; }
  void setPlannerParityLogsEmitted(int emitted) {
    plannerParityLogsEmitted_ = max(0, emitted);
  }
  AgentTaskPath findPathSegment(ConstraintTable& constraintTable, int startTime,
                                int stage, int lb) override;

 private:
  int agent_;
  bool plannerParityCheck_ = false;
  int plannerParityMaxLogs_ = 10;
  int plannerParityLogsEmitted_ = 0;
};
